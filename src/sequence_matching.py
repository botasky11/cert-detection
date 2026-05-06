"""
基于行为序列匹配的内部威胁检测算法 (v3 — 数据驱动改进).

经过对 CMU-CERT r4.2 全量数据 token 频率的分析, 我们得到了
"恶意/良性" 比例最显著的几类信号 (倍率):

    HTTP_LEAK_AH      51.5x   <-- 超强
    HTTP_LEAK_WH      39.2x   <-- 超强
    HTTP_KEYLOG_WH    37.9x   <-- 超强
    USB_CONN_AH        5.3x   <-- 强
    FILE_EXEC_AH       3.5x   <-- 强
    LOGON_AH_OTHER     1.7x   <-- 中
    HTTP_JOBHUNT_WH    1.1x   <-- 几乎无区分度 (83% 良性用户也访问)

模型由 4 个互补检测器加权融合, 全部"越大越可疑":
    1) PatternRule        —— 经过区分度加权的 N-gram 模板匹配
    2) NGramAnomaly       —— 良性用户 n-gram 语言模型 perplexity
    3) SequenceSimilarity —— 与恶意签名序列的 LCS / 高敏感命中
    4) PeerDeviation      —— 高判别力 token 的 IDF 加权稀有度
"""
import math
from collections import Counter, defaultdict


# ============================================================
# 1) Token 区分度: 数据驱动的 token-level "威胁强度" 权重
#    数值 = log(P(t|malicious) / P(t|benign)) 的近似
# ============================================================
TOKEN_THREAT_WEIGHT = {
    # 真正稀有, 极强信号
    'HTTP_LEAK_AH':       8.0,
    'HTTP_LEAK_WH':       8.0,
    'HTTP_KEYLOG_WH':     8.0,
    'HTTP_KEYLOG_AH':     8.0,
    # 中等强度
    'USB_CONN_AH':        2.5,
    'USB_DISC_AH':        1.5,
    'FILE_EXEC_AH':       2.0,
    'FILE_DOC_AH':        1.5,
    'LOGON_AH_OTHER':     2.0,
    'LOGON_AH_OWN':       0.8,
    # 弱信号
    'HTTP_JOBHUNT_AH':    0.7,
    'HTTP_JOBHUNT_WH':    0.0,        # 等同噪声 (83% 用户都有)
    'EMAIL_EXT_ATT_AH':   0.5,
    'EMAIL_EXT_ATT_WH':   0.0,
    'FILE_EXEC_WH':       0.3,
    'FILE_DOC_WH':        0.0,
    'FILE_OTH_AH':        0.3,
    'EMAIL_EXT_AH':       0.3,
}

HIGH_SENSITIVE = {
    'HTTP_LEAK_AH', 'HTTP_LEAK_WH',
    'HTTP_KEYLOG_AH', 'HTTP_KEYLOG_WH',
}

# ------------------------------------------------------------
# 红队场景的"行为剧本" N-gram 模板
# ------------------------------------------------------------
SCENARIO_TEMPLATES = {
    # 场景 1: 下班后用 U 盘 + wikileaks 上传 (Sabotage / Leak)
    'S1_LEAK_VIA_USB': [
        ['LOGON_AH_OWN',   'USB_CONN_AH', 'FILE_DOC_AH'],
        ['LOGON_AH_OWN',   'USB_CONN_AH', 'HTTP_LEAK_AH'],
        ['LOGON_AH_OTHER', 'USB_CONN_AH', 'FILE_DOC_AH'],
        ['LOGON_AH_OTHER', 'USB_CONN_AH', 'FILE_EXEC_AH'],
        ['USB_CONN_AH',    'FILE_DOC_AH', 'HTTP_LEAK_AH'],
        ['USB_CONN_AH',    'FILE_EXEC_AH','HTTP_LEAK_AH'],
        ['HTTP_LEAK_AH',   'USB_CONN_AH'],
        ['USB_CONN_AH',    'HTTP_LEAK_AH'],
    ],
    # 场景 2: 求职 + 离职前数据外发 (注意 JOBHUNT 区分度低,
    #          所以更看重 USB / FILE / 外发邮件 与之的组合)
    'S2_DEPARTING_DATA_THEFT': [
        ['HTTP_JOBHUNT_AH', 'FILE_DOC_WH'],
        ['HTTP_JOBHUNT_AH', 'USB_CONN_WH'],
        ['HTTP_JOBHUNT_AH', 'EMAIL_EXT_ATT_WH'],
        ['USB_CONN_WH',     'FILE_DOC_WH', 'EMAIL_EXT_ATT_WH'],
        ['HTTP_JOBHUNT_WH', 'USB_CONN_WH', 'FILE_DOC_WH'],
        ['HTTP_JOBHUNT_WH', 'FILE_DOC_WH', 'EMAIL_EXT_ATT_WH'],
    ],
    # 场景 3: 系统管理员安装键盘记录器
    'S3_SYSADMIN_KEYLOG': [
        ['HTTP_KEYLOG_WH', 'FILE_EXEC_WH'],
        ['HTTP_KEYLOG_AH', 'FILE_EXEC_AH'],
        ['HTTP_KEYLOG_WH'],
        ['HTTP_KEYLOG_AH'],
        ['LOGON_AH_OTHER', 'FILE_EXEC_AH'],
        ['HTTP_KEYLOG_WH', 'FILE_EXEC_WH', 'EMAIL_EXT_WH'],
        ['HTTP_KEYLOG_WH', 'EMAIL_EXT_WH'],
    ],
}


# ============================================================
# 工具函数
# ============================================================
def ngrams(seq, n):
    if len(seq) < n:
        return []
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def subsequence_count(seq, pattern, window=15):
    """有序、不重叠子序列匹配 (最大窗口=window)."""
    n, m = len(seq), len(pattern)
    if m == 0 or n < m:
        return 0
    count = 0
    i = 0
    while i <= n - m:
        end = min(n, i + window)
        idx = 0
        last = -1
        for j in range(i, end):
            if seq[j] == pattern[idx]:
                idx += 1
                last = j
                if idx == m:
                    count += 1
                    i = last + 1
                    break
        else:
            i += 1
            continue
        if idx < m:
            i += 1
    return count


def _lcs_length(a, b):
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ============================================================
# 检测器 1: PatternRuleDetector
# ============================================================
class PatternRuleDetector:
    """
    专家规则: 对每个场景模板做有序子序列匹配, 命中数取 log(1+x).
    模式权重 = (长度^1.5) * Σ token 区分度 (TOKEN_THREAT_WEIGHT).
    """

    def __init__(self, templates=None, window=15):
        self.templates = templates or SCENARIO_TEMPLATES
        self.window = window

    def _pattern_weight(self, pattern):
        len_w = len(pattern) ** 1.5
        thr_w = sum(TOKEN_THREAT_WEIGHT.get(t, 0.1) for t in pattern)
        return len_w * (1.0 + thr_w)

    def score(self, seq):
        details = {}
        total = 0.0
        for sid, pats in self.templates.items():
            s = 0.0
            for p in pats:
                hits = subsequence_count(seq, p, window=self.window)
                if hits == 0:
                    continue
                s += math.log(1 + hits) * self._pattern_weight(p)
            details[sid] = s
            total += s
        return total, details


# ============================================================
# 检测器 2: NGramAnomalyDetector
# ============================================================
class NGramAnomalyDetector:
    """
    基于良性用户 N-gram 语言模型的 perplexity 异常检测。
    被测序列 perplexity 越高 → 越偏离良性群体 → 越可疑。
    """

    def __init__(self, n=3, smoothing=0.5, sample_per_user=2000):
        self.n = n
        self.alpha = smoothing
        self.sample_per_user = sample_per_user
        self.context_counter = Counter()
        self.full_counter = Counter()
        self.vocab = set()
        self.fitted = False

    def fit(self, sequences):
        for seq in sequences:
            seq = list(seq)
            if len(seq) > self.sample_per_user:
                seq = seq[-self.sample_per_user:]
            self.vocab.update(seq)
            for ng in ngrams(seq, self.n):
                self.full_counter[ng] += 1
                self.context_counter[ng[:-1]] += 1
        self.V = max(len(self.vocab), 1)
        self.fitted = True
        return self

    def perplexity(self, seq):
        if not self.fitted or len(seq) < self.n:
            return 1.0
        seq = list(seq)
        if len(seq) > 5000:
            seq = seq[-5000:]
        log_p = 0.0
        N = 0
        for ng in ngrams(seq, self.n):
            num = self.full_counter.get(ng, 0) + self.alpha
            den = self.context_counter.get(ng[:-1], 0) + self.alpha * self.V
            log_p += math.log(num / den)
            N += 1
        if N == 0:
            return 1.0
        return math.exp(-log_p / N)

    def score(self, seq):
        return self.perplexity(seq)


# ============================================================
# 检测器 3: SequenceSimilarityDetector
# ============================================================
class SequenceSimilarityDetector:
    """
    与场景"恶意签名序列"的相似度.
        score = 0.4 * LCS_ratio + 0.6 * weighted_high_sensitive_hit
    高敏感 token (HTTP_LEAK / HTTP_KEYLOG) 命中得分极高.
    """

    def __init__(self, templates=None, max_seq=3000):
        self.templates = templates or SCENARIO_TEMPLATES
        self.max_seq = max_seq
        self.signatures = {
            sid: [tok for pat in pats for tok in pat]
            for sid, pats in self.templates.items()
        }
        # 每个场景的高敏感 token 集合 (区分度高的 token)
        self.hi_tokens = {}
        for sid, pats in self.templates.items():
            hi = set()
            for p in pats:
                for t in p:
                    if TOKEN_THREAT_WEIGHT.get(t, 0) >= 2.0:
                        hi.add(t)
            self.hi_tokens[sid] = hi

    def score(self, seq):
        seq_full = list(seq)
        seq = seq_full[-self.max_seq:]
        seq_set = set(seq)
        seq_counter = Counter(seq)

        total = 0.0
        details = {}
        for sid, sig in self.signatures.items():
            lcs = _lcs_length(seq, sig)
            lcs_ratio = lcs / max(len(sig), 1)

            hi = self.hi_tokens[sid]
            if hi:
                hi_score = 0.0
                for t in hi:
                    if seq_counter[t] > 0:
                        hi_score += (
                            math.log(1 + seq_counter[t]) *
                            TOKEN_THREAT_WEIGHT.get(t, 1.0)
                        )
                hi_score /= len(hi) * 5.0      # 归一化到 ~ [0, 2]
            else:
                hi_score = 0.0

            s = 0.4 * lcs_ratio + 0.6 * hi_score
            details[sid] = s
            total = max(total, s)
        return total, details


# ============================================================
# 检测器 4: PeerDeviationDetector
# ============================================================
class PeerDeviationDetector:
    """
    "稀有 token IDF 加权"偏离度. 不再用普通办公 token 计分,
    避免长序列良性用户被误伤.
        score = Σ_t  threat_weight(t) * IDF(t) * log(1 + count(t))
    """

    # 仅计入 "区分度 ≥ 1.5" 的 token
    RARE_TOKENS = {t for t, w in TOKEN_THREAT_WEIGHT.items() if w >= 1.5}

    def __init__(self):
        self.idf = {}
        self.fitted = False

    def fit(self, sequences):
        df = Counter()
        N = 0
        for seq in sequences:
            N += 1
            present = set(seq) & self.RARE_TOKENS
            for tok in present:
                df[tok] += 1
        self.N = N
        self.idf = {tok: math.log((N + 1) / (df.get(tok, 0) + 1))
                    for tok in self.RARE_TOKENS}
        self.fitted = True
        return self

    def score(self, seq):
        c = Counter(seq)
        s = 0.0
        for tok in self.RARE_TOKENS:
            cnt = c.get(tok, 0)
            if cnt > 0:
                s += (math.log(1 + cnt) *
                      self.idf.get(tok, 1.0) *
                      TOKEN_THREAT_WEIGHT.get(tok, 1.0))
        return s


# ============================================================
# 融合
# ============================================================
class InsiderThreatScorer:
    """
        R(u) = w1 * z(rule) + w2 * z(perplexity)
             + w3 * z(similarity) + w4 * z(peer_deviation)
    """

    def __init__(self,
                 weights=(0.30, 0.10, 0.30, 0.30),
                 ngram_n=3):
        self.w = weights
        self.rule = PatternRuleDetector()
        self.lm   = NGramAnomalyDetector(n=ngram_n)
        self.sim  = SequenceSimilarityDetector()
        self.peer = PeerDeviationDetector()

    def fit(self, train_sequences):
        self.lm.fit(train_sequences)
        self.peer.fit(train_sequences)
        return self

    def raw_scores(self, sequences_dict):
        out = {}
        for u, seq in sequences_dict.items():
            r_total, r_details = self.rule.score(seq)
            s_total, s_details = self.sim.score(seq)
            out[u] = {
                'rule':         r_total,
                'rule_details': r_details,
                'lm':           self.lm.score(seq),
                'sim':          s_total,
                'sim_details':  s_details,
                'peer':         self.peer.score(seq),
                'len':          len(seq),
            }
        return out

    @staticmethod
    def _minmax(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [0.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    def aggregate(self, raw):
        users = list(raw.keys())
        rule = [raw[u]['rule'] for u in users]
        lm   = [raw[u]['lm']   for u in users]
        sim  = [raw[u]['sim']  for u in users]
        peer = [raw[u]['peer'] for u in users]

        z_rule = self._minmax(rule)
        z_lm   = self._minmax(lm)
        z_sim  = self._minmax(sim)
        z_peer = self._minmax(peer)

        w1, w2, w3, w4 = self.w
        out = {}
        for i, u in enumerate(users):
            r = (w1 * z_rule[i] +
                 w2 * z_lm[i]   +
                 w3 * z_sim[i]  +
                 w4 * z_peer[i])
            out[u] = {
                'risk':   r,
                'z_rule': z_rule[i],
                'z_lm':   z_lm[i],
                'z_sim':  z_sim[i],
                'z_peer': z_peer[i],
                **raw[u],
            }
        return out
