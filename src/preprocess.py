"""
CMU-CERT r4.2 数据预处理：将多源日志（logon/device/file/http/email）
合并为按用户、按时间排序的事件序列，并提取行为 token。

输出：
  outputs/user_sequences.pkl      —— 每个用户的完整事件序列（带时间戳）
  outputs/user_daily_seq.pkl      —— 每个用户每天的行为 token 序列
  outputs/user_meta.csv           —— 用户元信息（角色、所属业务部门等）
"""
import os
import re
import pickle
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / 'data' / 'r4.2')
OUTPUT_DIR = str(PROJECT_ROOT / 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== 行为 token 词汇表 ========================
# 我们将原始事件抽象为"动作 token"，便于序列建模。
#
#   工作时间(WH)    : 周一~周五 07:00~18:00
#   非工作时间(AH) : 其它时段（周末、节假日、深夜、清晨、傍晚）
#
# Token 格式：<动作>_<时段>_<上下文>，如:
#   LOGON_AH_OWN     : 非工作时段登录到自己的 PC
#   LOGON_AH_OTHER   : 非工作时段登录到他人 PC
#   USB_CONNECT_AH   : 非工作时段插入 U 盘
#   FILE_COPY_AH     : 非工作时段拷贝文件到 U 盘
#   HTTP_JOBHUNT     : 访问求职网站
#   HTTP_LEAK        : 访问 wikileaks 等泄露站点
#   HTTP_KEYLOG      : 访问 keylogger 工具站
#   EMAIL_EXT_ATT    : 向公司外发件人发送附件邮件
#   ...

# 工作时间定义
WORK_START_HOUR = 7
WORK_END_HOUR = 18

# 敏感网站类别关键词
LEAK_DOMAINS   = ['wikileaks.org', 'wikileaks']
KEYLOG_DOMAINS = ['keylogger', 'keylogger.com']
JOBHUNT_DOMAINS = [
    'monster.com', 'careerbuilder', 'job-hunt', 'jobhuntersbible',
    'indeed.com', 'simplyhired', 'craigslist.org/jjj',
    'linkedin.com/jobs', 'jobs',
]

DTAA_EMAIL_SUFFIX = '@dtaa.com'


def parse_dt(s):
    """解析 'mm/dd/yyyy HH:MM:SS' 格式."""
    return datetime.strptime(s, '%m/%d/%Y %H:%M:%S')


def is_after_hours(dt):
    """非工作时间判定: 周末 或 不在 07:00-18:00."""
    if dt.weekday() >= 5:
        return True
    if dt.hour < WORK_START_HOUR or dt.hour >= WORK_END_HOUR:
        return True
    return False


# ======================== 用户元信息 ========================
def load_user_pc_mapping():
    """从 logon.csv 推断每个用户最常用的 PC（"自己的 PC"）。"""
    print("[Preprocess] Building user->own_pc mapping from logon.csv ...")
    user_pc_count = defaultdict(lambda: defaultdict(int))
    chunks = pd.read_csv(os.path.join(DATA_DIR, 'logon.csv'),
                         chunksize=200_000)
    for ch in chunks:
        for u, p in zip(ch['user'], ch['pc']):
            user_pc_count[u][p] += 1
    own_pc = {u: max(d, key=d.get) for u, d in user_pc_count.items()}
    print(f"[Preprocess] Found {len(own_pc)} users.")
    return own_pc


def load_ldap():
    """从最新一份 LDAP CSV 中读取每个用户的角色 / 部门信息。"""
    ldap_dir = os.path.join(DATA_DIR, 'LDAP')
    files = sorted(os.listdir(ldap_dir))
    ldap_path = os.path.join(ldap_dir, files[-1])  # 取最新月份
    ld = pd.read_csv(ldap_path)
    ld = ld.rename(columns={'user_id': 'user'})
    return ld[['user', 'role', 'business_unit', 'department', 'team']]


# ======================== 事件抽取 ========================
def stream_logon_events(own_pc):
    print("[Preprocess] Streaming logon events ...")
    chunks = pd.read_csv(os.path.join(DATA_DIR, 'logon.csv'),
                         chunksize=200_000)
    for ch in chunks:
        for _, r in ch.iterrows():
            try:
                dt = parse_dt(r['date'])
            except Exception:
                continue
            ah = is_after_hours(dt)
            same = (r['pc'] == own_pc.get(r['user']))
            if r['activity'] == 'Logon':
                tag = 'OWN' if same else 'OTHER'
                token = f"LOGON_{'AH' if ah else 'WH'}_{tag}"
            else:
                token = f"LOGOFF_{'AH' if ah else 'WH'}"
            yield (r['user'], dt, token, 'logon')


def stream_device_events(own_pc):
    print("[Preprocess] Streaming device events ...")
    chunks = pd.read_csv(os.path.join(DATA_DIR, 'device.csv'),
                         chunksize=200_000)
    for ch in chunks:
        for _, r in ch.iterrows():
            try:
                dt = parse_dt(r['date'])
            except Exception:
                continue
            ah = is_after_hours(dt)
            if r['activity'] == 'Connect':
                token = f"USB_CONN_{'AH' if ah else 'WH'}"
            else:
                token = f"USB_DISC_{'AH' if ah else 'WH'}"
            yield (r['user'], dt, token, 'device')


def stream_file_events():
    print("[Preprocess] Streaming file events ...")
    chunks = pd.read_csv(os.path.join(DATA_DIR, 'file.csv'),
                         chunksize=200_000,
                         usecols=['date', 'user', 'pc', 'filename'])
    for ch in chunks:
        for _, r in ch.iterrows():
            try:
                dt = parse_dt(r['date'])
            except Exception:
                continue
            ah = is_after_hours(dt)
            ext = str(r['filename']).split('.')[-1].lower()
            if ext in ('exe', 'jar', 'zip', 'rar', '7z'):
                kind = 'EXEC'
            elif ext in ('doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'pdf'):
                kind = 'DOC'
            else:
                kind = 'OTH'
            token = f"FILE_{kind}_{'AH' if ah else 'WH'}"
            yield (r['user'], dt, token, 'file')


def classify_url(url):
    if not isinstance(url, str):
        return None
    u = url.lower()
    for d in LEAK_DOMAINS:
        if d in u:
            return 'HTTP_LEAK'
    for d in KEYLOG_DOMAINS:
        if d in u:
            return 'HTTP_KEYLOG'
    for d in JOBHUNT_DOMAINS:
        if d in u:
            return 'HTTP_JOBHUNT'
    return None


def stream_http_events():
    """为节省时间和内存，仅保留命中敏感关键词的 HTTP 行为。"""
    # 优先使用 http_slim.csv, 不存在则回退到 http.csv
    slim_path = os.path.join(DATA_DIR, 'http_slim.csv')
    full_path = os.path.join(DATA_DIR, 'http.csv')
    if os.path.exists(slim_path):
        http_path = slim_path
    elif os.path.exists(full_path):
        http_path = full_path
    else:
        print("[Preprocess] WARNING: neither http_slim.csv nor http.csv found, skipping.")
        return
    print(f"[Preprocess] Streaming {os.path.basename(http_path)} (sensitive URLs only) ...")
    chunks = pd.read_csv(http_path,
                         chunksize=500_000,
                         usecols=['date', 'user', 'pc', 'url'])
    for ch in chunks:
        ch = ch.dropna(subset=['url'])
        ch['cls'] = ch['url'].apply(classify_url)
        ch = ch[ch['cls'].notna()]
        for _, r in ch.iterrows():
            try:
                dt = parse_dt(r['date'])
            except Exception:
                continue
            ah = is_after_hours(dt)
            base = r['cls']
            token = f"{base}_{'AH' if ah else 'WH'}"
            yield (r['user'], dt, token, 'http')


def stream_email_events():
    """关注外发(向非 dtaa 邮箱)、附件邮件。"""
    print("[Preprocess] Streaming email.csv ...")
    chunks = pd.read_csv(os.path.join(DATA_DIR, 'email.csv'),
                         chunksize=200_000,
                         usecols=['date', 'user', 'to', 'cc', 'bcc',
                                  'from', 'attachments'])
    for ch in chunks:
        for _, r in ch.iterrows():
            try:
                dt = parse_dt(r['date'])
            except Exception:
                continue
            ah = is_after_hours(dt)
            recipients = ' '.join(
                str(r.get(c) or '') for c in ('to', 'cc', 'bcc')
            ).lower()
            external = False
            for tok in re.split(r'[,\s;]+', recipients):
                if tok and '@' in tok and DTAA_EMAIL_SUFFIX not in tok:
                    external = True
                    break
            try:
                n_att = int(r.get('attachments') or 0)
            except Exception:
                n_att = 0
            if external and n_att > 0:
                tk = 'EMAIL_EXT_ATT'
            elif external:
                tk = 'EMAIL_EXT'
            elif n_att > 0:
                tk = 'EMAIL_INT_ATT'
            else:
                tk = 'EMAIL_INT'
            token = f"{tk}_{'AH' if ah else 'WH'}"
            yield (r['user'], dt, token, 'email')


# ======================== 主流程 ========================
def main():
    own_pc = load_user_pc_mapping()
    ldap = load_ldap()
    ldap.to_csv(os.path.join(OUTPUT_DIR, 'user_meta.csv'), index=False)

    user_events = defaultdict(list)        # user -> [(dt, token, source)]

    for src in (stream_logon_events(own_pc),
                stream_device_events(own_pc),
                stream_file_events(),
                stream_http_events(),
                stream_email_events()):
        for u, dt, tok, source in src:
            user_events[u].append((dt, tok, source))

    print(f"[Preprocess] Sorting events for {len(user_events)} users ...")
    for u in user_events:
        user_events[u].sort(key=lambda x: x[0])

    with open(os.path.join(OUTPUT_DIR, 'user_sequences.pkl'), 'wb') as f:
        pickle.dump(dict(user_events), f, protocol=4)
    print(f"[Preprocess] Saved user_sequences.pkl "
          f"({sum(len(v) for v in user_events.values()):,} events).")

    # ----- 按天分组 -----
    user_daily = defaultdict(lambda: defaultdict(list))
    for u, evs in user_events.items():
        for dt, tok, _ in evs:
            day = dt.date()
            user_daily[u][day].append(tok)
    user_daily = {u: dict(d) for u, d in user_daily.items()}

    with open(os.path.join(OUTPUT_DIR, 'user_daily_seq.pkl'), 'wb') as f:
        pickle.dump(user_daily, f, protocol=4)
    print(f"[Preprocess] Saved user_daily_seq.pkl.")


if __name__ == "__main__":
    main()
