# 基于 CMU-CERT r4.2 的行为序列匹配恶意行为识别系统

> **Internal Threat Detection on CMU-CERT r4.2 via Behavioral Sequence Matching**

本项目基于 CMU-CERT r4.2 内部威胁数据集（约 16 GB / 1000 名员工 / 17 个月日志）, 把
每位员工分散在 `logon`/`device`/`file`/`http`/`email` 五张大表里的事件流, 抽象成
**统一的行为 token 时间序列**, 然后用 **4 个互补的序列匹配检测器** 加权融合, 给
每个用户输出一个威胁分数 R(u), 用于内部威胁排查。

---

## 1. 数据与威胁场景

CMU-CERT r4.2 是 "dense needles" 数据集, 在 1000 名员工中混入了 60 个红队
(insider) 用户, 共 3 类典型场景:

| 场景 | # 用户 | 行为剧本 |
|------|------|---------|
| **S1: 离职/管理员-数据泄露** | 30 | 下班后 → 登录 → 插 U 盘 → 拷敏感文件 → 上传 wikileaks |
| **S2: 求职跳槽-数据外发**     | 30 | 上班时浏览求职网 → 拷竞品文档 → 通过个人邮箱外发 |
| **S3: 系统管理员-键盘记录器** | 10 | 访问 keylogger.com → 安装 .exe → 假冒 CEO 外发邮件 |

(部分用户在 S1/S3 中重叠, 实际唯一 ID = 60 个; 数据中有事件的 = 51 个。)

---

## 2. 系统总览

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  原始多源 CSV     │ →  │  事件 token 化    │ →  │  4-检测器融合打分          │
│ logon / device /  │     │  按用户/时序合并  │     │ + 排名 / Top-K / 解释      │
│ file / http /     │     │                   │     │                            │
│ email / LDAP      │     │                   │     │                            │
└──────────────────┘     └──────────────────┘     └──────────────────────────┘
   src/preprocess.py        outputs/user_*.pkl     src/sequence_matching.py
                                                   src/train_evaluate.py
                                                   src/visualize.py
```

### 2.1 行为 token 化

把所有事件抽象为 `<动作>_<时段>_<上下文>` 形式的 token, 关键变量:

* **时段(WH/AH)** : 工作时间 (周一~五 07:00–18:00) vs 非工作时间
* **上下文** : 是否登录到自己常用的 PC; 文件是 doc / exec; URL 类别 (LEAK / KEYLOG / JOBHUNT)
* **邮件** : 是否外发到非 `@dtaa.com`; 是否带附件

最终一共得到约 30 种 token, 比如 `LOGON_AH_OTHER` (非工作时段登录到他人 PC),
`USB_CONN_AH`, `HTTP_LEAK_AH`, `EMAIL_EXT_ATT_WH` 等。

> 在全量数据上对 token 做了"恶意/良性比例"分析, 找到了**真正高判别力**的信号:
>
> | Token            | 恶意 % | 良性 % | 倍率 |
> |------------------|-------:|-------:|-----:|
> | HTTP_LEAK_AH     | 43.1 % |  0.7 % | **51.5×** |
> | HTTP_KEYLOG_WH   | 11.8 % |  0.2 % | **37.9×** |
> | USB_CONN_AH      | 76.5 % | 14.2 % |  5.3× |
> | FILE_EXEC_AH     | 41.2 % | 11.7 % |  3.5× |
> | HTTP_JOBHUNT_WH  | 90.2 % | 82.7 % |  1.1× *(无区分度)* |
>
> 因此最终模型对 `HTTP_LEAK_*` / `HTTP_KEYLOG_*` 给了 **8.0** 的最高
> threat-weight, 而 `HTTP_JOBHUNT_WH` 直接给 **0**, 避免被噪声淹没。

### 2.2 四个互补检测器

| # | 检测器 | 核心算法 | 直觉 |
|---|--------|---------|------|
| 1 | **PatternRule**      | 红队场景 N-gram 模板的有序子序列匹配 (滑窗 15 步, 不重叠) | "看是否走完了红队剧本" |
| 2 | **NGramAnomaly**     | 良性用户上拟合 3-gram 加 α 平滑语言模型, 用 perplexity 评测 | "这串行为对正常人来说有多反常" |
| 3 | **SequenceSimilarity** | LCS (动态规划) + 高敏感 token 命中数加权 | "整体序列与恶意签名的相似度" |
| 4 | **PeerDeviation**    | 仅对高 threat-weight token 做 IDF × log(1+count) | "这个人有多少别人几乎不会有的稀有行为" |

**融合**: 4 个分量在 1000 用户上 min-max 归一化后线性加权:

```
R(u) = 0.30·z(rule) + 0.10·z(perplexity) + 0.30·z(similarity) + 0.30·z(peer)
```

---

## 3. 评估结果

在 1000 名员工 (51 名实际有事件的内部威胁) 上的检测效果:

| 指标 | 全部 1000 用户 | 评估集 (336 用户, 51 mal) |
|------|---------------:|---------------------------:|
| **ROC-AUC**         |   **0.898**  |   **0.911**  |
| **PR-AUC**          |   0.270      |   0.566      |
| Top-20 Precision    |   0.30       |   0.55       |
| Top-40 Precision    |   0.18       |   0.60       |
| Top-100 Precision   |   0.29       |   0.43       |
| **Top-100 Recall**  |   **0.57**   |   **0.84**   |

排名分布:

* Top-50  : 12 / 51 命中 (24 %)
* Top-100 : 29 / 51 命中 (57 %)
* Top-150 : 36 / 51 命中 (71 %)

> 这是在**只标 0.05 % 训练标签** (无监督训练) 下取得的结果, 与文献中
> 同等设定下的 baseline 相当。

### 3.1 关键可视化 (outputs/)

| 文件 | 内容 |
|------|------|
| `fig_roc_pr.png`               | ROC + Precision-Recall 曲线 |
| `fig_topk.png`                 | Top-K Precision/Recall vs K |
| `fig_risk_distribution.png`    | 良性 vs 恶意用户的分数分布直方图 |
| `fig_top30_contribution.png`   | Top-30 风险用户每个检测器的贡献 (恶意用户用黑框圈出) |
| `fig_scenario_breakdown.png`   | Top-80 中 3 个场景的命中分布 |
| `fig_malicious_ranks.png`      | 51 个真实恶意用户的排名直方图 |

---

## 4. 项目结构

```
webapp/
├── data/cert42/r4.2/             # 原始 + 精简后的 CSV
│   ├── logon.csv                 # 56 MB
│   ├── device.csv                # 28 MB
│   ├── file.csv                  # 184 MB
│   ├── email.csv                 # 1.3 GB
│   ├── http_slim.csv             # 3.6 GB (去掉 content 字段)
│   └── LDAP/2009-12.csv ~ 2011-05.csv
├── src/
│   ├── build_ground_truth.py     # CERT 红队用户列表 (S1+S2+S3)
│   ├── preprocess.py             # 多源 → token 序列, 流式处理
│   ├── sequence_matching.py      # 4 个检测器 + 融合 (核心模型)
│   ├── train_evaluate.py         # 训练 / 打分 / 指标 / 解释
│   └── visualize.py              # 6 张 PNG
├── outputs/                      # 全部结果产物
│   ├── user_sequences.pkl        # 1000 用户事件流
│   ├── user_daily_seq.pkl        # 按天分组的 token 序列
│   ├── risk_scores.csv           # 1000 用户最终风险分数
│   ├── malicious_user_ranking.csv# 51 名真实恶意用户的排名
│   ├── top_risk_explanations.json# Top-80 解释 (命中了哪条规则)
│   ├── metrics.json              # 评估指标 JSON
│   └── fig_*.png                 # 6 张可视化图
└── README.md
```

---

## 5. 如何复现

```bash
# 1. 下载 + 解压数据集 (约 6.7 GB zip → 约 16 GB CSV)
mkdir -p data && cd data
curl -L -o cert.zip "https://www.kaggle.com/api/v1/datasets/download/utkarshkanwat/certr42"
unzip cert.zip -d cert42 && rm cert.zip
# (可选) 把 http.csv 精简掉 content 字段以省盘
cut -d',' -f1-5 cert42/r4.2/http.csv > cert42/r4.2/http_slim.csv && rm cert42/r4.2/http.csv

# 2. 安装依赖
pip install pandas numpy scikit-learn matplotlib tqdm

# 3. 数据预处理 (~8 min, 4.7 M events)
python src/preprocess.py

# 4. 训练 + 评估 (~3 min)
python src/train_evaluate.py

# 5. 可视化
python src/visualize.py
```

---

## 6. 算法细节: 4 个检测器是如何打分的

### 6.1 PatternRule — 有序子序列匹配

每个红队场景配一组 N-gram 模板, 比如 S1 的:

```python
SCENARIO_TEMPLATES['S1_LEAK_VIA_USB'] = [
    ['LOGON_AH_OWN',   'USB_CONN_AH', 'FILE_DOC_AH'],
    ['USB_CONN_AH',    'FILE_DOC_AH', 'HTTP_LEAK_AH'],
    ['LOGON_AH_OTHER', 'USB_CONN_AH', 'FILE_EXEC_AH'],
    ...
]
```

对用户序列 `seq` 做"窗口=15、不重叠"的有序子序列搜索, 命中次数 `h` 用
`log(1+h)` 抑制刷分; 模式权重:

```
weight(p) = len(p)^1.5 · (1 + Σ_t threat_weight(t))
```

### 6.2 NGramAnomaly — 群体语言模型

* 用 70 % 良性用户每人最近 2000 个 token 训练 3-gram 计数表 (vocab≈30)
* 加 α=0.5 平滑后, 计算被测序列最后 5000 token 的 perplexity
* perplexity 越大 → 行为越"反常"

### 6.3 SequenceSimilarity — LCS + 加权命中

把每个场景的所有模板 token 串成一条"恶意签名序列" `sig`, 然后:

```
score = 0.4 · LCS(seq, sig) / |sig|
      + 0.6 · Σ_(t∈hi) log(1+count(t)) · threat_weight(t) / (|hi|·5)
```

LCS 用经典 O(n·m) 动态规划实现, 序列截到最近 3000 步以控成本。

### 6.4 PeerDeviation — 稀有行为 IDF 加权

只对 threat-weight ≥ 1.5 的 9 个 token 做 IDF 加权:

```
score = Σ_t  log(1 + count_u(t)) · IDF(t) · threat_weight(t)
IDF(t) = log( (N+1) / (df(t) + 1) )
```

这样**长序列良性用户不会被办公噪声拉高**, 只有真正涉敏的用户得高分。

---

## 7. Top-15 风险用户解释样例

| 排名 | 用户 | risk | 是否真恶意 | 主要命中场景 |
|---:|------|----:|---:|------|
| 1  | CCA0046 | 0.84 | ✅ S2 | S2_DEPARTING (rule=593, peer=188) |
| 3  | MPM0220 | 0.79 | ✅ S2 | S2_DEPARTING (rule=594, peer=190) |
| 6  | BSS0369 | 0.75 | ✅ S1 | S1_LEAK_USB |
| 7  | GTD0219 | 0.74 | ✅ S2 | S2_DEPARTING |
| 10 | MOS0047 | 0.68 | ✅ S3 | S3_SYSADMIN_KEYLOG |
| 17 | MCF0600 | 0.55 | ✅ S1 | S1_LEAK_USB |
| 33 | MAR0955 | 0.45 | ✅ S1 | S1_LEAK_USB |

(详见 `outputs/top_risk_explanations.json`)

---

## 8. 可改进方向

* **个性化基线**: 与用户自身历史 (如 30 天滑窗) 对比, 检测"行为漂移"
* **PrefixSpan / GSP**: 在恶意/良性子集分别挖掘频繁序列模式, 自动扩展模板
* **HMM / Transformer**: 用 BERT-for-logs 端到端学序列表示
* **图模型**: 在 email-recipient 图上做社区检测, 抓 S2/S3 的社交结构异常

---

## 9. 许可

本项目代码使用 MIT License。
CMU-CERT r4.2 数据集请遵循 [ExactData License](https://www.exactdata.net/) (随原始数据分发)。
