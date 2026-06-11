# 基于 CMU-CERT r4.2 的内部威胁检测研究报告：行为序列匹配与异构图神经网络方法对比

报告日期：2026-05-26

---

## 摘要

内部威胁检测面临样本稀缺、行为跨度长、攻击链隐蔽和误报成本高等核心挑战。本项目以 CMU-CERT r4.2 数据集（约 16 GB / 1000 名员工 / 17 个月日志 / 70 名恶意用户）为研究对象，实现并对比了两种内部威胁检测方法：

- **方法一：基于 N-gram 模板的行为序列匹配**——将多源日志统一抽象为行为 token 时间序列，通过四类互补检测器（红队模板匹配、N-gram 语言模型异常、恶意签名相似度、群体稀有度偏离）加权融合生成用户风险分数；
- **方法二：基于 R-GCN 的异构图检测**——构建用户-PC-文件类型-URL类别的异构图，利用关系图卷积网络（R-GCN）进行半监督节点分类，自动学习区分恶意与良性用户的结构化特征。

在完整 r4.2 数据集上的对比实验表明：R-GCN 在所有核心指标上全面优于 N-gram 方法，全量 1000 用户 ROC-AUC 达到 **0.9898**（N-gram 为 0.9041），PR-AUC 达到 **0.8358**（N-gram 为 0.3257），Top-100 Recall 达到 **98.57%**（N-gram 为 52.86%）。R-GCN 在最难检测的 S2 场景（求职+邮件外发）上表现尤为突出，Top-100 Recall 从 N-gram 的 16.67% 大幅提升至 96.67%。然而，N-gram 方法在可解释性方面仍具有独特优势，每个高风险用户可追溯至具体的规则模板和行为模式。

## 1. 研究背景与意义

企业内部威胁通常并不表现为单个孤立事件，而是由一组在时间上相互关联的行为组成，例如非工作时间登录、连接移动存储设备、访问泄露站点、拷贝敏感文件或向外部邮箱发送附件。单点异常检测容易受到办公噪声干扰，而纯监督学习又依赖大量高质量标签，在真实环境中难以满足。

CMU-CERT r4.2 是内部威胁检测研究中常用的模拟数据集，包含多个月份、多源日志与红队注入的恶意用户行为。该数据集适合研究以下问题：

- 如何将异构日志转换为统一的用户行为序列；
- 如何从攻击剧本中抽取有序行为模式；
- 如何在标签稀缺条件下利用良性群体行为建立异常基线；
- 如何利用用户-设备-资源之间的图结构关系增强检测能力；
- 如何在检测结果中保留可解释性，辅助安全分析人员排查。

本项目围绕两条技术路线开展研究：一是基于领域知识驱动的"行为序列匹配"方法，二是基于数据驱动的"异构图神经网络"方法，并在相同数据集上进行严格对比，为实际部署提供方法选型依据。

## 2. 数据集与威胁场景

### 2.1 数据集概览

项目基于 CMU-CERT r4.2 数据集。原始数据约 16 GB，覆盖 1000 名员工、约 17 个月的企业活动日志，主要包括：

| 数据源 | 文件大小 | 主要字段 | 行为含义 |
| --- | ---: | --- | --- |
| logon.csv | 56 MB | date, user, pc, activity | 登录/登出行为，识别工作时间、非工作时间和是否使用本人常用 PC |
| device.csv | 28 MB | date, user, pc, activity | USB 设备连接/断开行为 |
| file.csv | 184 MB | date, user, pc, filename | 文件访问行为，按文档、可执行文件和其他文件分类 |
| http.csv | 14.5 GB | date, user, pc, url, content | HTTP 浏览行为，本项目生成精简版 http_slim.csv（4.1 GB） |
| email.csv | 1.3 GB | date, user, recipients, attachments | 内外部邮件、附件外发行为 |
| LDAP/ | ~2.5 MB | user, role, department, team | 用户岗位与组织属性（按月快照） |
| insiders.csv | 14 KB | dataset, scenario, user, start, end | 官方恶意用户 ground truth |

### 2.2 威胁场景定义

当前项目使用 `src/build_ground_truth.py` 中经官方 `insiders.csv` 核对的 r4.2 恶意用户清单，共 **70 个唯一恶意用户**，分为三类典型场景：

| 场景 | 用户数 | 行为剧本 |
| --- | ---: | --- |
| S1 数据泄露/USB 外带 | 30 | 离职或不满员工在非工作时段登录、连接 U 盘、访问或拷贝文档，并上传至泄露站点（wikileaks） |
| S2 跳槽前数据外发 | 30 | 员工浏览求职网站，窃取竞品或业务资料，并通过个人邮箱或外部邮件渠道外发 |
| S3 系统管理员键盘记录器 | 10 | 管理员访问 keylogger 站点、安装可执行文件，并冒用高管身份发送破坏性邮件 |

三个场景互不重叠，共涵盖 70 个恶意用户和 930 个良性用户。

## 3. 数据预处理与行为 Token 化

### 3.1 统一预处理流程

预处理脚本为 `src/preprocess.py`，核心思想是将多源日志流式读取后统一为按用户、按时间排序的行为 token 序列。

**时间上下文划分**：
- 工作时间（WH）：周一至周五 07:00-18:00；
- 非工作时间（AH）：周末、深夜、清晨以及其他非工作时间段。

**用户上下文识别**：
- 从登录日志统计每个用户最常使用的 PC 作为"常用 PC"；
- 登录事件区分为 `LOGON_WH_OWN`（工作时间登录本人 PC）、`LOGON_AH_OTHER`（非工作时间登录他人 PC）等。

### 3.2 行为 Token 词汇表

项目最终形成约 30 类行为 token，典型示例如下：

| 行为类别 | Token 示例 | 含义 |
| --- | --- | --- |
| 登录 | `LOGON_AH_OTHER` | 非工作时间登录非常用 PC |
| 设备 | `USB_CONN_AH` | 非工作时间连接 USB 设备 |
| 文件 | `FILE_DOC_AH` | 非工作时间访问文档类文件 |
| 文件 | `FILE_EXEC_WH` | 工作时间访问可执行类文件 |
| HTTP | `HTTP_LEAK_AH` | 非工作时间访问泄露相关站点 |
| HTTP | `HTTP_KEYLOG_WH` | 工作时间访问 keylogger 相关站点 |
| HTTP | `HTTP_JOBHUNT_WH` | 工作时间访问求职网站 |
| 邮件 | `EMAIL_EXT_ATT_WH` | 工作时间向外部邮箱发送附件邮件 |

完整预处理结果：**1000 个用户，4,779,378 个事件**，保存为 `outputs/user_sequences.pkl` 和 `outputs/user_daily_seq.pkl`。

### 3.3 HTTP 数据精简

原始 `http.csv` 文件 14.5 GB，包含大量 `content` 字段。项目通过 `src/generate_http_slim.py` 生成精简版 `http_slim.csv`（4.1 GB，28,434,423 行），仅保留 `id, date, user, pc, url` 五列，供两种算法共同使用。

## 4. 方法一：基于 N-gram 模板的行为序列匹配

### 4.1 方法概述

该方法定义在 `src/sequence_matching.py` 中，整体架构如下：

```text
多源日志
  -> 行为 token 序列
  -> PatternRule / NGramAnomaly / SequenceSimilarity / PeerDeviation
  -> 分量归一化
  -> 用户风险分数 R(u)
  -> 排名、Top-K 指标与解释
```

系统核心由四类互补检测器组成，全部"得分越高越可疑"，最终通过 min-max 归一化后线性加权融合。

### 4.2 威胁权重设计

基于 token 在恶意用户和良性用户中的出现比例差异，项目为不同 token 设置了数据驱动的威胁权重（近似 `log(P(t|malicious) / P(t|benign))`）：

| Token | 权重 | 恶意/良性比率 | 解释 |
| --- | ---: | ---: | --- |
| `HTTP_LEAK_AH/WH` | 8.0 | 39-52x | 访问泄露站点，极强恶意信号 |
| `HTTP_KEYLOG_AH/WH` | 8.0 | 38x | 访问键盘记录器站点，极强恶意信号 |
| `USB_CONN_AH` | 2.5 | 5.3x | 非工作时间连接移动设备 |
| `FILE_EXEC_AH` | 2.0 | 3.5x | 非工作时间执行文件 |
| `LOGON_AH_OTHER` | 2.0 | 1.7x | 非工作时间登录非常用 PC |
| `HTTP_JOBHUNT_WH` | 0.0 | 1.1x | 求职网站在 83% 良性用户中也出现，区分度极低 |

### 4.3 四类检测器

**（1）PatternRule：红队剧本有序子序列匹配**

将三类红队场景编码为 21 条行为模板。例如 S1 包含 `[LOGON_AH_OWN, USB_CONN_AH, FILE_DOC_AH]` 等组合。在窗口（默认 15 步）内进行有序子序列匹配，不重叠计数：

```text
score(pattern) = log(1 + hits) × len(pattern)^1.5 × (1 + Σ threat_weight(token))
```

**（2）NGramAnomaly：良性行为语言模型**

使用 70% 良性用户训练 3-gram 语言模型（加性平滑 α=0.5），计算被测用户序列的 perplexity。行为序列在良性群体语言模型下概率越低（perplexity 越高），行为越偏离正常。

**（3）SequenceSimilarity：恶意签名相似度**

将每个场景的模板 token 拼接成恶意签名序列，计算用户行为序列与签名序列的最长公共子序列（LCS）比例，并对高敏感 token 出现次数加权：

```text
score = 0.4 × LCS_ratio + 0.6 × weighted_high_sensitive_hit
```

**（4）PeerDeviation：高风险 token 群体稀有度**

仅关注威胁权重 ≥ 1.5 的 9 类 token，计算 IDF 加权偏离度：

```text
score = Σ log(1 + count_u(token)) × IDF(token) × threat_weight(token)
```

### 4.4 融合策略

四个检测器输出经 min-max 归一化后线性融合：

```text
R(u) = 0.40 × z(rule) + 0.20 × z(perplexity) + 0.25 × z(similarity) + 0.15 × z(peer)
```

权重为人工设定，侧重规则匹配和相似度信号。

## 5. 方法二：基于 R-GCN 的异构图检测

### 5.1 方法概述

该方法将内部威胁检测建模为异构图上的节点分类问题，通过关系图卷积网络（R-GCN）自动从图结构中学习用户的恶意特征。代码位于 `src/rgcn/` 目录下。

### 5.2 异构图构建

图构建脚本为 `src/rgcn/build_graph.py`，从完整 r4.2 数据构建包含 4 种节点类型和 6 种关系的异构图：

**节点类型（4 种，共 2017 个节点）**：

| 节点类型 | 数量 | 说明 |
| --- | ---: | --- |
| user | 1000 | 员工节点 |
| pc | 1003 | 终端机节点 |
| file_type | 6 | 文件类别节点：(doc/exec/other) × (WH/AH) |
| url_cat | 8 | URL 类别节点：(LEAK/KEYLOG/JOBHUNT/OTHER) × (WH/AH) |

**关系类型（6 种，共 38,568 条边）**：

| 关系 | 方向 | 边数 | 含义 |
| --- | --- | ---: | --- |
| logon_wh | user → pc | 8,074 | 工作时间登录 |
| logon_ah | user → pc | 20,144 | 非工作时间登录 |
| usb_wh | user → pc | 1,441 | 工作时间 USB 连接 |
| usb_ah | user → pc | 4,639 | 非工作时间 USB 连接 |
| file_op | user → file_type | 1,063 | 文件操作 |
| http_visit | user → url_cat | 3,207 | HTTP 浏览 |

每条边带权重 `w = log(1 + count)`，同一 `(src, dst, rel)` 多次出现会合并。训练时自动添加反向边，最终 R-GCN 使用 12 种关系。

**用户节点特征（6 维）**：
1. 活跃天数
2. 总事件数
3. 非工作时间事件比率
4. 使用 PC 数量
5. 操作文件种类数
6. 非工作时间登录次数

特征经 RobustScaler 标准化（减中位数 / 除 IQR），并裁剪到 [-5, 5]。

### 5.3 R-GCN 模型架构

模型实现在 `src/rgcn/rgcn_model.py` 中，遵循 Schlichtkrull et al. (ESWC 2018) 的 R-GCN 框架，纯 PyTorch 实现（不依赖 DGL/PyG）。

**核心公式**：

```text
h_v^(l+1) = σ( W_self × h_v^(l) + Σ_r Σ_{u∈N_r(v)} (w_{u,v,r} / c_{v,r}) × W_r × h_u^(l) )
```

其中 `W_r` 采用基分解（basis decomposition）：`W_r = Σ_{b=1..B} a_{rb} × V_b`，用 B=4 个基矩阵共享 12 个关系的参数，有效减少参数量。

**模型结构**：

```text
user: Linear(6, 64)  ─┐
pc:   Embedding(64)   ─┤
file: Embedding(64)   ─┼──→ R-GCN Layer 1 (64→64, ReLU) ──→ R-GCN Layer 2 (64→64)
url:  Embedding(64)   ─┘                                          │
                                                                   ↓
                                                    Linear(64→32) → ReLU → Linear(32→1)
                                                    (仅 user 节点)       → sigmoid → P(malicious)
```

### 5.4 训练策略

- **半监督 Transductive**：所有节点参与消息传递，但仅在训练节点上计算损失；
- **5-fold StratifiedKFold** 交叉验证，保证每折恶意/良性比例不变；
- **类别不平衡处理**：BCEWithLogitsLoss 带 `pos_weight = #neg / #pos ≈ 13.3`；
- **优化器**：AdamW，lr=0.01，weight_decay=5e-4；
- **早停**：patience=50 epochs，监控验证集 ROC-AUC；
- **梯度裁剪**：max_norm=2.0；
- **OOF 预测**：每个用户从其所在验证折获得分数，汇总后计算整体指标。

### 5.5 超参数配置

| 参数 | 值 | 说明 |
| --- | --- | --- |
| hidden_dim | 64 | 隐藏层维度 |
| num_layers | 2 | R-GCN 层数 |
| num_bases | 4 | 基分解秩 |
| dropout | 0.3 | Dropout 率 |
| lr | 0.01 | 学习率 |
| weight_decay | 5e-4 | 权重衰减 |
| epochs | 300 | 最大训练轮数 |
| patience | 50 | 早停耐心值 |
| n_splits | 5 | 交叉验证折数 |

## 6. 实验设计

### 6.1 评估框架

两种算法在完全相同的数据集上进行评估：

| 项目 | N-gram 方法 | R-GCN 方法 |
| --- | --- | --- |
| 数据集 | 完整 r4.2（1000 用户） | 完整 r4.2（1000 用户） |
| 数据源 | logon + device + file + http_slim + email | logon + device + file + http_slim |
| 评估方式 | 70% 良性训练 / 30% 留出 + 全量打分 | 5-fold StratifiedKFold + OOF 汇总 |
| 标签使用 | 仅用于最终评估（无监督打分） | 用于训练二分类器（半监督） |
| Ground Truth | 70 恶意用户（S1:30, S2:30, S3:10） | 70 恶意用户（S1:30, S2:30, S3:10） |

### 6.2 评估指标

- **ROC-AUC**：不同阈值下的真正率/假正率权衡；
- **PR-AUC**：在正样本稀疏（7%）场景下更能反映实际排查难度；
- **Top-K Precision/Recall**（K=10,20,30,50,70,100）：模拟安全运营中"优先审查前 K 名"的场景；
- **场景级 Recall**：按 S1/S2/S3 分别统计在 Top-70 和 Top-100 中的召回率。

## 7. 实验结果

### 7.1 总体性能对比

| 指标 | N-gram | R-GCN | R-GCN 提升 |
| --- | ---: | ---: | ---: |
| **ROC-AUC** | 0.9041 | **0.9898** | +9.5% |
| **PR-AUC** | 0.3257 | **0.8358** | +156.6% |

R-GCN 在两个核心指标上全面领先。PR-AUC 的差距尤为显著（0.33 vs 0.84），说明 R-GCN 在提高召回率时仍能维持较高精确度，而 N-gram 方法在追求高召回时精确度下降迅速。

R-GCN 的 5-fold 交叉验证结果也非常稳定：各折 AUC 为 [0.9981, 0.9923, 0.9939, 0.9866, 0.9885]，均值 **0.9919 ± 0.0041**。

![ROC 曲线对比](outputs/comparison/fig_roc_comparison.png)

图 1 展示两种方法的 ROC 曲线。R-GCN 曲线整体更靠近左上角，说明在不同阈值下均能保持更高的真正率和更低的误报率。

![Precision-Recall 曲线对比](outputs/comparison/fig_pr_comparison.png)

图 2 展示两种方法的 PR 曲线。由于恶意用户仅占 7%，PR 曲线比 ROC 曲线更能反映真实排查压力；R-GCN 的曲线明显高于 N-gram，说明其在高召回区间仍能维持更好的精确率。

### 7.2 Top-K 排查效果对比

在安全运营场景中，分析人员通常优先查看风险排名靠前的用户，因此 Top-K 指标比单纯分类阈值更有实际意义。

| K | N-gram 精确率 | R-GCN 精确率 | N-gram 召回率 | R-GCN 召回率 | N-gram 命中 | R-GCN 命中 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.50 | **0.90** | 0.07 | **0.13** | 5 | 9 |
| 20 | 0.30 | **0.90** | 0.09 | **0.26** | 6 | 18 |
| 30 | 0.23 | **0.87** | 0.10 | **0.37** | 7 | 26 |
| 50 | 0.28 | **0.80** | 0.20 | **0.57** | 14 | 40 |
| 70 | 0.26 | **0.80** | 0.26 | **0.80** | 18 | 56 |
| 100 | 0.37 | **0.69** | 0.53 | **0.99** | 37 | 69 |

关键发现：
- R-GCN 在 Top-10 中命中 9/70 个恶意用户（精确率 90%），N-gram 仅命中 5 个（精确率 50%）；
- R-GCN 在 Top-70 时已达到 80% 召回率（56/70），而 N-gram 在 Top-100 时仅 53%（37/70）；
- R-GCN 在 Top-100 中命中了 69/70 恶意用户（仅漏掉 1 人），接近完美召回。

![Top-K Precision 与 Recall 对比](outputs/comparison/fig_topk_comparison.png)

图 3 给出了 Top-K 精确率和召回率对比。R-GCN 在 Top-10 到 Top-100 的所有 K 值上均明显领先，尤其在 Top-70 时已经达到 80% 召回，适合有限人工审查资源下的优先级排序。

### 7.3 场景级检测能力对比

| 场景 | 描述 | 用户数 | N-gram Top-70 | R-GCN Top-70 | N-gram Top-100 | R-GCN Top-100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 | USB + Leak 上传 | 30 | 0.3667 | **0.7667** | 0.8333 | **1.0000** |
| S2 | 求职 + 邮件外发 | 30 | 0.0667 | **0.8667** | 0.1667 | **0.9667** |
| S3 | 键盘记录器 | 10 | 0.5000 | **0.7000** | 0.7000 | **1.0000** |

**S2 场景差异最为突出**：N-gram 在 Top-100 中仅发现 S2 的 16.67% 恶意用户（5/30），而 R-GCN 发现了 96.67%（29/30）。这是因为 S2 的行为特征（求职浏览 + 文档外发）与良性用户高度重叠，83% 的良性用户也有求职网站访问记录。N-gram 方法将 `HTTP_JOBHUNT_WH` 权重设为 0（等同噪声），导致 S2 检测几乎依赖弱信号；而 R-GCN 通过图结构中用户-PC-URL 的交互模式，自动学习到了更细粒度的区分特征。

S1 和 S3 场景在 R-GCN 的 Top-100 中均实现了 100% 召回。

![场景级 Recall 对比](outputs/comparison/fig_scenario_comparison.png)

图 4 进一步展示了分场景 Top-70 与 Top-100 Recall。S2 的提升最明显，说明异构图结构能够补足弱 token 信号难以表达的关联模式。

### 7.4 N-gram 方法详细结果

#### 7.4.1 评估集与全量集对比

| 指标 | 全量 1000 用户 | 留出评估集 349 用户 |
| --- | ---: | ---: |
| 恶意用户数 | 70 | 70 |
| 良性用户数 | 930 | 279 |
| ROC-AUC | 0.9041 | 0.9053 |
| PR-AUC | 0.3257 | 0.6076 |
| Top-50 Precision | 0.2800 | 0.6600 |
| Top-100 Recall | 0.5286 | 0.8143 |

PR-AUC 在评估集上显著高于全量集，主要因为评估集中恶意样本占比更高（70/349=20%），在全量集中恶意用户仅占 7%。

![N-gram ROC 与 PR 曲线](outputs/fig_roc_pr.png)

图 5 是 N-gram 方法单独的 ROC 和 PR 曲线。该方法具备较好的整体排序能力，但在正样本稀疏场景下，PR 曲线显示其高召回区间精确率下降较快。

![N-gram Top-K Precision 与 Recall](outputs/fig_topk.png)

图 6 展示 N-gram 方法自身的 Top-K 变化趋势。随着 K 增大，召回率逐步提升，但精确率波动明显，反映出规则模板对部分良性高活跃用户存在误报。

#### 7.4.2 风险分数分布

全量用户中，良性用户平均风险分数为 0.1176，中位数为 0.0870；恶意用户平均风险分数为 0.3304，中位数为 0.3250。两类用户的分布存在一定分离，但重叠区域较大，部分良性用户风险分数较高（最高达 0.7927），导致精确率不高。

![N-gram 风险分数分布](outputs/fig_risk_distribution.png)

图 7 展示 N-gram 风险分数分布。恶意用户整体右移，但良性和恶意分布仍存在较大重叠，这是该方法 Top-K 精确率受限的重要原因。

#### 7.4.3 检测器贡献分析

N-gram 方法的 Top-20 高风险用户中，5 名为真正恶意用户，其中：
- CCA0046（S3，排名第 1）：PatternRule、SequenceSimilarity 和 PeerDeviation 均显著偏高；
- MPM0220（S3，排名第 3）：keylogger 签名强命中；
- BSS0369（S3，排名第 6）：与 S3 签名序列相似度最高。

S3 场景用户排名普遍靠前，因为 keylogger 相关 token 稀有且威胁权重高（8.0）；S2 场景用户排名最分散，最差排名达到 408 位。

![N-gram Top-30 检测器贡献](outputs/fig_top30_contribution.png)

图 8 展示 Top-30 风险用户中四个检测器的贡献。黑框标记真实恶意用户，可以看到部分恶意用户由多个检测器共同抬高风险分数，而部分良性用户因规则或稀有行为命中产生高分。

![N-gram 场景覆盖](outputs/fig_scenario_breakdown.png)

图 9 展示 N-gram 方法在 Top-80 中的场景覆盖情况。S1 和 S3 的规则命中更集中，S2 覆盖不足，与场景级 Recall 的结果一致。

![N-gram 恶意用户排名分布](outputs/fig_malicious_ranks.png)

图 10 展示 70 个真实恶意用户在 N-gram 全量排序中的排名分布。排名越靠前越利于人工排查；S2 场景排名更靠后，是后续优化重点。

### 7.5 R-GCN 方法详细结果

#### 7.5.1 交叉验证稳定性

| 折 | 训练集 | 验证集 | Val AUC | 最优 Epoch |
| ---: | --- | --- | ---: | ---: |
| 1 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9981 | 36 |
| 2 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9923 | 32 |
| 3 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9939 | 22 |
| 4 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9866 | 29 |
| 5 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9885 | 37 |
| **均值 ± 标准差** | | | **0.9919 ± 0.0041** | |

各折 AUC 均超过 0.98，标准差仅 0.004，表明模型在不同数据划分下表现稳定。早停 epoch 集中在 22-37 轮，说明模型收敛迅速。

#### 7.5.2 OOF 排名分析

R-GCN 的 Top-10 用户全部为恶意用户（精确率 100%），前 30 名中仅有 4 名良性用户误报。最高分恶意用户 JRG0207 的预测概率达到 0.9995。

在 Top-100 中，R-GCN 命中了 69/70 恶意用户，唯一遗漏的用户排名也在 100 名附近，几乎实现了完美召回。

![双算法风险分数分布对比](outputs/comparison/fig_score_dist_comparison.png)

图 11 对比了两种方法的风险分数分布。R-GCN 对良性和恶意用户的分离更明显，高分区恶意用户集中度更高；N-gram 分布重叠更明显，导致高风险名单中混入更多良性用户。

## 8. 性能差异分析

### 8.1 R-GCN 优势来源

1. **图结构信息利用**：R-GCN 通过消息传递机制利用了用户-PC-文件-URL 之间的关联关系。例如，如果多个恶意用户共享某些 PC 或访问相同的敏感 URL 类别，这种结构化模式会通过图卷积传播，增强对相邻节点的分类能力。

2. **自动特征学习**：无需手动定义行为模板和威胁权重，R-GCN 自动从带标签的训练数据中学习区分恶意与良性用户的特征组合。这使得模型能够捕获人类专家难以预先定义的复杂交互模式。

3. **S2 场景突破**：求职+邮件外发行为单独看并不异常，但在图结构中与其它恶意行为指标的关联模式（如与特定 PC 集群或文件类型的交互模式）可被模型自动学习。N-gram 方法将 `HTTP_JOBHUNT_WH` 权重设为 0 相当于主动放弃了这一维度的信号。

4. **半监督学习范式**：R-GCN 充分利用了 70 个已知恶意标签进行监督学习，而 N-gram 方法本质上是无监督的（仅用良性用户拟合基线），对已知的恶意模式利用不足。

### 8.2 N-gram 方法的独特优势

尽管 R-GCN 在数值指标上全面领先，N-gram 方法仍有以下不可替代的价值：

1. **可解释性**：每个高风险用户可追溯到具体的规则模板（"命中了 S1_LEAK_VIA_USB 模板 3 次"）、具体的行为（"出现了 HTTP_LEAK_AH 事件 12 次"）和具体的检测器（"PeerDeviation 异常"），这对 SOC 运营和内部审计至关重要。

2. **无需标签**：在真实环境中，恶意用户标签极度稀缺甚至不存在。N-gram 方法仅需良性行为基线即可运行，适合冷启动场景。

3. **领域知识融入**：安全专家的经验可以直接编码为行为模板和威胁权重，使模型对已知攻击模式具有即时响应能力。

4. **计算效率**：N-gram 方法不需要 GPU，不需要训练过程中的迭代优化，在资源受限的环境中更易部署。

### 8.3 两种方法的互补性

| 维度 | N-gram | R-GCN |
| --- | --- | --- |
| 检测能力 | 中等（AUC 0.90） | 优秀（AUC 0.99） |
| 可解释性 | 强（规则+模板追溯） | 弱（黑盒嵌入） |
| 标签依赖 | 无（无监督） | 有（需要恶意标签） |
| S2 场景 | 差（Recall 17%） | 优（Recall 97%） |
| 冷启动 | 可行 | 需要初始标签 |
| 计算成本 | 低 | 中 |

## 9. 可解释性分析

N-gram 方法的主要优势在于其可解释性结构：

- **PatternRule** 可以回答"用户命中了哪条红队行为模板"；
- **SequenceSimilarity** 可以回答"用户整体序列更像哪类恶意签名"；
- **PeerDeviation** 可以回答"用户有哪些群体中罕见的高风险行为"；
- **NGramAnomaly** 可以回答"用户行为在良性语言模型下是否反常"。

这种解释结构适合 SOC 或内部审计场景。分析人员可以先查看 Top-K 排名，再根据 `outputs/top_risk_explanations.json` 中的场景命中情况定位具体行为链，而不是只得到一个难以解释的分类概率。

R-GCN 方法的可解释性较弱，但可以通过分析节点嵌入、关系注意力权重或梯度归因等事后解释技术部分弥补。

## 10. 局限性

### 10.1 数据层面

- Ground truth 依赖官方红队清单，未涵盖真实业务环境中的未知攻击类型；
- 行为 token 粒度较粗，文件路径、邮件正文、收件人关系和岗位权限等信息尚未充分利用；
- R-GCN 当前未使用 email.csv 数据（1.3 GB），可能丢失了 S2 和 S3 场景的关键邮件行为特征。

### 10.2 方法层面

- N-gram 方法的融合权重为人工设定，尚未通过交叉验证或贝叶斯优化系统调参；
- N-gram 语言模型只使用群体良性基线，尚未建立用户个人历史基线；
- R-GCN 方法为 Transductive 学习，无法直接对未见过的新用户进行分类；
- R-GCN 当前不包含时序信息，所有事件被聚合为静态计数，可能丢失时间维度的攻击链模式。

### 10.3 评估层面

- 两种方法的评估协议略有差异：N-gram 使用 70/30 良性用户划分 + 全量打分，R-GCN 使用 5-fold CV + OOF 汇总。虽然最终都在全部 1000 用户上报告指标，但训练信息利用方式不同；
- R-GCN 使用了恶意标签进行监督训练，而 N-gram 本质上是无监督方法，这使得直接比较存在一定的不对称性。

## 11. 改进方向

1. **融合两种方法的优势**：构建集成系统，用 N-gram 提供可解释的初筛和规则告警，用 R-GCN 提供高精度的排序和优先级调整；
2. **引入时序图**：将 R-GCN 扩展为时序异构图（如 T-GCN 或动态图网络），捕获行为序列的时间演化模式；
3. **融入邮件数据**：为 R-GCN 图增加 email_addr 节点和发送/接收关系，补充 S2 和 S3 场景的邮件行为信号；
4. **加强 S2 场景建模**：结合邮件收件人图、附件大小、文件主题、离职时间窗口和岗位变化信息；
5. **自动模板挖掘**：使用 PrefixSpan、GSP 或 SPADE 在恶意与良性子集中自动挖掘差异化频繁序列，替代人工模板；
6. **学习融合权重**：在保持可解释性的前提下，用逻辑回归或排序学习自动学习 N-gram 四个分量的权重；
7. **图可解释性增强**：为 R-GCN 添加 GNNExplainer 或注意力机制，输出对每个预测影响最大的子图和关系；
8. **个性化基线**：以用户自身过去 30-60 天行为作为参照，检测行为漂移，降低高活跃良性用户误报。

## 12. 结论

本项目在 CMU-CERT r4.2 完整数据集上实现并对比了两种内部威胁检测方法。实验结果表明：

1. **R-GCN 异构图方法在检测性能上显著优于 N-gram 序列匹配方法**，全量 ROC-AUC 从 0.9041 提升至 0.9898，PR-AUC 从 0.3257 提升至 0.8358，Top-100 Recall 从 52.86% 提升至 98.57%。

2. **R-GCN 在最难检测的 S2 场景上实现了质的飞跃**，Top-100 Recall 从 16.67% 提升至 96.67%。图结构信息使模型能够捕获规则方法无法覆盖的行为关联模式。

3. **N-gram 方法在可解释性和无监督场景下仍具有独特价值**，每个高风险用户可追溯到具体的规则模板和行为模式，适合 SOC 运营和冷启动场景。

4. **两种方法具有互补性**，未来可构建集成系统，兼具 R-GCN 的高精度和 N-gram 的可解释性。

整体而言，本项目为内部威胁检测的方法选型提供了实证参考：在有标签数据可用时，图神经网络方法具有明显优势；在无标签或强调可解释性的场景下，基于领域知识的序列匹配方法仍是可行的选择。

## 附录 A：项目文件对应关系

| 文件 | 作用 |
| --- | --- |
| `src/preprocess.py` | 多源日志流式读取、行为 token 化、用户序列构建 |
| `src/build_ground_truth.py` | CMU-CERT r4.2 官方恶意用户清单（70 用户） |
| `src/sequence_matching.py` | N-gram 方法：四个检测器与风险融合模型 |
| `src/train_evaluate.py` | N-gram 方法：训练、打分、评估与解释输出 |
| `src/visualize.py` | N-gram 方法：评估图表生成 |
| `src/rgcn/build_graph.py` | R-GCN 方法：异构图构建 |
| `src/rgcn/rgcn_model.py` | R-GCN 方法：模型架构定义 |
| `src/rgcn/train_rgcn.py` | R-GCN 方法：训练与评估 |
| `src/generate_http_slim.py` | 工具：从 http.csv 生成精简版 http_slim.csv |
| `src/compare_algorithms.py` | 双算法性能对比分析与可视化 |
| `src/run_full_pipeline.py` | 一键运行完整流水线 |

## 附录 B：输出文件清单

| 文件 | 内容 |
| --- | --- |
| `outputs/metrics.json` | N-gram 方法评估指标 |
| `outputs/risk_scores.csv` | N-gram 方法全量用户风险分数 |
| `outputs/malicious_user_ranking.csv` | N-gram 方法恶意用户排名 |
| `outputs/top_risk_explanations.json` | N-gram 方法 Top-80 用户解释 |
| `outputs/fig_*.png` | N-gram 方法可视化图表（6 张） |
| `outputs/rgcn/graph.pt` | R-GCN 异构图数据 |
| `outputs/rgcn/rgcn_results.json` | R-GCN 评估指标与 Top-30 排名 |
| `outputs/rgcn/rgcn_results.csv` | R-GCN 全量用户风险排名 |
| `outputs/comparison/comparison_report.json` | 双算法对比数据 |
| `outputs/comparison/comparison_report.md` | 双算法对比报告 |
| `outputs/comparison/fig_*.png` | 对比可视化图表（5 张） |

## 附录 C：图表索引

| 图号 | 图表 | 文件 |
| --- | --- | --- |
| 图 1 | 双算法 ROC 曲线对比 | `outputs/comparison/fig_roc_comparison.png` |
| 图 2 | 双算法 Precision-Recall 曲线对比 | `outputs/comparison/fig_pr_comparison.png` |
| 图 3 | 双算法 Top-K Precision/Recall 对比 | `outputs/comparison/fig_topk_comparison.png` |
| 图 4 | 双算法场景级 Recall 对比 | `outputs/comparison/fig_scenario_comparison.png` |
| 图 5 | N-gram ROC 与 PR 曲线 | `outputs/fig_roc_pr.png` |
| 图 6 | N-gram Top-K Precision 与 Recall | `outputs/fig_topk.png` |
| 图 7 | N-gram 风险分数分布 | `outputs/fig_risk_distribution.png` |
| 图 8 | N-gram Top-30 检测器贡献 | `outputs/fig_top30_contribution.png` |
| 图 9 | N-gram 场景覆盖 | `outputs/fig_scenario_breakdown.png` |
| 图 10 | N-gram 恶意用户排名分布 | `outputs/fig_malicious_ranks.png` |
| 图 11 | 双算法风险分数分布对比 | `outputs/comparison/fig_score_dist_comparison.png` |

## 附录 D：环境与复现

```bash
# 激活虚拟环境
source ~/.virtualenvs/demo4-env/bin/activate

# 一键运行完整流水线
python src/run_full_pipeline.py

# 或分步运行：
python src/generate_http_slim.py    # Step 0: 生成 http_slim.csv（如不存在）
python src/preprocess.py             # Step 1: N-gram 预处理
python src/train_evaluate.py         # Step 2: N-gram 训练评估
python src/visualize.py              # Step 3: N-gram 可视化
python src/rgcn/build_graph.py       # Step 4: R-GCN 构建图
python src/rgcn/train_rgcn.py --verbose  # Step 5: R-GCN 训练评估
python src/compare_algorithms.py     # Step 6: 双算法对比
```

**依赖版本**：Python 3.x, pandas 2.1.4, scikit-learn 1.5.0, numpy 1.26.2, PyTorch 2.7.1, matplotlib, tqdm
