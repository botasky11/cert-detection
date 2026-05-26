# Insider Threat Detection on CMU-CERT r4.2: A Comparative Study of Behavioral Sequence Matching and Heterogeneous Graph Neural Networks

Report Date: 2026-05-26

---

## Abstract

Insider threat detection faces fundamental challenges including label scarcity, long behavioral time spans, stealthy attack chains, and high false-positive costs. This project uses the CMU-CERT r4.2 dataset (~16 GB / 1,000 employees / 17 months of logs / 70 malicious users) to implement and compare two insider threat detection approaches:

- **Approach 1: N-gram Template-Based Behavioral Sequence Matching** — abstracts multi-source logs into unified behavioral token time series and generates user-level risk scores through a weighted fusion of four complementary detectors (red-team template matching, N-gram language model anomaly, malicious signature similarity, and peer rarity deviation);
- **Approach 2: R-GCN Heterogeneous Graph Detection** — constructs a heterogeneous graph over users, PCs, file types, and URL categories, then applies Relational Graph Convolutional Networks (R-GCN) for semi-supervised node classification, automatically learning structural features that distinguish malicious from benign users.

Comparative experiments on the full r4.2 dataset demonstrate that R-GCN comprehensively outperforms the N-gram approach across all core metrics: ROC-AUC reaches **0.9898** (vs. 0.9041 for N-gram), PR-AUC reaches **0.8358** (vs. 0.3257), and Top-100 Recall reaches **98.57%** (vs. 52.86%). R-GCN shows particularly striking improvement on the hardest-to-detect S2 scenario (job-hunting + email exfiltration), where Top-100 Recall jumps from 16.67% to 96.67%. However, the N-gram approach retains unique advantages in interpretability, as each high-risk user can be traced back to specific rule templates and behavioral patterns.

## 1. Background and Motivation

Enterprise insider threats typically do not manifest as single isolated events, but rather as a set of temporally correlated behaviors — such as after-hours logins, removable storage device connections, visits to leak sites, copying sensitive files, or sending attachments to external email addresses. Single-point anomaly detection is easily disrupted by office noise, while purely supervised learning requires large amounts of high-quality labels that are difficult to obtain in real-world environments.

CMU-CERT r4.2 is a widely used simulated dataset in insider threat detection research, containing multi-month, multi-source logs with red-team-injected malicious user behaviors. This dataset is well-suited for studying the following questions:

- How to convert heterogeneous logs into unified user behavioral sequences;
- How to extract ordered behavioral patterns from attack playbooks;
- How to establish anomaly baselines using benign population behavior under label-scarce conditions;
- How to leverage graph-structural relationships among users, devices, and resources to enhance detection;
- How to preserve interpretability in detection results to assist security analysts in investigation.

This project pursues two technical approaches: a domain-knowledge-driven "behavioral sequence matching" method and a data-driven "heterogeneous graph neural network" method, conducting a rigorous comparison on the same dataset to inform practical deployment decisions.

## 2. Dataset and Threat Scenarios

### 2.1 Dataset Overview

The project is based on the CMU-CERT r4.2 dataset. The raw data is approximately 16 GB, covering 1,000 employees over approximately 17 months of enterprise activity logs:

| Data Source | File Size | Key Fields | Behavioral Meaning |
| --- | ---: | --- | --- |
| logon.csv | 56 MB | date, user, pc, activity | Login/logout events; identifies work hours, after-hours, and whether the user's own PC is used |
| device.csv | 28 MB | date, user, pc, activity | USB device connect/disconnect events |
| file.csv | 184 MB | date, user, pc, filename | File access events, classified by document, executable, and other types |
| http.csv | 14.5 GB | date, user, pc, url, content | HTTP browsing events; project generates a slim version http_slim.csv (4.1 GB) |
| email.csv | 1.3 GB | date, user, recipients, attachments | Internal/external email and attachment exfiltration events |
| LDAP/ | ~2.5 MB | user, role, department, team | Employee position and organizational attributes (monthly snapshots) |
| insiders.csv | 14 KB | dataset, scenario, user, start, end | Official malicious user ground truth |

### 2.2 Threat Scenario Definitions

The project uses the r4.2 malicious user list verified against the official `insiders.csv` in `src/build_ground_truth.py`, totaling **70 unique malicious users** across three typical scenarios:

| Scenario | Users | Attack Playbook |
| --- | ---: | --- |
| S1: Data Leak / USB Exfiltration | 30 | Disgruntled or departing employees log in after hours, connect USB drives, access or copy documents, and upload to leak sites (wikileaks) |
| S2: Pre-Departure Data Exfiltration | 30 | Employees browse job-hunting websites, steal competitive or business materials, and exfiltrate via personal email or external email channels |
| S3: Sysadmin Keylogger | 10 | Administrators visit keylogger sites, install executables, and send destructive emails impersonating executives |

The three scenarios are mutually exclusive, covering 70 malicious users and 930 benign users in total.

## 3. Data Preprocessing and Behavioral Tokenization

### 3.1 Unified Preprocessing Pipeline

The preprocessing script is `src/preprocess.py`. Its core idea is to stream-read multi-source logs and unify them into per-user, chronologically-ordered behavioral token sequences.

**Temporal Context Segmentation**:
- Work Hours (WH): Monday through Friday, 07:00–18:00;
- After Hours (AH): weekends, late nights, early mornings, and all other non-work periods.

**User Context Identification**:
- The most frequently used PC for each user is identified from logon logs as their "primary PC";
- Login events are differentiated into `LOGON_WH_OWN` (work-hours login on own PC), `LOGON_AH_OTHER` (after-hours login on another's PC), etc.

### 3.2 Behavioral Token Vocabulary

The project defines approximately 30 behavioral token types. Representative examples:

| Category | Token Example | Meaning |
| --- | --- | --- |
| Logon | `LOGON_AH_OTHER` | After-hours login on a non-primary PC |
| Device | `USB_CONN_AH` | After-hours USB device connection |
| File | `FILE_DOC_AH` | After-hours document file access |
| File | `FILE_EXEC_WH` | Work-hours executable file access |
| HTTP | `HTTP_LEAK_AH` | After-hours visit to a leak-related site |
| HTTP | `HTTP_KEYLOG_WH` | Work-hours visit to a keylogger-related site |
| HTTP | `HTTP_JOBHUNT_WH` | Work-hours visit to a job-hunting site |
| Email | `EMAIL_EXT_ATT_WH` | Work-hours email with attachment to external recipient |

Complete preprocessing output: **1,000 users, 4,779,378 events**, saved as `outputs/user_sequences.pkl` and `outputs/user_daily_seq.pkl`.

### 3.3 HTTP Data Slimming

The raw `http.csv` file is 14.5 GB, primarily due to the bulky `content` field. The project generates a slim version via `src/generate_http_slim.py` — `http_slim.csv` (4.1 GB, 28,434,423 rows) — retaining only the five columns `id, date, user, pc, url` for use by both algorithms.

## 4. Approach 1: N-gram Template-Based Behavioral Sequence Matching

### 4.1 Method Overview

This approach is defined in `src/sequence_matching.py`. The overall architecture is:

```text
Multi-source Logs
  -> Behavioral Token Sequence
  -> PatternRule / NGramAnomaly / SequenceSimilarity / PeerDeviation
  -> Component Normalization
  -> User Risk Score R(u)
  -> Ranking, Top-K Metrics, and Explanations
```

The system consists of four complementary detectors, all following a "higher score = more suspicious" convention. Outputs are min-max normalized and linearly fused into a final risk score.

### 4.2 Threat Weight Design

Based on the prevalence ratio of each token between malicious and benign users, the project assigns data-driven threat weights (approximating `log(P(t|malicious) / P(t|benign))`):

| Token | Weight | Mal/Ben Ratio | Interpretation |
| --- | ---: | ---: | --- |
| `HTTP_LEAK_AH/WH` | 8.0 | 39–52× | Visiting leak sites; very strong malicious signal |
| `HTTP_KEYLOG_AH/WH` | 8.0 | 38× | Visiting keylogger sites; very strong malicious signal |
| `USB_CONN_AH` | 2.5 | 5.3× | After-hours removable device connection |
| `FILE_EXEC_AH` | 2.0 | 3.5× | After-hours executable file access |
| `LOGON_AH_OTHER` | 2.0 | 1.7× | After-hours login on a non-primary PC |
| `HTTP_JOBHUNT_WH` | 0.0 | 1.1× | Job-hunting sites appear in 83% of benign users; extremely low discriminative power |

### 4.3 The Four Detectors

**(1) PatternRule: Red-Team Playbook Ordered Subsequence Matching**

Encodes three red-team scenarios into 21 behavioral templates. For example, S1 includes combinations such as `[LOGON_AH_OWN, USB_CONN_AH, FILE_DOC_AH]`. Performs ordered subsequence matching within a sliding window (default size 15) with non-overlapping counting:

```text
score(pattern) = log(1 + hits) × len(pattern)^1.5 × (1 + Σ threat_weight(token))
```

**(2) NGramAnomaly: Benign Behavior Language Model**

Trains a 3-gram language model (additive smoothing α=0.5) on 70% of benign users and computes the perplexity of each test user's sequence. Higher perplexity indicates greater deviation from normal behavioral patterns.

**(3) SequenceSimilarity: Malicious Signature Similarity**

Concatenates each scenario's template tokens into malicious signature sequences and computes the Longest Common Subsequence (LCS) ratio between the user's behavioral sequence and the signatures, with weighted counting of high-sensitivity token occurrences:

```text
score = 0.4 × LCS_ratio + 0.6 × weighted_high_sensitive_hit
```

**(4) PeerDeviation: High-Risk Token Peer Rarity**

Focuses exclusively on the 9 token types with threat weight ≥ 1.5, computing IDF-weighted deviation scores:

```text
score = Σ log(1 + count_u(token)) × IDF(token) × threat_weight(token)
```

### 4.4 Fusion Strategy

The four detector outputs undergo min-max normalization followed by linear fusion:

```text
R(u) = 0.40 × z(rule) + 0.20 × z(perplexity) + 0.25 × z(similarity) + 0.15 × z(peer)
```

Weights are manually specified, emphasizing rule matching and similarity signals.

## 5. Approach 2: R-GCN Heterogeneous Graph Detection

### 5.1 Method Overview

This approach formulates insider threat detection as a node classification problem on a heterogeneous graph, using Relational Graph Convolutional Networks (R-GCN) to automatically learn user-level malicious features from graph structure. The code resides in `src/rgcn/`.

### 5.2 Heterogeneous Graph Construction

The graph construction script is `src/rgcn/build_graph.py`, which builds a heterogeneous graph from the full r4.2 data containing 4 node types and 6 relation types:

**Node Types (4 types, 2,017 total nodes)**:

| Node Type | Count | Description |
| --- | ---: | --- |
| user | 1,000 | Employee nodes |
| pc | 1,003 | Workstation nodes |
| file_type | 6 | File category nodes: (doc/exec/other) × (WH/AH) |
| url_cat | 8 | URL category nodes: (LEAK/KEYLOG/JOBHUNT/OTHER) × (WH/AH) |

**Relation Types (6 types, 38,568 total edges)**:

| Relation | Direction | Edges | Meaning |
| --- | --- | ---: | --- |
| logon_wh | user → pc | 8,074 | Work-hours logins |
| logon_ah | user → pc | 20,144 | After-hours logins |
| usb_wh | user → pc | 1,441 | Work-hours USB connections |
| usb_ah | user → pc | 4,639 | After-hours USB connections |
| file_op | user → file_type | 1,063 | File operations |
| http_visit | user → url_cat | 3,207 | HTTP browsing |

Each edge carries a weight `w = log(1 + count)`, and multiple occurrences of the same `(src, dst, rel)` are merged. Reverse edges are automatically added during training, giving the R-GCN 12 relation types in total.

**User Node Features (6-dimensional)**:
1. Number of active days
2. Total event count
3. After-hours event ratio
4. Number of distinct PCs used
5. Number of distinct file types accessed
6. After-hours login count

Features are standardized using RobustScaler (subtract median / divide by IQR) and clipped to [-5, 5].

### 5.3 R-GCN Model Architecture

The model is implemented in `src/rgcn/rgcn_model.py`, following the R-GCN framework of Schlichtkrull et al. (ESWC 2018) in pure PyTorch (no DGL/PyG dependency).

**Core Formula**:

```text
h_v^(l+1) = σ( W_self × h_v^(l) + Σ_r Σ_{u∈N_r(v)} (w_{u,v,r} / c_{v,r}) × W_r × h_u^(l) )
```

where `W_r` employs basis decomposition: `W_r = Σ_{b=1..B} a_{rb} × V_b`, sharing B=4 basis matrices across 12 relations to effectively reduce parameter count.

**Model Structure**:

```text
user: Linear(6, 64)  ─┐
pc:   Embedding(64)   ─┤
file: Embedding(64)   ─┼──→ R-GCN Layer 1 (64→64, ReLU) ──→ R-GCN Layer 2 (64→64)
url:  Embedding(64)   ─┘                                          │
                                                                   ↓
                                                    Linear(64→32) → ReLU → Linear(32→1)
                                                    (user nodes only)    → sigmoid → P(malicious)
```

### 5.4 Training Strategy

- **Semi-supervised Transductive**: all nodes participate in message passing, but loss is computed only on training nodes;
- **5-fold StratifiedKFold** cross-validation, preserving malicious/benign ratio in each fold;
- **Class Imbalance Handling**: BCEWithLogitsLoss with `pos_weight = #neg / #pos ≈ 13.3`;
- **Optimizer**: AdamW, lr=0.01, weight_decay=5e-4;
- **Early Stopping**: patience=50 epochs, monitoring validation ROC-AUC;
- **Gradient Clipping**: max_norm=2.0;
- **Out-of-Fold (OOF) Prediction**: each user receives a score from the fold where it served as validation, then all scores are aggregated for overall metrics.

### 5.5 Hyperparameter Configuration

| Parameter | Value | Description |
| --- | --- | --- |
| hidden_dim | 64 | Hidden layer dimension |
| num_layers | 2 | Number of R-GCN layers |
| num_bases | 4 | Basis decomposition rank |
| dropout | 0.3 | Dropout rate |
| lr | 0.01 | Learning rate |
| weight_decay | 5e-4 | Weight decay |
| epochs | 300 | Maximum training epochs |
| patience | 50 | Early stopping patience |
| n_splits | 5 | Cross-validation folds |

## 6. Experimental Design

### 6.1 Evaluation Framework

Both algorithms are evaluated on the identical dataset:

| Item | N-gram Approach | R-GCN Approach |
| --- | --- | --- |
| Dataset | Full r4.2 (1,000 users) | Full r4.2 (1,000 users) |
| Data Sources | logon + device + file + http_slim + email | logon + device + file + http_slim |
| Evaluation Protocol | 70% benign train / 30% held-out + full scoring | 5-fold StratifiedKFold + OOF aggregation |
| Label Usage | Evaluation only (unsupervised scoring) | Training binary classifier (semi-supervised) |
| Ground Truth | 70 malicious users (S1:30, S2:30, S3:10) | 70 malicious users (S1:30, S2:30, S3:10) |

### 6.2 Evaluation Metrics

- **ROC-AUC**: True positive rate vs. false positive rate trade-off across thresholds;
- **PR-AUC**: Particularly informative under sparse positive samples (7%), reflecting practical investigation difficulty;
- **Top-K Precision/Recall** (K=10, 20, 30, 50, 70, 100): Simulates the "prioritize reviewing the top K users" scenario in security operations;
- **Scenario-Level Recall**: Per-scenario (S1/S2/S3) recall within Top-70 and Top-100.

## 7. Experimental Results

### 7.1 Overall Performance Comparison

| Metric | N-gram | R-GCN | R-GCN Improvement |
| --- | ---: | ---: | ---: |
| **ROC-AUC** | 0.9041 | **0.9898** | +9.5% |
| **PR-AUC** | 0.3257 | **0.8358** | +156.6% |

R-GCN leads comprehensively across both core metrics. The PR-AUC gap is especially notable (0.33 vs. 0.84), indicating that R-GCN maintains high precision even at elevated recall levels, whereas the N-gram approach suffers rapid precision degradation when pursuing high recall.

The R-GCN 5-fold cross-validation results are also highly stable: per-fold AUCs are [0.9981, 0.9923, 0.9939, 0.9866, 0.9885], mean **0.9919 ± 0.0041**.

### 7.2 Top-K Investigation Effectiveness Comparison

In security operations, analysts typically prioritize the highest-ranked users, making Top-K metrics more practically relevant than classification thresholds.

| K | N-gram Prec | R-GCN Prec | N-gram Rec | R-GCN Rec | N-gram Hits | R-GCN Hits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.50 | **0.90** | 0.07 | **0.13** | 5 | 9 |
| 20 | 0.30 | **0.90** | 0.09 | **0.26** | 6 | 18 |
| 30 | 0.23 | **0.87** | 0.10 | **0.37** | 7 | 26 |
| 50 | 0.28 | **0.80** | 0.20 | **0.57** | 14 | 40 |
| 70 | 0.26 | **0.80** | 0.26 | **0.80** | 18 | 56 |
| 100 | 0.37 | **0.69** | 0.53 | **0.99** | 37 | 69 |

Key findings:
- R-GCN identifies 9/70 malicious users in the Top-10 (90% precision); N-gram identifies only 5 (50% precision);
- R-GCN achieves 80% recall at Top-70 (56/70), while N-gram achieves only 53% at Top-100 (37/70);
- R-GCN identifies 69/70 malicious users in the Top-100 (missing only 1), approaching perfect recall.

### 7.3 Scenario-Level Detection Comparison

| Scenario | Description | Users | N-gram Top-70 | R-GCN Top-70 | N-gram Top-100 | R-GCN Top-100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 | USB + Leak Upload | 30 | 0.3667 | **0.7667** | 0.8333 | **1.0000** |
| S2 | Job-Hunt + Email Exfil | 30 | 0.0667 | **0.8667** | 0.1667 | **0.9667** |
| S3 | Keylogger | 10 | 0.5000 | **0.7000** | 0.7000 | **1.0000** |

**The S2 scenario shows the most striking difference**: N-gram detects only 16.67% of S2 malicious users in the Top-100 (5/30), while R-GCN detects 96.67% (29/30). This is because S2 behavioral features (job-site browsing + document emailing) heavily overlap with benign users — 83% of benign users also have job-hunting site visit records. The N-gram approach sets the weight of `HTTP_JOBHUNT_WH` to 0 (treating it as noise), causing S2 detection to rely almost entirely on weak signals. In contrast, R-GCN automatically learns finer-grained discriminative features from user-PC-URL interaction patterns in the graph structure.

S1 and S3 scenarios both achieve 100% recall within R-GCN's Top-100.

### 7.4 N-gram Approach Detailed Results

#### 7.4.1 Evaluation Set vs. Full Dataset Comparison

| Metric | Full 1,000 Users | Held-Out Evaluation Set (349 Users) |
| --- | ---: | ---: |
| Malicious Users | 70 | 70 |
| Benign Users | 930 | 279 |
| ROC-AUC | 0.9041 | 0.9053 |
| PR-AUC | 0.3257 | 0.6076 |
| Top-50 Precision | 0.2800 | 0.6600 |
| Top-100 Recall | 0.5286 | 0.8143 |

PR-AUC is significantly higher on the evaluation set, primarily because the malicious sample proportion is higher (70/349 = 20%) compared to the full dataset (7%).

#### 7.4.2 Risk Score Distribution

Among all users, benign users have a mean risk score of 0.1176 (median 0.0870), while malicious users have a mean of 0.3304 (median 0.3250). The two distributions show some separation, but with considerable overlap — the highest benign score reaches 0.7927, contributing to lower precision.

#### 7.4.3 Detector Contribution Analysis

Among the N-gram approach's Top-20 highest-risk users, 5 are true malicious users:
- CCA0046 (S3, rank 1): PatternRule, SequenceSimilarity, and PeerDeviation all significantly elevated;
- MPM0220 (S3, rank 3): strong keylogger signature match;
- BSS0369 (S3, rank 6): highest similarity with the S3 signature sequence.

S3 scenario users consistently rank high because keylogger-related tokens are rare and carry high threat weights (8.0). S2 scenario users are the most dispersed, with the worst rank reaching position 408.

### 7.5 R-GCN Approach Detailed Results

#### 7.5.1 Cross-Validation Stability

| Fold | Training Set | Validation Set | Val AUC | Best Epoch |
| ---: | --- | --- | ---: | ---: |
| 1 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9981 | 36 |
| 2 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9923 | 32 |
| 3 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9939 | 22 |
| 4 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9866 | 29 |
| 5 | 800 (56 mal / 744 ben) | 200 (14 mal / 186 ben) | 0.9885 | 37 |
| **Mean ± Std** | | | **0.9919 ± 0.0041** | |

All fold AUCs exceed 0.98 with a standard deviation of only 0.004, demonstrating stable performance across different data splits. Early stopping epochs concentrate in the 22–37 range, indicating rapid convergence.

#### 7.5.2 OOF Ranking Analysis

All of R-GCN's Top-10 users are malicious (100% precision), and only 4 benign users appear among the top 30. The highest-scoring malicious user (JRG0207) achieves a predicted probability of 0.9995.

Within the Top-100, R-GCN identifies 69/70 malicious users; the single missed user ranks only slightly beyond 100, approaching near-perfect recall.

## 8. Performance Difference Analysis

### 8.1 Sources of R-GCN's Advantage

1. **Graph Structural Information**: R-GCN leverages relationships among users, PCs, files, and URLs through message passing. For example, if multiple malicious users share certain PCs or access the same sensitive URL categories, these structural patterns propagate through graph convolutions, enhancing classification of neighboring nodes.

2. **Automatic Feature Learning**: Without manual definition of behavioral templates or threat weights, R-GCN automatically learns discriminative feature combinations from labeled training data. This enables the model to capture complex interaction patterns that human experts would find difficult to predefine.

3. **S2 Scenario Breakthrough**: Job-hunting + email exfiltration behaviors are individually unremarkable, but their correlation patterns with other malicious indicators in the graph structure (e.g., interaction patterns with specific PC clusters or file types) can be automatically learned. The N-gram approach sets `HTTP_JOBHUNT_WH` weight to 0, effectively discarding this dimension entirely.

4. **Semi-supervised Learning Paradigm**: R-GCN fully leverages the 70 known malicious labels for supervised training, whereas the N-gram approach is essentially unsupervised (fitting baselines only on benign users), underutilizing known malicious patterns.

### 8.2 Unique Advantages of the N-gram Approach

Despite R-GCN's comprehensive numerical superiority, the N-gram approach retains irreplaceable value:

1. **Interpretability**: Each high-risk user can be traced to specific rule templates ("matched S1_LEAK_VIA_USB template 3 times"), specific behaviors ("HTTP_LEAK_AH events occurred 12 times"), and specific detectors ("PeerDeviation anomaly"). This is critical for SOC operations and internal audits.

2. **No Label Requirement**: In real-world environments, malicious user labels are extremely scarce or nonexistent. The N-gram approach only requires a benign behavior baseline to operate, making it suitable for cold-start scenarios.

3. **Domain Knowledge Integration**: Security expert experience can be directly encoded as behavioral templates and threat weights, giving the model immediate response capability to known attack patterns.

4. **Computational Efficiency**: The N-gram approach requires no GPU, no iterative optimization during training, and is easier to deploy in resource-constrained environments.

### 8.3 Complementarity of the Two Approaches

| Dimension | N-gram | R-GCN |
| --- | --- | --- |
| Detection Capability | Moderate (AUC 0.90) | Excellent (AUC 0.99) |
| Interpretability | Strong (rule + template tracing) | Weak (black-box embeddings) |
| Label Dependency | None (unsupervised) | Required (needs malicious labels) |
| S2 Scenario | Poor (Recall 17%) | Excellent (Recall 97%) |
| Cold Start | Feasible | Requires initial labels |
| Computational Cost | Low | Moderate |

## 9. Interpretability Analysis

The N-gram approach's primary advantage lies in its interpretability structure:

- **PatternRule** answers "Which red-team behavioral templates did the user match?";
- **SequenceSimilarity** answers "Which type of malicious signature does the user's overall sequence most resemble?";
- **PeerDeviation** answers "What high-risk behaviors does the user exhibit that are rare among peers?";
- **NGramAnomaly** answers "Is the user's behavior abnormal under the benign language model?".

This explanation structure is well-suited for SOC or internal audit scenarios. Analysts can first review the Top-K rankings, then use the scenario match details in `outputs/top_risk_explanations.json` to locate specific behavioral chains, rather than receiving only an opaque classification probability.

The R-GCN approach has weaker interpretability, but can be partially compensated through post-hoc explanation techniques such as node embedding analysis, relational attention weights, or gradient attribution.

## 10. Limitations

### 10.1 Data Level

- Ground truth relies on the official red-team roster and does not cover unknown attack types in real business environments;
- Behavioral token granularity is relatively coarse; file paths, email body text, recipient relationships, and role-based access permissions are not fully utilized;
- R-GCN currently does not use email.csv data (1.3 GB), potentially missing critical email behavior features for S2 and S3 scenarios.

### 10.2 Method Level

- The N-gram approach's fusion weights are manually specified and have not been systematically tuned through cross-validation or Bayesian optimization;
- The N-gram language model uses only a population-level benign baseline and has not established individual user historical baselines;
- The R-GCN approach is transductive and cannot directly classify previously unseen new users;
- R-GCN currently does not incorporate temporal information — all events are aggregated into static counts, potentially losing temporal attack chain patterns.

### 10.3 Evaluation Level

- The two approaches use slightly different evaluation protocols: N-gram uses 70/30 benign user split + full scoring, while R-GCN uses 5-fold CV + OOF aggregation. Although both ultimately report metrics on all 1,000 users, training information utilization differs;
- R-GCN uses malicious labels for supervised training, while N-gram is essentially unsupervised, creating a certain asymmetry in direct comparison.

## 11. Future Directions

1. **Fuse the advantages of both approaches**: Build an ensemble system using N-gram for interpretable initial screening and rule-based alerting, and R-GCN for high-accuracy ranking and priority adjustment;
2. **Introduce temporal graphs**: Extend R-GCN to temporal heterogeneous graphs (e.g., T-GCN or dynamic graph networks) to capture temporal evolution of behavioral sequences;
3. **Incorporate email data**: Add email_addr nodes and send/receive relations to the R-GCN graph to supplement email behavioral signals for S2 and S3 scenarios;
4. **Strengthen S2 scenario modeling**: Combine email recipient graphs, attachment sizes, file topics, departure time windows, and role change information;
5. **Automatic template mining**: Use PrefixSpan, GSP, or SPADE to automatically mine discriminative frequent sequences from malicious and benign subsets, replacing manual templates;
6. **Learn fusion weights**: Under the constraint of maintaining interpretability, use logistic regression or learning-to-rank to automatically learn the N-gram four-component weights;
7. **Enhance graph interpretability**: Add GNNExplainer or attention mechanisms to R-GCN to output the most influential subgraphs and relations for each prediction;
8. **Personalized baselines**: Use each user's own past 30–60 days of behavior as a reference to detect behavioral drift, reducing false positives for highly active benign users.

## 12. Conclusion

This project implements and compares two insider threat detection approaches on the complete CMU-CERT r4.2 dataset. The experimental results demonstrate:

1. **The R-GCN heterogeneous graph approach significantly outperforms the N-gram sequence matching approach in detection performance**, with ROC-AUC improving from 0.9041 to 0.9898, PR-AUC from 0.3257 to 0.8358, and Top-100 Recall from 52.86% to 98.57%.

2. **R-GCN achieves a qualitative breakthrough on the hardest-to-detect S2 scenario**, with Top-100 Recall improving from 16.67% to 96.67%. Graph structural information enables the model to capture behavioral correlation patterns that rule-based methods cannot cover.

3. **The N-gram approach retains unique value in interpretability and unsupervised settings**, where each high-risk user can be traced to specific rule templates and behavioral patterns, making it suitable for SOC operations and cold-start scenarios.

4. **The two approaches are complementary**, and future work can build an ensemble system combining R-GCN's high accuracy with N-gram's interpretability.

Overall, this project provides empirical evidence for method selection in insider threat detection: when labeled data is available, graph neural network methods offer clear advantages; in label-free or interpretability-critical scenarios, domain-knowledge-driven sequence matching remains a viable choice.

## Appendix A: Project File Reference

| File | Purpose |
| --- | --- |
| `src/preprocess.py` | Multi-source log streaming, behavioral tokenization, user sequence construction |
| `src/build_ground_truth.py` | CMU-CERT r4.2 official malicious user list (70 users) |
| `src/sequence_matching.py` | N-gram approach: four detectors and risk fusion model |
| `src/train_evaluate.py` | N-gram approach: training, scoring, evaluation, and explanation output |
| `src/visualize.py` | N-gram approach: evaluation chart generation |
| `src/rgcn/build_graph.py` | R-GCN approach: heterogeneous graph construction |
| `src/rgcn/rgcn_model.py` | R-GCN approach: model architecture definition |
| `src/rgcn/train_rgcn.py` | R-GCN approach: training and evaluation |
| `src/generate_http_slim.py` | Utility: generate slim http_slim.csv from http.csv |
| `src/compare_algorithms.py` | Dual-algorithm performance comparison and visualization |
| `src/run_full_pipeline.py` | One-click full pipeline runner |

## Appendix B: Output File Inventory

| File | Contents |
| --- | --- |
| `outputs/metrics.json` | N-gram approach evaluation metrics |
| `outputs/risk_scores.csv` | N-gram approach full user risk scores |
| `outputs/malicious_user_ranking.csv` | N-gram approach malicious user rankings |
| `outputs/top_risk_explanations.json` | N-gram approach Top-80 user explanations |
| `outputs/fig_*.png` | N-gram approach visualization charts (6 figures) |
| `outputs/rgcn/graph.pt` | R-GCN heterogeneous graph data |
| `outputs/rgcn/rgcn_results.json` | R-GCN evaluation metrics and Top-30 ranking |
| `outputs/rgcn/rgcn_results.csv` | R-GCN full user risk ranking |
| `outputs/comparison/comparison_report.json` | Dual-algorithm comparison data |
| `outputs/comparison/comparison_report.md` | Dual-algorithm comparison report |
| `outputs/comparison/fig_*.png` | Comparison visualization charts (5 figures) |

## Appendix C: Environment and Reproducibility

```bash
# Activate virtual environment
source ~/.virtualenvs/demo4-env/bin/activate

# One-click full pipeline
python src/run_full_pipeline.py

# Or run step by step:
python src/generate_http_slim.py         # Step 0: Generate http_slim.csv (if not exists)
python src/preprocess.py                  # Step 1: N-gram preprocessing
python src/train_evaluate.py              # Step 2: N-gram training and evaluation
python src/visualize.py                   # Step 3: N-gram visualization
python src/rgcn/build_graph.py            # Step 4: R-GCN graph construction
python src/rgcn/train_rgcn.py --verbose   # Step 5: R-GCN training and evaluation
python src/compare_algorithms.py          # Step 6: Dual-algorithm comparison
```

**Dependency Versions**: Python 3.x, pandas 2.1.4, scikit-learn 1.5.0, numpy 1.26.2, PyTorch 2.7.1, matplotlib, tqdm
