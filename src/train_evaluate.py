"""
训练并评估行为序列匹配模型。

流程:
  1. 加载预处理后的用户事件序列 + 红队 ground truth
  2. 用良性用户的序列拟合 N-gram 语言模型 / TF-IDF
  3. 对全部 1000 个用户打分
  4. 计算 ROC-AUC, PR-AUC, Top-K Precision/Recall
  5. 输出每个被命中恶意用户的解释 (命中了哪个场景/模板)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter

import sys
sys.path.append(os.path.dirname(__file__))

from sequence_matching import InsiderThreatScorer
from build_ground_truth import get_malicious_users

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = str(PROJECT_ROOT / 'outputs')


def load_data():
    print("[Eval] Loading sequences ...")
    with open(os.path.join(OUT_DIR, 'user_sequences.pkl'), 'rb') as f:
        events = pickle.load(f)
    # 只保留 token 序列
    seqs = {u: [tok for _, tok, _ in evs] for u, evs in events.items()}
    print(f"[Eval] {len(seqs)} users loaded.")
    return seqs


def split_train_users(seqs, malicious_set, train_ratio=0.7, seed=42):
    """
    用 70% 良性用户的序列做语言模型训练,
    剩下的 + 全部恶意用户做评估.
    """
    rng = np.random.RandomState(seed)
    benign_users = [u for u in seqs if u not in malicious_set]
    rng.shuffle(benign_users)
    cut = int(len(benign_users) * train_ratio)
    train_users = benign_users[:cut]
    eval_users = benign_users[cut:] + list(malicious_set & set(seqs))
    return train_users, eval_users


def metrics(scores_df, malicious_set, k_list=(10, 20, 30, 50, 70, 100)):
    """计算 ROC-AUC / PR-AUC / Top-K Precision&Recall."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_true = scores_df['user'].isin(malicious_set).astype(int).values
    y_score = scores_df['risk'].values

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return {}

    res = {
        'n_users':       len(y_true),
        'n_malicious':   n_pos,
        'n_benign':      int(len(y_true) - n_pos),
        'roc_auc':       float(roc_auc_score(y_true, y_score)),
        'pr_auc':        float(average_precision_score(y_true, y_score)),
    }

    order = np.argsort(-y_score)
    for K in k_list:
        topk = order[:K]
        hit = int(y_true[topk].sum())
        res[f'top{K}_precision'] = hit / K
        res[f'top{K}_recall']    = hit / n_pos
    return res


def explain_user(record, top_n=3):
    """为单个用户构造解释字符串."""
    rd = record.get('rule_details', {})
    sd = record.get('sim_details', {})
    rule_top = sorted(rd.items(), key=lambda x: -x[1])[:top_n]
    sim_top  = sorted(sd.items(), key=lambda x: -x[1])[:top_n]
    return {
        'rule_hits': [{'scenario': k, 'score': float(v)} for k, v in rule_top if v > 0],
        'sim_top':   [{'scenario': k, 'score': float(v)} for k, v in sim_top],
    }


def main():
    malicious = get_malicious_users()                 # {uid: scenario}
    malicious_set = set(malicious.keys())

    seqs = load_data()
    train_users, eval_users = split_train_users(seqs, malicious_set)
    print(f"[Eval] Train users: {len(train_users)}  "
          f"| Eval users: {len(eval_users)}  "
          f"| Malicious in eval: {sum(u in malicious_set for u in eval_users)}")

    scorer = InsiderThreatScorer(weights=(0.40, 0.20, 0.25, 0.15), ngram_n=3)

    print("[Eval] Fitting N-gram + stats on benign training users ...")
    scorer.fit([seqs[u] for u in train_users])

    print("[Eval] Scoring all users (incl. train+eval) for ranking ...")
    raw = scorer.raw_scores({u: seqs[u] for u in seqs})
    final = scorer.aggregate(raw)

    rows = []
    for u, r in final.items():
        rows.append({
            'user':     u,
            'risk':     r['risk'],
            'z_rule':   r['z_rule'],
            'z_lm':     r['z_lm'],
            'z_sim':    r['z_sim'],
            'z_peer':   r['z_peer'],
            'rule':     r['rule'],
            'lm':       r['lm'],
            'sim':      r['sim'],
            'peer':     r['peer'],
            'len':      r['len'],
            'is_malicious': u in malicious_set,
            'scenario':     malicious.get(u, 0),
        })
    df = pd.DataFrame(rows).sort_values('risk', ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUT_DIR, 'risk_scores.csv'), index=False)

    # ---- 评估指标 (在 eval 用户子集上) ----
    df_eval = df[df['user'].isin(eval_users)].copy()
    res_eval = metrics(df_eval, malicious_set)
    res_all  = metrics(df, malicious_set)

    print("\n=== Metrics on EVAL set (held-out benign + all malicious) ===")
    for k, v in res_eval.items():
        print(f"  {k:<22s}: {v:.4f}" if isinstance(v, float) else f"  {k:<22s}: {v}")

    print("\n=== Metrics on ALL 1000 users ===")
    for k, v in res_all.items():
        print(f"  {k:<22s}: {v:.4f}" if isinstance(v, float) else f"  {k:<22s}: {v}")

    with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'eval': res_eval, 'all': res_all}, f, indent=2)

    # ---- Top-K 风险用户 + 解释 ----
    top = df.head(80).copy()
    explanations = []
    for _, row in top.iterrows():
        u = row['user']
        explanations.append({
            'user':     u,
            'risk':     row['risk'],
            'is_malicious': bool(row['is_malicious']),
            'scenario':     int(row['scenario']),
            'len':          int(row['len']),
            **explain_user(final[u]),
        })
    with open(os.path.join(OUT_DIR, 'top_risk_explanations.json'), 'w') as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)

    # ---- 命中情况一览 ----
    hits = df[df['is_malicious']].copy()
    hits['rank'] = hits.index + 1
    hits.to_csv(os.path.join(OUT_DIR, 'malicious_user_ranking.csv'), index=False)
    print(f"\n[Eval] All malicious users with their ranks saved to "
          f"outputs/malicious_user_ranking.csv")
    print(f"[Eval] Top-K explanations saved to outputs/top_risk_explanations.json")

    # 打印 top 20
    print("\n=== Top 20 risky users ===")
    print(df.head(20)[['user', 'risk', 'rule', 'lm', 'sim', 'peer',
                       'is_malicious', 'scenario']].to_string(index=False))

    return df, res_eval, res_all


if __name__ == "__main__":
    main()
