"""
模型评估结果可视化:
  - 风险分数分布 (恶意 vs 良性)
  - ROC / PR 曲线
  - Top-K 用户表
  - 各检测器贡献雷达 (Top 恶意命中样例)
  - 场景命中分布
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent.parent
OUT_DIR = str(PROJECT_ROOT / 'outputs')
plt.rcParams['font.sans-serif']    = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_score_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, df['risk'].max(), 50)
    ax.hist(df.loc[~df['is_malicious'], 'risk'], bins=bins,
            alpha=0.6, label='Benign (n={})'.format((~df['is_malicious']).sum()),
            color='#4C72B0')
    ax.hist(df.loc[df['is_malicious'], 'risk'], bins=bins,
            alpha=0.8, label='Malicious (n={})'.format(df['is_malicious'].sum()),
            color='#C44E52')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('# Users')
    ax.set_title('Risk Score Distribution: Benign vs Malicious')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_risk_distribution.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('[viz]', out)


def plot_roc_pr(df):
    y = df['is_malicious'].astype(int).values
    s = df['risk'].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y, s)
    ax = axes[0]
    ax.plot(fpr, tpr, color='#C44E52', lw=2,
            label=f'ROC AUC = {auc(fpr, tpr):.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (1000 users)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    pr, rc, _ = precision_recall_curve(y, s)
    ax = axes[1]
    ax.plot(rc, pr, color='#4C72B0', lw=2,
            label=f'PR  AUC = {auc(rc, pr):.3f}')
    base = y.mean()
    ax.axhline(base, ls='--', color='gray', lw=1,
               label=f'Random = {base:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve (1000 users)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_roc_pr.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('[viz]', out)


def plot_topk_curve(df):
    """Top-K Precision / Recall vs K."""
    y = df['is_malicious'].astype(int).values
    order = np.argsort(-df['risk'].values)
    n_pos = int(y.sum())
    Ks = np.arange(5, 251, 5)
    prec, rec = [], []
    for K in Ks:
        topk = order[:K]
        h = int(y[topk].sum())
        prec.append(h / K)
        rec.append(h / n_pos)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ks, prec, '-o', label='Top-K Precision',
            color='#4C72B0', markersize=4)
    ax.plot(Ks, rec, '-s', label='Top-K Recall',
            color='#C44E52', markersize=4)
    ax.set_xlabel('K (top users alerted)')
    ax.set_ylabel('Score')
    ax.set_title('Top-K Precision & Recall (n={} malicious / 1000)'.format(n_pos))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_topk.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('[viz]', out)


def plot_detector_contribution(df):
    """Stacked bar: 每个 top-30 风险用户的 4 个检测器贡献."""
    top = df.head(30).copy()
    weights = (0.30, 0.10, 0.30, 0.30)
    top['contrib_rule'] = weights[0] * top['z_rule']
    top['contrib_lm']   = weights[1] * top['z_lm']
    top['contrib_sim']  = weights[2] * top['z_sim']
    top['contrib_peer'] = weights[3] * top['z_peer']

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(top))
    bottom = np.zeros(len(top))
    for col, color, lab in [
        ('contrib_rule', '#C44E52', 'Pattern Rule'),
        ('contrib_sim',  '#4C72B0', 'Sequence Similarity'),
        ('contrib_peer', '#55A868', 'Peer Deviation (rare-token IDF)'),
        ('contrib_lm',   '#DD8452', 'N-gram Anomaly'),
    ]:
        ax.bar(x, top[col].values, bottom=bottom, label=lab, color=color)
        bottom += top[col].values

    # 用红框圈出真正的恶意用户
    for i, ism in enumerate(top['is_malicious'].values):
        if ism:
            ax.bar(i, bottom[i], facecolor='none',
                   edgecolor='black', lw=1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top['user'], rotation=70)
    ax.set_ylabel('Risk Score (weighted z-score)')
    ax.set_title('Top-30 Risk Users — Detector Contribution\n(Black border = ground-truth malicious)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_top30_contribution.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('[viz]', out)


def plot_scenario_breakdown(explanations):
    """Top-K 用户的场景命中分布: 哪类场景被命中最多."""
    sc = {'S1_LEAK_VIA_USB': 0, 'S2_DEPARTING_DATA_THEFT': 0,
          'S3_SYSADMIN_KEYLOG': 0}
    sc_mal = {k: 0 for k in sc}
    for e in explanations[:80]:
        is_mal = e['is_malicious']
        for h in e.get('rule_hits', []):
            sid = h['scenario']
            if h['score'] > 0:
                sc[sid] += 1
                if is_mal:
                    sc_mal[sid] += 1

    labels = list(sc.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, [sc[k] for k in labels], 0.35, label='all top-80',
           color='#4C72B0')
    ax.bar(x + 0.18, [sc_mal[k] for k in labels], 0.35,
           label='malicious top-80', color='#C44E52')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel('# users with rule hit')
    ax.set_title('Scenario coverage in Top-80 (rule hits > 0)')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_scenario_breakdown.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('[viz]', out)


def plot_malicious_rank_hist(df):
    """所有 60 个恶意用户的排名直方图 (排名越靠前越好)."""
    mal_ranks = df.index[df['is_malicious']].tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(mal_ranks, bins=20, color='#C44E52', edgecolor='black')
    ax.set_xlabel('Rank (lower = higher risk)')
    ax.set_ylabel('# malicious users')
    ax.set_title('Distribution of malicious users\' ranks (1000 total)')
    ax.axvline(60, ls='--', color='black', label='top-60')
    ax.axvline(100, ls=':', color='black', label='top-100')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig_malicious_ranks.png')
    plt.savefig(out, dpi=120)
    plt.close()
    print('[viz]', out)


def main():
    df = pd.read_csv(os.path.join(OUT_DIR, 'risk_scores.csv'))
    with open(os.path.join(OUT_DIR, 'top_risk_explanations.json')) as f:
        explanations = json.load(f)
    plot_score_distribution(df)
    plot_roc_pr(df)
    plot_topk_curve(df)
    plot_detector_contribution(df)
    plot_scenario_breakdown(explanations)
    plot_malicious_rank_hist(df)
    print('\nAll figures saved to', OUT_DIR)


if __name__ == "__main__":
    main()
