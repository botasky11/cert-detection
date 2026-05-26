"""
对比 N-gram 行为序列匹配 和 R-GCN 异构图检测 两种算法的性能.

读取两种算法的评估结果, 生成统一的对比报告和可视化图表.

用法:
    python src/compare_algorithms.py

输出:
    outputs/comparison/
        ├── comparison_report.json    — 结构化对比数据
        ├── comparison_report.md      — Markdown 格式对比报告
        ├── fig_roc_comparison.png    — ROC 曲线对比
        ├── fig_pr_comparison.png     — PR 曲线对比
        ├── fig_topk_comparison.png   — Top-K Precision/Recall 对比
        ├── fig_scenario_comparison.png — 场景 Recall 对比
        └── fig_score_dist_comparison.png — 风险分数分布对比
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    roc_auc_score, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ground_truth import get_malicious_users, get_all_malicious_user_ids

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'outputs'
COMP_DIR = OUT_DIR / 'comparison'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 统一颜色方案
COLOR_NGRAM = '#E74C3C'   # 红色系 (N-gram)
COLOR_RGCN  = '#3498DB'   # 蓝色系 (R-GCN)

K_LIST = [10, 20, 30, 50, 70, 100]


def top_k_metrics(scores, labels, k):
    """K 个最高分中的 precision / recall."""
    order = np.argsort(-scores)
    topk = order[:k]
    hit = int(labels[topk].sum())
    pos = int(labels.sum())
    prec = hit / k if k else 0.0
    rec  = hit / pos if pos else 0.0
    return prec, rec, hit


def load_ngram_results():
    """加载 N-gram 算法结果."""
    risk_path = OUT_DIR / 'risk_scores.csv'
    metrics_path = OUT_DIR / 'metrics.json'

    if not risk_path.exists():
        print(f"ERROR: {risk_path} 不存在. 请先运行 N-gram 算法.")
        return None

    df = pd.read_csv(risk_path)
    with open(metrics_path) as f:
        saved_metrics = json.load(f)

    malicious_set = get_all_malicious_user_ids()
    y_true = df['user'].isin(malicious_set).astype(int).values
    y_score = df['risk'].values

    return {
        'name': 'N-gram 行为序列匹配',
        'short_name': 'N-gram',
        'df': df,
        'y_true': y_true,
        'y_score': y_score,
        'saved_metrics': saved_metrics,
        'n_users': len(df),
    }


def load_rgcn_results():
    """加载 R-GCN 算法结果."""
    csv_path = OUT_DIR / 'rgcn' / 'rgcn_results.csv'
    json_path = OUT_DIR / 'rgcn' / 'rgcn_results.json'

    if not csv_path.exists():
        print(f"ERROR: {csv_path} 不存在. 请先运行 R-GCN 算法.")
        return None

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        saved_metrics = json.load(f)

    y_true = df['label'].values.astype(int)
    y_score = df['score'].values

    return {
        'name': 'R-GCN 异构图检测',
        'short_name': 'R-GCN',
        'df': df,
        'y_true': y_true,
        'y_score': y_score,
        'saved_metrics': saved_metrics,
        'n_users': len(df),
    }


def compute_unified_metrics(result):
    """为单个算法计算统一的评估指标."""
    y_true = result['y_true']
    y_score = result['y_score']
    n_pos = int(y_true.sum())

    m = {
        'n_users': int(len(y_true)),
        'n_malicious': n_pos,
        'n_benign': int(len(y_true) - n_pos),
        'roc_auc': float(roc_auc_score(y_true, y_score)),
        'pr_auc': float(average_precision_score(y_true, y_score)),
    }

    for k in K_LIST:
        if k > len(y_true):
            continue
        prec, rec, hit = top_k_metrics(y_score, y_true, k)
        m[f'top{k}_precision'] = round(prec, 4)
        m[f'top{k}_recall'] = round(rec, 4)
        m[f'top{k}_hit'] = hit

    # 场景级 recall
    mal_scenarios = get_malicious_users()
    order = np.argsort(-y_score)

    # 获取用户 ID 列表
    if 'user' in result['df'].columns:
        user_ids = result['df']['user'].values
    else:
        user_ids = result['df'].iloc[:, 1].values  # rank,user,score,label

    for s_id in [1, 2, 3]:
        s_users = {u for u, s in mal_scenarios.items() if s == s_id}
        s_total = len(s_users)
        for k in [70, 100]:
            if k > len(y_true):
                continue
            topk_users = set(user_ids[i] for i in order[:k])
            s_hit = len(s_users & topk_users)
            m[f'S{s_id}_top{k}_recall'] = round(s_hit / max(s_total, 1), 4)

    return m


# ============================================================
# 可视化
# ============================================================

def plot_roc_comparison(ngram, rgcn):
    """ROC 曲线对比."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for result, color, ls in [(ngram, COLOR_NGRAM, '-'), (rgcn, COLOR_RGCN, '--')]:
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_score'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, linestyle=ls,
                label=f'{result["short_name"]} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    out = COMP_DIR / 'fig_roc_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Compare] {out}')


def plot_pr_comparison(ngram, rgcn):
    """PR 曲线对比."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for result, color, ls in [(ngram, COLOR_NGRAM, '-'), (rgcn, COLOR_RGCN, '--')]:
        precision, recall, _ = precision_recall_curve(
            result['y_true'], result['y_score'])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2.5, linestyle=ls,
                label=f'{result["short_name"]} (AUC = {pr_auc:.4f})')

    base = ngram['y_true'].mean()
    ax.axhline(base, ls=':', color='gray', lw=1,
               label=f'Random baseline = {base:.3f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    plt.tight_layout()
    out = COMP_DIR / 'fig_pr_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Compare] {out}')


def plot_topk_comparison(ngram_metrics, rgcn_metrics):
    """Top-K Precision 和 Recall 对比柱状图."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    valid_ks = [k for k in K_LIST
                if f'top{k}_precision' in ngram_metrics
                and f'top{k}_precision' in rgcn_metrics]

    x = np.arange(len(valid_ks))
    width = 0.35

    # Precision
    ax = axes[0]
    ngram_prec = [ngram_metrics[f'top{k}_precision'] for k in valid_ks]
    rgcn_prec  = [rgcn_metrics[f'top{k}_precision']  for k in valid_ks]
    bars1 = ax.bar(x - width/2, ngram_prec, width, label='N-gram',
                   color=COLOR_NGRAM, alpha=0.85)
    bars2 = ax.bar(x + width/2, rgcn_prec,  width, label='R-GCN',
                   color=COLOR_RGCN, alpha=0.85)
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Top-K Precision', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in valid_ks])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    # 标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    # Recall
    ax = axes[1]
    ngram_rec = [ngram_metrics[f'top{k}_recall'] for k in valid_ks]
    rgcn_rec  = [rgcn_metrics[f'top{k}_recall']  for k in valid_ks]
    bars1 = ax.bar(x - width/2, ngram_rec, width, label='N-gram',
                   color=COLOR_NGRAM, alpha=0.85)
    bars2 = ax.bar(x + width/2, rgcn_rec,  width, label='R-GCN',
                   color=COLOR_RGCN, alpha=0.85)
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Top-K Recall', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in valid_ks])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out = COMP_DIR / 'fig_topk_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Compare] {out}')


def plot_scenario_comparison(ngram_metrics, rgcn_metrics):
    """按场景的 Top-70 / Top-100 Recall 对比."""
    scenarios = ['S1', 'S2', 'S3']
    scenario_names = ['S1: USB+Leak\n(30 users)',
                      'S2: Job+Email\n(30 users)',
                      'S3: Keylogger\n(10 users)']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, k in enumerate([70, 100]):
        ax = axes[ax_idx]
        x = np.arange(len(scenarios))
        width = 0.35

        ngram_vals = [ngram_metrics.get(f'{s}_top{k}_recall', 0) for s in scenarios]
        rgcn_vals  = [rgcn_metrics.get(f'{s}_top{k}_recall', 0)  for s in scenarios]

        bars1 = ax.bar(x - width/2, ngram_vals, width, label='N-gram',
                       color=COLOR_NGRAM, alpha=0.85)
        bars2 = ax.bar(x + width/2, rgcn_vals,  width, label='R-GCN',
                       color=COLOR_RGCN, alpha=0.85)

        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title(f'Scenario Recall @ Top-{k}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.15])

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out = COMP_DIR / 'fig_scenario_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Compare] {out}')


def plot_score_distribution_comparison(ngram, rgcn):
    """风险分数分布对比."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, result, color, title in [
        (axes[0], ngram, COLOR_NGRAM, 'N-gram'),
        (axes[1], rgcn, COLOR_RGCN, 'R-GCN'),
    ]:
        y = result['y_true']
        s = result['y_score']

        bins = 40
        ax.hist(s[y == 0], bins=bins, alpha=0.6, color='#95A5A6',
                label=f'Benign (n={(y==0).sum()})')
        ax.hist(s[y == 1], bins=bins, alpha=0.85, color=color,
                label=f'Malicious (n={(y==1).sum()})')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('# Users')
        ax.set_title(f'{title} — Score Distribution', fontweight='bold')
        ax.legend()
        ax.set_yscale('log')

    plt.tight_layout()
    out = COMP_DIR / 'fig_score_dist_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'[Compare] {out}')


def generate_markdown_report(ngram_m, rgcn_m):
    """生成 Markdown 格式对比报告."""
    lines = []
    lines.append('# 算法性能对比报告')
    lines.append('')
    lines.append('## 基本信息')
    lines.append('')
    lines.append(f'| 指标 | N-gram 行为序列匹配 | R-GCN 异构图检测 |')
    lines.append(f'|------|-------------------|-----------------|')
    lines.append(f'| 用户总数 | {ngram_m["n_users"]} | {rgcn_m["n_users"]} |')
    lines.append(f'| 恶意用户数 | {ngram_m["n_malicious"]} | {rgcn_m["n_malicious"]} |')
    lines.append(f'| 良性用户数 | {ngram_m["n_benign"]} | {rgcn_m["n_benign"]} |')
    lines.append('')

    lines.append('## 整体性能指标')
    lines.append('')
    lines.append(f'| 指标 | N-gram | R-GCN | 优胜 |')
    lines.append(f'|------|--------|-------|------|')
    for metric, label in [
        ('roc_auc', 'ROC-AUC'),
        ('pr_auc', 'PR-AUC'),
    ]:
        nv = ngram_m[metric]
        rv = rgcn_m[metric]
        winner = '**N-gram**' if nv > rv else ('**R-GCN**' if rv > nv else '平局')
        lines.append(f'| {label} | {nv:.4f} | {rv:.4f} | {winner} |')
    lines.append('')

    lines.append('## Top-K 性能对比')
    lines.append('')
    lines.append(f'| K | N-gram Prec | R-GCN Prec | N-gram Rec | R-GCN Rec |')
    lines.append(f'|---|-------------|------------|------------|-----------|')
    for k in K_LIST:
        np_ = ngram_m.get(f'top{k}_precision', '-')
        rp_ = rgcn_m.get(f'top{k}_precision', '-')
        nr_ = ngram_m.get(f'top{k}_recall', '-')
        rr_ = rgcn_m.get(f'top{k}_recall', '-')
        np_s = f'{np_:.4f}' if isinstance(np_, float) else np_
        rp_s = f'{rp_:.4f}' if isinstance(rp_, float) else rp_
        nr_s = f'{nr_:.4f}' if isinstance(nr_, float) else nr_
        rr_s = f'{rr_:.4f}' if isinstance(rr_, float) else rr_
        lines.append(f'| Top-{k} | {np_s} | {rp_s} | {nr_s} | {rr_s} |')
    lines.append('')

    lines.append('## 场景级 Recall 对比')
    lines.append('')
    scenario_info = {
        'S1': ('USB + Leak 上传', 30),
        'S2': ('求职 + 邮件外发', 30),
        'S3': ('键盘记录器', 10),
    }
    lines.append(f'| 场景 | 描述 | 用户数 | N-gram Top-70 | R-GCN Top-70 | N-gram Top-100 | R-GCN Top-100 |')
    lines.append(f'|------|------|--------|---------------|--------------|----------------|---------------|')
    for sid, (desc, cnt) in scenario_info.items():
        n70 = ngram_m.get(f'{sid}_top70_recall', '-')
        r70 = rgcn_m.get(f'{sid}_top70_recall', '-')
        n100 = ngram_m.get(f'{sid}_top100_recall', '-')
        r100 = rgcn_m.get(f'{sid}_top100_recall', '-')
        n70s = f'{n70:.4f}' if isinstance(n70, float) else n70
        r70s = f'{r70:.4f}' if isinstance(r70, float) else r70
        n100s = f'{n100:.4f}' if isinstance(n100, float) else n100
        r100s = f'{r100:.4f}' if isinstance(r100, float) else r100
        lines.append(f'| {sid} | {desc} | {cnt} | {n70s} | {r70s} | {n100s} | {r100s} |')
    lines.append('')

    return '\n'.join(lines)


# ============================================================
# 主函数
# ============================================================
def main():
    COMP_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  算法性能对比: N-gram vs R-GCN')
    print('=' * 60)

    # 1. 加载结果
    print('\n[1] 加载 N-gram 结果 ...')
    ngram = load_ngram_results()
    if ngram is None:
        return

    print('[2] 加载 R-GCN 结果 ...')
    rgcn = load_rgcn_results()
    if rgcn is None:
        return

    print(f'\n    N-gram: {ngram["n_users"]} 用户')
    print(f'    R-GCN:  {rgcn["n_users"]} 用户')

    # 2. 计算统一指标
    print('\n[3] 计算统一评估指标 ...')
    ngram_m = compute_unified_metrics(ngram)
    rgcn_m  = compute_unified_metrics(rgcn)

    print(f'\n    {"指标":<22s}  {"N-gram":>8s}  {"R-GCN":>8s}')
    print(f'    {"-"*22}  {"-"*8}  {"-"*8}')
    print(f'    {"ROC-AUC":<22s}  {ngram_m["roc_auc"]:8.4f}  {rgcn_m["roc_auc"]:8.4f}')
    print(f'    {"PR-AUC":<22s}  {ngram_m["pr_auc"]:8.4f}  {rgcn_m["pr_auc"]:8.4f}')
    for k in K_LIST:
        pk = f'top{k}_precision'
        rk = f'top{k}_recall'
        if pk in ngram_m and pk in rgcn_m:
            print(f'    {"Top-"+str(k)+" Precision":<22s}  '
                  f'{ngram_m[pk]:8.4f}  {rgcn_m[pk]:8.4f}')
            print(f'    {"Top-"+str(k)+" Recall":<22s}  '
                  f'{ngram_m[rk]:8.4f}  {rgcn_m[rk]:8.4f}')

    # 3. 生成可视化
    print('\n[4] 生成对比可视化 ...')
    plot_roc_comparison(ngram, rgcn)
    plot_pr_comparison(ngram, rgcn)
    plot_topk_comparison(ngram_m, rgcn_m)
    plot_scenario_comparison(ngram_m, rgcn_m)
    plot_score_distribution_comparison(ngram, rgcn)

    # 4. 保存结构化报告
    print('\n[5] 保存对比报告 ...')
    report = {
        'ngram': ngram_m,
        'rgcn': rgcn_m,
    }
    json_path = COMP_DIR / 'comparison_report.json'
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'    {json_path}')

    # 5. Markdown 报告
    md_report = generate_markdown_report(ngram_m, rgcn_m)
    md_path = COMP_DIR / 'comparison_report.md'
    md_path.write_text(md_report, encoding='utf-8')
    print(f'    {md_path}')

    print('\n' + '=' * 60)
    print('  对比完成! 详见 outputs/comparison/')
    print('=' * 60)


if __name__ == '__main__':
    main()
