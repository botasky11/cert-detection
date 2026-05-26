"""
在 r4.2 full dataset 上训练并评估 R-GCN.

设定:
  * 半监督二分类: user 节点中 70 mal / 100 benign
  * 5-fold StratifiedKFold (保证每折 mal/benign 比例不变)
  * 每折: 训练集~136 节点 (56 mal / 80 ben), 验证集~34 节点 (14 mal / 20 ben)
  * 训练时, 用所有节点 (包括 val) 都参与 message-passing,
    只在 train 节点上算 loss, val 节点上算指标 (典型 transductive 半监督)
  * 类别不平衡: BCE 上加 pos_weight = (#benign / #mal) ≈ 1.43

指标 (在 OOF 预测上汇总):
  ROC-AUC, PR-AUC, Top-K Precision, Top-K Recall (K = 20/40/100)

用法:
  python src/rgcn/train_rgcn.py [--hidden 64 --epochs 300 --lr 0.01 --seed 42]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rgcn.rgcn_model import RGCNClassifier, add_reverse_edges


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def top_k_metrics(scores: np.ndarray, labels: np.ndarray, k: int):
    """K 个最高分中的 precision / recall."""
    order = np.argsort(-scores)
    topk = order[:k]
    hit = int(labels[topk].sum())
    pos = int(labels.sum())
    prec = hit / k if k else 0.0
    rec  = hit / pos if pos else 0.0
    return prec, rec, hit


# ---------------------------------------------------------------------------
# 单折训练
# ---------------------------------------------------------------------------
def train_one_fold(
    graph, train_idx, val_idx, *,
    hidden=64, num_layers=2, num_bases=4, dropout=0.3,
    lr=0.01, weight_decay=5e-4, epochs=300,
    patience=50, device='cpu', verbose=False,
):
    user_feats = graph['user_feats'].to(device)
    labels = graph['labels'].to(device).float()

    # edges: 添加反向边, 然后全部搬到 device
    edges_list, _ = add_reverse_edges(graph['edges'], graph['relation_names'])
    edges_list = [
        (s.to(device), d.to(device), w.to(device)) for s, d, w in edges_list
    ]

    model = RGCNClassifier(
        node_offsets=graph['node_offsets'],
        user_feat_dim=user_feats.size(1),
        emb_dim=hidden,
        num_relations=len(edges_list),
        num_layers=num_layers,
        num_bases=num_bases,
        dropout=dropout,
    ).to(device)

    # 类别不平衡: pos_weight = #neg / #pos (用 train 集统计)
    y_tr = labels[train_idx]
    n_pos = float(y_tr.sum().item())
    n_neg = float((1 - y_tr).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t   = torch.as_tensor(val_idx,   dtype=torch.long, device=device)
    y_val_np = labels[val_idx].detach().cpu().numpy()

    best_auc = -1.0
    best_scores = None
    best_epoch = -1
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(user_feats, edges_list)        # [N_user]
        loss = loss_fn(logits[train_idx_t], labels[train_idx_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optim.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(user_feats, edges_list)
            val_scores = torch.sigmoid(val_logits[val_idx_t]).cpu().numpy()
        try:
            auc = roc_auc_score(y_val_np, val_scores)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_scores = val_scores.copy()
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if verbose and ep % 20 == 0:
            print(f'    ep {ep:3d}  loss={loss.item():.4f}  '
                  f'val_auc={auc:.4f}  best={best_auc:.4f}@{best_epoch}')

        if no_improve >= patience:
            if verbose:
                print(f'    early stop at {ep}, best auc={best_auc:.4f}@{best_epoch}')
            break

    return best_scores, best_auc, best_epoch


# ---------------------------------------------------------------------------
# 主流程: 5-fold CV
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default='outputs/rgcn/graph.pt')
    ap.add_argument('--out',   default='outputs/rgcn/rgcn_results.json')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--num-bases',  type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr',      type=float, default=0.01)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--epochs',  type=int, default=300)
    ap.add_argument('--patience', type=int, default=50)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--device',  default='cpu')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print('========= R-GCN training on r4.2 full dataset =========')
    print(f'config: {vars(args)}')

    set_seed(args.seed)

    graph = torch.load(args.graph, weights_only=False)
    labels = graph['labels'].numpy()        # [N_user]
    N = labels.shape[0]
    n_mal = int(labels.sum())
    n_ben = N - n_mal
    print(f'users: total={N}, mal={n_mal}, ben={n_ben}')

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                          random_state=args.seed)

    oof_scores = np.zeros(N, dtype=np.float64)      # 每个用户的 OOF 分数
    fold_aucs = []
    fold_epochs = []
    t0 = time.time()

    for fi, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(N), labels), 1):
        n_tr_mal = int(labels[train_idx].sum()); n_tr_ben = len(train_idx) - n_tr_mal
        n_va_mal = int(labels[val_idx].sum());   n_va_ben = len(val_idx)   - n_va_mal
        print(f'\n--- fold {fi}/{args.n_splits} : '
              f'train {len(train_idx)} ({n_tr_mal} mal / {n_tr_ben} ben) '
              f' val {len(val_idx)} ({n_va_mal} mal / {n_va_ben} ben) ---')

        scores, auc, best_ep = train_one_fold(
            graph, train_idx, val_idx,
            hidden=args.hidden,
            num_layers=args.num_layers,
            num_bases=args.num_bases,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            verbose=args.verbose,
        )
        oof_scores[val_idx] = scores
        fold_aucs.append(auc)
        fold_epochs.append(best_ep)
        print(f'    fold {fi} val ROC-AUC = {auc:.4f} (best epoch {best_ep})')

    print(f'\n--- all folds done in {time.time()-t0:.1f}s ---')
    print(f'per-fold AUCs : {[f"{a:.4f}" for a in fold_aucs]}')
    print(f'mean ± std    : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}')

    # ------ 用 OOF 预测计算整体指标 (整张表上一份分数) ------
    overall_auc = roc_auc_score(labels, oof_scores)
    overall_ap  = average_precision_score(labels, oof_scores)
    topk = {}
    for k in (10, 20, 30, 50, 70, 100):
        prec, rec, hit = top_k_metrics(oof_scores, labels, k)
        topk[k] = {'precision': round(prec, 4),
                   'recall':    round(rec, 4),
                   'hit':       hit}

    print(f'\n========= OOF metrics ({N} users) =========')
    print(f'  ROC-AUC : {overall_auc:.4f}')
    print(f'  PR-AUC  : {overall_ap:.4f}')
    for k, m in topk.items():
        print(f'  Top-{k:<3} : P={m["precision"]:.4f}  R={m["recall"]:.4f}  hits={m["hit"]}/{n_mal}')

    # ------ Scenario-level recall ------
    from build_ground_truth import get_malicious_users
    mal_scenarios = get_malicious_users()  # {user: scenario_id}
    user_ids = graph['node_ids']['user']
    order = np.argsort(-oof_scores)
    scenario_recall = {}
    for s_id in [1, 2, 3]:
        s_users = {u for u, s in mal_scenarios.items() if s == s_id}
        s_total = len(s_users)
        for k in [70, 100]:
            topk_users = set(user_ids[i] for i in order[:k])
            s_hit = len(s_users & topk_users)
            scenario_recall[f'S{s_id}_top{k}_recall'] = round(s_hit / max(s_total, 1), 4)
    print('\n  Scenario-level recall:')
    for key, val in scenario_recall.items():
        print(f'    {key}: {val:.4f}')

    # ------ 保存结果 ------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 按分数降序输出 ranking, 方便人工审查
    ranking = [
        {'rank': r + 1, 'user': user_ids[i],
         'score': float(oof_scores[i]), 'label': int(labels[i])}
        for r, i in enumerate(order)
    ]
    payload = {
        'config': vars(args),
        'n_users': N, 'n_mal': n_mal, 'n_ben': n_ben,
        'fold_aucs': [float(a) for a in fold_aucs],
        'fold_best_epochs': fold_epochs,
        'mean_auc': float(np.mean(fold_aucs)),
        'std_auc':  float(np.std(fold_aucs)),
        'overall': {
            'roc_auc': float(overall_auc),
            'pr_auc':  float(overall_ap),
            'top_k':   topk,
        },
        'scenario_recall': scenario_recall,
        'ranking_top30': ranking[:30],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\nSaved to {out_path}')

    # ranking csv (含全部 all users)
    csv_path = out_path.with_suffix('.csv')
    with csv_path.open('w', encoding='utf-8') as f:
        f.write('rank,user,score,label\n')
        for r in ranking:
            f.write(f'{r["rank"]},{r["user"]},{r["score"]:.6f},{r["label"]}\n')
    print(f'Saved ranking csv to {csv_path}')


if __name__ == '__main__':
    main()
