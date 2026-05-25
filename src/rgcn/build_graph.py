"""
从 r4.2-mini-v1 CSV 构建异构图, 给 R-GCN 用。

==== 图结构 ====
节点类型 (4):
  user       — 员工
  pc         — 终端机
  file_type  — (doc/exec/other) × (WH/AH)
  url_cat    — (LEAK/KEYLOG/JOBHUNT/OTHER) × (WH/AH)

关系类型 (6, 实际 R-GCN R=6;
shared_pc 在训练时再从 logon 派生, 避免在这里就把图弄爆):
  logon_wh    user --[WH]-->  pc
  logon_ah    user --[AH]-->  pc
  usb_wh      user --[WH]-->  pc        (device connect)
  usb_ah      user --[AH]-->  pc
  file_op     user -----+--> file_type  (按 (doc/exec/other, WH/AH) 折叠)
  http_visit  user -----+--> url_cat    (按 (LEAK/KEYLOG/JOBHUNT/OTHER, WH/AH) 折叠)

每条边带 weight = log(1+count). 同一 (src, dst, rel) 多次出现会合并.

==== 节点特征 ====
user      : 6 维浅特征 (活跃天数 / total / ah_ratio / distinct_pc /
            distinct_files / ah_logon_cnt), 经 StandardScaler 标准化
pc/file_type/url_cat : 类型 ID + Embedding (在模型里学习)

==== 输出 ====
outputs/rgcn/graph.pt  : dict
   node_offsets : {ntype: (start, end)} 在 flat index 上的范围
   node_ids     : {ntype: list[str]}    原始 id
   num_nodes    : int                   总节点数
   user_feats   : FloatTensor [N_user, 6]
   labels       : LongTensor  [N_user]  (1 mal / 0 benign)
   edges        : {rel: (LongTensor src_flat, LongTensor dst_flat,
                         FloatTensor weight)}
   relation_names : list[str]            R 个关系名 (顺序固定)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# 让 csv 字段大小够大 (file.csv 的 content 可能很长)
csv.field_size_limit(10_000_000)

# 项目内 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_ground_truth import get_all_malicious_user_ids  # noqa: E402


# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
DATA_DIR = Path('/home/user/webapp/data/cert42/r4.2-mini-v1')
OUT_DIR = Path('/home/user/webapp/outputs/rgcn')

# 工作日 7:00-18:00 视作 WH (work hours), 其它为 AH (after hours).
WORK_DAYS = {0, 1, 2, 3, 4}   # Mon..Fri
WORK_HOUR_START = 7
WORK_HOUR_END = 18

# URL 类别关键字 (复用 preprocess.py 中的领域先验)
URL_KEYS = {
    'LEAK':    ['wikileaks', 'pastebin', 'dropbox', 'mega.nz'],
    'KEYLOG':  ['keylog', 'keystroke', 'spyware', 'rootkit'],
    'JOBHUNT': ['indeed', 'monster', 'careerbuilder', 'simplyhired',
                'linkedin/jobs', 'jobsearch'],
}
URL_CATS = ['LEAK', 'KEYLOG', 'JOBHUNT', 'OTHER']

# 文件类别
FILE_CATS = ['doc', 'exec', 'other']
TIMEBINS = ['WH', 'AH']

RELATION_NAMES = [
    'logon_wh', 'logon_ah',
    'usb_wh',   'usb_ah',
    'file_op',  'http_visit',
]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def parse_dt(s: str) -> datetime | None:
    """CMU-CERT 日期格式: '01/02/2010 06:49:00' (M/D/Y H:M:S)."""
    try:
        return datetime.strptime(s, '%m/%d/%Y %H:%M:%S')
    except Exception:
        return None


def time_bin(dt: datetime) -> str:
    if dt.weekday() in WORK_DAYS and WORK_HOUR_START <= dt.hour < WORK_HOUR_END:
        return 'WH'
    return 'AH'


def file_cat(filename: str) -> str:
    f = (filename or '').lower()
    if f.endswith(('.exe', '.bat', '.dll', '.com', '.scr')):
        return 'exec'
    if f.endswith(('.doc', '.docx', '.pdf', '.xls', '.xlsx',
                   '.ppt', '.pptx', '.txt', '.csv')):
        return 'doc'
    return 'other'


def url_cat(url: str) -> str:
    u = (url or '').lower()
    for cat, keys in URL_KEYS.items():
        if any(k in u for k in keys):
            return cat
    return 'OTHER'


# ---------------------------------------------------------------------------
# 第 1 步: 扫描 logon.csv 得 user/pc 节点 + logon_{wh,ah} 边
# ---------------------------------------------------------------------------
def scan_logon(path: Path):
    """返回 user→pc 边计数 (按 wh/ah 分桶), 同时收集 user/pc 集合."""
    users: set[str] = set()
    pcs: set[str] = set()
    edges_wh: Counter = Counter()  # (user, pc) -> cnt
    edges_ah: Counter = Counter()
    user_stats: dict[str, dict] = defaultdict(lambda: {
        'total': 0, 'ah_cnt': 0, 'logon_ah_cnt': 0,
        'pcs': set(), 'days': set(),
    })

    print(f'[1] scan {path.name}', flush=True)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, pc = row['user'], row['pc']
            dt = parse_dt(row['date'])
            if not (u and pc and dt):
                continue
            users.add(u)
            pcs.add(pc)
            tb = time_bin(dt)
            (edges_wh if tb == 'WH' else edges_ah)[(u, pc)] += 1
            st = user_stats[u]
            st['total'] += 1
            st['pcs'].add(pc)
            st['days'].add(dt.date())
            if tb == 'AH':
                st['ah_cnt'] += 1
                # logon 单独记 (是否 logon 在 ah)
                if row.get('activity', '').lower().startswith('logon'):
                    st['logon_ah_cnt'] += 1
    print(f'    users={len(users)}  pcs={len(pcs)}  '
          f'wh_edges={len(edges_wh)}  ah_edges={len(edges_ah)}')
    return users, pcs, edges_wh, edges_ah, user_stats


# ---------------------------------------------------------------------------
# 第 2 步: device.csv -> usb_{wh,ah} 边 (只看 Connect)
# ---------------------------------------------------------------------------
def scan_device(path: Path, users: set[str], pcs: set[str], user_stats: dict):
    edges_wh: Counter = Counter()
    edges_ah: Counter = Counter()
    print(f'[2] scan {path.name}', flush=True)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('activity', '').lower() != 'connect':
                continue
            u, pc = row['user'], row['pc']
            dt = parse_dt(row['date'])
            if not (u in users and pc and dt):
                continue
            pcs.add(pc)
            tb = time_bin(dt)
            (edges_wh if tb == 'WH' else edges_ah)[(u, pc)] += 1
            user_stats[u]['total'] += 1
            if tb == 'AH':
                user_stats[u]['ah_cnt'] += 1
    print(f'    usb_wh_edges={len(edges_wh)}  usb_ah_edges={len(edges_ah)}')
    return edges_wh, edges_ah


# ---------------------------------------------------------------------------
# 第 3 步: file.csv -> file_op 边
# ---------------------------------------------------------------------------
def scan_file(path: Path, users: set[str], user_stats: dict):
    edges: Counter = Counter()  # (user, "doc_WH") -> cnt
    file_set = defaultdict(set)  # user -> set of file names (for distinct_files)
    print(f'[3] scan {path.name}', flush=True)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row['user']
            if u not in users:
                continue
            dt = parse_dt(row['date'])
            if not dt:
                continue
            cat = file_cat(row.get('filename', ''))
            tb = time_bin(dt)
            edges[(u, f'{cat}_{tb}')] += 1
            file_set[u].add(row.get('filename', ''))
            user_stats[u]['total'] += 1
            if tb == 'AH':
                user_stats[u]['ah_cnt'] += 1
    # 把 distinct files 数写回 user_stats
    for u, fs in file_set.items():
        user_stats[u]['distinct_files'] = len(fs)
    print(f'    file_op_edges={len(edges)}')
    return edges


# ---------------------------------------------------------------------------
# 第 4 步: http_slim.csv -> http_visit 边 (3.6M 行, 流式)
# ---------------------------------------------------------------------------
def scan_http(path: Path, users: set[str], user_stats: dict):
    edges: Counter = Counter()
    print(f'[4] scan {path.name} (streaming, ~3.6M rows)', flush=True)
    t0 = time.time()
    # http_slim 没有 quoted 字段 (url 中没逗号? 不一定, 但实测 split(',',4) 能拿到)
    # 用 reader 直接读会被 ','在 url 中的 instances 影响, 但概率极低;
    # 这里仍用 csv 模块以求稳, 单次 3-4 min 可以接受.
    n = 0
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        u_idx = header.index('user')
        d_idx = header.index('date')
        url_idx = header.index('url')
        for row in reader:
            n += 1
            if len(row) <= url_idx:
                continue
            u = row[u_idx]
            if u not in users:
                continue
            dt = parse_dt(row[d_idx])
            if not dt:
                continue
            cat = url_cat(row[url_idx])
            tb = time_bin(dt)
            edges[(u, f'{cat}_{tb}')] += 1
            user_stats[u]['total'] += 1
            if tb == 'AH':
                user_stats[u]['ah_cnt'] += 1
            if n % 1_000_000 == 0:
                print(f'    ... {n:,} rows  ({time.time()-t0:.1f}s)', flush=True)
    print(f'    http rows={n:,}  http_visit_edges={len(edges)}  '
          f'({time.time()-t0:.1f}s)')
    return edges


# ---------------------------------------------------------------------------
# 第 5 步: 组装 flat-index 异构图
# ---------------------------------------------------------------------------
def assemble_graph(users, pcs, user_stats,
                   logon_wh, logon_ah, usb_wh, usb_ah,
                   file_edges, http_edges):

    # ---- 节点 id 排序, 给出稳定的 idx ----
    user_list = sorted(users)
    pc_list   = sorted(pcs)
    file_list = [f'{c}_{t}' for c in FILE_CATS for t in TIMEBINS]  # 6
    url_list  = [f'{c}_{t}' for c in URL_CATS  for t in TIMEBINS]  # 8

    # flat index: [user | pc | file_type | url_cat]
    offsets = {}
    cur = 0
    for name, lst in [('user', user_list), ('pc', pc_list),
                      ('file_type', file_list), ('url_cat', url_list)]:
        offsets[name] = (cur, cur + len(lst))
        cur += len(lst)
    num_nodes = cur
    print(f'[5] flat nodes: total={num_nodes}, offsets={offsets}')

    def gidx(ntype, key, lookup):
        return offsets[ntype][0] + lookup[key]

    u_idx = {u: i for i, u in enumerate(user_list)}
    p_idx = {p: i for i, p in enumerate(pc_list)}
    f_idx = {k: i for i, k in enumerate(file_list)}
    h_idx = {k: i for i, k in enumerate(url_list)}

    def to_edges(counter, src_lookup, src_type, dst_lookup, dst_type):
        if not counter:
            return (torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.float))
        srcs, dsts, ws = [], [], []
        for (s, d), c in counter.items():
            if s not in src_lookup or d not in dst_lookup:
                continue
            srcs.append(gidx(src_type, s, src_lookup))
            dsts.append(gidx(dst_type, d, dst_lookup))
            ws.append(math.log1p(c))
        return (torch.tensor(srcs, dtype=torch.long),
                torch.tensor(dsts, dtype=torch.long),
                torch.tensor(ws, dtype=torch.float))

    edges = {
        'logon_wh':   to_edges(logon_wh, u_idx, 'user', p_idx, 'pc'),
        'logon_ah':   to_edges(logon_ah, u_idx, 'user', p_idx, 'pc'),
        'usb_wh':     to_edges(usb_wh,   u_idx, 'user', p_idx, 'pc'),
        'usb_ah':     to_edges(usb_ah,   u_idx, 'user', p_idx, 'pc'),
        'file_op':    to_edges(file_edges, u_idx, 'user', f_idx, 'file_type'),
        'http_visit': to_edges(http_edges, u_idx, 'user', h_idx, 'url_cat'),
    }
    for r in RELATION_NAMES:
        s, _, _ = edges[r]
        print(f'    edge[{r}]: {s.numel():,}')

    # ---- 用户浅特征: 6 维 ----
    feat = np.zeros((len(user_list), 6), dtype=np.float32)
    for i, u in enumerate(user_list):
        st = user_stats.get(u, {})
        total = st.get('total', 0)
        ah = st.get('ah_cnt', 0)
        feat[i, 0] = len(st.get('days', []))                 # active_days
        feat[i, 1] = total                                   # total_events
        feat[i, 2] = ah / total if total > 0 else 0.0        # ah_ratio
        feat[i, 3] = len(st.get('pcs', []))                  # distinct_pc
        feat[i, 4] = st.get('distinct_files', 0)             # distinct_files
        feat[i, 5] = st.get('logon_ah_cnt', 0)               # ah_logon_cnt

    # 标准化 (RobustScaler 思路: 减中位数 / IQR, 抗离群)
    med = np.median(feat, axis=0)
    q25 = np.quantile(feat, 0.25, axis=0)
    q75 = np.quantile(feat, 0.75, axis=0)
    iqr = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    feat_norm = (feat - med) / iqr
    feat_norm = np.clip(feat_norm, -5, 5)  # 防离群

    user_feats = torch.from_numpy(feat_norm).float()

    # ---- 标签 ----
    mal = get_all_malicious_user_ids()
    labels = torch.tensor(
        [1 if u in mal else 0 for u in user_list], dtype=torch.long
    )
    n_mal_in_graph = labels.sum().item()
    print(f'    labels: malicious={n_mal_in_graph}/{len(user_list)} '
          f'(expected 70 if all 70 mal are in user_list)')

    return {
        'node_offsets': offsets,
        'node_ids': {
            'user': user_list, 'pc': pc_list,
            'file_type': file_list, 'url_cat': url_list,
        },
        'num_nodes': num_nodes,
        'user_feats': user_feats,         # [N_user, 6]
        'labels': labels,                 # [N_user]
        'edges': edges,                   # {rel: (src, dst, w)}
        'relation_names': RELATION_NAMES,
        'user_stats_raw': dict(feat=feat,  # 留作可解释性输出
                               med=med, iqr=iqr),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(DATA_DIR))
    ap.add_argument('--out', default=str(OUT_DIR / 'graph.pt'))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    users, pcs, logon_wh, logon_ah, user_stats = scan_logon(data_dir / 'logon.csv')
    usb_wh, usb_ah = scan_device(data_dir / 'device.csv', users, pcs, user_stats)
    file_edges = scan_file(data_dir / 'file.csv', users, user_stats)
    http_edges = scan_http(data_dir / 'http_slim.csv', users, user_stats)

    graph = assemble_graph(
        users, pcs, user_stats,
        logon_wh, logon_ah, usb_wh, usb_ah,
        file_edges, http_edges,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out_path)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f'\nSaved graph to {out_path} ({size_mb:.1f} MB)')

    # 打印一下规模与正负样本
    print('\n========= GRAPH SUMMARY =========')
    print(f'  total nodes : {graph["num_nodes"]}')
    for nt, (s, e) in graph['node_offsets'].items():
        print(f'    {nt:10s} : {e - s}  (flat [{s},{e}))')
    for r in graph['relation_names']:
        s, _, _ = graph['edges'][r]
        print(f'  edges[{r:11s}] : {s.numel():,}')
    print(f'  user_feats  : {tuple(graph["user_feats"].shape)}')
    print(f'  labels      : mal={int(graph["labels"].sum())}, '
          f'ben={int((graph["labels"]==0).sum())}')
    print('=================================')


if __name__ == '__main__':
    main()
