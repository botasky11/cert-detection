"""
构建 r4.2-mini-v1 数据子集, 用于在本地 20GB 磁盘限制下做实验。

抽样规则:
  - 保留全部 70 个 malicious users (S1+S2+S3, 来自 build_ground_truth.py)
  - 随机抽取 N_BENIGN 个 benign users (默认 100)
  - 保留 logon / device / file / http 四类事件
  - 暂时跳过 email (它是全量数据中最大的两个之一: 1.3 GB)
  - 用 streaming 方式按行过滤 http_slim.csv (3.6 GB), 避免一次性读入

输出:
  data/cert42/r4.2-mini-v1/
    ├── logon.csv
    ├── device.csv
    ├── file.csv
    ├── http_slim.csv
    ├── selected_users.txt        # 选中的用户 ID, 每行一个
    └── subset_info.json          # 元数据: seed/统计/规则
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# 项目内
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ground_truth import get_all_malicious_user_ids  # noqa: E402

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'r4.2'
DEFAULT_OUT_NAME = 'r4.2-mini-v1'

# 待抽取的事件源 (file_name, 是否使用 csv 模块读取而不是按行 split)
#   logon/device/file 小, 用 csv 模块更稳;
#   http_slim 3.6 GB, 用按行扫描 user 字段位置, 避免 csv 解析 content 大字符串.
EVENT_FILES = [
    ('logon.csv',      'csv'),
    ('device.csv',     'csv'),
    ('file.csv',       'csv'),     # 184 MB, 含 content 字段, 但 csv 解析能搞定
    ('http_slim.csv',  'fastline'),  # 3.6 GB, 按行 split, user 是第 3 列
]


# ---------------------------------------------------------------------------
# 第 1 步: 收集全集用户 (来自 logon.csv, 它是用户全集最稳的来源)
# ---------------------------------------------------------------------------
def collect_all_users(logon_path: Path) -> set[str]:
    """扫描 logon.csv, 收集所有出现过的 user_id."""
    print(f'[1/4] 扫描 {logon_path.name} 收集全集用户 ...', flush=True)
    users: set[str] = set()
    with logon_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row.get('user')
            if u:
                users.add(u)
    print(f'      logon.csv 内出现的唯一用户: {len(users)}')
    return users


# ---------------------------------------------------------------------------
# 第 2 步: 决定要保留的用户集合
# ---------------------------------------------------------------------------
def select_users(all_users: set[str], n_benign: int, seed: int) -> tuple[set[str], list[str]]:
    malicious = get_all_malicious_user_ids()
    missing = malicious - all_users
    if missing:
        print(f'      ⚠ 有 {len(missing)} 个 malicious users 不在 logon.csv 里: {sorted(missing)[:5]}...')

    benign_pool = sorted(all_users - malicious)  # 排序后再 sample, 保证 seed 可复现
    if n_benign > len(benign_pool):
        print(f'      ⚠ benign 池只有 {len(benign_pool)} 人, 不够抽 {n_benign}, 全部取走.')
        n_benign = len(benign_pool)

    rng = random.Random(seed)
    benign_chosen = rng.sample(benign_pool, n_benign)

    kept = set(malicious) | set(benign_chosen)
    print(f'[2/4] 选定保留用户: malicious={len(malicious)}, benign={len(benign_chosen)}, '
          f'total={len(kept)} (seed={seed})')
    return kept, sorted(benign_chosen)


# ---------------------------------------------------------------------------
# 第 3 步: 流式过滤每个事件 CSV
# ---------------------------------------------------------------------------
def filter_csv_module(
    src: Path, dst: Path, kept_users: set[str], user_field: str = 'user'
) -> tuple[int, int]:
    """对 logon/device/file 这类有 quoted content 字段的 CSV, 用 csv 模块过滤."""
    in_rows = 0
    out_rows = 0
    with src.open('r', encoding='utf-8', newline='') as fin, \
         dst.open('w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        try:
            user_idx = header.index(user_field)
        except ValueError:
            raise RuntimeError(f'{src}: 缺少字段 {user_field}, header={header}')

        for row in reader:
            in_rows += 1
            # 防御性: row 长度可能与 header 不一致 (极少数转义异常)
            if len(row) > user_idx and row[user_idx] in kept_users:
                writer.writerow(row)
                out_rows += 1
    return in_rows, out_rows


def filter_fastline(
    src: Path, dst: Path, kept_users: set[str], user_col_idx: int = 2
) -> tuple[int, int]:
    """对 http_slim.csv (无 quoted content) 做按行 split 过滤, 性能高.

    http_slim.csv header: id,date,user,pc,url   -> user 在第 3 列 (idx=2)
    URL 字段可能包含逗号, 但 user 在 url 前面, 所以 split(',', 4) 安全.
    """
    in_rows = 0
    out_rows = 0
    with src.open('r', encoding='utf-8', newline='') as fin, \
         dst.open('w', encoding='utf-8', newline='') as fout:
        header_line = fin.readline()
        fout.write(header_line)
        header = header_line.rstrip('\r\n').split(',')
        if header[user_col_idx] != 'user':
            raise RuntimeError(f'{src}: 预期 user 在第 {user_col_idx} 列, header={header}')

        for line in fin:
            in_rows += 1
            # split 至多 user_col_idx + 2 段就够定位 user
            parts = line.split(',', user_col_idx + 1)
            if len(parts) > user_col_idx and parts[user_col_idx] in kept_users:
                fout.write(line)
                out_rows += 1
            if in_rows % 5_000_000 == 0:
                print(f'      ... 已扫描 {in_rows:,} 行 (写入 {out_rows:,})', flush=True)
    return in_rows, out_rows


def filter_all(out_dir: Path, kept_users: set[str]) -> dict:
    stats: dict = {}
    for fname, mode in EVENT_FILES:
        src = RAW_DIR / fname
        dst = out_dir / fname
        if not src.exists():
            print(f'      ⚠ 源文件不存在, 跳过: {src}')
            continue
        src_size_mb = src.stat().st_size / 1024 / 1024
        print(f'[3/4] 过滤 {fname} ({src_size_mb:.1f} MB, mode={mode}) ...', flush=True)
        t0 = time.time()
        if mode == 'csv':
            inn, out = filter_csv_module(src, dst, kept_users)
        else:
            inn, out = filter_fastline(src, dst, kept_users)
        dt = time.time() - t0
        dst_size_mb = dst.stat().st_size / 1024 / 1024
        keep_ratio = (out / inn) if inn else 0.0
        stats[fname] = {
            'in_rows': inn,
            'out_rows': out,
            'keep_ratio': round(keep_ratio, 4),
            'in_size_mb': round(src_size_mb, 1),
            'out_size_mb': round(dst_size_mb, 1),
            'elapsed_sec': round(dt, 1),
        }
        print(f'      -> {fname}: {inn:,} → {out:,} 行 '
              f'({keep_ratio*100:.2f}%), {dst_size_mb:.1f} MB, {dt:.1f}s', flush=True)
    return stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-benign', type=int, default=100,
                    help='要随机抽取的 benign 用户数 (默认 100; 后续可扩到 200/500)')
    ap.add_argument('--seed', type=int, default=42,
                    help='抽样随机种子 (默认 42, 保证可复现)')
    ap.add_argument('--out-name', default=DEFAULT_OUT_NAME,
                    help='输出目录名 (默认 r4.2-mini-v1)')
    ap.add_argument('--include-email', action='store_true',
                    help='同时抽取 email.csv (默认跳过, 因为它 1.3 GB)')
    args = ap.parse_args()

    out_dir = RAW_DIR.parent / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'输出目录: {out_dir}')

    if args.include_email:
        EVENT_FILES.append(('email.csv', 'csv'))

    # 1) 全集用户
    all_users = collect_all_users(RAW_DIR / 'logon.csv')

    # 2) 选 70 + N_BENIGN 个用户
    kept_users, benign_chosen = select_users(all_users, args.n_benign, args.seed)

    # 把选中用户列表落盘 (按 malicious / benign 分块, 方便人工审查)
    sel_path = out_dir / 'selected_users.txt'
    malicious_sorted = sorted(get_all_malicious_user_ids())
    with sel_path.open('w', encoding='utf-8') as f:
        f.write('# malicious_users (n={})\n'.format(len(malicious_sorted)))
        for u in malicious_sorted:
            f.write(u + '\n')
        f.write('# benign_users (n={}, seed={})\n'.format(len(benign_chosen), args.seed))
        for u in benign_chosen:
            f.write(u + '\n')
    print(f'      用户列表已写入 {sel_path}')

    # 3) 过滤所有事件 CSV
    file_stats = filter_all(out_dir, kept_users)

    # 4) 元数据 json
    info = {
        'name': args.out_name,
        'seed': args.seed,
        'n_malicious': len(get_all_malicious_user_ids()),
        'n_benign': len(benign_chosen),
        'n_total_users': len(kept_users),
        'include_email': args.include_email,
        'event_files': [f for f, _ in EVENT_FILES],
        'file_stats': file_stats,
    }
    info_path = out_dir / 'subset_info.json'
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[4/4] 元数据已写入 {info_path}')

    # 总结
    total_out_mb = sum(s['out_size_mb'] for s in file_stats.values())
    print('\n========== r4.2-mini-v1 抽样完成 ==========')
    print(f'用户: malicious=70 + benign={len(benign_chosen)} = {len(kept_users)}')
    print(f'输出: {out_dir}')
    print(f'总输出大小: {total_out_mb:.1f} MB')
    print('===========================================')


if __name__ == '__main__':
    main()
