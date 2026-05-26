"""
从 http.csv (14.5 GB, 含 content 字段) 生成 http_slim.csv (仅保留 id,date,user,pc,url).

用法:
    python src/generate_http_slim.py [--data-dir data/r4.2]

输出:
    data/r4.2/http_slim.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 允许 http.csv 的 content 字段很大
csv.field_size_limit(100_000_000)

KEEP_COLS = ['id', 'date', 'user', 'pc', 'url']


def main():
    ap = argparse.ArgumentParser(description='Generate http_slim.csv from http.csv')
    ap.add_argument('--data-dir',
                    default=str(PROJECT_ROOT / 'data' / 'r4.2'),
                    help='数据目录 (含 http.csv)')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    src_path = data_dir / 'http.csv'
    dst_path = data_dir / 'http_slim.csv'

    if not src_path.exists():
        print(f'ERROR: {src_path} 不存在')
        sys.exit(1)

    if dst_path.exists():
        src_size = src_path.stat().st_size
        dst_size = dst_path.stat().st_size
        # 如果 slim 文件已存在且大小合理 (> 1GB), 跳过
        if dst_size > 1_000_000_000:
            print(f'http_slim.csv 已存在 ({dst_size / 1e9:.2f} GB), 跳过生成.')
            print(f'如需重新生成, 请先删除 {dst_path}')
            return
        else:
            print(f'http_slim.csv 已存在但较小 ({dst_size / 1e6:.1f} MB), 将重新生成...')

    src_size_gb = src_path.stat().st_size / (1024**3)
    print(f'源文件: {src_path} ({src_size_gb:.2f} GB)')
    print(f'输出:   {dst_path}')
    print(f'保留列: {KEEP_COLS}')
    print('开始处理 (预计 10-30 分钟) ...\n')

    t0 = time.time()
    n_in = 0
    n_out = 0

    with open(src_path, 'r', encoding='utf-8', newline='') as fin, \
         open(dst_path, 'w', encoding='utf-8', newline='') as fout:

        reader = csv.DictReader(fin)
        # 验证所需列都存在
        missing = [c for c in KEEP_COLS if c not in reader.fieldnames]
        if missing:
            print(f'ERROR: http.csv 缺少列: {missing}')
            print(f'实际列: {reader.fieldnames}')
            sys.exit(1)

        writer = csv.DictWriter(fout, fieldnames=KEEP_COLS,
                                extrasaction='ignore')
        writer.writeheader()

        for row in reader:
            n_in += 1
            slim = {c: row[c] for c in KEEP_COLS}
            writer.writerow(slim)
            n_out += 1

            if n_in % 2_000_000 == 0:
                elapsed = time.time() - t0
                rate = n_in / elapsed
                print(f'  已处理 {n_in:>12,} 行  '
                      f'({elapsed:.0f}s, {rate:,.0f} rows/s)',
                      flush=True)

    elapsed = time.time() - t0
    dst_size_gb = dst_path.stat().st_size / (1024**3)
    print(f'\n完成!')
    print(f'  输入:  {n_in:,} 行')
    print(f'  输出:  {n_out:,} 行')
    print(f'  耗时:  {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)')
    print(f'  大小:  {dst_size_gb:.2f} GB')
    print(f'  路径:  {dst_path}')


if __name__ == '__main__':
    main()
