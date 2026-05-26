"""
一键运行完整流水线: N-gram + R-GCN + 对比.

用法:
    python src/run_full_pipeline.py [--skip-preprocess] [--skip-ngram] [--skip-rgcn]

步骤:
    1. 生成 http_slim.csv (如不存在)
    2. N-gram: 预处理 → 训练评估 → 可视化
    3. R-GCN: 构建图 → 训练评估
    4. 对比分析
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
DATA_DIR = PROJECT_ROOT / 'data' / 'r4.2'
OUT_DIR = PROJECT_ROOT / 'outputs'


def run_step(name: str, cmd: list[str], cwd: str | None = None):
    """运行一个步骤, 打印输出."""
    print(f'\n{"=" * 60}')
    print(f'  步骤: {name}')
    print(f'  命令: {" ".join(cmd)}')
    print(f'{"=" * 60}\n')

    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd or str(PROJECT_ROOT),
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'\n❌ {name} 失败 (exit code {result.returncode})')
        print(f'   耗时: {elapsed:.1f}s')
        return False

    print(f'\n✅ {name} 完成 (耗时: {elapsed:.1f}s)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-preprocess', action='store_true',
                    help='跳过预处理 (如已有 user_sequences.pkl)')
    ap.add_argument('--skip-ngram', action='store_true',
                    help='跳过 N-gram 算法')
    ap.add_argument('--skip-rgcn', action='store_true',
                    help='跳过 R-GCN 算法')
    ap.add_argument('--skip-compare', action='store_true',
                    help='跳过对比分析')
    args = ap.parse_args()

    python = sys.executable
    total_t0 = time.time()

    print('=' * 60)
    print('  CMU-CERT r4.2 双算法对比流水线')
    print(f'  数据目录: {DATA_DIR}')
    print(f'  输出目录: {OUT_DIR}')
    print('=' * 60)

    # ---- Step 0: 检查 http_slim.csv ----
    slim_path = DATA_DIR / 'http_slim.csv'
    if not slim_path.exists():
        ok = run_step(
            '生成 http_slim.csv',
            [python, str(SRC_DIR / 'generate_http_slim.py')]
        )
        if not ok:
            return
    else:
        print(f'\n✅ http_slim.csv 已存在 ({slim_path.stat().st_size / 1e9:.2f} GB)')

    # ---- Step 1: N-gram 预处理 ----
    if not args.skip_ngram:
        if not args.skip_preprocess:
            ok = run_step(
                'N-gram 预处理 (preprocess.py)',
                [python, str(SRC_DIR / 'preprocess.py')]
            )
            if not ok:
                return
        else:
            print('\n⏭ 跳过预处理')

        # ---- Step 2: N-gram 训练评估 ----
        ok = run_step(
            'N-gram 训练评估 (train_evaluate.py)',
            [python, str(SRC_DIR / 'train_evaluate.py')]
        )
        if not ok:
            return

        # ---- Step 3: N-gram 可视化 ----
        ok = run_step(
            'N-gram 可视化 (visualize.py)',
            [python, str(SRC_DIR / 'visualize.py')]
        )
        if not ok:
            print('  (可视化失败, 但不影响后续步骤)')
    else:
        print('\n⏭ 跳过 N-gram 算法')

    # ---- Step 4: R-GCN 构建图 ----
    if not args.skip_rgcn:
        ok = run_step(
            'R-GCN 构建异构图 (build_graph.py)',
            [python, str(SRC_DIR / 'rgcn' / 'build_graph.py')]
        )
        if not ok:
            return

        # ---- Step 5: R-GCN 训练评估 ----
        ok = run_step(
            'R-GCN 训练评估 (train_rgcn.py)',
            [python, str(SRC_DIR / 'rgcn' / 'train_rgcn.py'), '--verbose']
        )
        if not ok:
            return
    else:
        print('\n⏭ 跳过 R-GCN 算法')

    # ---- Step 6: 对比分析 ----
    if not args.skip_compare:
        ok = run_step(
            '算法性能对比 (compare_algorithms.py)',
            [python, str(SRC_DIR / 'compare_algorithms.py')]
        )
        if not ok:
            return
    else:
        print('\n⏭ 跳过对比分析')

    total_elapsed = time.time() - total_t0
    print(f'\n{"=" * 60}')
    print(f'  全部完成! 总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f} 分钟)')
    print(f'  结果目录: {OUT_DIR}')
    print(f'  对比报告: {OUT_DIR / "comparison" / "comparison_report.md"}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
