#!/usr/bin/env python3
"""删除 log 下每个第一级文件夹中 ckpt/ckpts 目录下的非 Best_mae 的 .pth 文件。"""

import os
import re
import argparse

LOG_DIR = "log"
CKPT_DIRS = ("ckpt", "ckpts")  # 支持 ckpt 或 ckpts

# 保留的文件名模式（不区分大小写）
KEEP_PATTERNS = (
    re.compile(r"best.*mae", re.I),
    re.compile(r"Best_mae_test\.pth", re.I),
)


def is_keep_file(name: str) -> bool:
    """判断是否应保留该文件。"""
    for pat in KEEP_PATTERNS:
        if pat.search(name):
            return True
    return False


def cleanup_ckpts(log_root: str, dry_run: bool = False):
    """遍历 log 下第一级目录，删除 ckpt/ckpts 中非 best_mae 的 .pth。"""
    log_root = os.path.abspath(log_root)
    if not os.path.isdir(log_root):
        print(f"目录不存在: {log_root}")
        return

    deleted = []
    kept = []

    for name in sorted(os.listdir(log_root)):
        path = os.path.join(log_root, name)
        if not os.path.isdir(path):
            continue

        for ckpt_dir in CKPT_DIRS:
            ckpt_path = os.path.join(path, ckpt_dir)
            if not os.path.isdir(ckpt_path):
                continue

            for f in os.listdir(ckpt_path):
                if not f.endswith(".pth"):
                    continue
                full_path = os.path.join(ckpt_path, f)
                if is_keep_file(f):
                    kept.append(full_path)
                else:
                    deleted.append(full_path)
                    if not dry_run:
                        os.remove(full_path)

    if dry_run:
        print("[DRY RUN] 以下文件将被删除:")
        for p in deleted:
            print(f"  - {p}")
        print(f"\n共 {len(deleted)} 个文件将被删除，{len(kept)} 个文件保留。")
    else:
        for p in deleted:
            print(f"已删除: {p}")
        print(f"\n已删除 {len(deleted)} 个文件，保留 {len(kept)} 个 Best_mae 文件。")


def main():
    parser = argparse.ArgumentParser(description="删除 log 下非 Best_mae 的 checkpoint")
    parser.add_argument(
        "--log_dir",
        default=os.path.join(os.path.dirname(__file__), "..", LOG_DIR),
        help="log 根目录",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印将要删除的文件，不实际删除",
    )
    args = parser.parse_args()
    cleanup_ckpts(args.log_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
