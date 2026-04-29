#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wandb 日志同步脚本：每隔 SYNC_INTERVAL 秒执行 wandb sync 上传离线日志
"""

import os
import subprocess
import sys
import time

# ==================== 配置区域 ====================
WANDB_DIR = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/wandb/offline-run-20260402_171351-0gq1e0ja"
WANDB_PROJECT = "flashmtp_training"
SYNC_INTERVAL = 5 * 60  # 秒
# ================================================


def sync_wandb() -> bool:
    cmd = f"wandb sync --project {WANDB_PROJECT} {WANDB_DIR}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=300,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def main():
    if not os.path.exists(WANDB_DIR):
        sys.exit(1)
    try:
        while True:
            sync_wandb()
            time.sleep(SYNC_INTERVAL)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
