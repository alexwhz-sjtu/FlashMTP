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
WANDB_DIR = "/data/wanghanzhen/FlashMTP_v2/wandb/run-20260726_174925-flashmtp_v1.1_n16_nlayers5_block_16_mhrnn_direct_r256_n40000_epochs6_Qwen3-8B"
WANDB_PROJECT = "flashmtp-training-exp"
SYNC_INTERVAL = 5 * 60  # 秒
# ================================================


def sync_wandb() -> bool:
    cmd = f"wandb sync --no-skip-online --project {WANDB_PROJECT} {WANDB_DIR}"
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
