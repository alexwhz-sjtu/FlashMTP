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
WANDB_DIR = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP_v2/wandb/offline-run-20260808_062242-flashmtp_v2_n16_nlayers5_block_8_leftshift0_mhrnn_easy_direct_r512_ce0.1_tv1.0_wb_0.06_bgemma_12_n2360k_aug3_epochs8_Qwen3-8B2"
WANDB_PROJECT = "flashmtp-training-v2"
SYNC_INTERVAL = 5 * 60  # 秒
# ================================================


def sync_wandb() -> bool:
    cmd = f"wandb sync --id flashmtp_v2_n16_nlayers5_block_8_mhrnn_easy_direct_r512_ce0.1_tv1.0_wb_0.06_bgemma_12_n2360k_aug3_8b_epochs8_Qwen3-8B --project {WANDB_PROJECT} {WANDB_DIR}"
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
