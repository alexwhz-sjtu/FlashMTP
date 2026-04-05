#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wandb 日志同步脚本
功能：每隔 5 分钟执行 wandb sync 上传离线日志
"""

import subprocess
import time
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ==================== 配置区域 ====================
# wandb 离线日志目录
WANDB_DIR = "/inspire/hdd/project/inference-chip/xujiaming-253308120313/whz/FlashMTP/wandb/offline-run-20260402_171351-0gq1e0ja"

# wandb project 名称
WANDB_PROJECT = "flashmtp_training"

# 同步间隔（秒）
SYNC_INTERVAL = 5 * 60  # 5 分钟

# 日志文件路径
LOG_FILE = "./wandb_sync_2.log"

# 锁文件路径（防止重复执行）
LOCK_FILE = "./wandb_sync_2.lock"
# ================================================


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def check_lock():
    """检查是否有锁文件（防止重复执行）"""
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, 'r') as f:
            pid = f.read().strip()
            if pid.isdigit() and is_process_running(int(pid)):
                logging.warning(f"检测到同步进程正在运行 (PID: {pid})，跳过本次执行")
                return False
        # 锁文件存在但进程已结束，删除旧锁
        os.remove(LOCK_FILE)
    return True


def create_lock():
    """创建锁文件"""
    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))


def remove_lock():
    """删除锁文件"""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)


def is_process_running(pid):
    """检查进程是否正在运行"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def sync_wandb():
    """执行 wandb sync 命令"""
    cmd = f"wandb sync --project {WANDB_PROJECT} {WANDB_DIR}"
    
    logging.info(f"开始执行同步命令：{cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 分钟超时
        )
        
        if result.returncode == 0:
            logging.info(f"✓ 同步成功 | stdout: {result.stdout.strip()}")
            return True
        else:
            logging.error(f"✗ 同步失败 | returncode: {result.returncode}")
            logging.error(f"stderr: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("✗ 同步超时（超过 5 分钟）")
        return False
    except Exception as e:
        logging.error(f"✗ 同步异常：{str(e)}")
        return False


def main():
    """主函数"""
    setup_logging()
    
    logging.info("=" * 60)
    logging.info("wandb 日志同步脚本启动")
    logging.info(f"日志目录：{WANDB_DIR}")
    logging.info(f"Project: {WANDB_PROJECT}")
    logging.info(f"同步间隔：{SYNC_INTERVAL} 秒")
    logging.info("=" * 60)
    
    # 检查目录是否存在
    if not os.path.exists(WANDB_DIR):
        logging.error(f"错误：日志目录不存在：{WANDB_DIR}")
        sys.exit(1)
    
    sync_count = 0
    success_count = 0
    
    try:
        while True:
            sync_count += 1
            logging.info(f"\n[第 {sync_count} 次同步] 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 检查锁
            if check_lock():
                create_lock()
                try:
                    if sync_wandb():
                        success_count += 1
                finally:
                    remove_lock()
            else:
                logging.info("跳过本次同步")
            
            # 成功率统计
            if sync_count > 0:
                rate = success_count / sync_count * 100
                logging.info(f"当前成功率：{rate:.1f}% ({success_count}/{sync_count})")
            
            # 等待下次执行
            logging.info(f"等待 {SYNC_INTERVAL} 秒后执行下一次同步...")
            time.sleep(SYNC_INTERVAL)
            
    except KeyboardInterrupt:
        logging.info("\n收到中断信号，停止同步")
    except Exception as e:
        logging.error(f"发生异常：{str(e)}")
    finally:
        remove_lock()
        logging.info("脚本已退出")


if __name__ == "__main__":
    main()
    
    
'''
# 找到进程 PID
ps aux | grep wandb_sync_daemon

# 杀死进程
kill -9 <PID>
'''