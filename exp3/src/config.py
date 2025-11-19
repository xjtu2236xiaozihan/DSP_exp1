"""
配置文件
存放所有全局配置和参数
"""

import os

# 基础路径（以本项目 exp3 目录为根）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据与模板目录
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")          # 统一后的音频数据目录
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")        # 模板存储目录

# 标签列表（数字 0-9 + 姓氏拼音）
LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'cheng', 'dai', 'fang', 'gu', 'han', 'hao', 'qi',
    'wang', 'wei', 'xiang', 'xiao', 'yu', 'zheng', 'zi'
]

# 训练文件数量（前8个用于训练，后2个用于测试）
TRAIN_FILE_COUNT = 8

# MFCC参数配置（实验二要求）
MFCC_PARAMS = {
    'n_mfcc': 13,          # 倒谱系数数量
    'n_fft': 2048,         # FFT窗口大小
    'hop_length': 512,     # 帧移
    'preemph_coef': 0.97   # 预加重系数
}

