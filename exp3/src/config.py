"""
配置文件
存放所有全局配置和参数
"""

import os

# 路径配置
VAD_DIR = "../dataset/VAD/"  # VAD处理后的音频文件目录
TEMPLATE_DIR = "templates/"  # 模板存储目录

# 数字列表（只识别0-9）
DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 训练文件数量（前4个用于训练，第5个用于测试）
TRAIN_FILE_COUNT = 4

# MFCC参数配置（实验二要求）
MFCC_PARAMS = {
    'n_mfcc': 13,          # 倒谱系数数量
    'n_fft': 2048,         # FFT窗口大小
    'hop_length': 512,     # 帧移
    'preemph_coef': 0.97   # 预加重系数
}

