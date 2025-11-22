"""
配置文件 (ASR标准参数版)
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")

LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'cheng', 'dai', 'fang', 'gu', 'han', 'hao', 'qi',
    'wang', 'wei', 'xiang', 'xiao', 'yu', 'zheng', 'zi'
]

TRAIN_FILE_COUNT = 8

# 1. 采样率保持 16k
SAMPLE_RATE = 16000 

# 2. [FIX] 采用语音识别黄金标准参数 (25ms窗口, 10ms帧移)
# 这会生成比之前多 2 倍的特征帧，大幅提升短词识别率
MFCC_PARAMS = {
    'n_mfcc': 13,
    'n_fft': 512,          # FFT 计算点数 (保持 2^9)
    'win_length': 400,     # [新增] 窗口长度 25ms (16000 * 0.025)
    'hop_length': 160,     # [修改] 帧移 10ms (16000 * 0.010)
    'preemph_coef': 0.97
}