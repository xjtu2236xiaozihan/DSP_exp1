"""
特征提取模块
提取MFCC特征（实验二内容）
"""

import os
from . import config

# 处理 numba cache 在 macOS / Python 3.13 上的限制
NUMBA_CACHE_DIR = os.path.join(config.PROJECT_ROOT, ".numba_cache")
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")

import librosa
import numpy as np


def extract_mfcc(file_path):
    """
    从音频文件中提取MFCC特征 (包含静音消除和标准化)
    
    Args:
        file_path: 音频文件路径
    
    Returns:
        numpy.ndarray: MFCC特征数组，形状为 (n_mfcc, n_frames)
    """
    # 1. 加载音频文件（保持原始采样率）
    y, sr = librosa.load(file_path, sr=None)
    
    # --- 优化点1：静音消除 ---
    # 去除首尾低于 30dB 的静音部分
    # 这能防止大量静音帧干扰 DTW 匹配
    y, _ = librosa.effects.trim(y, top_db=30)
    
    # 安全检查：如果剪切后音频太短（比如全是噪音被剪没了），这就重新加载原音频或保留部分
    if len(y) < 1024:
        y, sr = librosa.load(file_path, sr=None)

    # 2. 不应用预加重 消融实验揭示不使用准确率更高
    # y = librosa.effects.preemphasis(y, coef=config.MFCC_PARAMS['preemph_coef'])
    
    # 3. 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.MFCC_PARAMS['n_mfcc'],
        n_fft=config.MFCC_PARAMS['n_fft'],
        hop_length=config.MFCC_PARAMS['hop_length']
    )
    
    # --- 优化点2：特征标准化 (CMVN) ---
    # Cepstral Mean and Variance Normalization
    # 减去均值，除以标准差。消除信道噪声和音量差异的影响。
    # axis=1 表示沿时间轴计算均值和方差
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8) # 加极小值防止除零
    
    return mfcc