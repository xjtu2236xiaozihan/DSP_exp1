"""
特征提取模块
提取MFCC特征（实验二内容）
"""

import librosa
import numpy as np
from . import config


def extract_mfcc(file_path):
    """
    从音频文件中提取MFCC特征
    
    Args:
        file_path: 音频文件路径
    
    Returns:
        numpy.ndarray: MFCC特征数组，形状为 (n_mfcc, n_frames)
    """
    # 加载音频文件（保持原始采样率）
    y, sr = librosa.load(file_path, sr=None)
    
    # 应用预加重
    y = librosa.effects.preemphasis(y, coef=config.MFCC_PARAMS['preemph_coef'])
    
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.MFCC_PARAMS['n_mfcc'],
        n_fft=config.MFCC_PARAMS['n_fft'],
        hop_length=config.MFCC_PARAMS['hop_length']
    )
    
    return mfcc

