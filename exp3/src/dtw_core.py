"""
DTW核心模块
实现动态时间规整算法（实验三核心）
"""

import numpy as np
from dtw import dtw


def calculate_dtw_distance(template_mfcc, input_mfcc):
    """
    计算两个MFCC特征序列之间的DTW距离
    
    Args:
        template_mfcc: 模板MFCC特征，形状为 (n_mfcc, n_frames)
        input_mfcc: 输入MFCC特征，形状为 (n_mfcc, n_frames)
    
    Returns:
        float: DTW累计最小距离
    """
    # 注意：librosa输出为(n_features, n_frames)
    # 而dtw-python期望输入为(n_frames, n_features)
    # 因此需要转置
    
    # 使用欧式距离计算DTW
    alignment = dtw(
        template_mfcc.T, 
        input_mfcc.T, 
        keep_internals=False,
        dist_method='euclidean'
    )
    
    return alignment.distance

