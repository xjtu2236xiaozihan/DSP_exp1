"""
DTW核心模块
实现动态时间规整算法（实验三核心）
"""

import numpy as np
from dtw import dtw


def calculate_dtw_distance(template_mfcc, input_mfcc):
    """
    计算两个MFCC特征序列之间的DTW距离 (经过长度归一化)
    
    Args:
        template_mfcc: 模板MFCC特征，形状为 (n_mfcc, n_frames)
        input_mfcc: 输入MFCC特征，形状为 (n_mfcc, n_frames)
    
    Returns:
        float: 归一化后的DTW距离
    """
    # 获取序列长度（帧数）
    # shape[0]是特征维数(13)，shape[1]是时间帧数
    len_template = template_mfcc.shape[1]
    len_input = input_mfcc.shape[1]
    
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
    
    # --- 核心修改：距离归一化 ---
    # 原始 alignment.distance 是累积路径距离。
    # 序列越长，累积距离越大，导致算法偏向于匹配短的模板。
    # 除以两者长度之和 (或者路径长度) 进行归一化，使其具有可比性。
    normalized_distance = alignment.distance / (len_template + len_input)
    
    return normalized_distance