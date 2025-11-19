"""
数据处理模块
处理所有文件路径的查找、过滤和拆分
"""

import os
import re
import glob
import numpy as np
from . import config


def get_audio_files():
    """
    扫描DATASET_DIR目录，获取所有符合 label_serial.wav 命名的音频
    
    Returns:
        dict: 键为标签字符串，值为对应的文件路径列表
              例如: {'0': ['path/0_1.wav', ...], 'cheng': [...], ...}
    """
    file_dict = {label: [] for label in config.LABELS}
    
    # 获取数据集目录下的所有wav文件
    pattern = os.path.join(config.DATASET_DIR, "*.wav")
    all_files = glob.glob(pattern)
    
    # 正则表达式匹配 label_serial.wav 格式
    label_pattern = re.compile(r'^([a-zA-Z0-9]+)_(\d+)\.wav$')
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        match = label_pattern.match(filename)
        
        if match:
            label = match.group(1).lower()
            
            if label in config.LABELS:
                file_dict[label].append(file_path)
    
    return file_dict


def split_train_test(file_dict):
    """
    将文件字典拆分为训练集和测试集
    序号 < TRAIN_FILE_COUNT (例如 0-7) 的文件用于训练 (模板)，
    序号 >= TRAIN_FILE_COUNT (例如 8-9) 的文件用于测试。
    
    Args:
        file_dict: get_audio_files()返回的字典
    
    Returns:
        tuple: (train_files, test_files) 两个字典
    """
    # 1. 初始化空的字典
    train_files = {label: [] for label in config.LABELS}
    test_files = {label: [] for label in config.LABELS}
    
    # 2. 用于从文件名中提取序号的正则表达式（label_serial.wav）
    seq_pattern = re.compile(r'_(\d+)\.wav$')
    
    def get_sequence_number(file_path):
        """辅助函数：从文件名提取序号"""
        filename = os.path.basename(file_path)
        match = seq_pattern.search(filename)
        if match:
            return int(match.group(1))
        return float("inf")  # 返回一个无效值
    
    # 3. 遍历所有找到的文件
    for label, file_list in file_dict.items():
        sorted_files = sorted(file_list, key=get_sequence_number)
        
        for idx, file_path in enumerate(sorted_files):
            if idx < config.TRAIN_FILE_COUNT:
                train_files[label].append(file_path)
            else:
                test_files[label].append(file_path)

    return train_files, test_files


def load_templates():
    """
    从TEMPLATE_DIR加载所有保存的模板MFCC特征
    
    Returns:
        dict: 键为数字字符串，值为该数字的MFCC特征数组列表
              例如: {'0': [mfcc_array_1, mfcc_array_2, ...], '1': [...], ...}
    """
    templates = {}
    
    for label in config.LABELS:
        label_dir = os.path.join(config.TEMPLATE_DIR, label)
        templates[label] = []
        
        # 如果目录存在，加载该标签的所有模板
        if os.path.exists(label_dir):
            npy_files = glob.glob(os.path.join(label_dir, "*.npy"))
            
            for npy_file in sorted(npy_files):
                mfcc = np.load(npy_file)
                templates[label].append(mfcc)
    
    return templates

