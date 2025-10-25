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
    扫描VAD_DIR目录，获取所有数字音频文件
    
    Returns:
        dict: 键为数字字符串，值为对应的文件路径列表
              例如: {'0': ['path/0_1_xiugai2.wav', ...], '1': [...], ...}
    """
    file_dict = {digit: [] for digit in config.DIGITS}
    
    # 获取VAD目录下的所有wav文件
    pattern = os.path.join(config.VAD_DIR, "*.wav")
    all_files = glob.glob(pattern)
    
    # 正则表达式匹配数字_数字_*.wav格式
    digit_pattern = re.compile(r'^(\d+)_(\d+)_.+\.wav$')
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        match = digit_pattern.match(filename)
        
        if match:
            digit = match.group(1)  # 第一个数字（标签）
            
            # 只保留0-9的数字文件
            if digit in config.DIGITS:
                file_dict[digit].append(file_path)
    
    return file_dict


def split_train_test(file_dict):
    """
    将文件字典拆分为训练集和测试集
    前TRAIN_FILE_COUNT个文件用于训练，剩余的用于测试
    
    Args:
        file_dict: get_audio_files()返回的字典
    
    Returns:
        tuple: (train_files, test_files) 两个字典
    """
    train_files = {}
    test_files = {}
    
    # 用于从文件名中提取序号的正则表达式
    seq_pattern = re.compile(r'_(\d+)_')
    
    for digit, file_list in file_dict.items():
        # 根据文件名中的第二个数字（序号）排序
        def get_sequence_number(file_path):
            filename = os.path.basename(file_path)
            match = seq_pattern.search(filename)
            if match:
                return int(match.group(1))
            return 0
        
        sorted_files = sorted(file_list, key=get_sequence_number)
        
        # 前TRAIN_FILE_COUNT个用于训练
        train_files[digit] = sorted_files[:config.TRAIN_FILE_COUNT]
        # 剩余的用于测试（应该只有第5个）
        test_files[digit] = sorted_files[config.TRAIN_FILE_COUNT:]
    
    return train_files, test_files


def load_templates():
    """
    从TEMPLATE_DIR加载所有保存的模板MFCC特征
    
    Returns:
        dict: 键为数字字符串，值为该数字的MFCC特征数组列表
              例如: {'0': [mfcc_array_1, mfcc_array_2, ...], '1': [...], ...}
    """
    templates = {}
    
    for digit in config.DIGITS:
        digit_dir = os.path.join(config.TEMPLATE_DIR, digit)
        templates[digit] = []
        
        # 如果目录存在，加载该数字的所有模板
        if os.path.exists(digit_dir):
            npy_files = glob.glob(os.path.join(digit_dir, "*.npy"))
            
            for npy_file in sorted(npy_files):
                mfcc = np.load(npy_file)
                templates[digit].append(mfcc)
    
    return templates

