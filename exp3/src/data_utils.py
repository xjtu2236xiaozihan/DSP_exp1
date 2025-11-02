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
    序号 < TRAIN_FILE_COUNT (例如 0-7) 的文件用于训练 (模板)，
    序号 >= TRAIN_FILE_COUNT (例如 8-9) 的文件用于测试。
    
    Args:
        file_dict: get_audio_files()返回的字典
    
    Returns:
        tuple: (train_files, test_files) 两个字典
    """
    # 1. 初始化空的字典
    train_files = {digit: [] for digit in config.DIGITS}
    test_files = {digit: [] for digit in config.DIGITS}
    
    # 2. 用于从文件名中提取序号的正则表达式
    # 匹配 num_ORDER_...
    seq_pattern = re.compile(r'_(\d+)_')
    
    def get_sequence_number(file_path):
        """辅助函数：从文件名提取序号"""
        filename = os.path.basename(file_path)
        match = seq_pattern.search(filename)
        if match:
            return int(match.group(1))
        return -1 # 返回一个无效值

    # 3. 遍历所有找到的文件
    for digit, file_list in file_dict.items():
        for file_path in file_list:
            seq_num = get_sequence_number(file_path)
            
            if seq_num == -1:
                continue # 忽略不匹配的文件
            
            # 4. 核心修改：
            # 按序号判断，而不是按文件列表的索引
            # 假设 config.TRAIN_FILE_COUNT = 8
            # 序号 0-7 (< 8) 进入训练集
            if seq_num < config.TRAIN_FILE_COUNT:
                train_files[digit].append(file_path)
            # 序号 8-9 (>= 8) 进入测试集
            else:
                test_files[digit].append(file_path)
        
        # （可选，但推荐）对列表进行排序，确保顺序一致
        train_files[digit].sort(key=get_sequence_number)
        test_files[digit].sort(key=get_sequence_number)

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

