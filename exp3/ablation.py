"""
消融试验脚本
测试移除 '预加重' (Pre-emphasis) 步骤对识别率的影响
"""

import os
import numpy as np
import librosa  # 需要额外导入
from src import config
from src import data_utils
from src import dtw_core # 使用你原来的欧氏距离

# --- 消融特征提取 ---

def extract_mfcc_no_preemphasis(file_path):
    """
    (消融版) 提取MFCC特征，但不使用预加重
    """
    # 加载音频文件
    y, sr = librosa.load(file_path, sr=None)
    
    # ！！！ 移除了预加重步骤 ！！！
    # y = librosa.effects.preemphasis(y, coef=config.MFCC_PARAMS['preemph_coef'])
    
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.MFCC_PARAMS['n_mfcc'],
        n_fft=config.MFCC_PARAMS['n_fft'],
        hop_length=config.MFCC_PARAMS['hop_length']
    )
    return mfcc

# --- 消融版模板加载 ---

def build_templates_ablation(train_files):
    """
    (消融版) 使用 'extract_mfcc_no_preemphasis' 在内存中构建模板
    """
    templates = {digit: [] for digit in config.DIGITS}
    
    for digit in config.DIGITS:
        for file_path in train_files[digit]:
            # 使用消融版的特征提取
            mfcc = extract_mfcc_no_preemphasis(file_path)
            templates[digit].append(mfcc)
            
    return templates

# --- 主函数 ---

def main():
    """
    执行消融试验
    """
    print("="*60)
    print("DTW语音识别系统 - 消融试验 (移除预加重)")
    print("="*60)
    
    # 1. 获取所有音频文件
    print("\n[步骤 1/4] 扫描音频文件...")
    all_files = data_utils.get_audio_files()
    train_files, test_files = data_utils.split_train_test(all_files)
    
    total_train = sum(len(files) for files in train_files.values())
    total_test = sum(len(files) for files in test_files.values())
    print(f"找到 {total_train} 个训练文件, {total_test} 个测试文件")

    # 2. 加载模板库 (使用消融版特征)
    print("\n[步骤 2/4] 使用 (消融版) 特征在内存中构建模板...")
    templates = build_templates_ablation(train_files)
    print(f"成功构建 {total_train} 个消融模板")
    
    # 3. 执行识别 (使用消融版特征)
    print("\n[步骤 3/4] 开始识别 (使用消融版特征)...")
    print("-"*60)
    
    correct_count = 0
    total_count = 0
    
    for true_digit in config.DIGITS:
        for test_file_path in test_files[true_digit]:
            total_count += 1
            filename = os.path.basename(test_file_path)
            
            # 提取测试音频的 (消融版) MFCC特征
            test_mfcc = extract_mfcc_no_preemphasis(test_file_path)
            
            min_distance = float('inf')
            predicted_digit = None
            
            # 与所有 (消融版) 模板计算DTW距离
            for template_digit in config.DIGITS:
                for template_mfcc in templates[template_digit]:
                    distance = dtw_core.calculate_dtw_distance(
                        template_mfcc, 
                        test_mfcc
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        predicted_digit = template_digit
            
            # 统计结果
            is_correct = (predicted_digit == true_digit)
            if is_correct:
                correct_count += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} [{total_count:2d}] {filename:25s} | "
                  f"真实: {true_digit} | 预测: {predicted_digit}")

    # 4. 打印统计报告
    print("-"*60)
    print("\n[步骤 4/4] 识别结果统计 (消融版)")
    print("="*60)
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"  测试样本数: {total_count}")
    print(f"  识别正确数: {correct_count}")
    print(f"  识别错误数: {total_count - correct_count}")
    print(f"  基线准确率: 100.00% (来自 run_recognition.py)")
    print(f"  消融准确率: {accuracy:.2f}% (无预加重)")
    print("="*60)


if __name__ == "__main__":
    # 确保安装了 librosa: pip install librosa
    main()