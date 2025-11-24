"""
模板构建脚本（训练阶段）
从前8个音频文件提取MFCC特征并保存为模板
"""

import os
import numpy as np
from src import config
from src import data_utils
from src import features


def main():
    """
    执行模板构建流程
    """
    print("="*60)
    print("DTW语音识别系统 - 模板构建")
    print("="*60)
    
    # 1. 获取所有音频文件
    print("\n[步骤 1/3] 扫描音频文件...")
    all_files = data_utils.get_audio_files()
    
    total_files = sum(len(files) for files in all_files.values())
    print(f"找到 {total_files} 个标签音频文件")
    for label in config.LABELS:
        print(f"  标签 {label}: {len(all_files[label])} 个文件")
    
    # 2. 分割训练集和测试集
    print("\n[步骤 2/3] 分割数据集...")
    train_files, test_files = data_utils.split_train_test(all_files)
    
    total_train = sum(len(files) for files in train_files.values())
    total_test = sum(len(files) for files in test_files.values())
    print(f"训练集: {total_train} 个文件 (每个标签 {config.TRAIN_FILE_COUNT} 个)")
    test_per_label = total_test // len(config.LABELS) if config.LABELS else 0
    print(f"测试集: {total_test} 个文件 (每个标签约 {test_per_label} 个)")
    
    # 3. 提取特征并保存模板
    print("\n[步骤 3/3] 提取MFCC特征并构建模板库...")
    
    template_count = 0
    
    for label in config.LABELS:
        # 创建标签的模板目录
        label_dir = os.path.join(config.TEMPLATE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # 遍历该标签的训练文件
        for file_path in train_files[label]:
            filename = os.path.basename(file_path)
            
            # 提取MFCC特征
            mfcc = features.extract_mfcc(file_path)
            
            # 生成保存路径（保持原文件名，但后缀改为.npy）
            save_name = filename.replace('.wav', '.npy')
            save_path = os.path.join(label_dir, save_name)
            
            # 保存MFCC特征
            np.save(save_path, mfcc)
            
            template_count += 1
            print(f"  [{template_count:2d}/{total_train}] 标签 {label}: {filename} -> MFCC形状: {mfcc.shape}")
    
    print("\n" + "="*60)
    print(f"✓ 模板构建完成！共创建 {template_count} 个模板")
    print(f"✓ 模板保存在: {config.TEMPLATE_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()

