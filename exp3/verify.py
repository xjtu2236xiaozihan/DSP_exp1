"""
算法验证脚本 (verify_algorithm.py)
功能：在本地数据集上执行标准的 Train/Test 分割测试，验证核心算法有效性。
不依赖 Flask，不依赖麦克风。
"""

import os
import shutil
import numpy as np
import time
from src import config
from src import data_utils
from src import features
from src import dtw_core

def main():
    print("="*60)
    print("DTW 核心算法本地自检程序")
    print(f"参数检查: SR={config.SAMPLE_RATE}, Win={config.MFCC_PARAMS.get('win_length')}, Hop={config.MFCC_PARAMS['hop_length']}")
    print("="*60)

    # 1. 准备数据
    print("\n[步骤 1] 扫描数据集...")
    all_files = data_utils.get_audio_files()
    train_files, test_files = data_utils.split_train_test(all_files)
    
    total_train = sum(len(f) for f in train_files.values())
    total_test = sum(len(f) for f in test_files.values())
    print(f"  - 训练集 (用于做模板): {total_train} 个 (每个标签前 {config.TRAIN_FILE_COUNT} 个)")
    print(f"  - 测试集 (用于跑分):   {total_test} 个")

    if total_train == 0:
        print("❌ 错误: 未找到训练文件，请检查 dataset 目录！")
        return

    # 2. 临时构建内存模板 (不覆盖硬盘上的发布版模板，以免影响 exe)
    print("\n[步骤 2] 提取训练集特征 (构建内存模板)...")
    memory_templates = {}
    build_start = time.time()
    
    for label in config.LABELS:
        memory_templates[label] = []
        for file_path in train_files[label]:
            try:
                # 提取特征 (会调用 features.py 中的最新逻辑：滤波、降噪等)
                mfcc = features.extract_mfcc(file_path)
                memory_templates[label].append(mfcc)
            except Exception as e:
                print(f"  ⚠️ 训练样本提取失败: {os.path.basename(file_path)} - {e}")
    
    print(f"  ✓ 模板构建完成，耗时 {time.time() - build_start:.2f}s")

    # 3. 执行识别测试
    print("\n[步骤 3] 开始批量识别测试...")
    correct_count = 0
    total_count = 0
    errors = [] # 记录错误详情

    test_start = time.time()

    for true_label in config.LABELS:
        for file_path in test_files[true_label]:
            total_count += 1
            filename = os.path.basename(file_path)
            
            try:
                # 提取测试样本特征
                test_mfcc = features.extract_mfcc(file_path)
                
                # DTW 匹配
                min_dist = float('inf')
                predicted = None
                
                for t_label, t_mfccs in memory_templates.items():
                    for t_mfcc in t_mfccs:
                        dist = dtw_core.calculate_dtw_distance(t_mfcc, test_mfcc)
                        if dist < min_dist:
                            min_dist = dist
                            predicted = t_label
                
                # 统计
                if predicted == true_label:
                    correct_count += 1
                else:
                    # 记录错误：文件名, 真实标签, 预测标签, 距离
                    errors.append((filename, true_label, predicted, min_dist))
                    print(f"  ❌ 错: {filename:<20} | 真: {true_label} -> 猜: {predicted} (dist: {min_dist:.2f})")

            except Exception as e:
                print(f"  ⚠️ 测试样本跳过: {filename} - {e}")

    # 4. 输出最终报告
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    duration = time.time() - test_start
    
    print("\n" + "="*60)
    print("测试结果摘要")
    print("="*60)
    print(f"总样本数: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"错误数量: {len(errors)}")
    print(f"平均耗时: {duration/total_count*1000:.1f} ms/个")
    print(f"---------------------------")
    print(f"最终准确率: {accuracy:.2f}%")
    print(f"---------------------------")
    
    if errors:
        print("\n[错误分析 - Top 10 典型错误]")
        # 简单展示前10个错误
        for i, (fname, true_l, pred_l, dist) in enumerate(errors[:10]):
            print(f"{i+1}. {fname}: 把 '{true_l}' 认成了 '{pred_l}'")
            
    if accuracy < 60:
        print("\n[诊断建议]")
        print("1. 准确率极低，说明特征参数(config.py)与数据集严重不匹配。")
        print("2. 可能是滤波器(80-7500Hz)切掉了关键信息，或者帧长(25ms)不适合该语速。")
        print("3. 请检查 features.extract_mfcc 中的 librosa.load 是否强制使用了 sr=16000。")
    elif accuracy > 90:
        print("\n[诊断结论]")
        print("✅ 核心算法健康！本地文件识别率很高。")
        print("👉 问题大概率出在：前端录音采样率不匹配、麦克风噪音、或 WebM 转码失真。")

if __name__ == "__main__":
    main()