"""
对比试验脚本
对比 'euclidean' 和 'cosine' 两种距离度量对识别率的影响
"""

import os
import numpy as np
from dtw import dtw  # 直接导入dtw库以支持不同度量
from src import config
from src import data_utils
from src import features
from src import visualization

def calculate_dtw_distance(template_mfcc, input_mfcc, dist_method):
    """
    计算两个MFCC特征序列之间的DTW距离 (可指定度量方法)
    
    Args:
        template_mfcc: 模板MFCC (n_mfcc, n_frames)
        input_mfcc: 输入MFCC (n_mfcc, n_frames)
        dist_method: 'euclidean' or 'cosine'
    
    Returns:
        float: DTW累计最小距离
    """
    # 转置以匹配 dtw-python 的输入 (n_frames, n_features)
    alignment = dtw(
        template_mfcc.T, 
        input_mfcc.T, 
        keep_internals=False,
        dist_method=dist_method
    )
    return alignment.distance

def run_recognition_test(templates, test_files, dist_method):
    """
    使用指定的距离度量执行一次完整的识别流程
    """
    print("\n" + "="*60)
    print(f"开始测试 - 距离度量: {dist_method.upper()}")
    print("="*60)
    
    correct_count = 0
    total_count = 0
    
    # 遍历每个数字的测试文件
    for true_digit in config.DIGITS:
        for test_file_path in test_files[true_digit]:
            total_count += 1
            filename = os.path.basename(test_file_path)
            
            # 提取测试音频的MFCC特征
            test_mfcc = features.extract_mfcc(test_file_path)
            
            min_distance = float('inf')
            predicted_digit = None
            
            # 与所有模板计算DTW距离
            for template_digit in config.DIGITS:
                for template_mfcc in templates[template_digit]:
                    distance = calculate_dtw_distance(
                        template_mfcc, 
                        test_mfcc, 
                        dist_method
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        predicted_digit = template_digit
            
            # 统计结果
            if predicted_digit == true_digit:
                correct_count += 1
            
            # 打印简略结果
            status = "✓" if (predicted_digit == true_digit) else "✗"
            print(f"{status} [{total_count:2d}] {filename:25s} | "
                  f"真实: {true_digit} | 预测: {predicted_digit}")

    # 4. 打印统计报告
    print("-"*60)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n测试完成: {dist_method.upper()}")
    print(f"  测试样本数: {total_count}")
    print(f"  识别正确数: {correct_count}")
    print(f"  识别准确率: {accuracy:.2f}%")
    print("="*60)
    return accuracy

def main():
    """
    执行对比试验
    """
    # 1. 加载模板库
    print("[步骤 1/2] 加载模板库 (使用 build_templates.py 的结果)...")
    templates = data_utils.load_templates()
    template_count = sum(len(mfcc_list) for mfcc_list in templates.values())
    print(f"成功加载 {template_count} 个模板")
    
    if template_count == 0:
        print("错误: 未找到模板文件！请先运行 build_templates.py")
        return
    
    # 2. 获取测试文件
    print("\n[步骤 2/2] 获取测试文件...")
    all_files = data_utils.get_audio_files()
    _, test_files = data_utils.split_train_test(all_files)
    total_test = sum(len(files) for files in test_files.values())
    print(f"找到 {total_test} 个测试文件")

    # 3. 运行对比测试
    metrics_to_test = ['euclidean', 'cosine']
    results = {}
    
    for metric in metrics_to_test:
        accuracy = run_recognition_test(templates, test_files, metric)
        results[metric] = accuracy
        
    print("\n\n" + "="*60)
    print("对比试验最终总结")
    print("="*60)
    for metric, acc in results.items():
        print(f"  - {metric.upper():10s} 准确率: {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()