"""
识别运行脚本（测试阶段）
使用DTW算法对测试音频进行识别
"""

import os
import numpy as np
from src import config
from src import data_utils
from src import features
from src import dtw_core
from src import visualization


def main():
    """
    执行识别流程
    """
    print("="*60)
    print("DTW语音识别系统 - 语音识别")
    print("="*60)
    
    # 1. 加载模板库
    print("\n[步骤 1/4] 加载模板库...")
    templates = data_utils.load_templates()
    
    template_count = sum(len(mfcc_list) for mfcc_list in templates.values())
    print(f"成功加载 {template_count} 个模板")
    for label in config.LABELS:
        print(f"  标签 {label}: {len(templates[label])} 个模板")
    
    if template_count == 0:
        print("\n错误: 未找到模板文件！")
        print("请先运行 build_templates.py 构建模板库")
        return
    
    # 2. 获取测试文件
    print("\n[步骤 2/4] 获取测试文件...")
    all_files = data_utils.get_audio_files()
    train_files, test_files = data_utils.split_train_test(all_files)
    
    total_test = sum(len(files) for files in test_files.values())
    print(f"找到 {total_test} 个测试文件")
    
    # 3. 执行识别
    print("\n[步骤 3/4] 开始识别...")
    print("-"*60)
    
    correct_count = 0
    total_count = 0
    
    # 用于保存混淆矩阵数据
    confusion_matrix = {true_label: {pred_label: 0 for pred_label in config.LABELS} 
                        for true_label in config.LABELS}
    
    # 创建可视化结果目录
    vis_dir = "results/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 遍历每个数字的测试文件
    for true_label in config.LABELS:
        for test_file_path in test_files[true_label]:
            total_count += 1
            filename = os.path.basename(test_file_path)
            
            # 提取测试音频的MFCC特征
            test_mfcc = features.extract_mfcc(test_file_path)
            
            # 初始化最小距离和预测结果
            min_distance = float('inf')
            predicted_label = None
            best_template_idx = None
            
            # 与所有模板计算DTW距离
            for template_label in config.LABELS:
                for idx, template_mfcc in enumerate(templates[template_label]):
                    # 计算DTW距离
                    distance = dtw_core.calculate_dtw_distance(template_mfcc, test_mfcc)
                    
                    # 更新最小距离
                    if distance < min_distance:
                        min_distance = distance
                        predicted_label = template_label
                        best_template_idx = idx
            
            # 统计结果
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_count += 1
            
            # 更新混淆矩阵
            confusion_matrix[true_label][predicted_label] += 1
            
            # 打印识别结果
            status = "✓" if is_correct else "✗"
            print(f"{status} [{total_count:2d}] 文件: {filename:25s} | "
                  f"真实: {true_label} | 预测: {predicted_label} | "
                  f"距离: {min_distance:.2f}")
            
            # 可选：为第一个测试样本生成可视化（可根据需要调整）
            if total_count <= 3:  # 只为前3个样本生成可视化
                best_template = templates[predicted_label][best_template_idx]
                vis_path = os.path.join(vis_dir, f"alignment_{filename.replace('.wav', '.png')}")
                visualization.plot_dtw_alignment(
                    best_template, 
                    test_mfcc, 
                    save_path=vis_path,
                    title=f"DTW对齐: {filename} (真实:{true_label}, 预测:{predicted_label})"
                )
    
    # 4. 打印统计报告
    print("-"*60)
    print("\n[步骤 4/4] 识别结果统计")
    print("="*60)
    
    # 计算准确率
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n总体识别结果:")
    print(f"  测试样本数: {total_count}")
    print(f"  识别正确数: {correct_count}")
    print(f"  识别错误数: {total_count - correct_count}")
    print(f"  识别准确率: {accuracy:.2f}%")
    
    # 打印各数字的识别准确率
    print(f"\n各标签识别准确率:")
    for label in config.LABELS:
        label_total = len(test_files[label])
        label_correct = confusion_matrix[label][label]
        label_accuracy = (label_correct / label_total * 100) if label_total > 0 else 0
        print(f"  标签 {label}: {label_correct}/{label_total} ({label_accuracy:.2f}%)")
    
    # 打印混淆矩阵
    print(f"\n混淆矩阵 (行:真实, 列:预测):")
    col_width = max(4, max(len(label) for label in config.LABELS))
    row_width = col_width
    print(" " * (row_width + 3), end="")
    for label in config.LABELS:
        print(f"{label:>{col_width}s}", end="")
    print()
    
    for true_label in config.LABELS:
        print(f"{true_label:>{row_width}s} |", end="")
        for pred_label in config.LABELS:
            count = confusion_matrix[true_label][pred_label]
            print(f"{count:>{col_width}d}", end="")
        print()
    
    print("\n" + "="*60)
    print(f"✓ 识别完成！最终准确率: {accuracy:.2f}%")
    if total_count <= 3:
        print(f"✓ 对齐可视化图已保存到: {vis_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()

