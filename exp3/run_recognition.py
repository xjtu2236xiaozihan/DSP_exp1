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
    for digit in config.DIGITS:
        print(f"  数字 {digit}: {len(templates[digit])} 个模板")
    
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
    confusion_matrix = {true_digit: {pred_digit: 0 for pred_digit in config.DIGITS} 
                       for true_digit in config.DIGITS}
    
    # 创建可视化结果目录
    vis_dir = "results/visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 遍历每个数字的测试文件
    for true_digit in config.DIGITS:
        for test_file_path in test_files[true_digit]:
            total_count += 1
            filename = os.path.basename(test_file_path)
            
            # 提取测试音频的MFCC特征
            test_mfcc = features.extract_mfcc(test_file_path)
            
            # 初始化最小距离和预测结果
            min_distance = float('inf')
            predicted_digit = None
            best_template_idx = None
            
            # 与所有模板计算DTW距离
            for template_digit in config.DIGITS:
                for idx, template_mfcc in enumerate(templates[template_digit]):
                    # 计算DTW距离
                    distance = dtw_core.calculate_dtw_distance(template_mfcc, test_mfcc)
                    
                    # 更新最小距离
                    if distance < min_distance:
                        min_distance = distance
                        predicted_digit = template_digit
                        best_template_idx = idx
            
            # 统计结果
            is_correct = (predicted_digit == true_digit)
            if is_correct:
                correct_count += 1
            
            # 更新混淆矩阵
            confusion_matrix[true_digit][predicted_digit] += 1
            
            # 打印识别结果
            status = "✓" if is_correct else "✗"
            print(f"{status} [{total_count:2d}] 文件: {filename:25s} | "
                  f"真实: {true_digit} | 预测: {predicted_digit} | "
                  f"距离: {min_distance:.2f}")
            
            # 可选：为第一个测试样本生成可视化（可根据需要调整）
            if total_count <= 3:  # 只为前3个样本生成可视化
                best_template = templates[predicted_digit][best_template_idx]
                vis_path = os.path.join(vis_dir, f"alignment_{filename.replace('.wav', '.png')}")
                visualization.plot_dtw_alignment(
                    best_template, 
                    test_mfcc, 
                    save_path=vis_path,
                    title=f"DTW对齐: {filename} (真实:{true_digit}, 预测:{predicted_digit})"
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
    print(f"\n各数字识别准确率:")
    for digit in config.DIGITS:
        digit_total = len(test_files[digit])
        digit_correct = confusion_matrix[digit][digit]
        digit_accuracy = (digit_correct / digit_total * 100) if digit_total > 0 else 0
        print(f"  数字 {digit}: {digit_correct}/{digit_total} ({digit_accuracy:.2f}%)")
    
    # 打印混淆矩阵
    print(f"\n混淆矩阵 (行:真实, 列:预测):")
    print("     ", end="")
    for digit in config.DIGITS:
        print(f"{digit:4s}", end="")
    print()
    
    for true_digit in config.DIGITS:
        print(f"  {true_digit} |", end="")
        for pred_digit in config.DIGITS:
            count = confusion_matrix[true_digit][pred_digit]
            print(f"{count:4d}", end="")
        print()
    
    print("\n" + "="*60)
    print(f"✓ 识别完成！最终准确率: {accuracy:.2f}%")
    if total_count <= 3:
        print(f"✓ 对齐可视化图已保存到: {vis_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()

