"""
可视化模块
绘制DTW对齐图
"""

import matplotlib.pyplot as plt
from dtw import dtw
import os


def plot_dtw_alignment(template_mfcc, input_mfcc, save_path=None, title=None):
    """
    绘制DTW对齐路径图
    
    Args:
        template_mfcc: 模板MFCC特征，形状为 (n_mfcc, n_frames)
        input_mfcc: 输入MFCC特征，形状为 (n_mfcc, n_frames)
        save_path: 保存路径（可选）
        title: 图表标题（可选）
    """
    # 计算DTW，保留内部信息用于绘图
    alignment = dtw(
        template_mfcc.T,
        input_mfcc.T,
        keep_internals=True,
        dist_method='euclidean'
    )
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制对齐路径
    alignment.plot(type="alignment")
    
    # 设置标题
    if title:
        plt.title(title, fontsize=14, pad=20)
    else:
        plt.title("DTW对齐路径", fontsize=14, pad=20)
    
    plt.xlabel("模板帧索引", fontsize=12)
    plt.ylabel("输入帧索引", fontsize=12)
    
    # 保存图形
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对齐图已保存到: {save_path}")
    
    plt.close()


def plot_dtw_cost_matrix(template_mfcc, input_mfcc, save_path=None, title=None):
    """
    绘制DTW代价矩阵
    
    Args:
        template_mfcc: 模板MFCC特征，形状为 (n_mfcc, n_frames)
        input_mfcc: 输入MFCC特征，形状为 (n_mfcc, n_frames)
        save_path: 保存路径（可选）
        title: 图表标题（可选）
    """
    # 计算DTW，保留内部信息用于绘图
    alignment = dtw(
        template_mfcc.T,
        input_mfcc.T,
        keep_internals=True,
        dist_method='euclidean'
    )
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制代价矩阵
    alignment.plot(type="twoway")
    
    # 设置标题
    if title:
        plt.suptitle(title, fontsize=14)
    else:
        plt.suptitle("DTW代价矩阵和对齐路径", fontsize=14)
    
    # 保存图形
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"代价矩阵图已保存到: {save_path}")
    
    plt.close()

