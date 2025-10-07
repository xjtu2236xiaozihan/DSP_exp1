#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征分析可视化程序
功能：分析特征CSV文件，生成可视化图表，对比原始/加噪特征差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class FeatureAnalyzer:
    """特征分析器"""
    
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # 由于现在只有processed音频，不需要区分clean/noise
        print(f"加载了 {len(self.df)} 个音频文件的特征数据")
    
    def plot_feature_distribution(self):
        """绘制特征分布图"""
        # 更新为简化后的特征
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['短时能量', '短时平均幅度', '短时过零率', '均方根', 
                        '峰值因子', '频谱质心', '频谱带宽']
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (feature, name) in enumerate(zip(feature_columns, feature_names)):
            ax = axes[i]
            
            # 绘制直方图
            ax.hist(self.df[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{name}分布')
            ax.set_xlabel('特征值')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    
    def plot_correlation_matrix(self):
        """绘制特征相关性矩阵"""
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['短时能量', '短时平均幅度', '短时过零率', '均方根', 
                        '峰值因子', '频谱质心', '频谱带宽']
        
        # 计算相关性矩阵
        corr_matrix = self.df[feature_columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   xticklabels=feature_names, yticklabels=feature_names)
        plt.title('特征相关性矩阵')
        plt.tight_layout()
        plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_statistics(self):
        """绘制特征统计图"""
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['短时能量', '短时平均幅度', '短时过零率', '均方根', 
                        '峰值因子', '频谱质心', '频谱带宽']
        
        # 计算特征统计
        stats_data = []
        for feature in feature_columns:
            mean_val = self.df[feature].mean()
            std_val = self.df[feature].std()
            stats_data.append([mean_val, std_val])
        
        # 绘制特征统计柱状图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 均值统计
        means = [data[0] for data in stats_data]
        ax1.bar(feature_names, means, color='skyblue', alpha=0.7)
        ax1.set_title('特征均值')
        ax1.set_ylabel('特征值')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 标准差统计
        stds = [data[1] for data in stats_data]
        ax2.bar(feature_names, stds, color='lightcoral', alpha=0.7)
        ax2.set_title('特征标准差')
        ax2.set_ylabel('标准差')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成特征分析报告"""
        print("=== 特征分析报告 ===")

        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['短时能量', '短时平均幅度', '短时过零率', '均方根', 
                        '峰值因子', '频谱质心', '频谱带宽']
        
        print(f"总文件数: {len(self.df)}")
        
        print("\n特征统计:")
        for feature, name in zip(feature_columns, feature_names):
            mean_val = self.df[feature].mean()
            std_val = self.df[feature].std()
            min_val = self.df[feature].min()
            max_val = self.df[feature].max()
            
            print(f"\n{name}:")
            print(f"  均值: {mean_val:.4f}")
            print(f"  标准差: {std_val:.4f}")
            print(f"  最小值: {min_val:.4f}")
            print(f"  最大值: {max_val:.4f}")
        
        # 计算特征重要性（基于方差）
        print("\n特征重要性分析:")
        importance_scores = {}
        for feature in feature_columns:
            # 基于方差的重要性
            variance = self.df[feature].var()
            importance_scores[feature] = variance
        
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_features:
            feature_name = feature_names[feature_columns.index(feature)]
            print(f"  {feature_name}: {score:.4f}")

def main():
    """主函数"""
    csv_file = os.path.join("dataset", "features", "audio_features.csv")
    
    # 创建分析器
    analyzer = FeatureAnalyzer(csv_file)
    print("开始特征分析...")
    
    # 生成各种分析图表
    print("1. 生成特征分布图...")
    analyzer.plot_feature_distribution()
    
    print("2. 生成特征相关性矩阵...")
    analyzer.plot_correlation_matrix()
    
    print("3. 生成特征统计图...")
    analyzer.plot_feature_statistics()
    
    print("4. 生成分析报告...")
    analyzer.generate_summary_report()
    
    print("\n分析完成！图表已保存到当前目录。")

if __name__ == "__main__":
    main()
