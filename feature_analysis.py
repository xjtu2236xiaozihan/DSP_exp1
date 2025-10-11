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
        # 分离不同噪声类型的数据
        self.clean_df = self.df[self.df['noise_type'] == 'clean']
        self.noise1_df = self.df[self.df['noise_type'] == 'noise1']
        self.noise2_df = self.df[self.df['noise_type'] == 'noise2']
        self.noise3_df = self.df[self.df['noise_type'] == 'noise3']
        
        print(f"加载了 {len(self.df)} 个音频文件的特征数据")
        print(f"  Clean: {len(self.clean_df)}")
        print(f"  Noise1: {len(self.noise1_df)}")
        print(f"  Noise2: {len(self.noise2_df)}")
        print(f"  Noise3: {len(self.noise3_df)}")
    
    def plot_feature_distribution(self):
        """绘制特征分布图（对比不同噪声类型）"""
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['短时能量', '短时平均幅度', '短时过零率', '均方根', 
                        '峰值因子', '频谱质心', '频谱带宽']
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()
        
        colors = {'clean': 'blue', 'noise1': 'red', 'noise2': 'green', 'noise3': 'orange'}
        labels = {'clean': 'Clean', 'noise1': 'Noise1', 'noise2': 'Noise2', 'noise3': 'Noise3'}
        
        for i, (feature, name) in enumerate(zip(feature_columns, feature_names)):
            ax = axes[i]
            
            # 绘制不同类型的直方图
            for noise_type, df in [('clean', self.clean_df), ('noise1', self.noise1_df), 
                                   ('noise2', self.noise2_df), ('noise3', self.noise3_df)]:
                if not df.empty:
                    ax.hist(df[feature].dropna(), bins=20, alpha=0.5, 
                           color=colors[noise_type], label=labels[noise_type])
            
            ax.set_title(f'{name}分布对比')
            ax.set_xlabel('特征值')
            ax.set_ylabel('频次')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    
    def plot_correlation_matrix(self):
        """绘制特征相关性矩阵（对比不同噪声类型）"""
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['能量', '幅度', '过零率', '均方根', '峰值', '质心', '带宽']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        datasets = [('Clean', self.clean_df), ('Noise1', self.noise1_df), 
                   ('Noise2', self.noise2_df), ('Noise3', self.noise3_df)]
        
        for ax, (title, df) in zip(axes, datasets):
            if not df.empty:
                corr_matrix = df[feature_columns].corr()
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           xticklabels=feature_names, yticklabels=feature_names, ax=ax,
                           vmin=-1, vmax=1)
                ax.set_title(f'{title} 特征相关性')
        
        plt.tight_layout()
        plt.savefig('feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_statistics(self):
        """绘制特征统计图（对比不同噪声类型）"""
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['能量', '幅度', '过零率', '均方根', '峰值', '质心', '带宽']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(feature_names))
        width = 0.2
        
        datasets = [('Clean', self.clean_df, -1.5), ('Noise1', self.noise1_df, -0.5),
                   ('Noise2', self.noise2_df, 0.5), ('Noise3', self.noise3_df, 1.5)]
        colors = ['blue', 'red', 'green', 'orange']
        
        # 均值对比
        for i, (label, df, offset) in enumerate(datasets):
            if not df.empty:
                means = [df[feature].mean() for feature in feature_columns]
                ax1.bar(x + offset * width, means, width, label=label, color=colors[i], alpha=0.7)
        
        ax1.set_title('特征均值对比')
        ax1.set_ylabel('特征值')
        ax1.set_xticks(x)
        ax1.set_xticklabels(feature_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 标准差对比
        for i, (label, df, offset) in enumerate(datasets):
            if not df.empty:
                stds = [df[feature].std() for feature in feature_columns]
                ax2.bar(x + offset * width, stds, width, label=label, color=colors[i], alpha=0.7)
        
        ax2.set_title('特征标准差对比')
        ax2.set_ylabel('标准差')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('feature_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """生成特征分析报告（对比不同噪声类型）"""
        print("=== 特征分析报告 ===")
        
        feature_columns = ['energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean', 
                          'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean']
        feature_names = ['短时能量', '短时平均幅度', '短时过零率', '均方根', 
                        '峰值因子', '频谱质心', '频谱带宽']
        
        print(f"\n总文件数: {len(self.df)}")
        print(f"  Clean: {len(self.clean_df)}")
        print(f"  Noise1: {len(self.noise1_df)}")
        print(f"  Noise2: {len(self.noise2_df)}")
        print(f"  Noise3: {len(self.noise3_df)}")
        
        print("\n特征统计对比:")
        for feature, name in zip(feature_columns, feature_names):
            print(f"\n{name}:")
            
            for noise_type, df in [('Clean', self.clean_df), ('Noise1', self.noise1_df),
                                  ('Noise2', self.noise2_df), ('Noise3', self.noise3_df)]:
                if not df.empty:
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    print(f"  {noise_type}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
        
        # 噪声影响分析
        print("\n噪声影响分析（相对于Clean）:")
        for feature, name in zip(feature_columns, feature_names):
            print(f"\n{name}:")
            clean_mean = self.clean_df[feature].mean()
            
            for noise_type, df in [('Noise1', self.noise1_df), ('Noise2', self.noise2_df),
                                  ('Noise3', self.noise3_df)]:
                if not df.empty and clean_mean != 0:
                    noise_mean = df[feature].mean()
                    change_percent = ((noise_mean - clean_mean) / clean_mean) * 100
                    print(f"  {noise_type}: {change_percent:+.2f}%")

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
