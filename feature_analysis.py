#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征分析可视化程序
功能：分析特征CSV文件，生成可视化图表

改动：
- 通过 --window 选择 rect/hamming/hanning
- 默认从 dataset/features_<window>/audio_features_<window>.csv 读取
- 所有图片输出到 /home/wdai/dw/DSP_exp1/plots/<window>/ 目录
- 图片文件名仍追加窗口后缀
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def _normalize_window_name(name: str) -> str:
    if name is None:
        return "hamming"
    name = name.strip().lower()
    if name in ("rect", "rectangular", "box", "boxcar", "矩形窗"):
        return "rect"
    if name in ("hamming", "hamm", "ham", "汉明窗"):
        return "hamming"
    if name in ("hann", "han", "hanning", "hannning", "汉宁窗", "海明窗"):
        return "hanning"
    return "hamming"


class FeatureAnalyzer:
    def __init__(self, csv_file: str, plots_dir: str, window_name: str, bins: int = 30, dpi: int = 300):
        self.csv_file = csv_file
        self.plots_dir = plots_dir
        self.window_name = window_name
        self.bins = bins
        self.dpi = dpi

        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(
                f"找不到特征CSV：{self.csv_file}\n"
                f"请确认你已使用相同窗口({self.window_name})运行过特征提取脚本，"
                f"默认输出应为 dataset/features_{self.window_name}/audio_features_{self.window_name}.csv"
            )

        os.makedirs(self.plots_dir, exist_ok=True)

        self.df = pd.read_csv(self.csv_file)
        print(f"[加载] {len(self.df)} 条记录 <- {self.csv_file}")

        self.feature_columns = [
            'energy_mean', 'magnitude_mean', 'zcr_mean', 'rms_mean',
            'peak_factor_mean', 'spectral_centroid_mean', 'spectral_bandwidth_mean'
        ]
        self.feature_names = [
            '短时能量', '短时平均幅度', '短时过零率', '均方根',
            '峰值因子', '频谱质心', '频谱带宽'
        ]

    def _savefig(self, basename: str):
        path = os.path.join(self.plots_dir, f"{basename}_{self.window_name}.png")
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        print(f"[保存] {path}")

    def plot_feature_distribution(self):
        n = len(self.feature_columns)
        rows, cols = 3, 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()

        for i, (feature, name) in enumerate(zip(self.feature_columns, self.feature_names)):
            ax = axes[i]
            data = self.df[feature].dropna()
            ax.hist(data, bins=self.bins, alpha=0.75, edgecolor='black')
            ax.set_title(f'{name}分布（{self.window_name}）')
            ax.set_xlabel('特征值')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        self._savefig('feature_distribution')
        plt.close(fig)

    def plot_correlation_matrix(self):
        corr_matrix = self.df[self.feature_columns].corr()
        fig = plt.figure(figsize=(10, 8))
        if _HAS_SEABORN:
            sns.heatmap(
                corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=self.feature_names, yticklabels=self.feature_names
            )
        else:
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=45, ha='right')
            plt.yticks(range(len(self.feature_names)), self.feature_names)

        plt.title(f'特征相关性矩阵（{self.window_name}）')
        plt.tight_layout()
        self._savefig('feature_correlation')
        plt.close(fig)

    def plot_feature_statistics(self):
        stats_data = []
        for feature in self.feature_columns:
            mean_val = self.df[feature].mean()
            std_val = self.df[feature].std()
            stats_data.append([mean_val, std_val])

        means = [d[0] for d in stats_data]
        stds = [d[1] for d in stats_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.bar(self.feature_names, means, alpha=0.8)
        ax1.set_title(f'特征均值（{self.window_name}）')
        ax1.set_ylabel('特征值')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        ax2.bar(self.feature_names, stds, alpha=0.8)
        ax2.set_title(f'特征标准差（{self.window_name}）')
        ax2.set_ylabel('标准差')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._savefig('feature_statistics')
        plt.close(fig)

    def generate_summary_report(self):
        print("\n=== 特征分析报告 ===")
        print(f"窗口类型: {self.window_name}")
        print(f"总文件数: {len(self.df)}")

        print("\n特征统计：")
        for feature, name in zip(self.feature_columns, self.feature_names):
            s = self.df[feature].dropna()
            print(f"\n{name}:")
            print(f"  均值: {s.mean():.4f}")
            print(f"  标准差: {s.std():.4f}")
            print(f"  最小值: {s.min():.4f}")
            print(f"  最大值: {s.max():.4f}")

        print("\n特征重要性（基于方差，降序）：")
        variances = {f: self.df[f].var() for f in self.feature_columns}
        sorted_items = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        for f, v in sorted_items:
            name = self.feature_names[self.feature_columns.index(f)]
            print(f"  {name}: {v:.4f}")


def parse_args():
    cur = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="特征分析可视化（窗口类型决定默认CSV输入路径；图片统一输出到 /home/wdai/dw/DSP_exp1/plots/<window>/）"
    )
    parser.add_argument("--window", type=str, default="hamming",
                        choices=["rect", "hamming", "hanning"],
                        help="rect(矩形窗) / hamming(汉明窗) / hanning(海明/汉宁窗)")
    parser.add_argument("--csv_file", type=str, default=None,
                        help=("自定义CSV路径。若不提供，默认读取 "
                              "dataset/features_<window>/audio_features_<window>.csv"))
    parser.add_argument("--base_dir", type=str, default=os.path.join(cur, "dataset"),
                        help="数据根目录（默认 ./dataset）")
    parser.add_argument("--bins", type=int, default=30, help="直方图柱数，默认 30")
    parser.add_argument("--dpi", type=int, default=300, help="图片DPI，默认 300")
    parser.add_argument("--plots_root", type=str, default="/home/wdai/dw/DSP_exp1/plots",
                        help="图片根目录（会在其下创建 <window> 子目录）")
    return parser.parse_args()


def main():
    args = parse_args()
    win = _normalize_window_name(args.window)

    # CSV 默认路径：dataset/features_<window>/audio_features_<window>.csv
    if args.csv_file is None:
        csv_file = os.path.join(args.base_dir, f"features_{win}", f"audio_features_{win}.csv")
    else:
        csv_file = args.csv_file

    # 图片输出固定：/home/wdai/dw/DSP_exp1/plots/<window>/
    plots_dir = os.path.join(args.plots_root, win)

    print(f"[参数] window={win}")
    print(f"[读取CSV] {csv_file}")
    print(f"[输出目录] {plots_dir}")

    analyzer = FeatureAnalyzer(
        csv_file=csv_file,
        plots_dir=plots_dir,
        window_name=win,
        bins=args.bins,
        dpi=args.dpi
    )

    print("1. 生成特征分布图...")
    analyzer.plot_feature_distribution()

    print("2. 生成特征相关性矩阵...")
    analyzer.plot_correlation_matrix()

    print("3. 生成特征统计图...")
    analyzer.plot_feature_statistics()

    print("4. 生成分析报告...")
    analyzer.generate_summary_report()

    print("\n分析完成！图片已保存到指定目录。")


if __name__ == "__main__":
    main()
