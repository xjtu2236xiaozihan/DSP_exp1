"""
全方位消融实验脚本 (Ablation Study)
用于验证不同信号处理组件对 DTW 语音识别系统的贡献
"""

import os
import numpy as np
import librosa
from src import config
from src import data_utils
from src import dtw_core 
from src import features # 导入基线特征提取

# ==========================================
# 1. 定义不同配置的特征提取函数 (覆盖 src.features)
# ==========================================

def extract_baseline(file_path):
    """基线：使用 src/features.py 中现有的最优逻辑"""
    return features.extract_mfcc(file_path)

def extract_no_trim(file_path):
    """消融：移除静音消除"""
    y, sr = librosa.load(file_path, sr=None)
    # [移除] y, _ = librosa.effects.trim(y, top_db=30)
    
    if len(y) < 1024: y, sr = librosa.load(file_path, sr=None) # 兜底
    
    y = librosa.effects.preemphasis(y, coef=config.MFCC_PARAMS['preemph_coef'])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.MFCC_PARAMS['n_mfcc'], 
                                n_fft=config.MFCC_PARAMS['n_fft'], hop_length=config.MFCC_PARAMS['hop_length'])
    # 保持 CMVN
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    return mfcc

def extract_no_preemphasis(file_path):
    """消融：移除预加重"""
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=30) # 保持去静音
    
    # [移除] y = librosa.effects.preemphasis(...)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.MFCC_PARAMS['n_mfcc'], 
                                n_fft=config.MFCC_PARAMS['n_fft'], hop_length=config.MFCC_PARAMS['hop_length'])
    # 保持 CMVN
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    return mfcc

def extract_no_cmvn(file_path):
    """消融：移除 CMVN 标准化"""
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y, coef=config.MFCC_PARAMS['preemph_coef'])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.MFCC_PARAMS['n_mfcc'], 
                                n_fft=config.MFCC_PARAMS['n_fft'], hop_length=config.MFCC_PARAMS['hop_length'])
    # [移除] CMVN 步骤
    return mfcc

# ==========================================
# 2. 定义 DTW 距离计算变体
# ==========================================

from dtw import dtw

def dtw_no_normalization(template_mfcc, test_mfcc):
    """消融：移除 DTW 长度归一化 (使用原始欧式距离累积和)"""
    alignment = dtw(template_mfcc.T, test_mfcc.T, keep_internals=False, dist_method='euclidean')
    return alignment.distance # 直接返回累积距离，不除以长度

def dtw_baseline(template_mfcc, test_mfcc):
    """基线：使用 src/dtw_core.py 中的归一化距离"""
    return dtw_core.calculate_dtw_distance(template_mfcc, test_mfcc)

# ==========================================
# 3. 实验控制逻辑
# ==========================================

EXPERIMENTS = [
    {"name": "Baseline (全优化)", "feat_func": extract_baseline,       "dtw_func": dtw_baseline},
    {"name": "无静音消除 (No Trim)",    "feat_func": extract_no_trim,        "dtw_func": dtw_baseline},
    {"name": "无预加重 (No Preemph)", "feat_func": extract_no_preemphasis, "dtw_func": dtw_baseline},
    {"name": "无标准化 (No CMVN)",    "feat_func": extract_no_cmvn,        "dtw_func": dtw_baseline},
    {"name": "无距离归一化 (No Norm)",  "feat_func": extract_baseline,       "dtw_func": dtw_no_normalization},
]

def run_experiment(exp_config, train_files, test_files):
    """运行单个实验配置"""
    name = exp_config['name']
    feat_func = exp_config['feat_func']
    dtw_func = exp_config['dtw_func']
    
    print(f"\n>> 正在运行: [{name}] ...")
    
    # 1. 动态构建模板 (关键：每次实验必须重新提取特征)
    templates = {label: [] for label in config.LABELS}
    for label in config.LABELS:
        for file_path in train_files[label]:
            try:
                mfcc = feat_func(file_path)
                templates[label].append(mfcc)
            except Exception as e:
                print(f"  警告: 训练文件 {os.path.basename(file_path)} 提取失败: {e}")

    # 2. 执行识别
    correct = 0
    total = 0
    
    for true_label in config.LABELS:
        for file_path in test_files[true_label]:
            total += 1
            try:
                test_mfcc = feat_func(file_path)
                
                min_dist = float('inf')
                pred = None
                
                for t_label in config.LABELS:
                    for t_mfcc in templates[t_label]:
                        dist = dtw_func(t_mfcc, test_mfcc)
                        if dist < min_dist:
                            min_dist = dist
                            pred = t_label
                
                if pred == true_label:
                    correct += 1
            except Exception as e:
                print(f"  测试文件出错: {e}")
                
    acc = (correct / total * 100) if total > 0 else 0
    print(f"   -> 准确率: {acc:.2f}% ({correct}/{total})")
    return acc

def main():
    print("="*60)
    print("DTW语音识别 - 多组件消融实验")
    print("="*60)
    
    # 准备数据
    all_files = data_utils.get_audio_files()
    train_files, test_files = data_utils.split_train_test(all_files)
    
    total_train = sum(len(f) for f in train_files.values())
    total_test = sum(len(f) for f in test_files.values())
    print(f"数据集: 训练 {total_train} / 测试 {total_test}")
    
    results = []
    
    # 循环运行所有实验
    for exp in EXPERIMENTS:
        acc = run_experiment(exp, train_files, test_files)
        results.append((exp['name'], acc))
        
    # 打印最终汇总
    print("\n" + "="*60)
    print(f"{'实验名称':<25} | {'准确率':<10}")
    print("-" * 40)
    for name, acc in results:
        print(f"{name:<25} | {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()