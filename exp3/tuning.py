"""
exp3/run_tuning.py
参数调优实验：网格搜索最佳的 MFCC 窗口和帧移参数
"""
import time
from src import config, data_utils, features, dtw_core

# 定义要搜索的参数网格
# 格式: (win_length, hop_length)
# 16000Hz下: 
# 25ms/10ms = 400/160 (ASR标准)
# 32ms/16ms = 512/256 (你的旧参数)
# 15ms/5ms  = 240/80  (高分辨率)
PARAM_GRID = [
    (512, 256, "低分辨率 (32ms/16ms)"),
    (400, 160, "标准ASR (25ms/10ms)"),
    (240, 80,  "高分辨率 (15ms/5ms)")
]

def run_test_with_params(win, hop):
    # 1. 动态修改配置
    config.MFCC_PARAMS['win_length'] = win
    config.MFCC_PARAMS['hop_length'] = hop
    # 确保 FFT 足够大
    config.MFCC_PARAMS['n_fft'] = max(512, int(2**np.ceil(np.log2(win))))
    
    # 2. 重新加载数据 (因为 split 逻辑固定，这里只拿路径)
    all_files = data_utils.get_audio_files()
    train_files, test_files = data_utils.split_train_test(all_files)
    total_test = sum(len(f) for f in test_files.values())

    # 3. 构建模板
    templates = {}
    for label in config.LABELS:
        templates[label] = []
        for path in train_files[label]:
            try:
                templates[label].append(features.extract_mfcc(path))
            except: pass
            
    # 4. 测试识别
    correct = 0
    start = time.time()
    for true_label in config.LABELS:
        for path in test_files[true_label]:
            try:
                test_feat = features.extract_mfcc(path)
                min_dist = float('inf')
                pred = None
                for t_label, t_list in templates.items():
                    for t_feat in t_list:
                        d = dtw_core.calculate_dtw_distance(t_feat, test_feat)
                        if d < min_dist:
                            min_dist = d
                            pred = t_label
                if pred == true_label: correct += 1
            except: pass
            
    acc = correct / total_test * 100
    avg_time = (time.time() - start) / total_test * 1000
    return acc, avg_time

import numpy as np # 补充导入

def main():
    print(f"{'Win/Hop (samples)':<20} | {'描述':<20} | {'准确率':<8} | {'耗时(ms/个)'}")
    print("-" * 75)
    
    for win, hop, desc in PARAM_GRID:
        acc, time_ms = run_test_with_params(win, hop)
        param_str = f"{win}/{hop}"
        print(f"{param_str:<20} | {desc:<20} | {acc:6.2f}% | {time_ms:6.1f}")

if __name__ == "__main__":
    main()