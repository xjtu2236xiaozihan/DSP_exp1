"""
exp3/run_metric_compare.py
对比实验：不同距离度量对 DTW 性能的影响
"""
import time
from dtw import dtw
from src import config, data_utils, features

# 覆盖 dtw_core 的计算函数，允许传入 method
def calc_dtw(t_mfcc, i_mfcc, method):
    # dtw-python 支持: 'euclidean', 'cosine', 'cityblock' (曼哈顿)
    alignment = dtw(t_mfcc.T, i_mfcc.T, keep_internals=False, dist_method=method)
    # 同样进行长度归一化
    return alignment.distance / (t_mfcc.shape[1] + i_mfcc.shape[1])

def main():
    # 准备数据
    all_files = data_utils.get_audio_files()
    train_files, test_files = data_utils.split_train_test(all_files)
    total_test = sum(len(f) for f in test_files.values())

    # 预先提取所有特征 (避免重复提取，控制变量)
    print("正在预提取所有特征...")
    train_feats = {}
    for k, files in train_files.items():
        train_feats[k] = [features.extract_mfcc(f) for f in files]
        
    test_data = [] # list of (true_label, mfcc)
    for k, files in test_files.items():
        for f in files:
            test_data.append((k, features.extract_mfcc(f)))

    metrics = ['euclidean', 'cityblock', 'cosine']
    
    print(f"\n{'距离度量方法':<15} | {'准确率':<8} | {'耗时(s)'}")
    print("-" * 45)

    for metric in metrics:
        start_time = time.time()
        correct = 0
        
        for true_label, test_mfcc in test_data:
            min_dist = float('inf')
            pred = None
            
            for t_label, t_list in train_feats.items():
                for t_mfcc in t_list:
                    d = calc_dtw(t_mfcc, test_mfcc, metric)
                    if d < min_dist:
                        min_dist = d
                        pred = t_label
            
            if pred == true_label:
                correct += 1
        
        acc = correct / total_test * 100
        duration = time.time() - start_time
        print(f"{metric:<15} | {acc:6.2f}% | {duration:6.2f}")

if __name__ == "__main__":
    main()