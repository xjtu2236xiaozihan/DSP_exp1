# exp3/build_final_app.py

import os
import shutil
import numpy as np
from src import config
from src import data_utils
from src import features

def main():
    print("="*50)
    print("构建发布版全量模板库 (基于 98.26% 最优配置)")
    print("="*50)
    
    # 1. 强制清理并重建模板目录
    # 注意：PyInstaller 打包时需要这个目录存在且包含数据
    if os.path.exists(config.TEMPLATE_DIR):
        shutil.rmtree(config.TEMPLATE_DIR)
    os.makedirs(config.TEMPLATE_DIR)
    
    # 2. 获取所有音频文件
    all_files = data_utils.get_audio_files()
    total_count = 0
    
    print("正在处理所有音频文件...")
    
    for label, file_list in all_files.items():
        label_dir = os.path.join(config.TEMPLATE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        for file_path in file_list:
            try:
                # 提取特征 (此时已去除了预加重)
                mfcc = features.extract_mfcc(file_path)
                
                # 保存为 .npy
                filename = os.path.basename(file_path)
                save_name = filename.replace('.wav', '.npy')
                np.save(os.path.join(label_dir, save_name), mfcc)
                total_count += 1
            except Exception as e:
                print(f"  [跳过] {filename}: {e}")
                
    print(f"\n✓ 成功构建 {total_count} 个模板！")
    print(f"✓ 模板库位置: {config.TEMPLATE_DIR}")
    print("现在可以运行 PyInstaller 进行打包了。")

if __name__ == "__main__":
    main()