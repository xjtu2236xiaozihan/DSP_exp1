# exp3/build_final_app.py

import os
import shutil
import numpy as np
import PyInstaller.__main__ # 导入 PyInstaller 模块
from src import config
from src import data_utils
from src import features

def main():
    print("="*50)
    print("1. 构建发布版全量模板库...")
    print("="*50)
    
    # ---原有逻辑：生成 .npy 模板---
    if os.path.exists(config.TEMPLATE_DIR):
        shutil.rmtree(config.TEMPLATE_DIR)
    os.makedirs(config.TEMPLATE_DIR)
    
    all_files = data_utils.get_audio_files()
    count = 0
    for label, file_list in all_files.items():
        label_dir = os.path.join(config.TEMPLATE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        for file_path in file_list:
            try:
                mfcc = features.extract_mfcc(file_path)
                save_name = os.path.basename(file_path).replace('.wav', '.npy')
                np.save(os.path.join(label_dir, save_name), mfcc)
                count += 1
            except Exception as e:
                print(f"跳过: {e}")
    print(f"模板构建完成：{count} 个文件。")

    # ---新增逻辑：调用 PyInstaller 打包---
    print("\n"+"="*50)
    print("2. 开始 PyInstaller 打包 (Python模式)...")
    print("="*50)

    # 自动获取当前系统的路径分隔符 (; 或 :)
    sep = os.pathsep 
    
    # 定义打包参数 (相当于之前的命令行参数)
    # 格式: '源路径{sep}目标路径'
    args = [
        'exp3/desktop_main.py',                        # 主程序入口
        '--name=DTW_Speech_System',                    # exe名字
        '--onefile',                                   # 单文件
        '--windowed',                                  # 无黑框
        '--noconfirm',                                 # 覆盖输出
        '--clean',                                     # 清理缓存
        
        # 数据资源映射
        f'--add-data=exp3/web_templates{sep}exp3/web_templates',
        f'--add-data=exp3/static{sep}exp3/static',
        f'--add-data=exp3/templates{sep}templates',     # 将生成的模板打包进根目录
        
        # 隐藏导入
        '--hidden-import=sklearn.utils._typedefs',
        '--hidden-import=sklearn.neighbors._partition_nodes',
    ]
    
    print(f"执行参数: {args}")
    
    # 运行打包
    PyInstaller.__main__.run(args)
    
    print("\n打包完成！请检查 dist/ 文件夹。")

if __name__ == "__main__":
    main()