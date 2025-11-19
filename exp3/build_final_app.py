import os
import sys
import shutil
import numpy as np
import PyInstaller.__main__ 

# 导入项目模块
from src import config
from src import data_utils
from src import features

def main():
    # [Fix 1] Use English logs to avoid UnicodeEncodeError on Windows Consoles
    print("="*50)
    print("STEP 1: Building Full Template Library (Release Version)...")
    print("="*50)
    
    # [Fix 2] Force Absolute Paths
    # 获取当前脚本所在的绝对路径 (.../DSP_exp1/exp3)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Working Directory (Absolute): {CURRENT_DIR}")

    # 1. 清理并重新构建模板
    # config.TEMPLATE_DIR 已经在 config.py 中使用了 os.path.abspath，所以它已经是绝对路径
    if os.path.exists(config.TEMPLATE_DIR):
        shutil.rmtree(config.TEMPLATE_DIR)
    os.makedirs(config.TEMPLATE_DIR)
    
    print(f"Target Template Directory: {config.TEMPLATE_DIR}")

    all_files = data_utils.get_audio_files()
    count = 0
    
    # 遍历所有文件生成特征
    for label, file_list in all_files.items():
        label_dir = os.path.join(config.TEMPLATE_DIR, label)
        os.makedirs(label_dir, exist_ok=True)
        
        for file_path in file_list:
            try:
                # 提取特征
                mfcc = features.extract_mfcc(file_path)
                # 保存
                filename = os.path.basename(file_path)
                save_name = filename.replace('.wav', '.npy')
                np.save(os.path.join(label_dir, save_name), mfcc)
                count += 1
            except Exception as e:
                # Use English error log
                print(f"Skipping file {filename}: {e}")
                
    print(f"Templates built successfully. Total count: {count}")

    # ---------------------------------------------------------
    # 2. PyInstaller 打包流程
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 2: Starting PyInstaller Build...")
    print("="*50)

    # 获取系统路径分隔符 (Windows是 ';', Mac/Linux是 ':')
    sep = os.pathsep 
    
    # [Fix 2] 定义绝对路径资源
    # 确保源文件路径都是绝对路径，防止相对路径在 CI 环境中跑偏
    src_web_tpl = os.path.join(CURRENT_DIR, 'web_templates')
    src_static = os.path.join(CURRENT_DIR, 'static')
    src_templates = config.TEMPLATE_DIR  # 这个本身就是绝对路径
    
    # 主程序入口的绝对路径
    script_path = os.path.join(CURRENT_DIR, 'desktop_main.py')
    
    print(f"Main Script: {script_path}")
    print(f"Resource Map 1: {src_web_tpl} -> exp3/web_templates")
    print(f"Resource Map 2: {src_static} -> exp3/static")
    print(f"Resource Map 3: {src_templates} -> templates")

    # 构建参数列表
    args = [
        script_path,
        '--name=DTW_Speech_System',     # 生成的 exe 名字
        '--onefile',                    # 单文件模式
        '--windowed',                   # 无控制台黑框
        '--noconfirm',                  # 覆盖输出目录不询问
        '--clean',                      # 清理缓存
        
        # 数据映射: "绝对源路径{分隔符}相对目标路径"
        f'--add-data={src_web_tpl}{sep}exp3/web_templates',
        f'--add-data={src_static}{sep}exp3/static',
        f'--add-data={src_templates}{sep}templates',
        
        # 强制隐藏导入，防止缺失模块
        '--hidden-import=sklearn.utils._typedefs',
        '--hidden-import=sklearn.neighbors._partition_nodes',
    ]
    
    # 执行打包
    print("Running PyInstaller...")
    PyInstaller.__main__.run(args)
    
    print("\nBuild Finished! Check the 'dist' folder in GitHub Artifacts.")

if __name__ == "__main__":
    main()