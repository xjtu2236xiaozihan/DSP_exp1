import os
import sys
import shutil
import numpy as np
import subprocess
import PyInstaller.__main__ 

# 导入项目模块
from src import config
from src import data_utils
from src import features

def install_dependencies_force():
    """
    [Robustness Fix] 
    Force install critical dependencies before building.
    This ensures they exist in the GitHub Actions Windows environment
    even if requirements.txt is missing or incomplete.
    """
    print("\n" + "="*50)
    print("PRE-STEP: Checking & Installing Dependencies...")
    print("="*50)
    
    # 关键依赖列表 (包含 Flask 全家桶 和 科学计算库)
    pkgs = [
        'flask', 'werkzeug', 'jinja2', 'click', 'itsdangerous', 'markupsafe',
        'numpy', 'scipy', 'scikit-learn', 'librosa', 'soundfile', 'dtw-python'
    ]
    
    try:
        # 使用当前 python 解释器强制安装
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + pkgs)
        print("Dependencies check passed.")
    except Exception as e:
        print(f"Warning: Dependency install failed: {e}")
        print("Trying to proceed anyway...")

def main():
    # 1. 强制安装依赖 (防止环境缺失)
    install_dependencies_force()

    print("\n" + "="*50)
    print("STEP 1: Building Full Template Library...")
    print("="*50)
    
    # 获取绝对路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Working Directory: {CURRENT_DIR}")

    # 清理并重建模板
    if os.path.exists(config.TEMPLATE_DIR):
        shutil.rmtree(config.TEMPLATE_DIR)
    os.makedirs(config.TEMPLATE_DIR)
    
    all_files = data_utils.get_audio_files()
    count = 0
    
    # 生成特征
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
                print(f"Skipping: {e}")
                
    print(f"Templates built: {count}")

    # ---------------------------------------------------------
    # 2. PyInstaller 打包 (加入隐式导入)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 2: PyInstaller Build (With Hidden Imports)...")
    print("="*50)

    sep = os.pathsep 
    src_web_tpl = os.path.join(CURRENT_DIR, 'web_templates')
    src_static = os.path.join(CURRENT_DIR, 'static')
    src_templates = config.TEMPLATE_DIR
    script_path = os.path.join(CURRENT_DIR, 'desktop_main.py')
    
    # 构建参数
    args = [
        script_path,
        '--name=DTW_Speech_System',
        '--onefile',
        '--windowed',
        '--noconfirm',
        '--clean',
        
        # [Robustness Fix] Flask 依赖
        '--hidden-import=flask',
        '--hidden-import=werkzeug',
        '--hidden-import=jinja2',
        
        # [New] Librosa/Scipy/Sklearn 深度依赖补全
        '--hidden-import=sklearn.tree',
        '--hidden-import=sklearn.neighbors.typedefs',
        '--hidden-import=sklearn.neighbors.quad_tree',
        '--hidden-import=sklearn.tree._utils',
        '--hidden-import=scipy.signal',             
        '--hidden-import=scipy.signal.bsplines',    
        '--hidden-import=scipy.special',
        '--hidden-import=scipy.special.cython_special',
        '--hidden-import=scipy.spatial.transform._rotation_groups',
        '--hidden-import=scipy.integrate.lsoda',  # 常见缺失
        '--hidden-import=librosa',
        
        # 数据映射
        f'--add-data={src_web_tpl}{sep}exp3/web_templates',
        f'--add-data={src_static}{sep}exp3/static',
        f'--add-data={src_templates}{sep}templates',
    ]
    
    print("Running PyInstaller...")
    PyInstaller.__main__.run(args)
    print("\nBuild Success!")

if __name__ == "__main__":
    main()