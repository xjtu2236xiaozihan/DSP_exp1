# 文件路径: exp3/desktop_main.py

import os
import sys
import threading
import webbrowser
import time
from server import app, load_resources  # 导入 app 和加载函数
from src import config                  # 导入配置对象以进行补丁

def start_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:54321/')

if __name__ == "__main__":
    # -------------------------------------------------
    # 路径重定向核心逻辑
    # -------------------------------------------------
    if getattr(sys, 'frozen', False):
        # 【打包模式】
        # PyInstaller 将文件解压到了 sys._MEIPASS 临时目录
        base_dir = sys._MEIPASS
        print(f"[启动模式] 打包运行 (临时目录: {base_dir})")
        
        # 1. 修正 Flask 寻找 html/css 的路径
        #    对应 spec/cmd 中的 add-data 目标路径
        app.template_folder = os.path.join(base_dir, 'exp3', 'web_templates')
        app.static_folder = os.path.join(base_dir, 'exp3', 'static')
        
        # 2. 修正 DTW 寻找 .npy 模板的路径
        #    对应 spec/cmd 中的 add-data 目标路径 "templates" (根目录)
        config.TEMPLATE_DIR = os.path.join(base_dir, 'templates')
        
    else:
        # 【源码模式】
        print("[启动模式] 源码运行")
        # 源码模式下 config.py 自动推算的路径通常是正确的 (exp3/templates)
        # 但为了保险，这里可以显式打印一下
        print(f"默认模板路径: {config.TEMPLATE_DIR}")

    # -------------------------------------------------
    # 初始化资源 & 启动
    # -------------------------------------------------
    # 显式加载模板 (此时 config.TEMPLATE_DIR 已经被修正)
    with app.app_context():
        load_resources()

    threading.Thread(target=start_browser, daemon=True).start()
    
    print("正在启动语音识别引擎...")
    # 关闭 debug 模式，防止重载器导致多次启动
    app.run(host='127.0.0.1', port=54321, debug=False)