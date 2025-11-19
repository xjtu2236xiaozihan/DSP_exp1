# 文件路径: exp3/server.py

import os
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import tempfile
import logging

# --- 关键改动：从同级目录的 src 导入 ---
#
from src import data_utils
from src import features
from src import dtw_core
from src import config

# --- 1. 初始化 Flask 应用 ---
# 
# 关键改动：
# 1. 告诉 Flask 网页模板在 'web_templates' 文件夹中，
#    避免与你现有的 'templates' (DTW模板) 文件夹冲突。
# 2. 告诉 Flask 静态文件 (CSS, JS) 在 'static' 文件夹中。
#
app = Flask(__name__, 
            template_folder='web_templates', 
            static_folder='static')

app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO)

# --- 2. 在服务器启动时加载一次模板库 ---
try:
    app.logger.info("正在加载 DTW 模板库...")
    #
    templates = data_utils.load_templates()
    template_count = sum(len(v) for v in templates.values())
    if template_count == 0:
        raise RuntimeError("模板库为空，请先运行 build_templates.py")
    app.logger.info(f"模板库加载完毕，共 {template_count} 个模板。")
except Exception as e:
    app.logger.error(f"启动失败: {e}")
    templates = None 

# --- 3. 网页服务 (渲染 index.html) ---
@app.route('/')
def index():
    """渲染我们的科幻风格主页"""
    return render_template('index.html') 

# --- 4. 识别 API 端口 ---
@app.route('/recognize', methods=['POST'])
def recognize_speech():
    """接收上传的音频，返回识别结果"""
    if 'audio_file' not in request.files:
        app.logger.warning("请求中未找到 'audio_file'")
        return jsonify({'error': '未找到音频文件'}), 400
    
    if templates is None:
        app.logger.error("模板库未加载")
        return jsonify({'error': '服务器模板库未加载'}), 500

    file = request.files['audio_file']
    
    # 1. 保存临时音频文件
    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename or 'temp_audio.wav')
    temp_audio_path = os.path.join(temp_dir, filename)
    file.save(temp_audio_path)
    app.logger.info(f"临时文件已保存: {temp_audio_path}")

    try:
        # 2. 提取 MFCC 特征
        test_mfcc = features.extract_mfcc(temp_audio_path)
        
        # 3. 执行 DTW 识别 (来自 run_recognition.py 的核心逻辑)
        min_distance = float('inf')
        predicted_label = None
        
        for template_label in config.LABELS:
            if template_label in templates:
                for template_mfcc in templates[template_label]:
                    #
                    distance = dtw_core.calculate_dtw_distance(template_mfcc, test_mfcc)
                    
                    if distance < min_distance:
                        min_distance = distance
                        predicted_label = template_label
        
        app.logger.info(f"识别结果: {predicted_label}, 距离: {min_distance}")

        # 4. 返回 JSON 结果
        return jsonify({
            'digit': predicted_label,
            'distance': float(min_distance) # 确保为JSON兼容的float
        })

    except Exception as e:
        app.logger.error(f"识别出错: {e}", exc_info=True)
        return jsonify({'error': f'处理音频时出错: {e}'}), 500
    finally:
        # 5. 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- 5. 启动服务器 ---
if __name__ == '__main__':
    # 运行: python exp3/server.py
    app.run(debug=True, port=5000)