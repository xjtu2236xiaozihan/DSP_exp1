# 文件路径: exp3/server.py

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tempfile
import logging
from src import data_utils
from src import features
from src import dtw_core
from src import config

app = Flask(__name__, 
            template_folder='web_templates', 
            static_folder='static')

app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO)

# --- 修改点：将模板加载改为全局变量 + 加载函数 ---
templates = None

def load_resources():
    """
    在主程序配置好路径后，显式调用此函数加载模板
    """
    global templates
    try:
        app.logger.info(f"正在从 {config.TEMPLATE_DIR} 加载 DTW 模板库...")
        templates = data_utils.load_templates()
        
        template_count = sum(len(v) for v in templates.values())
        if template_count == 0:
            app.logger.error("错误: 模板库为空！")
        else:
            app.logger.info(f"模板库加载完毕，共 {template_count} 个模板。")
    except Exception as e:
        app.logger.error(f"模板加载失败: {e}")
        templates = {}

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    # 确保模板已加载
    global templates
    if templates is None or len(templates) == 0:
        load_resources()
        if templates is None or len(templates) == 0:
            return jsonify({'error': '服务器模板库加载失败，请检查日志'}), 500

    if 'audio_file' not in request.files:
        return jsonify({'error': '未找到音频文件'}), 400
    
    file = request.files['audio_file']
    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename or 'temp_audio.wav')
    temp_audio_path = os.path.join(temp_dir, filename)
    file.save(temp_audio_path)

    try:
        test_mfcc = features.extract_mfcc(temp_audio_path)
        min_distance = float('inf')
        predicted_label = None
        
        for template_label in config.LABELS:
            if template_label in templates:
                for template_mfcc in templates[template_label]:
                    distance = dtw_core.calculate_dtw_distance(template_mfcc, test_mfcc)
                    if distance < min_distance:
                        min_distance = distance
                        predicted_label = template_label
        
        app.logger.info(f"识别结果: {predicted_label}, 距离: {min_distance}")
        return jsonify({'digit': predicted_label, 'distance': float(min_distance)})

    except Exception as e:
        app.logger.error(f"识别出错: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == '__main__':
    # 开发模式下自动加载
    load_resources()
    app.run(debug=True, port=5000)