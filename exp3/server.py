import os
import sys
import shutil
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tempfile
import logging
import traceback

from src import data_utils
from src import features
from src import dtw_core
from src import config

app = Flask(__name__, template_folder='web_templates', static_folder='static')

# 系统临时目录
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'dtw_app_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 桌面调试路径
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

log_path = os.path.join(get_desktop_path(), "dtw_debug_log.txt")
debug_wav_path = os.path.join(get_desktop_path(), "debug_last_input.wav")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# 全局变量初始化为 None
templates = None

def load_resources():
    global templates
    try:
        logger.info(f"加载模板库... (参数: sr={config.SAMPLE_RATE}, n_fft={config.MFCC_PARAMS['n_fft']})")
        templates = data_utils.load_templates()
        count = sum(len(v) for v in templates.values())
        logger.info(f"模板库加载完毕: {count} 个")
    except Exception as e:
        logger.error(f"模板加载失败: {e}")
        logger.error(traceback.format_exc())
        templates = {} # 失败时兜底为空字典

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/recognize', methods=['POST'])
def recognize_speech():
    global templates
    # 1. 懒加载检查
    if templates is None:
        load_resources()
    
    # 2. [FIX] 关键修复：创建安全的局部引用
    # 如果 templates 依然是 None（极端情况），则使用空字典 {}
    # 这能消除 "Operator 'in' not supported for 'None'" 错误
    active_templates = templates if templates is not None else {}

    if 'audio_file' not in request.files:
        return jsonify({'error': '无音频文件'}), 400
    
    file = request.files['audio_file']
    filename = secure_filename(file.filename or 'temp.wav')
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # 保存并备份录音
        file.save(save_path)
        try:
            shutil.copy(save_path, debug_wav_path)
        except:
            pass

        # 提取特征
        test_mfcc = features.extract_mfcc(save_path)
        
        # DTW 匹配
        min_distance = float('inf')
        predicted_label = None
        top_candidates = []

        for template_label in config.LABELS:
            # [FIX] 这里使用安全的 active_templates
            if template_label in active_templates:
                for idx, template_mfcc in enumerate(active_templates[template_label]):
                    dist = dtw_core.calculate_dtw_distance(template_mfcc, test_mfcc)
                    
                    if dist < min_distance:
                        min_distance = dist
                        predicted_label = template_label
                    
                    # 记录前3名
                    if len(top_candidates) < 3 or dist < top_candidates[-1][0]:
                        top_candidates.append((dist, template_label))
                        top_candidates.sort(key=lambda x: x[0])
                        top_candidates = top_candidates[:3]
        
        logger.info(f"识别结果: {predicted_label} (距离: {min_distance:.4f})")
        
        return jsonify({'digit': predicted_label, 'distance': float(min_distance)})

    except ValueError as ve:
        # [New] 专门处理音量过低/无效音频的错误
        logger.warning(f"无效输入拦截: {ve}")
        return jsonify({'error': str(ve)}), 400 # 返回 400 Bad Request
    
    except Exception as e:
        logger.error(f"识别崩溃: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(save_path):
            try: os.remove(save_path)
            except: pass

if __name__ == '__main__':
    load_resources()
    app.run(debug=False, port=54321)