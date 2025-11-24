"""
特征提取模块 (高精度版)
"""
import os
import tempfile
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from typing import Tuple, Any

from . import config

# Numba Cache Fix
NUMBA_CACHE_DIR = os.path.join(tempfile.gettempdir(), "dtw_numba_cache")
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE_DIR
os.environ["NUMBA_DISABLE_CACHING"] = "0"

# --- 信号处理函数 ---
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # [Fix] 忽略 Pylance 的静态检查误报
    b, a = butter(order, [low, high], btype='band') # type: ignore
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    except Exception as e:
        raise RuntimeError(f"加载失败: {e}")

    # --- [Fix 1] 噪音门阈值微调 ---
    # 如果环境很安静，0.005 可能太高了，导致说话声被当成噪音。
    # 调低到 0.001 (更灵敏)
    rms = np.sqrt(np.mean(y**2))
    if rms < 0.001: 
        raise ValueError("声音太小，请靠近麦克风")

    # --- [Fix 2] 放宽带通滤波 (80Hz - 7500Hz) ---
    # 之前 300Hz 把男声低频切没了，改为 80Hz
    try:
        y = butter_bandpass_filter(y, 80, 7500, sr, order=4)
    except:
        pass 

    # --- [Fix 3] 静音消除 ---
    # top_db=30 是最安全的通用值，既能去噪又不会切掉字尾
    y, _ = librosa.effects.trim(y, top_db=30) # type: ignore
    
    if len(y) < 512: # 稍微放宽最小长度限制
        # 如果切完了没了，尝试只切两头极其安静的部分
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=15)

    try:
        # 使用 config 中的新参数 (win_length)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=config.MFCC_PARAMS['n_mfcc'],
            n_fft=config.MFCC_PARAMS['n_fft'],
            win_length=config.MFCC_PARAMS.get('win_length', 400), # 兼容性写法
            hop_length=config.MFCC_PARAMS['hop_length']
        )
    except Exception as e:
         raise RuntimeError(f"MFCC错误: {e}")
    
    # CMVN
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    
    return mfcc