#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音特征提取程序
功能：提取所有语音特征，分析噪声影响，进行特征归一化
"""

import numpy as np
import pandas as pd
import os
import wave
import contextlib
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore, skew, kurtosis
from scipy.fft import fft
import librosa

# 配置参数
current_dir = os.path.dirname(os.path.abspath(__file__))
# 音频文件目录
INPUT_DIRS = {
    'clean': os.path.join(current_dir, "dataset", "audio"),
    'noise1': os.path.join(current_dir, "dataset", "audio_noise1"),
    'noise2': os.path.join(current_dir, "dataset", "audio_noise2"),
    'noise3': os.path.join(current_dir, "dataset", "audio_noise3")
}
OUTPUT_DIR = os.path.join(current_dir, "dataset", "features")

# 特征提取参数
FRAME_LENGTH = 0.025  # 帧长 25ms
FRAME_SHIFT = 0.010   # 帧移 10ms
SAMPLE_RATE = 16000  # 采样率

class FeatureExtractor:
    """语音特征提取器"""
    
    def __init__(self, frame_length=FRAME_LENGTH, frame_shift=FRAME_SHIFT, sample_rate=SAMPLE_RATE):
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.sample_rate = sample_rate
        self.frame_size = int(frame_length * sample_rate)
        self.frame_shift_size = int(frame_shift * sample_rate)
    
    def read_audio(self, file_path):
        """读取音频文件"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def frame_signal(self, signal):
        """将信号分帧并加窗"""
        frames = []
        # 生成汉明窗 - 减少频谱泄漏，提高频谱分析精度
        hamming_window = np.hamming(self.frame_size)
        
        for i in range(0, len(signal) - self.frame_size + 1, self.frame_shift_size):
            frame = signal[i:i + self.frame_size]
            # 应用汉明窗 - 标准语音信号处理步骤
            windowed_frame = frame * hamming_window
            frames.append(windowed_frame)
        return np.array(frames)
    
    def short_time_energy(self, frames):
        """计算短时能量"""
        energy = np.sum(frames ** 2, axis=1)
        return energy
    
    def short_time_average_magnitude(self, frames):
        """计算短时平均幅度"""
        magnitude = np.mean(np.abs(frames), axis=1)
        return magnitude
    
    def short_time_zero_crossing_rate(self, frames):
        """计算短时过零率"""
        zcr = []
        for frame in frames:
            # 计算过零次数
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))))
            zcr.append(zero_crossings / (2 * len(frame)))
        return np.array(zcr)
    
    
    def extract_spectral_features(self, frames):
        """提取频谱特征（已加窗的帧）"""
        spectral_features = []
        
        for frame in frames:
            # 计算FFT（帧已经加窗）
            fft_frame = np.abs(fft(frame))
            fft_frame = fft_frame[:len(fft_frame)//2]  # 只取正频率部分
            
            # 避免除零错误
            if np.sum(fft_frame) == 0:
                spectral_centroid = 0
                spectral_bandwidth = 0
            else:
                # 频谱质心
                spectral_centroid = np.sum(np.arange(len(fft_frame)) * fft_frame) / np.sum(fft_frame)
                
                # 频谱带宽
                spectral_bandwidth = np.sqrt(np.sum(((np.arange(len(fft_frame)) - spectral_centroid) ** 2) * fft_frame) / np.sum(fft_frame))
            
            # 频谱滚降点
            cumsum = np.cumsum(fft_frame)
            spectral_rolloff = np.where(cumsum >= 0.85 * cumsum[-1])[0][0] if len(np.where(cumsum >= 0.85 * cumsum[-1])[0]) > 0 else len(fft_frame)-1
            
            # 频谱平坦度
            if np.mean(fft_frame) > 0:
                spectral_flatness = np.exp(np.mean(np.log(fft_frame + 1e-10))) / np.mean(fft_frame)
            else:
                spectral_flatness = 0
            
            spectral_features.append([spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness])
        
        return np.array(spectral_features)
    
    
    def extract_advanced_time_features(self, frames):
        """提取高级时域特征"""
        advanced_features = []
        
        for frame in frames:
            skewness = skew(frame)
            kurt = kurtosis(frame)
            rms = np.sqrt(np.mean(frame**2))
            peak_factor = np.max(np.abs(frame)) / (rms + 1e-10)
            waveform_factor = rms / (np.mean(np.abs(frame)) + 1e-10)
            impulse_factor = np.max(np.abs(frame)) / (np.mean(np.abs(frame)) + 1e-10)

            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))))
            zcr = zero_crossings / (2 * len(frame))
            
            advanced_features.append([
                skewness, kurt, rms, peak_factor, 
                waveform_factor, impulse_factor, zcr
            ])
        
        return np.array(advanced_features)
    
    
    def extract_all_features(self, audio_file):
        """提取核心特征"""
        # 读取音频
        audio = self.read_audio(audio_file)
        
        # 分帧
        frames = self.frame_signal(audio)
        
        # 基础时域特征
        energy = self.short_time_energy(frames)
        magnitude = self.short_time_average_magnitude(frames)
        zcr = self.short_time_zero_crossing_rate(frames)
        
        # 高级时域特征
        advanced_time_features = self.extract_advanced_time_features(frames)
        spectral_features = self.extract_spectral_features(frames)
        
        # 统计特征
        features = {
            'file_name': os.path.basename(audio_file),
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'magnitude_mean': np.mean(magnitude),
            'magnitude_std': np.std(magnitude),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'rms_mean': np.mean(advanced_time_features[:, 2]),
            'peak_factor_mean': np.mean(advanced_time_features[:, 3]),
            'spectral_centroid_mean': np.mean(spectral_features[:, 0]),
            'spectral_bandwidth_mean': np.mean(spectral_features[:, 1]),
            'frame_count': len(frames)
        }
        
        return features
    
    def normalize_features(self, features_dict):
        """特征归一化（z-score）"""
        normalized_features = {}
        
        for feature_name, values in features_dict.items():
            if feature_name in ['file_name', 'frame_count']:
                normalized_features[feature_name] = values
            else:
                # Z-score归一化
                normalized_values = zscore(values)
                normalized_features[feature_name] = normalized_values
        
        return normalized_features
    
    def combine_features(self, features_dict, weights=None):
        """特征组合"""
        if weights is None:
            # 默认权重：能量+幅度，减少过零率权重
            weights = {
                'energy': 0.4,
                'magnitude': 0.4,
                'zcr': 0.1,
                'rms': 0.05,
                'spectral_centroid': 0.05
            }
        
        # 归一化特征
        normalized_features = self.normalize_features(features_dict)
        
        # 组合特征
        combined_feature = np.zeros(len(normalized_features['energy']))
        for feature_name, weight in weights.items():
            if feature_name in normalized_features:
                combined_feature += weight * normalized_features[feature_name]
        
        return combined_feature

def process_directory(directory_path, extractor, noise_type='clean'):
    """处理目录中的所有音频文件"""
    features_list = []
    
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return features_list
    
    audio_files = [f for f in os.listdir(directory_path) 
                   if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac'))]
    
    print(f"处理目录: {directory_path}")
    print(f"找到 {len(audio_files)} 个音频文件")
    
    for i, filename in enumerate(audio_files):
        file_path = os.path.join(directory_path, filename)
        print(f"处理文件 {i+1}/{len(audio_files)}: {filename}")
        
        try:
            # 提取特征
            features = extractor.extract_all_features(file_path)
            features['noise_type'] = noise_type  # 添加噪声类型标记
            features_list.append(features)
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
    
    return features_list

def main():
    """主函数"""
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    
    extractor = FeatureExtractor()
    
    # 处理所有类型的音频
    all_features = []
    
    for noise_type, directory_path in INPUT_DIRS.items():
        print(f"\n=== 处理 {noise_type} 音频 ===")
        features = process_directory(directory_path, extractor, noise_type)
        all_features.extend(features)
    
    if not all_features:
        print("未找到音频文件或处理失败")
        return
    
    df = pd.DataFrame(all_features)
    output_file = os.path.join(OUTPUT_DIR, "audio_features.csv")
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n特征已保存到: {output_file}")
    
    print(f"\n=== 特征统计 ===")
    print(f"总文件数: {len(all_features)}")
    print(f"特征维度: {len(df.columns)}")
    
    # 按噪声类型统计
    print("\n各类型文件数:")
    for noise_type in ['clean', 'noise1', 'noise2', 'noise3']:
        count = len(df[df['noise_type'] == noise_type])
        print(f"  {noise_type}: {count}")
    
    print("\n特征提取完成！")

if __name__ == "__main__":
    main()