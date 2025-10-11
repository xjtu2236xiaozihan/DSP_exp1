#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音特征提取程序
功能：提取所有语音特征，分析噪声影响，进行特征归一化
改动：
1) 分帧加窗支持三种窗口：rect(矩形窗)、hamming(汉明窗)、hanning(汉宁/海明窗)
2) 使用 argparse 参数化所有可配参数
3) 输出目录与输出文件名在末尾追加窗口名后缀，便于区分
"""

import numpy as np
import pandas as pd
import os
import wave
import contextlib
from pydub import AudioSegment  # 可能不直接用到，但保留以便扩展
import matplotlib.pyplot as plt  # 可能不直接用到，但保留以便扩展
from scipy import signal as sp_signal  # 避免与变量名冲突
from scipy.stats import zscore, skew, kurtosis
from scipy.fft import fft
import librosa
import argparse


# —— 默认配置（可被 argparse 覆盖） ——
_current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUDIO_DIR = os.path.join(_current_dir, "dataset", "audio_processed")
DEFAULT_OUTPUT_DIR = os.path.join(_current_dir, "dataset", "features")
DEFAULT_FRAME_LENGTH = 0.025  # 25 ms
DEFAULT_FRAME_SHIFT  = 0.010  # 10 ms
DEFAULT_SAMPLE_RATE  = 16000  # 16 kHz
DEFAULT_WINDOW       = "hamming"  # rect | hamming | hanning


class FeatureExtractor:
    """语音特征提取器"""

    def __init__(self,
                 frame_length: float = DEFAULT_FRAME_LENGTH,
                 frame_shift: float  = DEFAULT_FRAME_SHIFT,
                 sample_rate: int    = DEFAULT_SAMPLE_RATE,
                 window: str         = DEFAULT_WINDOW):
        """
        window: 'rect'（矩形窗）| 'hamming'（汉明窗）| 'hanning'（汉宁/海明窗）
        """
        self.frame_length = frame_length
        self.frame_shift  = frame_shift
        self.sample_rate  = sample_rate
        self.frame_size        = int(round(frame_length * sample_rate))
        self.frame_shift_size  = int(round(frame_shift  * sample_rate))
        self.window_type  = self._normalize_window_name(window)
        self.window_vec   = self._make_window(self.window_type, self.frame_size)

    @staticmethod
    def _normalize_window_name(name: str) -> str:
        if name is None:
            return "hamming"
        name = name.strip().lower()
        # 同义纠正：海明窗/汉宁窗常被混用，这里都映射到 hanning（Hann 窗）
        if name in ("hann", "han", "hanning", "hannning", "汉宁窗", "海明窗"):
            return "hanning"
        if name in ("hamm", "ham", "hamming", "汉明窗"):
            return "hamming"
        if name in ("rect", "rectangular", "box", "boxcar", "矩形窗"):
            return "rect"
        # 默认用 hamming，防止输错
        return "hamming"

    @staticmethod
    def _make_window(window_type: str, frame_size: int) -> np.ndarray:
        if window_type == "rect":
            return np.ones(frame_size, dtype=np.float32)
        elif window_type == "hamming":
            return np.hamming(frame_size).astype(np.float32)
        elif window_type == "hanning":
            return np.hanning(frame_size).astype(np.float32)
        else:
            # 兜底
            return np.hamming(frame_size).astype(np.float32)

    def read_audio(self, file_path: str) -> np.ndarray:
        """读取音频文件（单声道、指定采样率）"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio.astype(np.float32)

    def frame_signal(self, sig: np.ndarray) -> np.ndarray:
        """将信号分帧并加窗"""
        frames = []
        win = self.window_vec
        N   = self.frame_size
        H   = self.frame_shift_size

        if len(sig) < N:
            return np.empty((0, N), dtype=np.float32)

        for i in range(0, len(sig) - N + 1, H):
            frame = sig[i:i + N]
            frames.append(frame * win)
        return np.stack(frames, axis=0).astype(np.float32)

    def short_time_energy(self, frames: np.ndarray) -> np.ndarray:
        """计算短时能量"""
        return np.sum(frames ** 2, axis=1)

    def short_time_average_magnitude(self, frames: np.ndarray) -> np.ndarray:
        """计算短时平均幅度"""
        return np.mean(np.abs(frames), axis=1)

    def short_time_zero_crossing_rate(self, frames: np.ndarray) -> np.ndarray:
        """计算短时过零率"""
        zcr = []
        for frame in frames:
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))))
            zcr.append(zero_crossings / (2 * len(frame)))
        return np.array(zcr, dtype=np.float32)

    def extract_spectral_features(self, frames: np.ndarray) -> np.ndarray:
        """提取频谱特征（已加窗的帧）"""
        spectral_features = []
        eps = 1e-10

        for frame in frames:
            fft_frame = np.abs(fft(frame))
            fft_frame = fft_frame[:len(fft_frame)//2]  # 正频率
            S = np.sum(fft_frame)

            if S <= eps:
                spectral_centroid = 0.0
                spectral_bandwidth = 0.0
                spectral_rolloff = 0
                spectral_flatness = 0.0
            else:
                idx = np.arange(len(fft_frame), dtype=np.float32)
                spectral_centroid = float(np.sum(idx * fft_frame) / S)
                spectral_bandwidth = float(np.sqrt(np.sum(((idx - spectral_centroid) ** 2) * fft_frame) / S))

                cumsum = np.cumsum(fft_frame)
                thr = 0.85 * cumsum[-1]
                pos = np.where(cumsum >= thr)[0]
                spectral_rolloff = int(pos[0]) if len(pos) > 0 else len(fft_frame) - 1

                spectral_flatness = float(np.exp(np.mean(np.log(fft_frame + eps))) / (np.mean(fft_frame) + eps))

            spectral_features.append([spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness])

        return np.array(spectral_features, dtype=np.float32)

    def extract_advanced_time_features(self, frames: np.ndarray) -> np.ndarray:
        """提取高级时域特征"""
        advanced_features = []
        eps = 1e-10

        for frame in frames:
            sk = float(skew(frame))
            ku = float(kurtosis(frame))
            rms = float(np.sqrt(np.mean(frame ** 2) + eps))
            peak = float(np.max(np.abs(frame)))
            peak_factor = float(peak / (rms + eps))
            waveform_factor = float(rms / (np.mean(np.abs(frame)) + eps))
            impulse_factor  = float(peak / (np.mean(np.abs(frame)) + eps))

            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))))
            zcr = float(zero_crossings / (2 * len(frame)))

            advanced_features.append([sk, ku, rms, peak_factor, waveform_factor, impulse_factor, zcr])

        return np.array(advanced_features, dtype=np.float32)

    def extract_all_features(self, audio_file: str) -> dict:
        """提取一个文件的聚合统计特征"""
        audio = self.read_audio(audio_file)
        frames = self.frame_signal(audio)

        if frames.shape[0] == 0:
            # 极短音频
            return {
                'file_name': os.path.basename(audio_file),
                'window': self.window_type,
                'frame_count': 0,
                'energy_mean': 0, 'energy_std': 0,
                'magnitude_mean': 0, 'magnitude_std': 0,
                'zcr_mean': 0, 'zcr_std': 0,
                'rms_mean': 0, 'peak_factor_mean': 0,
                'spectral_centroid_mean': 0, 'spectral_bandwidth_mean': 0,
            }

        energy    = self.short_time_energy(frames)
        magnitude = self.short_time_average_magnitude(frames)
        zcr       = self.short_time_zero_crossing_rate(frames)

        advanced_time_features = self.extract_advanced_time_features(frames)
        spectral_features      = self.extract_spectral_features(frames)

        features = {
            'file_name': os.path.basename(audio_file),
            'window': self.window_type,
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy)),
            'magnitude_mean': float(np.mean(magnitude)),
            'magnitude_std': float(np.std(magnitude)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr)),
            'rms_mean': float(np.mean(advanced_time_features[:, 2])),
            'peak_factor_mean': float(np.mean(advanced_time_features[:, 3])),
            'spectral_centroid_mean': float(np.mean(spectral_features[:, 0])),
            'spectral_bandwidth_mean': float(np.mean(spectral_features[:, 1])),
            'frame_count': int(frames.shape[0]),
        }
        return features

    # 下列两个函数在当前主流程未用到，但保留供后续扩展（例如综合评分）
    def normalize_features(self, features_dict: dict) -> dict:
        """特征归一化（z-score）"""
        normalized_features = {}
        for feature_name, values in features_dict.items():
            if feature_name in ['file_name', 'frame_count']:
                normalized_features[feature_name] = values
            else:
                normalized_values = zscore(values)
                normalized_features[feature_name] = normalized_values
        return normalized_features

    def combine_features(self, features_dict: dict, weights=None) -> np.ndarray:
        """特征组合（可选功能）"""
        if weights is None:
            weights = {
                'energy': 0.4,
                'magnitude': 0.4,
                'zcr': 0.1,
                'rms': 0.05,
                'spectral_centroid': 0.05
            }
        normalized_features = self.normalize_features(features_dict)
        combined_feature = np.zeros(len(normalized_features['energy']))
        for feature_name, weight in weights.items():
            if feature_name in normalized_features:
                combined_feature += weight * normalized_features[feature_name]
        return combined_feature


def process_directory(directory_path: str, extractor: FeatureExtractor):
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
            features = extractor.extract_all_features(file_path)
            features_list.append(features)
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    return features_list


def parse_args():
    p = argparse.ArgumentParser(description="语音特征提取（支持 rect/hamming/hanning 窗口）")
    p.add_argument("--audio_dir", type=str, default=DEFAULT_AUDIO_DIR,
                   help=f"输入音频目录（默认：{DEFAULT_AUDIO_DIR}）")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}，程序会在末尾自动加 _<window> 后缀）")
    p.add_argument("--frame_length", type=float, default=DEFAULT_FRAME_LENGTH,
                   help=f"帧长（秒），默认 {DEFAULT_FRAME_LENGTH}")
    p.add_argument("--frame_shift", type=float, default=DEFAULT_FRAME_SHIFT,
                   help=f"帧移（秒），默认 {DEFAULT_FRAME_SHIFT}")
    p.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE,
                   help=f"采样率（Hz），默认 {DEFAULT_SAMPLE_RATE}")
    p.add_argument("--window", type=str, default=DEFAULT_WINDOW, choices=["rect", "hamming", "hanning"],
                   help="加窗类型：rect(矩形窗) / hamming(汉明窗) / hanning(汉宁/海明窗)")
    p.add_argument("--output_name", type=str, default="audio_features",
                   help="输出CSV 基础文件名（程序会追加 _<window>.csv）")
    return p.parse_args()


def main():
    args = parse_args()

    # 归一化窗口名，确保后续路径与文件名统一
    win_norm = FeatureExtractor._normalize_window_name(args.window)

    # 输出目录在末尾追加窗口名，便于不同窗口结果分开存放
    output_dir = f"{args.output_dir}_{win_norm}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    extractor = FeatureExtractor(
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        sample_rate=args.sample_rate,
        window=win_norm
    )

    print(f"=== 处理音频目录：{args.audio_dir} ===")
    print(f"窗口类型：{win_norm} | 采样率：{args.sample_rate} Hz | 帧长：{args.frame_length*1000:.1f} ms | 帧移：{args.frame_shift*1000:.1f} ms")

    features = process_directory(args.audio_dir, extractor)

    if not features:
        print("未找到音频文件或处理失败")
        return

    df = pd.DataFrame(features)
    output_file = os.path.join(output_dir, f"{args.output_name}_{win_norm}.csv")
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n特征已保存到: {output_file}")
    print(f"\n=== 特征统计 ===")
    print(f"总文件数: {len(features)}")
    print(f"特征维度: {len(df.columns)}")
    print("\n特征提取完成！")


if __name__ == "__main__":
    main()
