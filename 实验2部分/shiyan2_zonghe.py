import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import scipy.fftpack as fft
import warnings
import math

# =================================================================
# 1. 全局参数定义 (用于调参实验的关键变量)
# =================================================================

# --- 预处理参数 ---
PRE_EMPHASIS_COEFF = 0.97  # 预加重系数 (用于消融实验)

# --- 分帧参数 ---
FRAME_LEN_MS = 25     # 帧长 (ms)
FRAME_SHIFT_MS = 10   # 帧移 (ms)
WINDOW_TYPE = 'hamming' # 窗函数 (用于调参实验)

# --- VAD (端点检测) 参数 ---
# VAD参数可以独立于MFCC的帧参数，这里使用一组固定的经验值
VAD_FRAME_LEN_MS = 20    
VAD_FRAME_SHIFT_MS = 10  

# --- Mel 滤波器组和MFCC参数 (用于调参实验) ---
N_MELS = 26  # Mel滤波器数量
N_MFCC = 13  # MFCC系数阶数 (例如 12, 13, 16)

# =================================================================
# 2. 辅助函数
# =================================================================

def hz_to_mel(hz):
    """将频率从Hz转换为Mel尺度"""
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    """将频率从Mel尺度转换回Hz"""
    return 700 * (10 ** (mel / 2595.0) - 1)

def short_term_energy(frame):
    """计算单帧的短时能量 E_n"""
    return np.sum(frame**2)

def short_term_zcr(frame):
    """计算单帧的短时过零率 Z_n"""
    sgn = np.sign(frame)
    # 使用 np.diff 计算相邻元素差
    zcr = np.sum(np.abs(np.diff(sgn))) / (2 * len(frame))
    return zcr

def dual_threshold_vad(signal, sample_rate, frame_len_ms, frame_shift_ms):
    """
    实现基于双门限法的端点检测
    返回 (start_index, end_index)
    """
    
    frame_len = int(frame_len_ms / 1000 * sample_rate)
    frame_shift = int(frame_shift_ms / 1000 * sample_rate)
    num_samples = len(signal)
    
    # ----------------------------------------------------
    # ** 修正后的分帧逻辑 **：使用截断 (Truncation) 确保不越界
    if num_samples < frame_len:
        num_frames = 0
    else:
        # 确保最后一帧的起始点不会超出信号范围
        num_frames = int(np.floor((num_samples - frame_len) / frame_shift)) + 1
    # ----------------------------------------------------
        
    if num_frames <= 0:
        print("  [VAD警告] 信号过短，无法分帧。")
        return 0, len(signal)
        
    # 构建帧索引矩阵
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    
    # 直接用索引从信号中提取帧
    frames = signal[indices.astype(np.int32)]

    # 2. 计算短时能量和短时过零率
    energy = np.array([short_term_energy(frame) for frame in frames])
    zcr = np.array([short_term_zcr(frame) for frame in frames])

    # 归一化能量
    energy = energy / (np.max(energy) + 1e-10) # 加一个小量避免除零
    
    # 3. 确定门限 (经验值)
    T_E_high = 0.1   # 能量高门限 T2
    T_E_low = 0.02   # 能量低门限 T1
    T_ZCR = 0.15     # 过零率门限 T3

    # 4. 步骤 [1] & [2]: 能量高门限初判
    high_energy_frames = np.where(energy > T_E_high)[0]
    if len(high_energy_frames) == 0:
        return 0, len(signal) # 未找到高能量段

    start_N3 = high_energy_frames[0]
    end_N4 = high_energy_frames[-1]

    # 5. 步骤 [3]: 能量低门限扩展 (前向/后向搜索)
    
    # 扩展起点 N2
    start_N2 = start_N3
    while start_N2 > 0 and energy[start_N2] > T_E_low:
        start_N2 -= 1
        
    # 扩展终点 N5
    end_N5 = end_N4
    while end_N5 < (num_frames - 1) and energy[end_N5] > T_E_low:
        end_N5 += 1

    # 6. 步骤 [4]: 过零率精判 (这里采用最简化的过零率约束)
    # 起点 N1
    start_N1 = start_N2
    count_threshold = 3 # 连续3帧低于ZCR门限则停止
    count = 0
    while start_N1 > 0:
        if zcr[start_N1] < T_ZCR:
            count += 1
        else:
            count = 0
        
        if count >= count_threshold:
            start_N1 += count # 回退到ZCR开始低于门限的位置
            break
        start_N1 -= 1

    # 终点 N6
    end_N6 = end_N5
    count = 0
    while end_N6 < (num_frames - 1):
        if zcr[end_N6] < T_ZCR:
            count += 1
        else:
            count = 0

        if count >= count_threshold:
            end_N6 -= count # 回退到ZCR开始低于门限的位置
            break
        end_N6 += 1
        
    # 7. 转换回信号索引
    # N1 和 N6 已经是帧索引。转换为信号点的索引
    start_index = max(0, start_N1 * frame_shift)
    # N6 帧的结束点
    end_index = min(num_samples, end_N6 * frame_shift + frame_len)
    
    return start_index, end_index


def create_mel_filterbank(sample_rate, n_fft, n_mels=N_MELS, fmin=0, fmax=None):
    """
    创建Mel滤波器组
    """
    if fmax is None:
        fmax = sample_rate // 2

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    # M + 2 个点
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    # 转换为FFT bin索引
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(n_mels):
        left = fft_bins[m]
        center = fft_bins[m + 1]
        right = fft_bins[m + 2]
        
        # 实现三角形滤波器的左半边
        if left < center:
            filterbank[m, left:center] = np.linspace(0, 1, center - left)
        # 实现三角形滤波器的右半边
        if center < right:
            filterbank[m, center:right] = np.linspace(1, 0, right - center)

    return filterbank

def extract_mfcc(power_spectrum, sample_rate, n_fft, n_mfcc=N_MFCC, n_mels=N_MELS):
    """
    从功率谱中提取MFCC特征
    """
    # 1. 创建Mel滤波器组
    mel_filterbank = create_mel_filterbank(sample_rate, n_fft, n_mels)

    # 2. 应用Mel滤波器组，计算Mel频谱能量 S(i, m)
    # power_spectrum: (NumFrames, NFFT/2 + 1)
    # mel_filterbank: (N_MELS, NFFT/2 + 1) -> 转置后 (NFFT/2 + 1, N_MELS)
    mel_energy = np.dot(power_spectrum, mel_filterbank.T)

    # 3. 对Mel能量取对数
    # 加一个小值避免log(0)，1e-8
    log_mel_energy = np.log(mel_energy + 1e-8)

    # 4. 离散余弦变换(DCT)得到MFCC系数
    # 使用DCT-II类型
    mfcc = fft.dct(log_mel_energy, type=2, axis=1, norm='ortho')

    # 5. 保留前n_mfcc个系数
    return mfcc[:, :n_mfcc]

# =================================================================
# 3. 核心处理函数 (process_audio_file)
# =================================================================

def process_audio_file(file_path):
    """
    对单个音频文件执行完整的MFCC提取流程。
    """
    
    try:
        sample_rate, signal = wavfile.read(file_path)
    except Exception as e:
        print(f"  [错误] 无法读取文件 {file_path}: {e}。")
        return None, 0

    # 统一数据类型，并处理立体声
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1) # 取平均，转为单声道
    
    if len(signal) == 0:
        print(f"  [错误] 文件 {file_path} 为空。")
        return None, 0

    # 步骤 1a: 去直流
    processed_signal = signal - np.mean(signal)

    # 步骤 1b: 端点检测 (VAD)
    print(f"  [VAD] 正在执行端点检测...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # 忽略可能的空切片警告
        start_idx, end_idx = dual_threshold_vad(processed_signal, sample_rate, 
                                                VAD_FRAME_LEN_MS, VAD_FRAME_SHIFT_MS)
    
    vad_len = end_idx - start_idx
    if vad_len < 1000: # 经验值：小于1000个采样点认为太短
        print(f"  [VAD警告] 检测到的语音过短 (原始: {len(processed_signal)}点, VAD后: {vad_len}点)。跳过此文件。")
        return None, sample_rate

    processed_signal = processed_signal[start_idx:end_idx]
    print(f"  [VAD] 原始长度: {len(signal)}点, VAD后: {len(processed_signal)}点")

    # 步骤 2: 预加重 (用于消融实验)
    if PRE_EMPHASIS_COEFF > 0:
        processed_signal = np.append(processed_signal[0], 
                                     processed_signal[1:] - PRE_EMPHASIS_COEFF * processed_signal[:-1])

    # 步骤 3: 分帧
    frame_len_samples = int(FRAME_LEN_MS / 1000 * sample_rate)
    frame_shift_samples = int(FRAME_SHIFT_MS / 1000 * sample_rate)
    
    signal_len = len(processed_signal)
    
    # 使用零填充的帧数计算逻辑
    num_frames = 1 + int(np.ceil((signal_len - frame_len_samples) / frame_shift_samples))
    
    # 零填充，使信号长度恰好包含 num_frames 个帧
    pad_len = (num_frames - 1) * frame_shift_samples + frame_len_samples - signal_len
    padded_signal = np.pad(processed_signal, (0, int(pad_len)), 'constant', constant_values=0)

    # 构建帧索引矩阵
    indices = np.tile(np.arange(0, frame_len_samples), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift_samples, frame_shift_samples), (frame_len_samples, 1)).T
    
    frames = padded_signal[indices.astype(np.int32)]
    
    # 步骤 4: 加窗
    window = get_window(WINDOW_TYPE, frame_len_samples)
    windowed_frames = frames * window

    # 步骤 5: FFT (保证是 2 的幂次)
    N_FFT = 2**int(np.ceil(np.log2(frame_len_samples))) 
    fft_frames = np.fft.fft(windowed_frames, n=N_FFT, axis=1)

    # 步骤 6: 计算谱线能量 E(i, k)
    # 取前 N_FFT / 2 + 1 点 (对称性)
    power_spectrum_half = np.absolute(fft_frames[:, 0:(N_FFT // 2 + 1)])**2

    # 步骤 7: 提取 MFCC 
    mfcc_features = extract_mfcc(power_spectrum_half, sample_rate, N_FFT, N_MFCC, N_MELS)
    
    return mfcc_features, sample_rate

# =================================================================
# 4. 主程序 (Main)
# =================================================================
def main():
    # !! 路径设置：请根据你的实际路径修改 !!
    input_wav_path = r"C:\Users\肖梓涵\Desktop\全组音频_未处理\新建文件夹" # 包含 .wav 文件的文件夹
    output_mfcc_path = r"C:\Users\肖梓涵\Desktop\shiyan2_mfcc_features" # 存放 _mfcc.npz 文件的目标文件夹
    
    # 确保输出路径存在
    os.makedirs(output_mfcc_path, exist_ok=True)
    
    if not os.path.isdir(input_wav_path):
        print(f"错误：输入路径不存在 -> {input_wav_path}")
        return

    wav_files = glob.glob(os.path.join(input_wav_path, "*.wav"))
    
    if not wav_files:
        print(f"错误：在 {input_wav_path} 中未找到任何 .wav 文件。")
        return

    print(f"在 {input_wav_path} 中找到 {len(wav_files)} 个 .wav 文件。")
    print(f"MFCC结果将保存至: {output_mfcc_path}")
    print("-" * 50)

    success_count = 0

    for file_path in wav_files:
        file_name = os.path.basename(file_path)
        base_filename = os.path.splitext(file_name)[0]
        print(f"正在处理: {file_name}")

        # --- 调用核心处理函数 ---
        # 注意：这里的参数是全局变量，如果要进行调参实验，你需要修改全局变量或封装函数
        mfcc_features, sample_rate = process_audio_file(file_path)
        # ------------------------

        if mfcc_features is not None:
            output_filename = f"{base_filename}_mfcc.npz"
            output_path = os.path.join(output_mfcc_path, output_filename)
            
            try:
                # 保存最终的MFCC特征
                np.savez(output_path, 
                         mfcc=mfcc_features, 
                         sample_rate=sample_rate,
                         n_mfcc=N_MFCC,
                         n_mels=N_MELS)
                
                print(f"  处理完成。采样率: {sample_rate} Hz")
                print(f"  输出MFCC特征 E(i, n) 形状: {mfcc_features.shape}")
                print(f"  [成功] 已保存至: {output_path}")
                success_count += 1
                
            except Exception as e:
                print(f"  [错误] 保存文件 {output_path} 失败: {e}")
        else:
            print(f"  [失败] 文件 {file_name} 处理失败或被跳过。")
            
        print("-" * 50)

    print(f"\n全部处理完毕。共成功处理并保存 {success_count} 个文件。")

if __name__ == "__main__":
    main()