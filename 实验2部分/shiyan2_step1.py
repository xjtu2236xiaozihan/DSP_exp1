import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

# --- 1. 全局参数定义 (可根据实验需求调整) ---

# 步骤2: 预加重系数
PRE_EMPHASIS_COEFF = 0.97

# 步骤3: 分帧参数
FRAME_LEN_MS = 25  # 帧长 (ms)，建议范围 10-30ms
FRAME_SHIFT_MS = 10 # 帧移 (ms)，建议为帧长的 1/3 到 1/2

# 步骤4: 窗函数类型
# 可选: 'hamming' (汉明窗), 'hanning' (海宁窗), 'boxcar' (矩形窗)
WINDOW_TYPE = 'hamming'


def process_audio_file(file_path):
    """
    对单个音频文件执行实验二的成员一任务（6个步骤）。
    
    参数:
        file_path (str): .wav 文件的完整路径
        
    返回:
        power_spectrum (np.ndarray): 谱线能量 E(i, k)。
                                      形状为 (num_frames, N_FFT // 2 + 1)
        sample_rate (int):          音频的采样率
        None, 0:                    如果处理失败
    """
    
    # --- 准备工作: 读取音频 ---
    try:
        sample_rate, signal = wavfile.read(file_path)
    except ValueError as e:
        print(f"  [错误] 无法读取文件 {file_path}: {e}。可能是文件损坏或非PCM格式。")
        return None, 0
    except Exception as e:
        print(f"  [未知错误] 读取 {file_path} 时发生: {e}")
        return None, 0

    # 确保信号为浮点数以便进行数学运算
    signal = signal.astype(np.float32)
    
    # 如果是双声道，取平均值转为单声道
    if signal.ndim > 1:
        print(f"  [信息] 检测到双声道，已自动转为单声道。")
        signal = np.mean(signal, axis=1)

    # =================================================================
    # 步骤 1: 去直流 与 端点检测 (VAD)
    # =================================================================
    
    # 步骤 1a: 去直流
    # 目的：消除信号中的直流分量，防止其在FFT中产生干扰
    processed_signal = signal - np.mean(signal)

    # 步骤 1b: 端点检测 (VAD)
    # -------------------------
    # !! 注意 !!
    # [cite_start]根据实验原理 [cite: 94]，此处应对原始数据进行端点检测，以分离出实际发声部分。
    # [cite_start]实验一 详细描述了基于短时能量和过零率的“双门限法” [cite: 142, 160-178]。
    # 为集中精力于实验二的频域分析，此处 **暂时跳过 VAD**，对 **整个文件** 进行处理。
    # 在完整的系统中，您应在此处插入VAD代码，仅处理检测到的语音片段。
    # -------------------------
    
    
    # =================================================================
    # 步骤 2: 预加重 (Pre-emphasis)
    # =================================================================
    # [cite_start]目的：增强高频分量 [cite: 269, 270]。
    # [cite_start]方法：应用一阶高通滤波器 H(z) = 1 - a * z^(-1) [cite: 271]。
    processed_signal = np.append(processed_signal[0], 
                                  processed_signal[1:] - PRE_EMPHASIS_COEFF * processed_signal[:-1])

    # =================================================================
    # 步骤 3: 分帧 (Framing)
    # =================================================================
    # [cite_start]目的：将信号切分为短时平稳的帧 [cite: 109]。
    
    # 1. 计算帧长和帧移对应的采样点数
    frame_len_samples = int(FRAME_LEN_MS / 1000 * sample_rate)
    frame_shift_samples = int(FRAME_SHIFT_MS / 1000 * sample_rate)

    # 2. 计算总帧数
    signal_len = len(processed_signal)
    num_frames = 1 + int(np.ceil((signal_len - frame_len_samples) / frame_shift_samples))

    # 3. 对信号进行补零 (Padding)，确保所有帧都有足够的数据
    pad_len = (num_frames - 1) * frame_shift_samples + frame_len_samples - signal_len
    padded_signal = np.pad(processed_signal, (0, pad_len), 'constant', constant_values=0)

    # 4. 切分帧 (高效的Numpy索引)
    indices = np.tile(np.arange(0, frame_len_samples), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift_samples, frame_shift_samples), (frame_len_samples, 1)).T
    
    frames = padded_signal[indices.astype(np.int32)]
    
    # =================================================================
    # 步骤 4: 加窗 (Windowing)
    # =================================================================
    # [cite_start]目的：减少频谱泄漏 [cite: 110]。
    
    # [cite_start]1. 获取窗函数 (实验中提到了 汉明窗 [cite: 114][cite_start], 海宁窗 [cite: 118][cite_start], 矩形窗 [cite: 111])
    try:
        window = get_window(WINDOW_TYPE, frame_len_samples)
    except ValueError:
        print(f"  [警告] 不支持的窗函数 '{WINDOW_TYPE}', 将使用 'hamming'。")
        window = get_window('hamming', frame_len_samples)

    # 2. 应用窗函数
    windowed_frames = frames * window

    # =================================================================
    # 步骤 5: 快速傅里叶变换 (FFT)
    # =================================================================
    # [cite_start]目的：将时域信号转换为频域信号 [cite: 228-231]。
    
    # [cite_start]注意 1：FFT的点数N必须是2的整数幂 (2^L) [cite: 232]。
    # 我们选择大于等于帧长(frame_len_samples)的最小 2^L。
    N_FFT = 2**int(np.ceil(np.log2(frame_len_samples)))

    # 2. 对每一帧执行N点FFT
    # X_i(k)
    fft_frames = np.fft.fft(windowed_frames, n=N_FFT, axis=1)

    # =================================================================
    # 步骤 6: 计算谱线能量 (Power Spectrum)
    # =================================================================
    # [cite_start]目的：获得信号的能量谱，作为Mel滤波的输入 [cite: 276]。
    # [cite_start]方法：E(i, k) = [X_i(k)]^2 (在信号处理中指模值平方) [cite: 277]。
    
    power_spectrum = np.absolute(fft_frames)**2
    
    # 注意：对于实数信号（如语音），FFT的结果是对称的。
    # 我们只需要保留前半部分（0到fs/2）的能量。
    # N_FFT // 2 + 1 个点
    power_spectrum_half = power_spectrum[:, 0:(N_FFT // 2 + 1)]

    # --- 最终输出 ---
    return power_spectrum_half, sample_rate


# =================================================================
# 主程序 (Main) - 负责批量处理和保存文件
# =================================================================
def main():
    # !! 您的音频文件输入 和 处理结果输出 路径 !!
    base_path = r"C:\Users\肖梓涵\Desktop\全组音频_未处理\新建文件夹"
    
    # 检查路径是否存在
    if not os.path.isdir(base_path):
        print(f"错误：路径不存在 -> {base_path}")
        print("请检查 'base_path' 变量是否设置正确。")
        return

    # 查找所有 .wav 文件
    wav_files = glob.glob(os.path.join(base_path, "*.wav"))
    
    if not wav_files:
        print(f"错误：在 {base_path} 中未找到任何 .wav 文件。")
        return

    print(f"在 {base_path} 中找到 {len(wav_files)} 个 .wav 文件。")
    print("-" * 30)

    success_count = 0

    for file_path in wav_files:
        # 解析文件名
        file_name = os.path.basename(file_path)
        try:
            # 文件名_前的为字符，_后的为遍数
            parts = file_name.split('.')[0].split('_')
            character = parts[0]
            repetition_num = parts[1]
            print(f"正在处理: 字符='{character}', 遍数={repetition_num} (文件: {file_name})")
        except IndexError:
            print(f"正在处理: {file_name} (文件名格式无法解析，跳过解析)")

        # --- 调用核心处理函数 ---
        power_spectrum, sample_rate = process_audio_file(file_path)
        # ------------------------

        if power_spectrum is not None:
            # 1. 准备输出文件名
            # 获取不带 .wav 后缀的文件名 (例如 "0_1")
            base_filename = os.path.splitext(file_name)[0]
            # 定义输出文件名 (例如 "0_1_powerspec.npz")
            output_filename = f"{base_filename}_powerspec.npz"
            # 定义完整的输出路径
            output_path = os.path.join(base_path, output_filename)
            
            # 2. 保存处理结果 (谱线能量 和 采样率)
            try:
                # 使用 np.savez 保存多个数组
                np.savez(output_path, 
                         spectrum=power_spectrum, 
                         sample_rate=sample_rate)
                
                print(f"  处理完成。采样率: {sample_rate} Hz")
                print(f"  输出谱线能量 E(i, k) 形状: {power_spectrum.shape}")
                print(f"  [成功] 已保存至: {output_path}")
                success_count += 1
                
            except Exception as e:
                print(f"  [错误] 保存文件 {output_path} 失败: {e}")
                
            print("-" * 30)

    print(f"\n全部处理完毕。共成功处理并保存 {success_count} 个文件。")
    print(f"所有输出的 .npz 文件均已保存在: {base_path}")


if __name__ == "__main__":
    main()