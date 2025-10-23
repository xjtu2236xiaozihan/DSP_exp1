import os
import glob
import numpy as np
import scipy.fftpack as fft

# --- Mel滤波器组参数 ---
N_MELS = 26  # Mel滤波器数量
N_MFCC = 13  # MFCC系数阶数（通常取12-16）


def hz_to_mel(hz):
    """将频率从Hz转换为Mel尺度"""
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    """将频率从Mel尺度转换回Hz"""
    return 700 * (10 ** (mel / 2595.0) - 1)


def create_mel_filterbank(sample_rate, n_fft, n_mels=N_MELS, fmin=0, fmax=None):
    """
    创建Mel滤波器组

    参数:
        sample_rate: 采样率
        n_fft: FFT点数
        n_mels: Mel滤波器数量
        fmin: 最低频率 (Hz)
        fmax: 最高频率 (Hz)，默认为奈奎斯特频率

    返回:
        filterbank: Mel滤波器组，形状为 (n_mels, n_fft//2 + 1)
    """
    if fmax is None:
        fmax = sample_rate // 2  # 奈奎斯特频率

    # 1. 在Mel尺度上等间隔的点
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # 2. 将Mel点转换回Hz
    hz_points = mel_to_hz(mel_points)

    # 3. 将Hz点转换为FFT bin索引
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # 4. 创建三角形滤波器
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))

    for m in range(n_mels):
        # 当前滤波器的三个关键点
        left = fft_bins[m]
        center = fft_bins[m + 1]
        right = fft_bins[m + 2]

        # 创建三角形的左半边
        if left < center:
            filterbank[m, left:center] = np.linspace(0, 1, center - left)

        # 创建三角形的右半边
        if center < right:
            filterbank[m, center:right] = np.linspace(1, 0, right - center)

    return filterbank


def extract_mfcc(power_spectrum, sample_rate, n_mfcc=N_MFCC, n_mels=N_MELS):
    """
    从功率谱中提取MFCC特征

    参数:
        power_spectrum: 功率谱，形状为 (num_frames, n_fft//2 + 1)
        sample_rate: 采样率
        n_mfcc: MFCC系数阶数
        n_mels: Mel滤波器数量

    返回:
        mfcc: MFCC系数，形状为 (num_frames, n_mfcc)
    """
    num_frames, n_fft_half = power_spectrum.shape
    n_fft = (n_fft_half - 1) * 2  # 恢复原始FFT点数

    # 1. 创建Mel滤波器组
    mel_filterbank = create_mel_filterbank(sample_rate, n_fft, n_mels)

    # 2. 应用Mel滤波器组，计算Mel频谱能量
    # S(i, m) = Σ E(i, k) * H_m(k)
    mel_energy = np.dot(power_spectrum, mel_filterbank.T)

    # 3. 对Mel能量取对数（加一个小值避免log(0)）
    log_mel_energy = np.log(mel_energy + 1e-8)

    # 4. 离散余弦变换(DCT)得到MFCC系数
    # mfcc(i, n) = sqrt(2/M) * Σ ln[S(i, m)] * cos(πn(2m-1)/(2M))
    mfcc = fft.dct(log_mel_energy, type=2, axis=1, norm='ortho')

    # 5. 保留前n_mfcc个系数（通常去掉第0个系数，即能量项）
    mfcc = mfcc[:, :n_mfcc]

    return mfcc


def process_npz_file(file_path):
    """
    处理单个npz文件，提取MFCC特征

    参数:
        file_path: .npz文件的完整路径

    返回:
        mfcc_features: MFCC特征矩阵
        sample_rate: 采样率
        None, 0: 如果处理失败
    """
    try:
        # 加载谱线能量数据
        data = np.load(file_path)
        power_spectrum = data['spectrum']
        sample_rate = data['sample_rate']

        print(f"  加载数据: 谱线能量形状 {power_spectrum.shape}, 采样率 {sample_rate} Hz")

        # 提取MFCC特征
        mfcc_features = extract_mfcc(power_spectrum, sample_rate)

        print(f"  提取MFCC: 特征形状 {mfcc_features.shape}")

        return mfcc_features, sample_rate

    except Exception as e:
        print(f"  [错误] 处理文件 {file_path} 失败: {e}")
        return None, 0


def main():
    """
    主程序 - 批量处理所有谱线能量文件并提取MFCC特征
    """
    # 目标文件夹 (exp2预处理结果的集中地)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "exp2预处理")

    # 检查路径是否存在
    if not os.path.isdir(base_path):
        print(f"错误：路径不存在 -> {base_path}")
        return

    # 查找所有 _powerspec.npz 文件
    npz_files = glob.glob(os.path.join(base_path, "*_powerspec.npz"))

    if not npz_files:
        print(f"错误：在 {base_path} 中未找到任何 _powerspec.npz 文件。")
        return

    print(f"在 {base_path} 中找到 {len(npz_files)} 个谱线能量文件。")
    print("开始提取MFCC特征...")
    print("-" * 50)

    success_count = 0

    for file_path in npz_files:
        # 解析文件名
        file_name = os.path.basename(file_path)
        base_filename = file_name.replace('_powerspec.npz', '')

        print(f"正在处理: {file_name}")

        # --- 调用MFCC提取函数 ---
        mfcc_features, sample_rate = process_npz_file(file_path)

        output_dir = os.path.join(current_dir, "MFCC")
        os.makedirs(output_dir, exist_ok=True)
        if mfcc_features is not None:
            # 准备输出文件名
            output_filename = f"{base_filename}_mfcc.npz"
            output_path = os.path.join(output_dir, output_filename)

            # 保存MFCC特征
            try:
                np.savez(output_path,
                         mfcc=mfcc_features,
                         sample_rate=sample_rate,
                         n_mfcc=N_MFCC,
                         n_mels=N_MELS)

                print(f"  [成功] MFCC特征已保存至: {output_filename}")
                success_count += 1

            except Exception as e:
                print(f"  [错误] 保存文件 {output_path} 失败: {e}")

        print("-" * 50)

    print(f"\nMFCC提取完成！共成功处理 {success_count} 个文件。")
    print(f"所有输出的 _mfcc.npz 文件均已保存在: {output_dir}")

    # 显示MFCC特征示例
    if success_count > 0:
        print("\nMFCC特征示例:")
        print(f"  - 每个文件的MFCC特征形状: {mfcc_features.shape}")
        print(f"  - 特征维度: {mfcc_features.shape[1]} 阶MFCC系数")
        print(f"  - 时间帧数: {mfcc_features.shape[0]} 帧")
        print(f"  - 使用的Mel滤波器数量: {N_MELS}")
        print(f"  - 保留的MFCC系数: {N_MFCC} 阶")


if __name__ == "__main__":
    main()