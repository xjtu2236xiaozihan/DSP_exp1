import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from scipy.io import wavfile

# 设置中文字体和图形样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_mel_filterbank(sample_rate=16000, n_fft=512, n_mels=26):
    """
    绘制Mel滤波器组
    """

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    # 创建Mel滤波器组
    fmax = sample_rate // 2
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # 创建频率轴
    freqs = np.linspace(0, fmax, n_fft // 2 + 1)

    plt.figure(figsize=(12, 6))

    # 绘制每个三角形滤波器
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        # 创建三角形
        triangle = np.zeros_like(freqs)
        mask_left = (freqs >= left) & (freqs <= center)
        mask_right = (freqs >= center) & (freqs <= right)

        if np.any(mask_left):
            triangle[mask_left] = np.linspace(0, 1, np.sum(mask_left))
        if np.any(mask_right):
            triangle[mask_right] = np.linspace(1, 0, np.sum(mask_right))

        plt.plot(freqs, triangle, label=f'Filter {i + 1}' if i < 3 else "")

    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.title(f'Mel滤波器组 ({n_mels}个滤波器)')
    plt.xlim(0, fmax)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_mfcc_features(mfcc_file_path, original_wav_path=None):
    """
    可视化单个文件的MFCC特征

    参数:
        mfcc_file_path: MFCC特征文件路径
        original_wav_path: 原始音频文件路径（可选）
    """
    try:
        # 加载MFCC特征
        data = np.load(mfcc_file_path)
        mfcc_features = data['mfcc']
        sample_rate = data['sample_rate']

        print(f"MFCC特征形状: {mfcc_features.shape}")
        print(f"采样率: {sample_rate} Hz")

        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. MFCC热力图
        im = axes[0, 0].imshow(mfcc_features.T, aspect='auto', origin='lower',
                               cmap='viridis', interpolation='nearest')
        axes[0, 0].set_title('MFCC特征热力图')
        axes[0, 0].set_xlabel('时间帧')
        axes[0, 0].set_ylabel('MFCC系数')
        plt.colorbar(im, ax=axes[0, 0])

        # 2. 前几个MFCC系数的时间变化
        num_coeffs_to_plot = min(6, mfcc_features.shape[1])
        for i in range(num_coeffs_to_plot):
            axes[0, 1].plot(mfcc_features[:, i], label=f'MFCC {i + 1}')
        axes[0, 1].set_title('MFCC系数时间变化')
        axes[0, 1].set_xlabel('时间帧')
        axes[0, 1].set_ylabel('系数值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. MFCC统计分布
        mfcc_mean = np.mean(mfcc_features, axis=0)
        mfcc_std = np.std(mfcc_features, axis=0)

        x_pos = np.arange(len(mfcc_mean))
        axes[1, 0].bar(x_pos, mfcc_mean, yerr=mfcc_std, capsize=5,
                       alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1, 0].set_title('MFCC系数均值和标准差')
        axes[1, 0].set_xlabel('MFCC系数序号')
        axes[1, 0].set_ylabel('系数值')
        axes[1, 0].set_xticks(x_pos)

        # 4. MFCC的3D曲面图（可选）
        if mfcc_features.shape[0] > 10:  # 确保有足够的时间帧
            X, Y = np.meshgrid(np.arange(mfcc_features.shape[0]),
                               np.arange(mfcc_features.shape[1]))

            ax_3d = fig.add_subplot(224, projection='3d')
            surf = ax_3d.plot_surface(X.T, Y.T, mfcc_features,
                                      cmap=cm.viridis, linewidth=0,
                                      antialiased=True, alpha=0.8)
            ax_3d.set_title('MFCC 3D曲面图')
            ax_3d.set_xlabel('时间帧')
            ax_3d.set_ylabel('MFCC系数')
            ax_3d.set_zlabel('系数值')
            fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5)
        else:
            # 如果时间帧太少，显示相关系数矩阵
            corr_matrix = np.corrcoef(mfcc_features.T)
            im_corr = axes[1, 1].imshow(corr_matrix, cmap='coolwarm',
                                        vmin=-1, vmax=1, aspect='auto')
            axes[1, 1].set_title('MFCC系数相关性矩阵')
            axes[1, 1].set_xlabel('MFCC系数')
            axes[1, 1].set_ylabel('MFCC系数')
            plt.colorbar(im_corr, ax=axes[1, 1])

        plt.tight_layout()

        # 添加总标题
        file_name = os.path.basename(mfcc_file_path)
        fig.suptitle(f'MFCC特征可视化 - {file_name}', fontsize=14, y=0.98)
        plt.subplots_adjust(top=0.93)

        plt.show()

        # 如果有原始音频文件，显示波形图
        if original_wav_path and os.path.exists(original_wav_path):
            plot_audio_waveform(original_wav_path, mfcc_features.shape[0])

    except Exception as e:
        print(f"可视化MFCC特征时出错: {e}")


def plot_audio_waveform(wav_file_path, num_frames):
    """
    绘制原始音频波形图
    """
    try:
        sample_rate, signal = wavfile.read(wav_file_path)

        # 如果是双声道，取平均值
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        # 创建时间轴
        time_axis = np.linspace(0, len(signal) / sample_rate, len(signal))

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, signal, linewidth=0.5, color='blue', alpha=0.7)
        plt.title('原始音频波形')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.grid(True, alpha=0.3)

        # 标记帧的位置（近似）
        frame_duration = len(signal) / sample_rate / num_frames
        for i in range(0, num_frames, max(1, num_frames // 10)):  # 每10帧标记一个
            frame_time = i * frame_duration
            plt.axvline(x=frame_time, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"绘制音频波形时出错: {e}")


def compare_multiple_mfcc(mfcc_files, max_files=5):
    """
    比较多个文件的MFCC特征
    """
    if len(mfcc_files) > max_files:
        mfcc_files = mfcc_files[:max_files]
        print(f"注意: 只显示前{max_files}个文件的比较")

    plt.figure(figsize=(15, 10))

    # 1. 比较MFCC均值
    plt.subplot(2, 2, 1)
    for i, mfcc_file in enumerate(mfcc_files):
        try:
            data = np.load(mfcc_file)
            mfcc_features = data['mfcc']
            mfcc_mean = np.mean(mfcc_features, axis=0)

            file_label = os.path.basename(mfcc_file).replace('_mfcc.npz', '')
            plt.plot(mfcc_mean, label=file_label, linewidth=2, alpha=0.8)
        except Exception as e:
            print(f"读取文件 {mfcc_file} 时出错: {e}")

    plt.title('不同音频的MFCC均值比较')
    plt.xlabel('MFCC系数序号')
    plt.ylabel('均值')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. 比较MFCC标准差
    plt.subplot(2, 2, 2)
    for i, mfcc_file in enumerate(mfcc_files):
        try:
            data = np.load(mfcc_file)
            mfcc_features = data['mfcc']
            mfcc_std = np.std(mfcc_features, axis=0)

            file_label = os.path.basename(mfcc_file).replace('_mfcc.npz', '')
            plt.plot(mfcc_std, label=file_label, linewidth=2, alpha=0.8)
        except Exception as e:
            print(f"读取文件 {mfcc_file} 时出错: {e}")

    plt.title('不同音频的MFCC标准差比较')
    plt.xlabel('MFCC系数序号')
    plt.ylabel('标准差')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. 热力图比较（第一个文件的MFCC）
    plt.subplot(2, 2, 3)
    if mfcc_files:
        try:
            data = np.load(mfcc_files[0])
            mfcc_features = data['mfcc']
            im = plt.imshow(mfcc_features.T, aspect='auto', origin='lower',
                            cmap='viridis', interpolation='nearest')
            plt.title(f'MFCC热力图 - {os.path.basename(mfcc_files[0])}')
            plt.xlabel('时间帧')
            plt.ylabel('MFCC系数')
            plt.colorbar(im)
        except Exception as e:
            print(f"绘制热力图时出错: {e}")

    # 4. 特征分布直方图（所有MFCC系数的分布）
    plt.subplot(2, 2, 4)
    all_mfcc_values = []
    for mfcc_file in mfcc_files:
        try:
            data = np.load(mfcc_file)
            mfcc_features = data['mfcc']
            all_mfcc_values.extend(mfcc_features.flatten())
        except Exception as e:
            print(f"读取文件 {mfcc_file} 时出错: {e}")

    if all_mfcc_values:
        plt.hist(all_mfcc_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('MFCC系数值分布')
        plt.xlabel('MFCC系数值')
        plt.ylabel('频数')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 提供交互式可视化选项
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    feature_path = os.path.join(current_dir, "MFCC")

    # 检查路径是否存在
    if not os.path.isdir(feature_path):
        print(f"错误：路径不存在 -> {feature_path}")
        return

    # 查找所有MFCC特征文件
    mfcc_files = glob.glob(os.path.join(feature_path, "*_mfcc.npz"))

    if not mfcc_files:
        print("未找到MFCC特征文件，请先运行MFCC提取代码。")
        return

    print(f"找到 {len(mfcc_files)} 个MFCC特征文件")

    while True:
        print("\n" + "=" * 50)
        print("MFCC特征可视化工具")
        print("=" * 50)
        print("1. 显示Mel滤波器组")
        print("2. 可视化单个文件的MFCC特征")
        print("3. 比较多个文件的MFCC特征")
        print("4. 退出")

        choice = input("\n请选择操作 (1-4): ").strip()

        if choice == '1':
            # 显示Mel滤波器组
            plot_mel_filterbank()

        elif choice == '2':
            # 显示文件列表供选择
            print("\n可用的MFCC文件:")
            for i, file_path in enumerate(mfcc_files):
                print(f"{i + 1}. {os.path.basename(file_path)}")

            try:
                file_idx = int(input("\n请选择文件编号: ")) - 1
                if 0 <= file_idx < len(mfcc_files):
                    selected_file = mfcc_files[file_idx]

                    # 尝试找到对应的原始音频文件
                    base_name = os.path.basename(selected_file).replace('_mfcc.npz', '')
                    wav_file = os.path.join(feature_path, f"{base_name}.wav")
                    wav_path = wav_file if os.path.exists(wav_file) else None

                    visualize_mfcc_features(selected_file, wav_path)
                else:
                    print("无效的文件编号")
            except ValueError:
                print("请输入有效的数字")

        elif choice == '3':
            # 比较多个文件
            if len(mfcc_files) >= 2:
                compare_multiple_mfcc(mfcc_files)
            else:
                print("需要至少2个MFCC文件进行比较")

        elif choice == '4':
            print("退出可视化工具")
            break

        else:
            print("无效的选择，请重新输入")


if __name__ == "__main__":
    main()
