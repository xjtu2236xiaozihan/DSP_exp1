import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
import scipy.fftpack as fft
import warnings
import pickle
import random
from collections import defaultdict

# =================================================================
# 全局参数
# =================================================================

# 预加重系数
PRE_EMPHASIS_COEFF = 0.97

# MFCC参数
FRAME_LEN_MS = 25
FRAME_SHIFT_MS = 10
N_MELS = 26
N_MFCC = 12

# VAD参数
VAD_FRAME_LEN_MS = 20
VAD_FRAME_SHIFT_MS = 10

# 窗函数类型: 'hamming' (汉明窗), 'hann' (海宁窗), 'boxcar' (矩形窗)
WINDOW_TYPE = 'hamming'

# DTW参数
DTW_DISTANCE_METRIC = 'euclidean'


# =================================================================
# 核心函数
# =================================================================

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def short_term_energy(frame):
    return np.sum(frame ** 2)


def short_term_zcr(frame):
    sgn = np.sign(frame)
    return np.sum(np.abs(np.diff(sgn))) / (2 * len(frame))


def dual_threshold_vad(signal, sample_rate):
    frame_len = int(VAD_FRAME_LEN_MS / 1000 * sample_rate)
    frame_shift = int(VAD_FRAME_SHIFT_MS / 1000 * sample_rate)
    num_samples = len(signal)

    if num_samples < frame_len:
        return 0, len(signal)

    num_frames = int(np.floor((num_samples - frame_len) / frame_shift)) + 1
    if num_frames <= 0:
        return 0, len(signal)

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    frames = signal[indices.astype(np.int32)]

    energy = np.array([short_term_energy(frame) for frame in frames])
    zcr = np.array([short_term_zcr(frame) for frame in frames])
    energy = energy / (np.max(energy) + 1e-10)

    T_E_high, T_E_low, T_ZCR = 0.1, 0.02, 0.15

    high_energy_frames = np.where(energy > T_E_high)[0]
    if len(high_energy_frames) == 0:
        return 0, len(signal)

    start_N3, end_N4 = high_energy_frames[0], high_energy_frames[-1]

    start_N2 = start_N3
    while start_N2 > 0 and energy[start_N2] > T_E_low:
        start_N2 -= 1

    end_N5 = end_N4
    while end_N5 < (num_frames - 1) and energy[end_N5] > T_E_low:
        end_N5 += 1

    start_N1, end_N6 = start_N2, end_N5
    count_threshold = 3

    count = 0
    while start_N1 > 0:
        if zcr[start_N1] < T_ZCR:
            count += 1
        else:
            count = 0
        if count >= count_threshold:
            start_N1 += count
            break
        start_N1 -= 1

    count = 0
    while end_N6 < (num_frames - 1):
        if zcr[end_N6] < T_ZCR:
            count += 1
        else:
            count = 0
        if count >= count_threshold:
            end_N6 -= count
            break
        end_N6 += 1

    start_index = max(0, start_N1 * frame_shift)
    end_index = min(num_samples, end_N6 * frame_shift + frame_len)

    return start_index, end_index


def create_mel_filterbank(sample_rate, n_fft, n_mels=N_MELS):
    fmax = sample_rate // 2
    mel_min, mel_max = hz_to_mel(0), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        left, center, right = fft_bins[m], fft_bins[m + 1], fft_bins[m + 2]
        if left < center:
            filterbank[m, left:center] = np.linspace(0, 1, center - left)
        if center < right:
            filterbank[m, center:right] = np.linspace(1, 0, right - center)
    return filterbank


def extract_mfcc(power_spectrum, sample_rate, n_fft):
    mel_filterbank = create_mel_filterbank(sample_rate, n_fft)
    mel_energy = np.dot(power_spectrum, mel_filterbank.T)
    log_mel_energy = np.log(mel_energy + 1e-8)
    mfcc = fft.dct(log_mel_energy, type=2, axis=1, norm='ortho')
    return mfcc[:, :N_MFCC]


def process_audio_file(file_path):
    try:
        sample_rate, signal = wavfile.read(file_path)
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return None, 0

    if signal.dtype != np.float32:
        signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    if len(signal) == 0:
        return None, 0

    processed_signal = signal - np.mean(signal)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_idx, end_idx = dual_threshold_vad(processed_signal, sample_rate)

    if end_idx - start_idx < 1000:
        return None, sample_rate

    processed_signal = processed_signal[start_idx:end_idx]

    # 预加重
    if PRE_EMPHASIS_COEFF > 0:
        processed_signal = np.append(processed_signal[0],
                                     processed_signal[1:] - PRE_EMPHASIS_COEFF * processed_signal[:-1])

    # 分帧
    frame_len = int(FRAME_LEN_MS / 1000 * sample_rate)
    frame_shift = int(FRAME_SHIFT_MS / 1000 * sample_rate)
    signal_len = len(processed_signal)

    num_frames = 1 + int(np.ceil((signal_len - frame_len) / frame_shift))
    pad_len = (num_frames - 1) * frame_shift + frame_len - signal_len
    padded_signal = np.pad(processed_signal, (0, int(pad_len)), 'constant')

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    frames = padded_signal[indices.astype(np.int32)]

    # 加窗和FFT
    window = get_window(WINDOW_TYPE, frame_len)
    windowed_frames = frames * window
    N_FFT = 2 ** int(np.ceil(np.log2(frame_len)))
    fft_frames = np.fft.fft(windowed_frames, n=N_FFT, axis=1)
    power_spectrum = np.absolute(fft_frames[:, 0:(N_FFT // 2 + 1)]) ** 2

    # 提取MFCC
    mfcc_features = extract_mfcc(power_spectrum, sample_rate, N_FFT)

    return mfcc_features, sample_rate


# =================================================================
# DTW算法
# =================================================================

def dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.sqrt(np.sum((seq1[i - 1] - seq2[j - 1]) ** 2))
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    return dtw_matrix[n, m]


def dtw_classify(test_sample, templates):
    min_avg_distance = float('inf')
    predicted_label = None
    distances = {}

    for label, template_list in templates.items():
        label_distances = [dtw_distance(test_sample, template) for template in template_list]
        avg_distance = np.mean(label_distances)
        distances[label] = avg_distance

        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            predicted_label = label

    return predicted_label, distances


# =================================================================
# 识别器类
# =================================================================

class IsolatedWordRecognizer:
    def __init__(self):
        self.templates = defaultdict(list)
        self.labels = []

    def train_from_files(self, train_files):
        print(f"使用 {len(train_files)} 个文件进行训练...")

        for file_path in train_files:
            try:
                filename = os.path.basename(file_path)
                label = filename.split('_')[0]

                data = np.load(file_path)
                mfcc_features = data['mfcc']

                self.templates[label].append(mfcc_features)
                if label not in self.labels:
                    self.labels.append(label)

            except Exception as e:
                print(f"加载训练文件 {file_path} 失败: {e}")

        print(f"训练完成! 共学习 {len(self.labels)} 个类别")
        for label in self.labels:
            print(f"  {label}: {len(self.templates[label])} 个样本")

        return True

    def predict(self, test_mfcc_features):
        predicted_label, distances = dtw_classify(test_mfcc_features, self.templates)

        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        if len(sorted_distances) >= 2:
            confidence = sorted_distances[1][1] / (sorted_distances[0][1] + 1e-8)
        else:
            confidence = 1.0

        return predicted_label, confidence, distances

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump({
                'templates': dict(self.templates),
                'labels': self.labels
            }, f)
        print(f"模型已保存至: {model_path}")

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.templates = defaultdict(list, data['templates'])
            self.labels = data['labels']
        print(f"模型已加载: {len(self.labels)} 个类别")


# =================================================================
# 数据集管理
# =================================================================

def split_dataset(feature_dir, train_ratio=0.8):
    all_files = glob.glob(os.path.join(feature_dir, "*_mfcc.npz"))

    if not all_files:
        print(f"在 {feature_dir} 中未找到特征文件")
        return [], []

    print(f"找到 {len(all_files)} 个特征文件")

    # 按标签组织文件
    label_files = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        label = filename.split('_')[0]
        if label not in label_files:
            label_files[label] = []
        label_files[label].append(file_path)

    # 划分训练测试集
    train_files, test_files = [], []
    random.seed(42)

    for label, files in label_files.items():
        # random.shuffle(files)
        train_count = max(1, min(len(files) - 1, int(len(files) * train_ratio)))

        train_files.extend(files[:train_count])
        test_files.extend(files[train_count:])

    print(f"训练集: {len(train_files)} 个文件, 测试集: {len(test_files)} 个文件")

    # 显示分布
    print("\n各类别数据分布:")
    for label in label_files:
        train_count = sum(1 for f in train_files if f.split('_')[0] == label)
        test_count = sum(1 for f in test_files if f.split('_')[0] == label)
        print(f"  {label}: 训练{train_count}个, 测试{test_count}个")

    return train_files, test_files


# =================================================================
# 评估函数
# =================================================================

def evaluate_recognizer(recognizer, test_files):
    if not test_files:
        print("没有测试文件")
        return 0

    correct, total = 0, 0

    print(f"开始评估，共 {len(test_files)} 个测试文件")
    print("-" * 40)

    for file_path in test_files:
        try:
            filename = os.path.basename(file_path)
            true_label = filename.split('_')[0]

            data = np.load(file_path)
            test_features = data['mfcc']

            predicted_label, confidence, _ = recognizer.predict(test_features)

            is_correct = (predicted_label == true_label)
            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} {filename}: 真实={true_label}, 预测={predicted_label}, 置信度={confidence:.2f}")

        except Exception as e:
            print(f"处理测试文件 {file_path} 失败: {e}")

    accuracy = correct / total if total > 0 else 0

    print("-" * 40)
    print(f"评估完成! 准确率: {accuracy:.4f} ({correct}/{total})")

    return accuracy


# =================================================================
# 主程序
# =================================================================

def main():
    # 路径设置
    input_wav_path = r"C:\Users\Lenovo\Desktop\数字信号处理\实验1\全组音频_未处理\新建文件夹"
    feature_dir = r"C:\Users\Lenovo\Desktop\数字信号处理\实验2\DSP_exp1-xjtu2236xiaozihan-patch-1\实验2部分\MFCC"
    model_path = 'isolated_word_model.pkl'

    # 确保输出目录存在
    os.makedirs(feature_dir, exist_ok=True)

    # 1. 特征提取
    print("=" * 40)
    print("特征提取")
    print("=" * 40)

    if not os.path.isdir(input_wav_path):
        print(f"错误：输入路径不存在 -> {input_wav_path}")
        return

    wav_files = glob.glob(os.path.join(input_wav_path, "*.wav"))

    if not wav_files:
        print(f"错误：在 {input_wav_path} 中未找到任何 .wav 文件。")
        return

    print(f"找到 {len(wav_files)} 个 .wav 文件")
    success_count = 0

    for file_path in wav_files:
        file_name = os.path.basename(file_path)
        base_filename = os.path.splitext(file_name)[0]
        print(f"处理: {file_name}")

        mfcc_features, sample_rate = process_audio_file(file_path)

        if mfcc_features is not None:
            output_path = os.path.join(feature_dir, f"{base_filename}_mfcc.npz")
            np.savez(output_path, mfcc=mfcc_features, sample_rate=sample_rate)
            print(f"  完成。MFCC特征形状: {mfcc_features.shape}")
            success_count += 1
        else:
            print(f"  失败")

        print("-" * 30)

    print(f"特征提取完成! 共成功处理 {success_count} 个文件")

    # 2. 训练和评估
    print("\n" + "=" * 40)
    print("训练和评估")
    print("=" * 40)

    train_files, test_files = split_dataset(feature_dir)
    if not train_files:
        return

    recognizer = IsolatedWordRecognizer()

    # 训练
    print("\n训练模型...")
    recognizer.train_from_files(train_files)
    recognizer.save_model(model_path)

    # 评估
    print("\n评估模型...")
    accuracy = evaluate_recognizer(recognizer, test_files)

    print(f"\n最终准确率: {accuracy:.4f}")


if __name__ == "__main__":
    main()