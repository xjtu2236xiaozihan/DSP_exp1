import numpy as np
import soundfile as sf
from pathlib import Path

# --- 配置 ---
INPUT_DIR = Path("./data_clean")
OUTPUT_DIR = Path("./data_noise")
NOISE_LEVEL = 0.008 # 噪声强度 (0.0 到 1.0)

# --- 脚本 ---

def main():
    """
    向干净的音频文件添加高斯白噪声。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    clean_files = list(INPUT_DIR.glob("*_clean.wav"))
    print(f"Found {len(clean_files)} files to process in {INPUT_DIR}.")

    for clean_path in clean_files:
        # 1. 读取音频
        audio, sr = sf.read(clean_path)

        # 2. 生成高斯噪声
        # 噪声的标准差基于音频的标准差，通过 NOISE_LEVEL 缩放
        noise_std = np.std(audio) * NOISE_LEVEL
        noise = np.random.normal(0, noise_std, audio.shape)

        # 3. 混合并钳位 (Clipping)
        noisy_audio = audio + noise
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        # 4. 构造文件名并保存
        file_name = clean_path.name.replace("_clean.wav", "_noise.wav")
        output_path = OUTPUT_DIR / file_name
        sf.write(output_path, noisy_audio, sr)

    print(f"Noise addition complete. Files saved in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()