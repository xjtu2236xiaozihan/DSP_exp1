import torch
import soundfile as sf
from pathlib import Path
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

# --- 配置 ---
MODEL_DIR = Path("../models/speecht5_tts")
VOCODER_DIR = Path("../models/speecht5_hifigan")
OUTPUT_DIR = Path("./data_clean")
SAMPLES_PER_DIGIT = 10
SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 我们需要一个映射：
# 文件名使用数字 (key), TTS输入使用单词 (value)
TEXT_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine"
}

# --- 脚本 ---

def main():
    """
    使用SpeechT5模型生成数字0-9的语音。
    使用单词（"zero"）作为模型输入，但使用数字（"0"）作为文件名。
    """
    print(f"Using device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型和处理器
    processor = SpeechT5Processor.from_pretrained(MODEL_DIR)
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_DIR).to(DEVICE)
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_DIR).to(DEVICE)

    # 2. 加载说话人嵌入 (xvector)
    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", 
        split="validation", 
        trust_remote_code=True
    )
    speaker_embeddings = torch.tensor(
        embeddings_dataset[7306]["xvector"]
    ).unsqueeze(0).to(DEVICE)

    # 3. 生成语音
    print(f"Generating audio files in {OUTPUT_DIR}...")
    total_files = 0
    
    # ！！！ 修改循环 ！！！
    # 遍历 TEXT_MAP
    for digit, text_to_speak in TEXT_MAP.items():
        print(f"  Generating speech for: '{digit}' (as '{text_to_speak}')")
        for i in range(SAMPLES_PER_DIGIT):
            # 准备输入
            # 使用 text_to_speak ("zero") 作为输入
            inputs = processor(text=text_to_speak, return_tensors="pt").to(DEVICE)

            # 生成语音
            speech = model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings, 
                vocoder=vocoder
            )

            # 构造文件名
            # 使用 digit ("0") 作为文件名
            filename = f"{digit}_{i}_clean.wav"
            output_path = OUTPUT_DIR / filename
            sf.write(
                output_path, 
                speech.cpu().numpy(), 
                samplerate=SAMPLING_RATE
            )
            total_files += 1

    print(f"\nGeneration complete. {total_files} files created.")

if __name__ == "__main__":
    main()