import webrtcvad
from pydub import AudioSegment
import os
import wave
import contextlib

# --- 静态配置参数（主要用于文件批处理时的默认值） ---
# 原始语音文件所在的输入文件夹
INPUT_DIR = r"C:\Users\肖梓涵\Desktop\全组音频_未处理\新建文件夹"
# 处理后的语音文件保存的输出文件夹
OUTPUT_DIR = r"C:\Users\肖梓涵\Desktop\全组音频"

# VAD 模式：0 (非侵略性) 到 3 (最侵略性)
VAD_MODE = 3
# 帧长度（毫秒）。WebRTC VAD 接受 10, 20 或 30 ms。
FRAME_DURATION_MS = 30 
# 沉默时长阈值（帧数）。用于确定一个声音片段结束后的“静默期”。
SILENCE_THRESHOLD_FRAMES = 10 
# 最小语音片段长度（毫秒）。小于此长度的语音片段将被忽略。
MIN_SPEECH_SEGMENT_MS = 50 


def read_wave(path):
    """读取 WAV 文件并返回音频数据、采样率和帧数。"""
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "VAD 需要单声道文件。"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, "VAD 需要 16-bit PCM (2字节) 文件。"
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), "不支持的采样率。"
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio_data, sample_rate):
    """将 PCM 数据写入 WAV 文件。"""
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setnframes(len(audio_data) // 2) # 每帧2字节
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)


# --- 建议的新核心接口：vad_process ---
# 实现了“输入 pcm_data 和 sample_rate，返回 segmented_audio_data”的纯数据处理功能
def vad_process(pcm_data: bytes, sample_rate: int,
                vad_mode: int = VAD_MODE, 
                frame_duration_ms: int = FRAME_DURATION_MS,
                silence_threshold_frames: int = SILENCE_THRESHOLD_FRAMES,
                min_speech_segment_ms: int = MIN_SPEECH_SEGMENT_MS) -> bytes:
    """
    功能描述：只进行 VAD 核心处理。输入原始 PCM 数据和采样率，返回提取的有效语音数据。
    这实现了您团队集成模块所需的数据接口标准。
    """
    
    # 验证输入数据的基本要求
    assert sample_rate in (8000, 16000, 32000, 48000), "不支持的采样率。"
    assert frame_duration_ms in (10, 20, 30), "帧长必须是 10, 20 或 30 ms。"

    # 初始化 VAD
    vad = webrtcvad.Vad(vad_mode)
    
    # 计算帧大小
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    # 将 PCM 数据切割成帧
    frames = []
    # 2 bytes per sample for 16-bit PCM
    for i in range(0, len(pcm_data), frame_size * 2): 
        frame = pcm_data[i:i + frame_size * 2]
        if len(frame) < frame_size * 2:
            break # 忽略最后一帧不完整的
        frames.append(frame)

    # 2. VAD 检测与切片逻辑
    segments = [] # 存储 (start_frame, end_frame)
    current_segment_start_frame = -1
    silence_counter = 0

    for i, frame in enumerate(frames):
        # is_speech 返回 True 或 False
        is_speech = vad.is_speech(frame, sample_rate)
        
        if is_speech:
            silence_counter = 0
            if current_segment_start_frame == -1:
                # 语音开始
                current_segment_start_frame = i
        else:
            if current_segment_start_frame != -1:
                # 计数沉默帧
                silence_counter += 1
                if silence_counter >= silence_threshold_frames:
                    # 沉默超过阈值，视为语音结束
                    segment_end_frame = i - silence_threshold_frames
                    segments.append((current_segment_start_frame, segment_end_frame))
                    current_segment_start_frame = -1
                    silence_counter = 0 # 重置沉默计数器

    # 处理最后一个片段（如果文件以语音结束）
    if current_segment_start_frame != -1:
        segments.append((current_segment_start_frame, len(frames) - 1))

    # 3. 合并语音片段
    processed_audio_data = b''
    total_duration_ms = 0
    
    for start_frame, end_frame in segments:
        segment_duration_frames = end_frame - start_frame + 1
        segment_duration_ms = segment_duration_frames * frame_duration_ms
        
        if segment_duration_ms >= min_speech_segment_ms:
            # 提取 PCM 数据
            start_byte = start_frame * frame_size * 2
            end_byte = (end_frame + 1) * frame_size * 2
            processed_audio_data += pcm_data[start_byte:end_byte]
            total_duration_ms += segment_duration_ms
            
    # 返回提取出的语音 PCM 数据 (segmented_audio_data)
    return processed_audio_data


# --- 文件处理接口（保留原有批处理能力） ---
def vad_process_file(input_path, output_path):
    """对单个 WAV 文件进行 VAD 处理并保存结果。"""
    try:
        # 1. 读取并验证 WAV 文件
        pcm_data, sample_rate = read_wave(input_path)
    except AssertionError as e:
        print(f"  [跳过] 文件 {os.path.basename(input_path)} 格式不符合 VAD 要求: {e}")
        return
    except Exception as e:
        print(f"  [错误] 读取文件 {os.path.basename(input_path)} 失败: {e}")
        return
    
    print(f"  采样率: {sample_rate} Hz, 原始时长: {len(pcm_data) / (sample_rate * 2):.2f}s")
    
    # **关键步骤：调用新的核心接口**
    processed_audio_data = vad_process(
        pcm_data, 
        sample_rate, 
        vad_mode=VAD_MODE, # 使用静态配置参数作为默认值
        frame_duration_ms=FRAME_DURATION_MS,
        silence_threshold_frames=SILENCE_THRESHOLD_FRAMES,
        min_speech_segment_ms=MIN_SPEECH_SEGMENT_MS
    )

    # 4. 保存文件
    if processed_audio_data:
        write_wave(output_path, processed_audio_data, sample_rate)
        
        # 计算处理后时长
        total_duration_ms = len(processed_audio_data) / (sample_rate * 2) * 1000
        print(f"  [成功] 提取并保存。处理后时长: {total_duration_ms / 1000:.2f}s")
    else:
        print("  [警告] 未检测到有效语音片段，未生成文件。")


# --- 主程序入口（保持不变） ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出文件夹: {OUTPUT_DIR}")

    print(f"--- 开始处理文件夹: {INPUT_DIR} ---")

    # 循环遍历输入文件夹下的所有 WAV 文件
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(INPUT_DIR, filename)
            
            # 生成输出文件名
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_vad_processed{ext}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            print(f"\n-> 正在处理文件: {filename}")
            vad_process_file(input_path, output_path)

    print("\n--- 所有文件处理完成 ---")

if __name__ == "__main__":
    # 使用 pydub 检查 ffmpeg/libav 并修复潜在的依赖问题
    try:
        AudioSegment.empty()
    except Exception:
        print("警告: pydub 无法初始化。虽然本脚本主要使用 webrtcvad 和 wave 库，")
        print("但如果你的 wav 文件格式不标准，可能需要安装 ffmpeg 或 libav。")
        print("如果一切正常，请忽略此警告。")
        
    main()