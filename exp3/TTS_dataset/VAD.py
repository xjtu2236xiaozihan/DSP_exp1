import webrtcvad
from pydub import AudioSegment
import os
import wave
import contextlib

# --- 静态配置参数（主要用于文件批处理时的默认值） ---
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 原始语音文件所在的输入文件夹列表
INPUT_DIRS = [
    os.path.join(current_dir, "data_clean"),
    os.path.join(current_dir, "data_noise")
]
# 处理后的语音文件保存的输出文件夹
OUTPUT_DIR = os.path.join(current_dir, "VAD")

# VAD 模式：0 (非侵略性) 到 3 (最侵略性)
VAD_MODE = 3
# 帧长度（毫秒）。WebRTC VAD 接受 10, 20 或 30 ms。
FRAME_DURATION_MS = 30 
# 沉默时长阈值（帧数）。用于确定一个声音片段结束后的“静默期”。
SILENCE_THRESHOLD_FRAMES = 10 
# 最小语音片段长度（毫秒）。小于此长度的语音片段将被忽略。
MIN_SPEECH_SEGMENT_MS = 50 


def convert_audio_to_wav_format(input_path, temp_path):
    """使用pydub将音频文件转换为VAD兼容的格式"""
    audio = AudioSegment.from_file(input_path)
    
    # 转换为单声道
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # 转换为16kHz采样率（WebRTC VAD推荐）
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)   
    # 转换为16-bit PCM
    audio = audio.set_sample_width(2)
    # 导出为WAV格式
    audio.export(temp_path, format="wav")

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
    """对单个音频文件进行 VAD 处理并保存结果。"""
    import tempfile
    
    temp_wav_path = None
    try:
        # 1. 尝试直接读取WAV文件
        try:
            pcm_data, sample_rate = read_wave(input_path)
            print(f"  采样率: {sample_rate} Hz, 原始时长: {len(pcm_data) / (sample_rate * 2):.2f}s")
        except (AssertionError, Exception) as e:
            print(f"  [转换] 文件格式不符合VAD要求，正在转换: {e}")
            
            # 创建临时文件进行格式转换
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav_path = temp_file.name
            
            # 转换音频格式
            if not convert_audio_to_wav_format(input_path, temp_wav_path):
                print(f"  [跳过] 无法转换文件 {os.path.basename(input_path)}")
                return
            
            # 读取转换后的文件
            pcm_data, sample_rate = read_wave(temp_wav_path)
            print(f"  [转换完成] 采样率: {sample_rate} Hz, 原始时长: {len(pcm_data) / (sample_rate * 2):.2f}s")
    
    except Exception as e:
        print(f"  [错误] 处理文件 {os.path.basename(input_path)} 失败: {e}")
        return
    finally:
        # 清理临时文件
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except:
                pass
    
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


# --- 主程序入口 ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 处理所有输入目录
    for input_dir in INPUT_DIRS:
        if not os.path.exists(input_dir):
            print(f"目录不存在，跳过: {input_dir}")
            continue
            
        print(f"处理文件夹: {input_dir}")
        
        # 获取当前目录的所有音频文件
        audio_files = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.aac')):
                audio_files.append(filename)
        
        if not audio_files:
            print(f"在 {input_dir} 中未找到音频文件")
            continue
        
        # 处理当前目录的所有音频文件
        for filename in audio_files:
            input_path = os.path.join(input_dir, filename)
            
            # 生成输出文件名，添加目录标识
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_vad_processed.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            print(f"处理: {filename}")
            vad_process_file(input_path, output_path)
    
    print("处理完成")

if __name__ == "__main__":
    AudioSegment.empty()
    main()