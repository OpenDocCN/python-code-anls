# `.\voice_assistant_llm\app.py`

```
import os  # 导入操作系统相关的模块
import wave  # 导入处理 WAV 文件的模块
import pyaudio  # 导入 PyAudio 音频处理库
import numpy as np  # 导入数值计算库 NumPy
from scipy.io import wavfile  # 从 SciPy 中导入 WAV 文件处理模块 wavfile
from faster_whisper import WhisperModel  # 导入自定义的 WhisperModel 类

import voice_service as vs  # 导入声音服务模块 voice_service
from rag.AIVoiceAssistant import AIVoiceAssistant  # 从 rag 包中导入 AIVoiceAssistant 类

DEFAULT_MODEL_SIZE = "medium"  # 默认模型大小为 medium
DEFAULT_CHUNK_LENGTH = 10  # 默认音频块长度为 10 秒

ai_assistant = AIVoiceAssistant()  # 创建 AIVoiceAssistant 实例


def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))  # 计算音频数据的最大振幅
    return max_amplitude <= max_amplitude_threshold  # 返回是否静音的布尔值


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []  # 存储录制的音频帧
    for _ in range(0, int(16000 / 1024 * chunk_length)):  # 按指定长度录制音频块
        data = stream.read(1024)  # 从音频流中读取数据
        frames.append(data)  # 将数据帧添加到 frames 列表中

    temp_file_path = 'temp_audio_chunk.wav'  # 临时保存音频块的文件路径
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # 设置音频通道数为 1
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # 设置采样宽度
        wf.setframerate(16000)  # 设置帧速率
        wf.writeframes(b''.join(frames))  # 写入音频数据

    # 检查录制的音频块是否包含静音
    try:
        samplerate, data = wavfile.read(temp_file_path)  # 读取临时文件的采样率和数据
        if is_silence(data):  # 判断是否静音
            os.remove(temp_file_path)  # 如果是静音，删除临时文件
            return True  # 返回静音标志
        else:
            return False  # 返回非静音标志
    except Exception as e:
        print(f"Error while reading audio file: {e}")  # 如果读取文件时出错，打印错误信息
        return False  # 返回 False 表示发生错误


def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)  # 使用模型进行音频转录
    transcription = ' '.join(segment.text for segment in segments)  # 将转录结果连接成字符串
    return transcription  # 返回转录的文本


def main():
    
    model_size = DEFAULT_MODEL_SIZE + ".en"  # 设置模型大小
    model = WhisperModel(model_size, device="cuda", compute_type="float16", num_workers=10)  # 创建 WhisperModel 实例
    
    audio = pyaudio.PyAudio()  # 创建 PyAudio 实例
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000,  # 打开音频流
                        input=True, frames_per_buffer=1024)
    customer_input_transcription = ""  # 初始化客户输入转录文本

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"  # 指定存储音频块的文件名
            
            # 录制音频块
            print("_")
            if not record_audio_chunk(audio, stream):  # 如果录制的音频块不是静音
                # 转录音频
                transcription = transcribe_audio(model, chunk_file)  # 对音频进行转录
                os.remove(chunk_file)  # 删除临时音频文件
                print("Customer:{}".format(transcription))  # 打印客户的转录文本
                
                # 将客户输入添加到转录中
                customer_input_transcription += "Customer: " + transcription + "\n"
                
                # 处理客户输入并从 AI 助手获取响应
                output = ai_assistant.interact_with_llm(transcription)  # 与 AI 助手交互
                if output:  # 如果有输出
                    output = output.lstrip()  # 去除输出字符串左侧的空格
                    vs.play_text_to_speech(output)  # 播放 AI 助手的响应文本
                    print("AI Assistant:{}".format(output))  # 打印 AI 助手的响应文本

    except KeyboardInterrupt:  # 捕获键盘中断异常
        print("\nStopping...")  # 打印停止提示信息

    finally:
        stream.stop_stream()  # 停止音频流
        stream.close()  # 关闭音频流
        audio.terminate()  # 终止 PyAudio 实例

if __name__ == "__main__":
    main()  # 调用主函数
```