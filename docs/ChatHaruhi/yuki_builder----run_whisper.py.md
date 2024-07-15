# `.\Chat-Haruhi-Suzumiya\yuki_builder\run_whisper.py`

```py
#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
__ToDo："transcribe video to srt via OpenAI Whisper "
__info:"ASR + simplied chinese + noise reduced"
__author: "Aria:(https://github.com/ariafyy)"
"""

# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import os  # 提供与操作系统交互的功能
import pathlib  # 提供处理文件和目录路径的类
import torch  # 用于机器学习和深度学习任务
from typing import Iterator, TextIO  # 引入类型提示
try:
    import whisper  # 导入Whisper模块，用于语音识别
except ImportError:
    print("check requirements: yuki_builder/requirements_run_whisper.txt")  # 打印导入错误信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU（如果可用）或CPU
from hanziconv import HanziConv  # 导入繁简体中文转换库
from subprocess import CalledProcessError, run  # 导入子进程管理模块
import numpy as np  # 导入处理数组和矩阵的库
SAMPLE_RATE = 16000  # 设置音频采样率为16000Hz
TRANSCRIBE_MODE = ' '  # 设置转录模式（目前为空格，可选项为'noisereduce'）

class Video2Subtitles(object):
    def __init__(self):
        MODEL_WHISPER = "medium"  # 定义Whisper模型的大小或本地路径
        WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]  # 可用的Whisper模型列表
        print("---loading model in your local path or downloading now---")  # 打印模型加载信息
        self.model = whisper.load_model(MODEL_WHISPER)  # 加载指定的Whisper模型

    def srt_format_timestamp(self, seconds: float):
        assert seconds >= 0, "non-negative timestamp expected"  # 断言确保时间戳为非负数
        milliseconds = round(seconds * 1000.0)  # 将秒转换为毫秒

        hours = milliseconds // 3_600_000  # 计算小时部分
        milliseconds -= hours * 3_600_000  # 减去小时部分的毫秒

        minutes = milliseconds // 60_000  # 计算分钟部分
        milliseconds -= minutes * 60_000  # 减去分钟部分的毫秒

        seconds = milliseconds // 1_000  # 计算秒部分
        milliseconds -= seconds * 1_000  # 减去秒部分的毫秒

        return (f"{hours}:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"  # 返回格式化后的时间戳字符串

    def write_srt(self, transcript: Iterator[dict], file: TextIO):
        count = 0  # 初始化段落数
        for segment in transcript:  # 遍历转录文本中的每个段落
            count += 1  # 段落计数加一
            print(
                f"{self.srt_format_timestamp(segment['start'])} --> {self.srt_format_timestamp(segment['end'])}\n"
                f"{self.trad2simp(segment['text']).replace('-->', '->').strip()}\n",
                file=file,
                flush=True,
            )  # 打印格式化的时间戳和转换后的文本，写入到指定文件中

    def trad2simp(self,text):
        """
        # traditional chinese into simplified chinese
        :param text: 
        :return: 
        """
        simp = HanziConv.toSimplified(text)  # 将繁体中文转换为简体中文
        return simp  # 返回转换后的文本
    def transcribe(self, input_video: str, srt_folder: str):
        # 设定字幕格式为 SRT
        subtitle_format = "srt"
        # 设定语言为中文
        lang = "zh"
        # 是否显示详细信息
        verbose = True
        # 检查CUDA设备是否可用
        DEVICE = torch.cuda.is_available()
        # 获取模型对象
        model = self.model
        # 如果输入视频是字符串，则直接使用，否则获取其文件名
        input_video_ = input_video if isinstance(input_video, str) else input_video.name
        # 根据 TRANSCRIBE_MODE 的设置选择处理音频的方式
        if TRANSCRIBE_MODE == 'noisereduce':
            # 对音频进行降噪处理
            audio = self.audio_denoise(input_video_)
        else:
            audio = input_video_
        # 使用模型进行语音转录
        result = model.transcribe(
            audio=audio,
            task="transcribe",
            language=lang,
            verbose=verbose,
            initial_prompt=None,
            word_timestamps=True,
            no_speech_threshold=0.95,
            fp16=DEVICE
        )
        # 创建存放字幕文件的文件夹，如果不存在则创建
        os.makedirs(srt_folder, exist_ok=True)
        # 拼接字幕文件的路径
        subtitle_file = srt_folder + "/" + pathlib.Path(input_video).stem + "." + subtitle_format
        # 如果字幕格式为 SRT，则将结果写入到 SRT 文件中
        if subtitle_format == "srt":
            with open(subtitle_file, "w") as srt:
                self.write_srt(result["segments"], file=srt)
        # 打印生成的字幕文件路径
        print("\nsubtitle_file:", subtitle_file, "\n")
        # 返回生成的字幕文件路径
        return subtitle_file

    def load_audio(self, file: str, sr: int = SAMPLE_RATE):
        """
        Requires the ffmpeg CLI in PATH.
        fmt: off
        """
        # 构建用于加载音频的 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]
        # fmt: on
        try:
            # 执行 ffmpeg 命令并捕获输出
            out = run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            # 如果执行失败，则抛出异常
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        # 将输出转换为 numpy 数组，然后进行数据类型转换和标准化
        data = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        # 返回音频数据
        return data

    def audio_denoise(self, input_audio: str, ):
        """
        # reduce noise
        """
        try:
            # 导入 noisereduce 库
            import noisereduce as nr
        except ImportError:
            # 提示用户安装 noisereduce 库
            print("pip install noisereduce")
        # 设定采样率
        rate = SAMPLE_RATE
        # 加载音频数据
        data = self.load_audio(input_audio)
        # 使用 noisereduce 库对音频进行降噪处理
        reduced_audio = nr.reduce_noise(y=data, sr=rate)
        # 返回降噪后的音频数据
        return reduced_audio
def run_whisper(args):

    # 如果 verbose 参数为 True，则打印 'runing whisper'
    if args.verbose:
        print('runing whisper')

    # 检查输入的视频文件是否存在
    # 如果不存在，则打印 'input_video is not exist' 并返回
    if not os.path.isfile(args.input_video):
        print('input_video is not exist')
        return
    
    # 检查 srt_folder 是否为文件夹
    # 如果不存在，则打印 'warning srt_folder is not exist'
    # 并创建 srt_folder 文件夹，然后打印 'create folder' 后接文件夹名
    if not os.path.isdir(args.srt_folder):
        print('warning srt_folder is not exist')
        os.mkdir(args.srt_folder)
        print('create folder', args.srt_folder)

    # 运行 whisper
    # 将输入文件和 srt 文件夹路径传递给 Video2Subtitles 类的 transcribe 方法
    input_file = args.input_video
    srt_folder = args.srt_folder
    result = Video2Subtitles().transcribe(input_file, srt_folder)
    return result


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='video to chinese srt with medium ',
        epilog='author:Aria(https://github.com/ariafyy)'
    )
    # 添加 verbose 参数，类型为布尔值，作为存储动作
    parser.add_argument("verbose", type=bool, action="store")
    # 添加 input_video 参数，默认为 'input_file'，类型为字符串，必需的参数，帮助信息为 "video path"
    parser.add_argument('--input_video', default='input_file', type=str, required=True, help="video path")
    # 添加 srt_folder 参数，默认为 'out_folder'，类型为字符串，必需的参数，帮助信息为 "srt path"
    parser.add_argument('--srt_folder', default='out_folder', type=str, required=True, help="srt path")
    # 解析命令行参数
    args = parser.parse_args()
    # 打印帮助信息
    parser.print_help()
    # 运行 run_whisper 函数，并传入解析后的参数对象 args
    run_whisper(args)
```