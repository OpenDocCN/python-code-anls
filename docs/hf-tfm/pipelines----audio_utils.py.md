# `.\transformers\pipelines\audio_utils.py`

```
# 导入模块和库
# 2023 年版权声明，HuggingFace 团队保留所有权利
import datetime  # 导入处理日期时间的模块
import platform  # 导入获取平台信息的模块
import subprocess  # 导入子进程管理模块
from typing import Optional, Tuple, Union  # 导入类型提示相关模块

import numpy as np  # 导入数值计算库 NumPy


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    Helper function to read an audio file through ffmpeg.
    """
    # 设置音频文件的采样率和声道数
    ar = f"{sampling_rate}"
    ac = "1"
    format_for_conversion = "f32le"  # 音频格式为 32 位浮点数（小端）
    # 构建执行 FFmpeg 命令所需参数列表
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]

    try:
        # 执行 FFmpeg 命令以读取音频文件
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)  # 通过标准输入流发送音频数据并获取输出流
    except FileNotFoundError as error:
        # 如果找不到 FFmpeg，则抛出错误
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error
    out_bytes = output_stream[0]  # 获取输出流中的字节数据
    audio = np.frombuffer(out_bytes, np.float32)  # 将输出流中的字节数据转换为 NumPy 数组
    if audio.shape[0] == 0:
        # 如果音频数据为空，则抛出错误
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio  # 返回读取的音频数据的 NumPy 数组


def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
):
    """
    Helper function to read raw microphone data.
    """
    ar = f"{sampling_rate}"  # 设置麦克风音频采样率
    ac = "1"  # 设置声道数为 1
    if format_for_conversion == "s16le":
        size_of_sample = 2  # 如果音频格式为 16 位整数（小端），每个样本占用 2 字节
    elif format_for_conversion == "f32le":
        size_of_sample = 4  # 如果音频格式为 32 位浮点数（小端），每个样本占用 4 字节
    else:
        # 如果音频格式不被支持，则抛出错误
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    system = platform.system()  # 获取当前操作系统信息
    if system == "Linux":
        format_ = "alsa"  # 如果是 Linux 系统，使用 ALSA 格式
        input_ = "default"  # 使用默认音频输入设备
    elif system == "Darwin":
        format_ = "avfoundation"  # 如果是 macOS 系统，使用 AVFoundation 格式
        input_ = ":0"  # 使用默认音频输入设备
    elif system == "Windows":
        format_ = "dshow"  # 如果是 Windows 系统，使用 DirectShow 格式
        input_ = _get_microphone_name()  # 获取麦克风名称

    # 构建执行 FFmpeg 命令所需参数列表
    ffmpeg_command = [
        "ffmpeg",
        "-f",
        format_,
        "-i",
        input_,
        "-ac",
        ac,
        "-ar",
        ar,
        "-f",
        format_for_conversion,
        "-fflags",
        "nobuffer",
        "-hide_banner",
        "-loglevel",
        "quiet",
        "pipe:1",
    ]
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample  # 计算每个音频块的长度
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)  # 调用私有函数返回 FFmpeg 流的迭代器
    for item in iterator:
        yield item  # 生成每个音频块的迭代器


def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
):
    """
    # 定义一个用于通过ffmpeg从麦克风文件中读取音频的辅助函数
    # 从`stream_chunk_s`开始（如果已定义），输出重叠的部分，直到达到`chunk_length_s`。使用步进来避免在各个块的“边缘”出现错误。

    # 参数：
    # sampling_rate（`int`）：
    # 用于从麦克风读取数据时使用的采样率。尝试使用模型的采样率以避免后续重新采样。
    # chunk_length_s（`float`或`int`）：
    # 要发送和返回的最大音频块的长度。这包括最终的步进。
    # stream_chunk_s（`float`或`int`）
    # 要返回的临时音频的最短长度。
    # stride_length_s（`float`或`int`或（`float`，`float`），*可选*，默认为`None`）
    # 要使用的步进长度。步进用于向模型提供关于音频样本的（左，右）的上下文，但不使用该部分进行实际预测。设置这个不会改变块的长度。
    # format_for_conversion（`str`，默认为`f32le`）
    # 要由ffmpeg返回的音频样本格式的名称。标准是`f32le`，也可以使用`s16le`。

    # 返回：
    # 生成器，产生以下形式的字典

    # `{"sampling_rate": int, "raw": np.array(), "partial" bool}`，如果定义了`stride_length_s`，还有一个`"stride" (int, int)`键。

    # `stride`和`raw`都以`samples`表示，`partial`是一个布尔值，表示当前产生项是否是一个完整的块，还是一个部分临时结果，稍后将由另一个更大的块替换。
    """
    # 如果stream_chunk_s不为空，则chunk_s等于stream_chunk_s，否则等于chunk_length_s
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s

    # 使用ffmpeg_microphone函数读取麦克风的音频数据
    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)
    
    # 根据format_for_conversion的值确定dtype和size_of_sample
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    # 如果stride_length_s为None，则stride_length_s等于chunk_length_s的1/6
    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6
    
    # 计算chunk_len和stride的左右长度
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]
    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample
    
    # 记录生成结果的时间
    audio_time = datetime.datetime.now()
    # 计算时间差
    delta = datetime.timedelta(seconds=chunk_s)
    # 遍历通过麦克风生成的数据块迭代器
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        # 将所有内容还原为 numpy 的规模
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        # 调整步长为采样样本大小的比例
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        # 设置数据的采样率
        item["sampling_rate"] = sampling_rate
        # 累加音频时间
        audio_time += delta
        # 如果当前时间超过音频时间加上 10 个增量时间，则跳过
        if datetime.datetime.now() > audio_time + 10 * delta:
            # 我们晚了！！跳过
            continue
        # 生成数据块
        yield item
# 从一个迭代器中读取原始字节，并按长度 `chunk_len` 进行分块。可选地添加 `stride` 以获得重叠。
# `stream` 用于在有部分但不足一个完整的 `chunk_len` 时返回部分结果。
def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool = False):
    acc = b""  # 初始化一个空的字节串
    stride_left, stride_right = stride  # 解包 stride 元组
    if stride_left + stride_right >= chunk_len:
        raise ValueError(  # 如果 stride 大于等于 chunk_len，抛出异常
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )
    _stride_left = 0  # 初始化 _stride_left 为 0
    for raw in iterator:  # 遍历迭代器中的原始数据
        acc += raw  # 将原始数据添加到累加器中
        if stream and len(acc) < chunk_len:  # 当 stream 为 True 且累加器长度小于 chunk_len 时
            stride = (_stride_left, 0)  # 重设 stride 为 (_stride_left, 0)
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}  # 返回部分结果字典
        else:  # 否则
            while len(acc) >= chunk_len:  # 当累加器长度大于等于 chunk_len 时
                stride = (_stride_left, stride_right)  # 设置 stride 为 (_stride_left, stride_right)
                item = {"raw": acc[:chunk_len], "stride": stride}  # 创建结果字典
                if stream:  # 如果 stream 为 True
                    item["partial"] = False  # 设置部分结果标志为 False
                yield item  # 返回结果字典
                _stride_left = stride_left  # 更新 _stride_left 为 stride_left
                acc = acc[chunk_len - stride_left - stride_right :]  # 截取累加器中的剩余数据
    # 最后一个分块
    if len(acc) > stride_left:  # 如果累加器中的数据大于 stride_left
        item = {"raw": acc, "stride": (_stride_left, 0)}  # 创建最后一个分块的结果字典
        if stream:  # 如果 stream 为 True
            item["partial"] = False  # 设置部分结果标志为 False
        yield item  # 返回最后一个分块的结果字典


# 内部函数，通过 ffmpeg 创建数据生成器
def _ffmpeg_stream(ffmpeg_command, buflen: int):
    bufsize = 2**24  # 16Mb
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:  # 使用 subprocess 执行 ffmpeg 命令
            while True:  # 无限循环
                raw = ffmpeg_process.stdout.read(buflen)  # 从 ffmpeg 进程的标准输出读取数据
                if raw == b"":  # 如果读取的数据为空字节串
                    break  # 退出循环
                yield raw  # 返回读取的数据
    except FileNotFoundError as error:  # 捕获 FileNotFoundError 异常
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error  # 抛出异常


# 获取麦克风名称在 Windows 中的函数
def _get_microphone_name():
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]  # 创建获取麦克风名称的 ffmpeg 命令

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")  # 执行获取麦克风名称的 ffmpeg 命令
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]  # 从 stderr 中筛选包含 "(audio)" 的行

        if microphone_lines:  # 如果有麦克风行
            microphone_name = microphone_lines[0].split('"')[1]  # 获取麦克风名称
            print(f"Using microphone: {microphone_name}")  # 打印使用的麦克风名称
            return f"audio={microphone_name}"  # 返回麦克风名称
    except FileNotFoundError:  # 捕获 FileNotFoundError 异常
        print("ffmpeg was not found. Please install it or make sure it is in your system PATH.")  # 打印 ffmpeg 未找到的消息

    return "default"  # 返回默认麦克风名称
```