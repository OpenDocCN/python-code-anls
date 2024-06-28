# `.\pipelines\audio_utils.py`

```
# 版权声明及导入必要的库和模块
# 版权 2023 The HuggingFace Team. 保留所有权利。
import datetime
import platform
import subprocess
from typing import Optional, Tuple, Union

import numpy as np


def ffmpeg_read(bpayload: bytes, sampling_rate: int) -> np.array:
    """
    通过ffmpeg读取音频文件的辅助函数。
    """
    # 将采样率转换为字符串
    ar = f"{sampling_rate}"
    # 设置音频通道数为1
    ac = "1"
    # 设置转换格式为"f32le"，即32位浮点数，低端序
    format_for_conversion = "f32le"
    # 构建ffmpeg命令
    ffmpeg_command = [
        "ffmpeg",                # ffmpeg命令
        "-i", "pipe:0",          # 输入文件从标准输入(pipe:0)读取
        "-ac", ac,               # 设置音频通道数
        "-ar", ar,               # 设置采样率
        "-f", format_for_conversion,  # 设置输出格式为指定的转换格式
        "-hide_banner",          # 隐藏ffmpeg的banner信息
        "-loglevel", "quiet",    # 设置日志级别为静默模式，不输出日志信息
        "pipe:1",                # 输出音频数据到标准输出(pipe:1)
    ]

    try:
        # 使用subprocess.Popen启动ffmpeg进程，并通过stdin传输输入数据，stdout接收输出数据
        with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ffmpeg_process:
            output_stream = ffmpeg_process.communicate(bpayload)  # 传输音频数据并获取输出流
    except FileNotFoundError as error:
        # 若ffmpeg未找到，则抛出错误
        raise ValueError("ffmpeg was not found but is required to load audio files from filename") from error

    out_bytes = output_stream[0]   # 获取ffmpeg输出的音频数据字节流
    audio = np.frombuffer(out_bytes, np.float32)   # 将字节流转换为numpy数组，数据类型为32位浮点数

    # 如果音频数据长度为0，则抛出异常，说明音频文件格式不正确或损坏
    if audio.shape[0] == 0:
        raise ValueError(
            "Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has "
            "a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted. If reading from a remote "
            "URL, ensure that the URL is the full address to **download** the audio file."
        )
    return audio   # 返回解码后的音频数据数组


def ffmpeg_microphone(
    sampling_rate: int,
    chunk_length_s: float,
    format_for_conversion: str = "f32le",
):
    """
    读取原始麦克风数据的辅助函数。
    """
    # 将采样率转换为字符串
    ar = f"{sampling_rate}"
    # 设置音频通道数为1
    ac = "1"

    # 根据指定的转换格式确定每个音频样本的字节大小
    if format_for_conversion == "s16le":
        size_of_sample = 2   # 每个样本为16位整数，即2个字节
    elif format_for_conversion == "f32le":
        size_of_sample = 4   # 每个样本为32位浮点数，即4个字节
    else:
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    # 获取当前操作系统类型
    system = platform.system()
    if system == "Linux":
        format_ = "alsa"     # Linux系统使用alsa音频系统
        input_ = "default"   # 默认输入设备
    elif system == "Darwin":
        format_ = "avfoundation"   # macOS系统使用avfoundation音频系统
        input_ = ":0"        # 默认音频输入设备
    elif system == "Windows":
        format_ = "dshow"    # Windows系统使用dshow音频系统
        input_ = _get_microphone_name()  # 获取当前连接的麦克风设备名称

    # 构建ffmpeg命令
    ffmpeg_command = [
        "ffmpeg",                # ffmpeg命令
        "-f", format_,           # 指定输入格式为当前系统指定的音频系统
        "-i", input_,            # 指定输入来源，如默认设备或具体设备名称
        "-ac", ac,               # 设置音频通道数
        "-ar", ar,               # 设置采样率
        "-f", format_for_conversion,  # 设置输出格式为指定的转换格式
        "-fflags", "nobuffer",   # 设置fflags参数为nobuffer，禁用缓冲
        "-hide_banner",          # 隐藏ffmpeg的banner信息
        "-loglevel", "quiet",    # 设置日志级别为静默模式，不输出日志信息
        "pipe:1",                # 输出音频数据到标准输出(pipe:1)
    ]

    # 计算每个数据块的长度
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample

    # 使用私有函数_ffmpeg_stream迭代处理音频流
    iterator = _ffmpeg_stream(ffmpeg_command, chunk_len)

    # 生成器函数，逐项产生处理后的音频数据块
    for item in iterator:
        yield item


def ffmpeg_microphone_live(
    sampling_rate: int,
    chunk_length_s: float,
    stream_chunk_s: Optional[int] = None,
    stride_length_s: Optional[Union[Tuple[float, float], float]] = None,
    format_for_conversion: str = "f32le",
):
    """
    实时读取麦克风音频数据的辅助函数。
    """
    # 如果 stream_chunk_s 不为 None，则将其作为 chunk_s；否则使用 chunk_length_s 作为 chunk_s
    if stream_chunk_s is not None:
        chunk_s = stream_chunk_s
    else:
        chunk_s = chunk_length_s

    # 调用 ffmpeg_microphone 函数获取麦克风的音频流，使用指定的采样率和 chunk_s，同时指定音频格式为 format_for_conversion
    microphone = ffmpeg_microphone(sampling_rate, chunk_s, format_for_conversion=format_for_conversion)

    # 根据 format_for_conversion 的值选择相应的数据类型 dtype 和每个样本的大小 size_of_sample
    if format_for_conversion == "s16le":
        dtype = np.int16
        size_of_sample = 2
    elif format_for_conversion == "f32le":
        dtype = np.float32
        size_of_sample = 4
    else:
        # 如果 format_for_conversion 不是已处理的格式，则抛出 ValueError 异常
        raise ValueError(f"Unhandled format `{format_for_conversion}`. Please use `s16le` or `f32le`")

    # 如果未指定 stride_length_s，则将 chunk_length_s 的六分之一作为默认值
    if stride_length_s is None:
        stride_length_s = chunk_length_s / 6

    # 计算 chunk_length_s 对应的音频数据长度，并转换为字节数
    chunk_len = int(round(sampling_rate * chunk_length_s)) * size_of_sample

    # 如果 stride_length_s 是单个数字（int 或 float），则将其转换为左右两侧相同长度的列表
    if isinstance(stride_length_s, (int, float)):
        stride_length_s = [stride_length_s, stride_length_s]

    # 计算左右两侧 stride 对应的音频数据长度，并转换为字节数
    stride_left = int(round(sampling_rate * stride_length_s[0])) * size_of_sample
    stride_right = int(round(sampling_rate * stride_length_s[1])) * size_of_sample

    # 记录当前时间，用于计算音频的时间戳
    audio_time = datetime.datetime.now()

    # 计算时间增量 delta，表示每次处理的音频数据长度对应的时间长度
    delta = datetime.timedelta(seconds=chunk_s)
    # 使用 chunk_bytes_iter 函数从 microphone 生成的数据流中迭代获取每个音频片段
    for item in chunk_bytes_iter(microphone, chunk_len, stride=(stride_left, stride_right), stream=True):
        # 将 item 中的 "raw" 字段重新转换为 numpy 数组
        item["raw"] = np.frombuffer(item["raw"], dtype=dtype)
        # 调整 item 中的 "stride" 字段，使其单位符合采样样本的大小
        item["stride"] = (
            item["stride"][0] // size_of_sample,
            item["stride"][1] // size_of_sample,
        )
        # 设置 item 中的 "sampling_rate" 字段为指定的采样率
        item["sampling_rate"] = sampling_rate
        # 增加当前音频时间
        audio_time += delta
        # 如果当前时间超过音频时间加上 10 倍的时间间隔，则跳过当前音频片段
        if datetime.datetime.now() > audio_time + 10 * delta:
            # 我们已经迟到了！！跳过
            continue
        # 通过生成器返回当前处理的音频片段
        yield item
def chunk_bytes_iter(iterator, chunk_len: int, stride: Tuple[int, int], stream: bool = False):
    """
    Reads raw bytes from an iterator and does chunks of length `chunk_len`. Optionally adds `stride` to each chunks to
    get overlaps. `stream` is used to return partial results even if a full `chunk_len` is not yet available.
    """
    acc = b""  # 初始化一个空字节串，用于累积迭代器中的原始字节数据
    stride_left, stride_right = stride  # 将步长参数解包成左右两部分
    if stride_left + stride_right >= chunk_len:
        raise ValueError(
            f"Stride needs to be strictly smaller than chunk_len: ({stride_left}, {stride_right}) vs {chunk_len}"
        )  # 如果步长大于等于块长度，抛出数值错误异常
    _stride_left = 0  # 初始化一个内部步长变量为零
    for raw in iterator:  # 迭代处理输入的迭代器
        acc += raw  # 将迭代器中的原始数据累加到累积变量中
        if stream and len(acc) < chunk_len:
            stride = (_stride_left, 0)  # 如果流模式为真且累积数据长度小于块长度，则使用当前内部步长和零作为步长
            yield {"raw": acc[:chunk_len], "stride": stride, "partial": True}  # 生成部分结果字典，包含截取的原始数据、步长和部分结果标志
        else:
            while len(acc) >= chunk_len:
                # We are flushing the accumulator
                stride = (_stride_left, stride_right)  # 当累积数据长度大于等于块长度时，使用当前内部步长和右侧步长作为步长
                item = {"raw": acc[:chunk_len], "stride": stride}  # 创建包含截取的原始数据和步长的结果字典
                if stream:
                    item["partial"] = False  # 如果流模式为真，设置部分结果标志为假
                yield item  # 生成结果字典
                _stride_left = stride_left  # 更新内部步长为左侧步长
                acc = acc[chunk_len - stride_left - stride_right :]  # 更新累积变量，去除已处理的数据部分
    # Last chunk
    if len(acc) > stride_left:
        item = {"raw": acc, "stride": (_stride_left, 0)}  # 处理最后一个块，生成结果字典包含剩余的原始数据和步长
        if stream:
            item["partial"] = False  # 如果流模式为真，设置部分结果标志为假
        yield item  # 生成最后一个结果字典


def _ffmpeg_stream(ffmpeg_command, buflen: int):
    """
    Internal function to create the generator of data through ffmpeg
    """
    bufsize = 2**24  # 16Mo，设置缓冲区大小为 16MB
    try:
        with subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=bufsize) as ffmpeg_process:
            while True:
                raw = ffmpeg_process.stdout.read(buflen)  # 从 ffmpeg 进程的标准输出中读取指定长度的数据块
                if raw == b"":  # 如果读取到的数据为空字节串
                    break  # 跳出循环
                yield raw  # 生成读取到的原始数据块
    except FileNotFoundError as error:
        raise ValueError("ffmpeg was not found but is required to stream audio files from filename") from error
        # 如果捕获到文件未找到异常，抛出数值错误异常，指明需要安装 ffmpeg 以从文件名流式传输音频文件


def _get_microphone_name():
    """
    Retrieve the microphone name in Windows .
    """
    command = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", ""]  # 定义获取麦克风名称的命令

    try:
        ffmpeg_devices = subprocess.run(command, text=True, stderr=subprocess.PIPE, encoding="utf-8")  # 执行命令并捕获标准错误
        microphone_lines = [line for line in ffmpeg_devices.stderr.splitlines() if "(audio)" in line]
        # 过滤包含"(audio)"的标准错误输出行

        if microphone_lines:  # 如果找到匹配的麦克风行
            microphone_name = microphone_lines[0].split('"')[1]  # 解析麦克风名称
            print(f"Using microphone: {microphone_name}")  # 打印使用的麦克风名称
            return f"audio={microphone_name}"  # 返回 ffmpeg 需要的音频设备字符串
    except FileNotFoundError:
        print("ffmpeg was not found. Please install it or make sure it is in your system PATH.")
        # 如果捕获到文件未找到异常，打印消息提示用户安装或将其添加到系统路径中

    return "default"  # 默认返回字符串，表示使用默认音频设备
```