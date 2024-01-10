# `so-vits-svc\vencoder\whisper\utils.py`

```
# 导入所需的模块
import json
import os
import sys
import zlib
from typing import Callable, TextIO

# 获取系统默认编码
system_encoding = sys.getdefaultencoding()

# 如果系统编码不是 UTF-8，则定义一个函数用于替换不可表示的字符
if system_encoding != "utf-8":
    def make_safe(string):
        # 用系统默认编码替换任何不可表示的字符为 '?'
        # 避免 UnicodeEncodeError (https://github.com/openai/whisper/discussions/729)
        return string.encode(system_encoding, errors="replace").decode(system_encoding)
else:
    def make_safe(string):
        # UTF-8 可以编码任何 Unicode 码点，所以不需要进行往返编码
        return string

# 定义一个函数用于确保 x 能被 y 整除
def exact_div(x, y):
    assert x % y == 0
    return x // y

# 定义一个函数用于将字符串转换为布尔值
def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

# 定义一个函数用于将字符串转换为可选的整数
def optional_int(string):
    return None if string == "None" else int(string)

# 定义一个函数用于将字符串转换为可选的浮点数
def optional_float(string):
    return None if string == "None" else float(string)

# 定义一个函数用于计算文本的压缩比
def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

# 定义一个函数用于格式化时间戳
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"

# 定义一个类用于写入结果
class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    # 定义一个方法，接受一个字典类型的参数result和一个字符串类型的参数audio_path
    def __call__(self, result: dict, audio_path: str):
        # 获取音频文件的基本名称
        audio_basename = os.path.basename(audio_path)
        # 拼接输出路径，将输出目录和音频基本名称以及扩展名组合起来
        output_path = os.path.join(self.output_dir, audio_basename + "." + self.extension)
    
        # 打开输出路径的文件，以写入模式，编码为utf-8
        with open(output_path, "w", encoding="utf-8") as f:
            # 调用write_result方法，将result写入文件f
            self.write_result(result, file=f)
    
    # 定义一个方法，接受一个字典类型的参数result和一个文件类型的参数file
    def write_result(self, result: dict, file: TextIO):
        # 抛出一个未实现的错误，提示子类需要实现这个方法
        raise NotImplementedError
# 定义一个类 WriteTXT，继承自 ResultWriter
class WriteTXT(ResultWriter):
    # 定义属性 extension 为 "txt"
    extension: str = "txt"

    # 定义方法 write_result，接受 result 字典和 file 文本流作为参数
    def write_result(self, result: dict, file: TextIO):
        # 遍历 result 字典中的 "segments" 键对应的值
        for segment in result["segments"]:
            # 将 segment 中的 "text" 键对应的值去除首尾空格后写入 file 文本流中，并刷新缓冲区
            print(segment['text'].strip(), file=file, flush=True)


# 定义一个类 WriteVTT，继承自 ResultWriter
class WriteVTT(ResultWriter):
    # 定义属性 extension 为 "vtt"
    extension: str = "vtt"

    # 定义方法 write_result，接受 result 字典和 file 文本流作为参数
    def write_result(self, result: dict, file: TextIO):
        # 在 file 文本流中写入 "WEBVTT"，表示 Web 视频文本轨道格式，并换行
        print("WEBVTT\n", file=file)
        # 遍历 result 字典中的 "segments" 键对应的值
        for segment in result["segments"]:
            # 在 file 文本流中写入格式化后的时间戳和文本内容，并刷新缓冲区
            print(
                f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


# 定义一个类 WriteSRT，继承自 ResultWriter
class WriteSRT(ResultWriter):
    # 定义属性 extension 为 "srt"
    extension: str = "srt"

    # 定义方法 write_result，接受 result 字典和 file 文本流作为参数
    def write_result(self, result: dict, file: TextIO):
        # 遍历 result 字典中的 "segments" 键对应的值，并使用 enumerate 函数获取索引和对应的值
        for i, segment in enumerate(result["segments"], start=1):
            # 在 file 文本流中写入格式化后的时间戳和文本内容，并刷新缓冲区
            print(
                f"{i}\n"
                f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
                f"{segment['text'].strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


# 定义一个类 WriteTSV，继承自 ResultWriter
class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """
    # 定义属性 extension 为 "tsv"
    extension: str = "tsv"
    # 定义一个方法，用于将结果写入文件
    def write_result(self, result: dict, file: TextIO):
        # 在文件中打印标题行，包括"start", "end", "text"，使用制表符分隔，输出到指定文件对象
        print("start", "end", "text", sep="\t", file=file)
        # 遍历结果中的每个段落
        for segment in result["segments"]:
            # 在文件中打印每个段落的起始时间（乘以1000取整），使用制表符分隔，输出到指定文件对象
            print(round(1000 * segment['start']), file=file, end="\t")
            # 在文件中打印每个段落的结束时间（乘以1000取整），使用制表符分隔，输出到指定文件对象
            print(round(1000 * segment['end']), file=file, end="\t")
            # 在文件中打印每个段落的文本内容（去除首尾空格并替换制表符为空格），输出到指定文件对象，立即刷新缓冲区
            print(segment['text'].strip().replace("\t", " "), file=file, flush=True)
# 定义一个名为 WriteJSON 的类，继承自 ResultWriter 类
class WriteJSON(ResultWriter):
    # 定义类属性 extension，值为 "json"
    extension: str = "json"

    # 定义 write_result 方法，接受 result 和 file 两个参数
    def write_result(self, result: dict, file: TextIO):
        # 使用 json 模块将 result 写入到 file 中
        json.dump(result, file)


# 定义一个名为 get_writer 的函数，接受 output_format 和 output_dir 两个参数，返回类型为 Callable[[dict, TextIO], None]
def get_writer(output_format: str, output_dir: str) -> Callable[[dict, TextIO], None]:
    # 定义一个字典 writers，包含不同输出格式对应的写入类
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }

    # 如果 output_format 为 "all"，则创建所有输出格式对应的写入类实例
    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        # 定义一个名为 write_all 的函数，接受 result 和 file 两个参数
        def write_all(result: dict, file: TextIO):
            # 遍历所有写入类实例，将 result 写入到 file 中
            for writer in all_writers:
                writer(result, file)

        # 返回 write_all 函数
        return write_all

    # 如果 output_format 不为 "all"，则返回对应输出格式的写入类实例
    return writers[output_format](output_dir)
```