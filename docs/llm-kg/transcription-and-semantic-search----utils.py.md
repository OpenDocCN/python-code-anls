# `.\transcription-and-semantic-search\utils.py`

```
import os  # 导入操作系统接口模块
from typing import Optional  # 导入可选类型提示

import moviepy.editor as mp  # 导入 moviepy 库中的编辑模块，并重命名为 mp
from pytube import YouTube  # 导入 pytube 库中的 YouTube 类


def download_youtube_video(url: str, output_path: Optional[str] = None) -> None:
    """
    从指定的 URL 下载 YouTube 视频，并保存到指定的输出路径或当前目录。

    Args:
        url: 要下载的 YouTube 视频的 URL。
        output_path: 下载的视频将保存到的路径。如果为 None，则保存到当前目录。

    Returns:
        None
    """
    yt = YouTube(url)  # 创建 YouTube 对象，传入视频的 URL

    # 从 YouTube 视频中选择第一个分辨率最高、支持渐进下载且文件扩展名为 mp4 的视频流
    video_stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")  # 按分辨率排序
        .desc()  # 降序排列
        .first()  # 取第一个（最高分辨率）
    )

    if output_path:
        video_stream.download(output_path)  # 如果有指定输出路径，则下载视频到该路径
        print(f"Video successfully downloaded to {output_path}")  # 打印下载成功信息及路径
    else:
        video_stream.download()  # 否则，下载视频到当前目录
        print("Video successfully downloaded to the current directory")  # 打印下载成功信息


def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> None:
    """
    使用 FFmpeg 将音频文件转换为 WAV 格式。

    Args:
        input_file (str): 要转换的输入音频文件路径。
        output_file (str): 转换后的输出 WAV 文件路径。如果为 None，则将输入文件的扩展名替换为 ".wav"。

    Returns:
        None
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".wav"  # 如果未指定输出文件路径，则生成默认的输出文件路径

    clip = mp.VideoFileClip(input_file)  # 创建视频文件剪辑对象
    clip.audio.write_audiofile(output_file, codec="pcm_s16le")  # 将剪辑的音频部分写入到指定的输出文件中，使用 PCM S16LE 编解码器
```