# `.\gpt4-tts\utils.py`

```
# 导入必要的库
import base64  # 用于 base64 编解码
import cv2  # OpenCV 库，用于图像处理
import logging  # 日志记录库
import openai  # OpenAI API 访问库
import requests  # HTTP 请求库
import time  # 时间操作库
from typing import Optional  # 用于类型提示

from IPython.display import display, Image  # IPython 中用于显示图像
from moviepy.editor import VideoFileClip, AudioFileClip  # 用于处理视频和音频文件

# 配置日志记录，设置日志级别为 INFO
logging.basicConfig(level=logging.INFO)


def convert_frames_to_base64(path_to_video: str, resize_dim: Optional[tuple] = None) -> list:
    """
    从给定路径读取视频并将其转换为 base64 格式的帧列表
    如果提供了 resize_dim，则同时调整帧的大小
    Args:
        path_to_video (str): 视频文件路径
        resize_dim (tuple, optional): 调整后的尺寸。默认为 None。
    Returns:
        list: 包含 base64 编码帧的列表
    """
    # 打开视频文件
    video = cv2.VideoCapture(path_to_video)

    base64Frames = []
    while video.isOpened():
        # 读取视频中的一帧
        success, frame = video.read()

        # 检查是否成功读取了帧
        if not success:
            break

        # 如果提供了 resize_dim，则调整帧的大小
        if resize_dim is not None:
            frame = cv2.resize(frame, resize_dim)

        # 将帧编码为 base64 格式
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # 记录日志，记录读取了多少帧
    logging.getLogger().info(f"{len(base64Frames)}, frames read.")

    return base64Frames


def render_video(frames_list: list) -> None:
    """
    在笔记本单元格中渲染给定的帧列表
    Args:
        frames_list (list): base64 编码的帧列表
    """
    # 创建一个显示句柄，用于更新显示的图像
    display_handle = display(None, display_id=True)
    for img in frames_list:
        # 更新显示图像
        display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        time.sleep(0.025)


def attach_audio_to_video(audio_path: str, video_path: str, save_path: str) -> None:
    """
    将音频文件附加到视频文件中
    Args:
        audio_path (str): 音频文件路径
        video_path (str): 视频文件路径
        save_path (str): 保存合成后视频的路径
    """

    # 打开视频和音频文件
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    # 将音频与视频合并
    final_clip = video_clip.set_audio(audio_clip)

    # 保存合成后的视频文件
    final_clip.write_videofile(save_path)


def get_description(prompt: str, frame_list: list, frame_interval: int, open_ai_key: str, max_tokens: int) -> str:
    """
    调用 OpenAI API，基于文本和图像生成描述
    Args:
        prompt (str): 提供给 GPT-4 的指令字符串
        frame_list (list): 输入到 GPT-4 的帧列表
        frame_interval (int): 传递帧之间的间隔，减少传递的 token 数量
        open_ai_key (str): 你的 OpenAI API 密钥
        max_tokens (int): GPT-4 响应的最大 token 数量
    Returns:
        str: 来自 GPT-4 的输出描述
    """
    # 构造提示消息，包含用户角色和内容列表，内容包括文本提示和一系列帧的图像数据
    prompt_message = [
        {
            "role": "user",
            "content": [prompt,  # 用户提供的文本提示
                *map(lambda x: {"image": x}, frame_list[0::frame_interval]),  # 使用 lambda 函数将帧列表中的图像数据转换为字典形式
            ],
        },
    ]
    
    # 设置请求参数，包括模型名称、消息内容、API 密钥、自定义头部信息和生成的最大标记数
    params = {
        "model": "gpt-4-vision-preview",  # 使用的模型名称
        "messages": prompt_message,  # 上面构造的提示消息
        "api_key": open_ai_key,  # OpenAI API 密钥
        "headers": {"Openai-Version": "2020-11-07"},  # 自定义请求头部信息，指定 OpenAI 版本
        "max_tokens": max_tokens,  # 生成的最大标记数
    }
    
    # 发起对话完成请求，使用给定的参数创建对话完成对象
    result = openai.ChatCompletion.create(**params)
    
    # 记录返回结果的第一个选择的消息内容到日志中
    logging.getLogger().info(result.choices[0].message.content)
    
    # 返回第一个选择的消息内容作为函数的结果
    return result.choices[0].message.content
# 将文本描述转换为语音文件，并保存到指定路径
def transform_text_to_speech(description: str, open_ai_key: str, model: str, voice: str, save_path: str) -> bytes:
    """
    Receives a string and transform it into audio
    Args:
        description (str): output from GPT-4，GPT-4生成的文本描述
        open_ai_key (str): your open ai key，OpenAI的API密钥
        model (str): TTS model，文本转语音模型
        voice (str): voice from TTS OpenAI，语音的声音类型（来自TTS OpenAI）
        save_path (str): path to save audio，保存音频文件的路径
    """

    # 发送 POST 请求到 OpenAI 的语音生成 API
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {open_ai_key}",  # 设置授权头部信息
        },
        json={
            "model": model,         # 指定使用的文本转语音模型
            "input": description,   # 输入要转换的文本描述
            "voice": voice,         # 指定语音的声音类型
        },
    )

    audio = b""
    # 逐块接收响应内容，每块大小为 1MB，将其合并成完整的音频数据
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio += chunk

    # 将生成的音频数据写入到指定路径的文件中
    with open(save_path, 'wb') as file:
        file.write(audio)

    # 返回生成的音频数据
    return audio
```