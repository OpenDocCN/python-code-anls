# `.\cogvideo-finetune\inference\gradio_composite_demo\app.py`

```py
# 这是 Gradio 网页演示的主文件，使用 CogVideoX-5B 模型生成视频并进行增强
"""
THis is the main file for the gradio web demo. It uses the CogVideoX-5B model to generate videos gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

Usage:
    OpenAI_API_KEY=your_openai_api_key OPENAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

# 导入数学库
import math
# 导入操作系统相关的库
import os
# 导入随机数生成库
import random
# 导入多线程库
import threading
# 导入时间库
import time

# 导入计算机视觉库
import cv2
# 导入临时文件处理库
import tempfile
# 导入视频处理库
import imageio_ffmpeg
# 导入 Gradio 库用于构建界面
import gradio as gr
# 导入 PyTorch 库
import torch
# 导入图像处理库
from PIL import Image
# 导入 Diffusers 库中的模型和调度器
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDPMScheduler,
    CogVideoXVideoToVideoPipeline,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
# 导入视频和图像加载工具
from diffusers.utils import load_video, load_image
# 导入日期和时间处理库
from datetime import datetime, timedelta

# 导入图像处理器
from diffusers.image_processor import VaeImageProcessor
# 导入 OpenAI 库
from openai import OpenAI
# 导入视频编辑库
import moviepy.editor as mp
# 导入实用工具模块
import utils
# 导入 RIFE 模型的加载和推断工具
from rife_model import load_rife_model, rife_inference_with_latents
# 导入 Hugging Face Hub 的下载工具
from huggingface_hub import hf_hub_download, snapshot_download

# 检查是否可以使用 GPU，并设置设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义模型的名称
MODEL = "THUDM/CogVideoX-5b"

# 从 Hugging Face Hub 下载 Real-ESRGAN 模型权重
hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x4.pth", local_dir="model_real_esran")
# 从 Hugging Face Hub 下载 RIFE 模型快照
snapshot_download(repo_id="AlexWortega/RIFE", local_dir="model_rife")

# 从预训练模型中加载视频生成管道，并将其转移到指定设备
pipe = CogVideoXPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
# 设置调度器，使用配置文件中的参数
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
# 创建视频到视频的生成管道
pipe_video = CogVideoXVideoToVideoPipeline.from_pretrained(
    MODEL,
    transformer=pipe.transformer,
    vae=pipe.vae,
    scheduler=pipe.scheduler,
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    torch_dtype=torch.bfloat16,
).to(device)

# 创建图像到视频的生成管道
pipe_image = CogVideoXImageToVideoPipeline.from_pretrained(
    MODEL,
    transformer=CogVideoXTransformer3DModel.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=torch.bfloat16
    ),
    vae=pipe.vae,
    scheduler=pipe.scheduler,
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    torch_dtype=torch.bfloat16,
).to(device)

# 下面的行被注释掉，用于内存优化
# pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# pipe_image.transformer.to(memory_format=torch.channels_last)
# pipe_image.transformer = torch.compile(pipe_image.transformer, mode="max-autotune", fullgraph=True)

# 创建输出目录，如果不存在则创建
os.makedirs("./output", exist_ok=True)
# 创建临时文件夹用于 Gradio
os.makedirs("./gradio_tmp", exist_ok=True)

# 加载超分辨率模型
upscale_model = utils.load_sd_upscale("model_real_esran/RealESRGAN_x4.pth", device)
# 加载帧插值模型
frame_interpolation_model = load_rife_model("model_rife")

# 系统提示，用于指导生成视频的助手
sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.
```  
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
``` 
# 检查输入视频的尺寸是否适合要求，若不适合则进行处理
def resize_if_unfit(input_video, progress=gr.Progress(track_tqdm=True)):
    # 获取输入视频的宽度和高度
    width, height = get_video_dimensions(input_video)

    # 如果视频尺寸为720x480，直接使用原视频
    if width == 720 and height == 480:
        processed_video = input_video
    # 否则进行中心裁剪和调整大小
    else:
        processed_video = center_crop_resize(input_video)
    # 返回处理后的视频
    return processed_video


# 获取输入视频的尺寸信息
def get_video_dimensions(input_video_path):
    # 读取视频帧
    reader = imageio_ffmpeg.read_frames(input_video_path)
    # 获取视频元数据
    metadata = next(reader)
    # 返回视频尺寸
    return metadata["size"]


# 对视频进行中心裁剪和调整大小
def center_crop_resize(input_video_path, target_width=720, target_height=480):
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)

    # 获取原视频的宽度、高度和帧率
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算宽度和高度的缩放因子
    width_factor = target_width / orig_width
    height_factor = target_height / orig_height
    # 选择较大的缩放因子进行调整
    resize_factor = max(width_factor, height_factor)

    # 计算中间宽度和高度
    inter_width = int(orig_width * resize_factor)
    inter_height = int(orig_height * resize_factor)

    # 设置目标帧率
    target_fps = 8
    # 计算理想跳过的帧数
    ideal_skip = max(0, math.ceil(orig_fps / target_fps) - 1)
    # 限制跳过的帧数最大为5
    skip = min(5, ideal_skip)  # Cap at 5

    # 调整跳过的帧数，以确保足够的帧数
    while (total_frames / (skip + 1)) < 49 and skip > 0:
        skip -= 1

    processed_frames = []  # 存储处理后的帧
    frame_count = 0  # 记录已处理帧数
    total_read = 0  # 记录已读取帧数

    # 读取帧并进行处理，直到处理49帧或读取完成
    while frame_count < 49 and total_read < total_frames:
        ret, frame = cap.read()  # 读取一帧
        if not ret:  # 如果未成功读取，退出循环
            break

        # 只处理指定间隔的帧
        if total_read % (skip + 1) == 0:
            # 调整帧的大小
            resized = cv2.resize(frame, (inter_width, inter_height), interpolation=cv2.INTER_AREA)

            # 计算裁剪区域的起始位置
            start_x = (inter_width - target_width) // 2
            start_y = (inter_height - target_height) // 2
            # 裁剪帧
            cropped = resized[start_y : start_y + target_height, start_x : start_x + target_width]

            processed_frames.append(cropped)  # 将裁剪后的帧添加到列表
            frame_count += 1  # 更新处理帧数

        total_read += 1  # 更新已读取帧数

    cap.release()  # 释放视频捕获对象
    # 使用临时文件创建一个后缀为 .mp4 的文件，不会在关闭时删除
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            # 获取临时视频文件的路径
            temp_video_path = temp_file.name
            # 指定视频编码格式为 mp4v
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # 初始化视频写入对象，设置输出路径、编码格式、帧率和帧大小
            out = cv2.VideoWriter(temp_video_path, fourcc, target_fps, (target_width, target_height))
    
            # 遍历处理过的帧
            for frame in processed_frames:
                # 将每一帧写入视频文件
                out.write(frame)
    
            # 释放视频写入对象，完成文件写入
            out.release()
    
        # 返回临时视频文件的路径
        return temp_video_path
# 定义一个转换提示的函数，接受提示字符串和重试次数（默认为3）
def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    # 检查环境变量中是否存在 OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        # 如果没有 API 密钥，返回原始提示
        return prompt
    # 创建 OpenAI 客户端
    client = OpenAI()
    # 去掉提示字符串两端的空白字符
    text = prompt.strip()

    # 返回处理后的提示字符串
    return prompt


# 定义一个推断函数，接受多种参数
def infer(
    prompt: str,
    image_input: str,
    video_input: str,
    video_strenght: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = -1,
    progress=gr.Progress(track_tqdm=True),
):
    # 如果种子为 -1，随机生成一个种子
    if seed == -1:
        seed = random.randint(0, 2**8 - 1)

    # 如果有视频输入
    if video_input is not None:
        # 加载视频并限制为49帧
        video = load_video(video_input)[:49]  # Limit to 49 frames
        # 通过管道处理视频
        video_pt = pipe_video(
            video=video,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
            strength=video_strenght,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames
    # 如果有图像输入
    elif image_input is not None:
        # 将输入图像转换为PIL格式并调整大小
        image_input = Image.fromarray(image_input).resize(size=(720, 480))  # Convert to PIL
        # 加载图像
        image = load_image(image_input)
        # 通过管道处理图像
        video_pt = pipe_image(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            num_videos_per_prompt=1,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames
    # 如果没有图像或视频输入
    else:
        # 通过管道直接处理提示生成视频
        video_pt = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            output_type="pt",
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).frames

    # 返回生成的视频和种子
    return (video_pt, seed)


# 定义一个将视频转换为GIF的函数
def convert_to_gif(video_path):
    # 加载视频文件
    clip = mp.VideoFileClip(video_path)
    # 设置视频帧率为8
    clip = clip.set_fps(8)
    # 调整视频高度为240
    clip = clip.resize(height=240)
    # 创建GIF文件的路径
    gif_path = video_path.replace(".mp4", ".gif")
    # 将视频写入GIF文件
    clip.write_gif(gif_path, fps=8)
    # 返回生成的GIF文件路径
    return gif_path


# 定义一个删除旧文件的函数
def delete_old_files():
    # 无限循环以持续检查旧文件
    while True:
        # 获取当前时间
        now = datetime.now()
        # 计算10分钟前的时间
        cutoff = now - timedelta(minutes=10)
        # 定义要检查的目录
        directories = ["./output", "./gradio_tmp"]

        # 遍历每个目录
        for directory in directories:
            # 遍历目录中的每个文件
            for filename in os.listdir(directory):
                # 生成文件的完整路径
                file_path = os.path.join(directory, filename)
                # 检查是否为文件
                if os.path.isfile(file_path):
                    # 获取文件的最后修改时间
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    # 如果文件的修改时间早于截止时间，则删除文件
                    if file_mtime < cutoff:
                        os.remove(file_path)
        # 每600秒（10分钟）休眠一次
        time.sleep(600)


# 启动一个线程来执行删除旧文件的函数，设置为守护线程
threading.Thread(target=delete_old_files, daemon=True).start()
# 定义示例视频列表
examples_videos = [["example_videos/horse.mp4"], ["example_videos/kitten.mp4"], ["example_videos/train_running.mp4"]]
# 创建一个包含示例图片路径的列表，每个子列表包含一个图片路径
examples_images = [["example_images/beach.png"], ["example_images/street.png"], ["example_images/camping.png"]]

# 使用 Gradio 库创建一个块结构的界面
with gr.Blocks() as demo:
    # 添加一个 Markdown 组件，用于显示标题和链接
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               # 在页面中居中显示的标题，字体大小为 32px，粗体，并有底部间距
               CogVideoX-5B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               # 提供多个链接，指向 Huggingface 模型库和相关资源
               <a href="https://huggingface.co/THUDM/CogVideoX-5B">🤗 5B(T2V) Model Hub</a> |
               <a href="https://huggingface.co/THUDM/CogVideoX-5B-I2V">🤗 5B(I2V) Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> |
               <a href="https://arxiv.org/pdf/2408.06072">📜 arxiv </a>
           </div>
           <div style="text-align: center;display: flex;justify-content: center;align-items: center;margin-top: 1em;margin-bottom: .5em;">
              # 提示用户如果空间太忙，可以复制使用
              <span>If the Space is too busy, duplicate it to use privately</span>
              # 提供一个按钮图标，链接到复制空间的地址
              <a href="https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space?duplicate=true"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg.svg" width="160" style="
                margin-left: .75em;
            "></a>
           </div>
           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            # 显示一条警告信息，表明此演示仅用于学术研究和实验使用
            ⚠️ This demo is for academic research and experimental use only. 
            </div>
           """)
    # 创建一个行容器，用于排列子组件
    with gr.Row():
        # 创建一个列容器，用于排列图像和视频输入组件
        with gr.Column():
            # 创建一个折叠组件，用于图像输入，初始状态为关闭
            with gr.Accordion("I2V: Image Input (cannot be used simultaneously with video input)", open=False):
                # 创建图像输入组件，并设置标签
                image_input = gr.Image(label="Input Image (will be cropped to 720 * 480)")
                # 创建示例组件，供用户选择预设的图像示例
                examples_component_images = gr.Examples(examples_images, inputs=[image_input], cache_examples=False)
            # 创建一个折叠组件，用于视频输入，初始状态为关闭
            with gr.Accordion("V2V: Video Input (cannot be used simultaneously with image input)", open=False):
                # 创建视频输入组件，并设置标签
                video_input = gr.Video(label="Input Video (will be cropped to 49 frames, 6 seconds at 8fps)")
                # 创建滑块组件，用于调整强度，范围从0.1到1.0，默认值为0.8
                strength = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Strength")
                # 创建示例组件，供用户选择预设的视频示例
                examples_component_videos = gr.Examples(examples_videos, inputs=[video_input], cache_examples=False)
            # 创建文本框组件，用于输入提示，限制为200个单词
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            # 创建一个行容器，用于排列按钮和说明文本
            with gr.Row():
                # 创建一个Markdown组件，显示关于增强提示按钮的说明
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                # 创建一个按钮，用于增强提示，标记为可选
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")
            # 创建一个组容器，用于排列生成相关的参数设置
            with gr.Group():
                # 创建一个列容器，用于排列生成参数
                with gr.Column():
                    # 创建一个行容器，用于排列随机种子输入
                    with gr.Row():
                        # 创建一个数字输入组件，用于输入推理种子，-1表示随机
                        seed_param = gr.Number(
                            label="Inference Seed (Enter a positive number, -1 for random)", value=-1
                        )
                    # 创建一个行容器，用于排列复选框
                    with gr.Row():
                        # 创建复选框组件，表示启用超分辨率功能
                        enable_scale = gr.Checkbox(label="Super-Resolution (720 × 480 -> 2880 × 1920)", value=False)
                        # 创建复选框组件，表示启用帧插值功能
                        enable_rife = gr.Checkbox(label="Frame Interpolation (8fps -> 16fps)", value=False)
                    # 创建一个Markdown组件，显示关于使用的技术和工具的说明
                    gr.Markdown(
                        "✨In this demo, we use [RIFE](https://github.com/hzwer/ECCV2022-RIFE) for frame interpolation and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling(Super-Resolution).<br>&nbsp;&nbsp;&nbsp;&nbsp;The entire process is based on open-source solutions."
                    )

            # 创建一个按钮，用于生成视频
            generate_button = gr.Button("🎬 Generate Video")

        # 创建一个列容器，用于显示生成的视频输出
        with gr.Column():
            # 创建视频输出组件，用于显示生成的视频
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            # 创建一个行容器，用于排列下载按钮和种子显示
            with gr.Row():
                # 创建文件下载按钮，用于下载生成的视频，初始状态为不可见
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                # 创建文件下载按钮，用于下载生成的GIF，初始状态为不可见
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                # 创建数字输入组件，用于显示用于视频生成的种子，初始状态为不可见
                seed_text = gr.Number(label="Seed Used for Video Generation", visible=False)

    # 创建一个Markdown组件，显示表格的结束标签
    gr.Markdown("""
    </table>
        """)

    # 定义生成视频的函数，接收多个参数
    def generate(
        prompt,
        image_input,
        video_input,
        video_strength,
        seed_value,
        scale_status,
        rife_status,
        progress=gr.Progress(track_tqdm=True)
    ):
        # 调用 infer 函数获取潜在表示和随机种子
        latents, seed = infer(
            # 输入的提示文本
            prompt,
            # 输入的图像数据
            image_input,
            # 输入的视频数据
            video_input,
            # 视频强度参数
            video_strength,
            # 设置推理步数为 50
            num_inference_steps=50,  # NOT Changed
            # 设置引导比例为 7.0
            guidance_scale=7.0,  # NOT Changed
            # 使用给定的种子值
            seed=seed_value,
            # 进度显示参数
            progress=progress,
        )
        # 如果缩放状态为真，进行批量放大和拼接
        if scale_status:
            latents = utils.upscale_batch_and_concatenate(upscale_model, latents, device)
        # 如果 RIFE 状态为真，使用潜在表示进行插帧推理
        if rife_status:
            latents = rife_inference_with_latents(frame_interpolation_model, latents)

        # 获取潜在表示的批量大小
        batch_size = latents.shape[0]
        # 初始化存储视频帧的列表
        batch_video_frames = []
        # 遍历每个批次的索引
        for batch_idx in range(batch_size):
            # 获取当前批次的潜在图像
            pt_image = latents[batch_idx]
            # 将当前图像的每个通道堆叠成一个张量
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

            # 将 PyTorch 图像转换为 NumPy 格式
            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            # 将 NumPy 图像转换为 PIL 格式
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            # 将转换后的图像添加到视频帧列表中
            batch_video_frames.append(image_pil)

        # 保存视频，并计算每秒帧数
        video_path = utils.save_video(batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6))
        # 更新视频显示状态
        video_update = gr.update(visible=True, value=video_path)
        # 将视频转换为 GIF 格式
        gif_path = convert_to_gif(video_path)
        # 更新 GIF 显示状态
        gif_update = gr.update(visible=True, value=gif_path)
        # 更新种子显示状态
        seed_update = gr.update(visible=True, value=seed)

        # 返回视频路径和更新状态
        return video_path, video_update, gif_update, seed_update

    # 定义增强提示功能的函数
    def enhance_prompt_func(prompt):
        # 转换提示文本，并设置重试次数为 1
        return convert_prompt(prompt, retry_times=1)

    # 为生成按钮点击事件绑定生成函数
    generate_button.click(
        # 调用生成函数
        generate,
        # 输入参数列表
        inputs=[prompt, image_input, video_input, strength, seed_param, enable_scale, enable_rife],
        # 输出参数列表
        outputs=[video_output, download_video_button, download_gif_button, seed_text],
    )

    # 为增强按钮点击事件绑定增强提示函数
    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])
    # 处理视频输入的上传事件，调整大小以适应
    video_input.upload(resize_if_unfit, inputs=[video_input], outputs=[video_input])
# 判断当前模块是否为主程序入口
if __name__ == "__main__":
    # 初始化队列，设置最大大小为 15
    demo.queue(max_size=15)
    # 启动 demo 程序
    demo.launch()
```