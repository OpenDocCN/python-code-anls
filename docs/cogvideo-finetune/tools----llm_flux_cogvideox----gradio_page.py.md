# `.\cogvideo-finetune\tools\llm_flux_cogvideox\gradio_page.py`

```
# 导入操作系统模块
import os
# 导入 Gradio 库，用于构建用户界面
import gradio as gr
# 导入垃圾回收模块
import gc
# 导入随机数生成模块
import random
# 导入 PyTorch 库
import torch
# 导入 NumPy 库
import numpy as np
# 导入图像处理库
from PIL import Image
# 导入 Transformers 库
import transformers
# 从 Diffusers 库导入视频生成相关的类
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, DiffusionPipeline
# 从 Diffusers 库导入导出视频的工具
from diffusers.utils import export_to_video
# 从 Transformers 库导入自动分词器
from transformers import AutoTokenizer
# 导入日期和时间处理模块
from datetime import datetime, timedelta
# 导入多线程模块
import threading
# 导入时间模块
import time
# 导入 MoviePy 库进行视频编辑
import moviepy.editor as mp

# 设置浮点数矩阵乘法的精度为高
torch.set_float32_matmul_precision("high")

# 设置默认值
caption_generator_model_id = "/share/home/zyx/Models/Meta-Llama-3.1-8B-Instruct"  # 生成视频描述的模型路径
image_generator_model_id = "/share/home/zyx/Models/FLUX.1-dev"  # 生成图像的模型路径
video_generator_model_id = "/share/official_pretrains/hf_home/CogVideoX-5b-I2V"  # 生成视频的模型路径
seed = 1337  # 随机数种子

# 创建输出目录，若已存在则不报错
os.makedirs("./output", exist_ok=True)
# 创建临时目录，用于 Gradio
os.makedirs("./gradio_tmp", exist_ok=True)

# 从指定模型加载自动分词器
tokenizer = AutoTokenizer.from_pretrained(caption_generator_model_id, trust_remote_code=True)
# 创建文本生成管道，用于生成视频描述
caption_generator = transformers.pipeline(
    "text-generation",  # 指定任务为文本生成
    model=caption_generator_model_id,  # 指定模型
    device_map="balanced",  # 设置设备映射为平衡模式
    model_kwargs={  # 模型参数
        "local_files_only": True,  # 仅使用本地文件
        "torch_dtype": torch.bfloat16,  # 设置张量数据类型
    },
    trust_remote_code=True,  # 允许使用远程代码
    tokenizer=tokenizer  # 使用加载的分词器
)

# 从指定模型加载图像生成管道
image_generator = DiffusionPipeline.from_pretrained(
    image_generator_model_id,  # 指定图像生成模型
    torch_dtype=torch.bfloat16,  # 设置张量数据类型
    device_map="balanced"  # 设置设备映射为平衡模式
)
# image_generator.to("cuda")  # 可选择将生成器移动到 GPU（被注释掉）

# 从指定模型加载视频生成管道
video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
    video_generator_model_id,  # 指定视频生成模型
    torch_dtype=torch.bfloat16,  # 设置张量数据类型
    device_map="balanced"  # 设置设备映射为平衡模式
)

# 启用视频生成器的 VAE 切片功能
video_generator.vae.enable_slicing()
# 启用视频生成器的 VAE 平铺功能
video_generator.vae.enable_tiling()

# 设置视频生成器的调度器，使用自定义配置
video_generator.scheduler = CogVideoXDPMScheduler.from_config(
    video_generator.scheduler.config, timestep_spacing="trailing"  # 设置时间步长为后续模式
)

# 定义系统提示
SYSTEM_PROMPT = """
# 系统提示内容，说明视频生成任务和规则
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. Your task is to summarize the descriptions of videos provided by users and create detailed prompts to feed into the generative model.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure not to exceed the limit.

Your responses should just be the video generation prompt. Here are examples:
# 定义包含详细描述的字符串，描述玩具船在蓝色地毯上的场景
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
# 定义包含街头艺术家的字符串，描述其在城市墙壁上喷涂的情景
- "A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart of the city, holding a can of spray paint, spray-painting a colorful bird on a mottled wall."
# 去除多余的空格并保存为用户提示
""".strip()

# 定义用户提示模板，用于生成视频生成模型的提示
USER_PROMPT = """
Could you generate a prompt for a video generation model? Please limit the prompt to [{0}] words.
""".strip()

# 定义生成字幕的函数，接受一个提示参数
def generate_caption(prompt):
    # 随机选择字数（25、50、75或100）以限制生成的字幕长度
    num_words = random.choice([25, 50, 75, 100])
    # 格式化用户提示，将随机字数插入提示模板中
    user_prompt = USER_PROMPT.format(num_words)

    # 创建消息列表，包含系统角色和用户角色的内容
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n" + user_prompt},
    ]

    # 调用字幕生成器生成字幕，指定最大新令牌数和是否返回完整文本
    response = caption_generator(
        messages,
        max_new_tokens=226,
        return_full_text=False
    )
    # 获取生成的字幕文本
    caption = response[0]["generated_text"]
    # 如果字幕以双引号开头和结尾，去掉这两个引号
    if caption.startswith("\"") and caption.endswith("\""):
        caption = caption[1:-1]
    # 返回生成的字幕
    return caption

# 定义生成图像的函数，接受字幕和进度参数
def generate_image(caption, progress=gr.Progress(track_tqdm=True)):
    # 调用图像生成器生成图像，指定相关参数
    image = image_generator(
        prompt=caption,
        height=480,
        width=720,
        num_inference_steps=30,
        guidance_scale=3.5,
    ).images[0]
    # 返回生成的图像，重复一次以便于后续处理
    return image, image  # One for output One for State

# 定义生成视频的函数，接受字幕、图像和进度参数
def generate_video(
        caption,
        image,
        progress=gr.Progress(track_tqdm=True)
):
    # 创建一个随机种子生成器
    generator = torch.Generator().manual_seed(seed)
    # 调用视频生成器生成视频帧，指定相关参数
    video_frames = video_generator(
        image=image,
        prompt=caption,
        height=480,
        width=720,
        num_frames=49,
        num_inference_steps=50,
        guidance_scale=6,
        use_dynamic_cfg=True,
        generator=generator,
    ).frames[0]
    # 保存生成的视频并获取视频路径
    video_path = save_video(video_frames)
    # 将视频转换为 GIF 并获取 GIF 路径
    gif_path = convert_to_gif(video_path)
    # 返回视频路径和 GIF 路径
    return video_path, gif_path

# 定义保存视频的函数，接受张量作为参数
def save_video(tensor):
    # 获取当前时间戳以命名视频文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建视频文件路径
    video_path = f"./output/{timestamp}.mp4"
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 导出张量为视频文件，指定帧率
    export_to_video(tensor, video_path, fps=8)
    # 返回视频文件路径
    return video_path

# 定义将视频转换为 GIF 的函数，接受视频路径作为参数
def convert_to_gif(video_path):
    # 加载视频文件
    clip = mp.VideoFileClip(video_path)
    # 设置视频的帧率
    clip = clip.set_fps(8)
    # 调整视频的高度以进行 GIF 输出
    clip = clip.resize(height=240)
    # 创建 GIF 文件路径
    gif_path = video_path.replace(".mp4", ".gif")
    # 将视频写入 GIF 文件，指定帧率
    clip.write_gif(gif_path, fps=8)
    # 返回 GIF 文件路径
    return gif_path

# 定义删除旧文件的函数，功能尚未实现
def delete_old_files():
    # 无限循环，持续执行文件清理操作
        while True:
            # 获取当前日期和时间
            now = datetime.now()
            # 计算截止时间，当前时间减去10分钟
            cutoff = now - timedelta(minutes=10)
            # 定义要清理的目录列表
            directories = ["./output", "./gradio_tmp"]
    
            # 遍历目录列表
            for directory in directories:
                # 遍历当前目录中的文件名
                for filename in os.listdir(directory):
                    # 构造文件的完整路径
                    file_path = os.path.join(directory, filename)
                    # 检查路径是否为文件
                    if os.path.isfile(file_path):
                        # 获取文件的最后修改时间
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        # 如果文件的修改时间早于截止时间，删除该文件
                        if file_mtime < cutoff:
                            os.remove(file_path)
            # 暂停600秒（10分钟），然后继续循环
            time.sleep(600)
# 启动一个新线程来删除旧文件，设置为守护线程以便主程序退出时自动结束
threading.Thread(target=delete_old_files, daemon=True).start()

# 创建一个 Gradio 应用程序的界面
with gr.Blocks() as demo:
    # 添加一个 Markdown 组件，显示标题
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               LLM + FLUX + CogVideoX-I2V Space 🤗
            </div>
    """)
    # 创建一个行布局以排列组件
    with gr.Row():
        # 创建第一列布局
        with gr.Column():
            # 创建一个文本框用于输入提示
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=5)
            # 创建一个按钮用于生成字幕
            generate_caption_button = gr.Button("Generate Caption")
            # 创建一个文本框用于显示生成的字幕
            caption = gr.Textbox(label="Caption", placeholder="Caption will appear here", lines=5)
            # 创建一个按钮用于生成图像
            generate_image_button = gr.Button("Generate Image")
            # 创建一个图像组件用于显示生成的图像
            image_output = gr.Image(label="Generated Image")
            # 创建一个状态组件，用于保存图像状态
            state_image = gr.State()
            # 设置生成字幕按钮的点击事件，调用生成字幕函数
            generate_caption_button.click(fn=generate_caption, inputs=prompt, outputs=caption)
            # 设置生成图像按钮的点击事件，调用生成图像函数
            generate_image_button.click(fn=generate_image, inputs=caption, outputs=[image_output, state_image])
        # 创建第二列布局
        with gr.Column():
            # 创建一个视频组件用于显示生成的视频
            video_output = gr.Video(label="Generated Video", width=720, height=480)
            # 创建一个文件组件用于下载视频，初始设置为不可见
            download_video_button = gr.File(label="📥 Download Video", visible=False)
            # 创建一个文件组件用于下载 GIF，初始设置为不可见
            download_gif_button = gr.File(label="📥 Download GIF", visible=False)
            # 创建一个按钮用于从图像生成视频
            generate_video_button = gr.Button("Generate Video from Image")
            # 设置生成视频按钮的点击事件，调用生成视频函数
            generate_video_button.click(fn=generate_video, inputs=[caption, state_image],
                                        outputs=[video_output, download_gif_button])

# 如果当前模块是主程序，则启动 Gradio 应用程序
if __name__ == "__main__":
    demo.launch()
```