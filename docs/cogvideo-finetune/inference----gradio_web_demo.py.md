# `.\cogvideo-finetune\inference\gradio_web_demo.py`

```
"""
# 主文件用于 Gradio 网络演示，使用 CogVideoX-2B 模型生成视频
# 设置环境变量 OPENAI_API_KEY 使用 OpenAI API 增强提示

# 此演示仅支持文本到视频的生成模型。
# 如果希望使用图像到视频或视频到视频生成模型，
# 请使用 gradio_composite_demo 实现完整的 GUI 功能。

# 使用方法：
# OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

# 导入操作系统相关功能
import os
# 导入多线程功能
import threading
# 导入时间功能
import time

# 导入 Gradio 库以构建 Web 应用
import gradio as gr
# 导入 PyTorch 库进行深度学习
import torch
# 导入 CogVideoXPipeline 模型
from diffusers import CogVideoXPipeline
# 导入导出视频功能
from diffusers.utils import export_to_video
# 导入日期时间处理功能
from datetime import datetime, timedelta
# 导入 OpenAI 库以使用其 API
from openai import OpenAI
# 导入 MoviePy 库进行视频编辑
import moviepy.editor as mp

# 从预训练模型加载 CogVideoXPipeline，指定数据类型为 bfloat16，并移动到 GPU
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda")

# 启用 VAE 的切片功能
pipe.vae.enable_slicing()
# 启用 VAE 的平铺功能
pipe.vae.enable_tiling()

# 创建输出目录，如果已存在则不报错
os.makedirs("./output", exist_ok=True)
# 创建临时目录，如果已存在则不报错
os.makedirs("./gradio_tmp", exist_ok=True)

# 定义系统提示，指导视频生成的描述
sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""

# 定义转换提示的函数，接受提示和重试次数作为参数
def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    # 如果没有设置 OpenAI API 密钥，返回原始提示
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt

    # 创建 OpenAI 客户端
    client = OpenAI()
    # 去除提示两端的空白
    text = prompt.strip()

    # 返回原始提示
    return prompt

# 定义推断函数，接受提示、推断步骤和引导尺度
def infer(prompt: str, num_inference_steps: int, guidance_scale: float, progress=gr.Progress(track_tqdm=True)):
    # 清空 GPU 缓存
    torch.cuda.empty_cache()
    # 使用模型生成视频，指定相关参数
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        guidance_scale=guidance_scale,
    ).frames[0]

    # 返回生成的视频
    return video

# 定义保存视频的函数，接受张量作为参数
def save_video(tensor):
    # 获取当前时间戳，用于生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 定义视频保存路径
    video_path = f"./output/{timestamp}.mp4"
    # 创建视频保存目录，如果已存在则不报错
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 将张量导出为视频文件
    export_to_video(tensor, video_path)
    # 返回视频文件路径
    return video_path

# 定义将视频转换为 GIF 的函数，接受视频路径作为参数
def convert_to_gif(video_path):
    # 使用 MoviePy 加载视频文件
    clip = mp.VideoFileClip(video_path)
    # 设置视频的帧率为 8
    clip = clip.set_fps(8)
    # 调整剪辑的高度为 240 像素，保持宽高比
        clip = clip.resize(height=240)
        # 将视频路径中的 ".mp4" 后缀替换为 ".gif" 后缀，生成 GIF 文件路径
        gif_path = video_path.replace(".mp4", ".gif")
        # 将剪辑写入 GIF 文件，设置每秒帧数为 8
        clip.write_gif(gif_path, fps=8)
        # 返回生成的 GIF 文件路径
        return gif_path
# 定义删除旧文件的函数
def delete_old_files():
    # 无限循环，持续执行删除旧文件的任务
    while True:
        # 获取当前时间
        now = datetime.now()
        # 计算10分钟前的时间，用于判断文件是否过期
        cutoff = now - timedelta(minutes=10)
        # 定义需要清理的目录列表
        directories = ["./output", "./gradio_tmp"]

        # 遍历每个目录
        for directory in directories:
            # 遍历目录中的每个文件
            for filename in os.listdir(directory):
                # 构建文件的完整路径
                file_path = os.path.join(directory, filename)
                # 检查该路径是否为文件
                if os.path.isfile(file_path):
                    # 获取文件的最后修改时间
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    # 判断文件是否早于截止时间
                    if file_mtime < cutoff:
                        # 删除该文件
                        os.remove(file_path)
        # 每600秒（10分钟）暂停一次
        time.sleep(600)

# 启动一个线程来执行删除旧文件的函数，设置为守护线程
threading.Thread(target=delete_old_files, daemon=True).start()

# 使用 Gradio 创建用户界面
with gr.Blocks() as demo:
    # 创建 Markdown 组件，显示标题
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX Gradio Simple Space🤗
            """)

    # 创建一行布局
    with gr.Row():
        # 创建一列布局
        with gr.Column():
            # 创建文本框用于输入提示
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

            # 创建一个行布局
            with gr.Row():
                # 创建 Markdown 组件，说明增强提示按钮的功能
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                # 创建增强提示的按钮
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            # 创建另一列布局
            with gr.Column():
                # 创建 Markdown 组件，描述可选参数
                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                    "50 steps are recommended for most cases.<br>"
                )
                # 创建一行布局，包含推理步数和引导比例输入框
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                # 创建生成视频的按钮
                generate_button = gr.Button("🎬 Generate Video")

        # 创建另一列布局
        with gr.Column():
            # 创建视频输出组件
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            # 创建一行布局，包含下载按钮
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)

    # 定义生成视频的函数
    def generate(prompt, num_inference_steps, guidance_scale, model_choice, progress=gr.Progress(track_tqdm=True)):
        # 调用推理函数生成张量
        tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        # 保存生成的视频并获取其路径
        video_path = save_video(tensor)
        # 更新视频输出组件为可见，并设置视频路径
        video_update = gr.update(visible=True, value=video_path)
        # 将视频转换为 GIF 并获取其路径
        gif_path = convert_to_gif(video_path)
        # 更新 GIF 下载按钮为可见，并设置 GIF 路径
        gif_update = gr.update(visible=True, value=gif_path)

        # 返回视频路径和更新信息
        return video_path, video_update, gif_update

    # 定义增强提示的函数
    def enhance_prompt_func(prompt):
        # 转换提示并允许重试一次
        return convert_prompt(prompt, retry_times=1)
    # 为生成按钮添加点击事件，触发生成函数
        generate_button.click(
            # 绑定生成函数到点击事件
            generate,
            # 定义输入组件，包括提示文本、推理步骤数和引导尺度
            inputs=[prompt, num_inference_steps, guidance_scale],
            # 定义输出组件，包括视频输出和下载按钮
            outputs=[video_output, download_video_button, download_gif_button],
        )
    
    # 为增强按钮添加点击事件，触发增强提示函数
        enhance_button.click(enhance_prompt_func, 
            # 定义输入组件，包括提示文本
            inputs=[prompt], 
            # 定义输出组件，更新提示文本
            outputs=[prompt]
        )
# 检查当前模块是否为主程序入口
if __name__ == "__main__":
    # 调用 demo 对象的 launch 方法
    demo.launch()
```