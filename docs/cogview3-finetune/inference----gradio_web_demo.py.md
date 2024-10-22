# `.\cogview3-finetune\inference\gradio_web_demo.py`

```py
# 主文件用于 Gradio 网络演示，使用 CogView3-Plus-3B 模型生成图像
"""
THis is the main file for the gradio web demo. It uses the CogView3-Plus-3B model to generate images gradio web demo.
set environment variable OPENAI_API_KEY to use the OpenAI API to enhance the prompt.

Usage:
    OpenAI_API_KEY=your_openai_api_key OpenAI_BASE_URL=https://api.openai.com/v1 python inference/gradio_web_demo.py
"""

# 导入必要的库
import os  # 用于处理操作系统功能，如环境变量
import re  # 用于正则表达式处理字符串
import threading  # 用于多线程操作
import time  # 用于时间相关操作
from datetime import datetime, timedelta  # 用于日期和时间处理

import gradio as gr  # 导入 Gradio 库以创建用户界面
import random  # 用于生成随机数
from diffusers import CogView3PlusPipeline  # 导入用于图像生成的管道
import torch  # 导入 PyTorch 库以处理深度学习模型
from openai import OpenAI  # 导入 OpenAI 库以使用 API

import gc  # 导入垃圾回收模块

# 检查是否可以使用 GPU，设置设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的 CogView3-Plus-3B 模型并将其移到指定设备
pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.bfloat16).to(device)

# 创建用于临时文件的目录，如果已经存在则不报错
os.makedirs("./gradio_tmp", exist_ok=True)


# 定义函数以清理字符串
def clean_string(s):
    # 将字符串中的换行符替换为空格
    s = s.replace("\n", " ")
    # 去掉字符串开头和结尾的空白
    s = s.strip()
    # 用单个空格替换两个或更多的空白
    s = re.sub(r"\s{2,}", " ", s)
    # 返回清理后的字符串
    return s


# 定义函数以转换提示词
def convert_prompt(
    prompt: str,
    retry_times: int = 5,
) -> str:
    # 检查环境变量是否设置了 OPENAI_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        # 如果未设置，直接返回原始提示词
        return prompt
    # 创建 OpenAI 客户端实例
    client = OpenAI()
    # 定义系统指令，指导图像描述生成
    system_instruction = """
    You are part of a team of bots that creates images . You work with an assistant bot that will draw anything you say. 
    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an image of a forest morning , as described. 
    You will be prompted by people looking to create detailed , amazing images. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive. 
    There are a few rules to follow : 
    - Prompt should always be written in English, regardless of the input language. Please provide the prompts in English.
    - You will only ever output a single image description per user request.
    - Image descriptions must be detailed and specific, including keyword categories such as subject, medium, style, additional details, color, and lighting. 
    - When generating descriptions, focus on portraying the visual elements rather than delving into abstract psychological and emotional aspects. Provide clear and concise details that vividly depict the scene and its composition, capturing the tangible elements that make up the setting.
    - Do not provide the process and explanation, just return the modified English description . Image descriptions must be between 100-200 words. Extra words will be ignored. 
    """

    # 去除提示词两端的空白
    text = prompt.strip()
    # 返回原始提示词（此处未对提示词进行修改）
    return prompt


# 定义函数以删除旧文件
def delete_old_files():
    # 无限循环，用于持续检查和清理文件
        while True:
            # 获取当前的日期和时间
            now = datetime.now()
            # 计算截止时间，5分钟前的时间点
            cutoff = now - timedelta(minutes=5)
            # 定义需要检查的目录列表
            directories = ["./gradio_tmp"]
    
            # 遍历每个目录
            for directory in directories:
                # 列出目录中的所有文件
                for filename in os.listdir(directory):
                    # 生成文件的完整路径
                    file_path = os.path.join(directory, filename)
                    # 检查路径是否为文件
                    if os.path.isfile(file_path):
                        # 获取文件的最后修改时间
                        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        # 如果文件的修改时间早于截止时间，则删除该文件
                        if file_mtime < cutoff:
                            os.remove(file_path)
            # 暂停600秒（10分钟），然后继续循环
            time.sleep(600)
# 创建并启动一个后台线程用于删除旧文件
threading.Thread(target=delete_old_files, daemon=True).start()


# 定义推断函数，接受多个参数
def infer(
    # 输入提示词
    prompt,
    # 随机种子
    seed,
    # 是否随机化种子标志
    randomize_seed,
    # 图像宽度
    width,
    # 图像高度
    height,
    # 引导缩放参数
    guidance_scale,
    # 推断步骤数量
    num_inference_steps,
    # 进度条对象，跟踪进度
    progress=gr.Progress(track_tqdm=True),
):
    # 垃圾回收，释放内存
    gc.collect()
    # 清空 CUDA 的缓存
    torch.cuda.empty_cache()
    # 收集 CUDA 进程间通信的资源
    torch.cuda.ipc_collect()
    
    # 如果需要随机化种子
    if randomize_seed:
        # 生成一个新的随机种子
        seed = random.randint(0, 65536)

    # 使用管道进行推断，生成图像
    image = pipe(
        prompt=prompt,  # 输入提示词
        guidance_scale=guidance_scale,  # 指导缩放
        num_images_per_prompt=1,  # 每个提示生成一张图像
        num_inference_steps=num_inference_steps,  # 指定推断步骤
        width=width,  # 图像宽度
        height=height,  # 图像高度
        generator=torch.Generator().manual_seed(seed),  # 使用手动设置的种子生成器
    ).images[0]  # 获取生成的第一张图像
    # 返回生成的图像和种子
    return image, seed


# 示例提示词列表
examples = [
    # 描述一辆复古粉色敞篷车的场景
    "A vintage pink convertible with glossy chrome finishes and whitewall tires sits parked on an open road, surrounded by a field of wildflowers under a clear blue sky. The car's body is a delicate pastel pink, complementing the vibrant greens and colors of the meadow. Its interior boasts cream leather seats and a polished wooden dashboard, evoking a sense of classic elegance. The sun casts a soft light on the vehicle, highlighting its curves and shiny surfaces, creating a picture of nostalgia mixed with dreamy escapism.",
    # 描述一只黑色拉布拉多犬在草地上的场景
    "A noble black Labrador retriever sits serenely in a sunlit meadow, its glossy coat absorbing the golden rays of a late afternoon sun. The dog's intelligent eyes sparkle with a mixture of curiosity and loyalty, as it gazes off into the distance where the meadow meets a line of tall, slender birch trees. The dog's posture is regal, yet approachable, with its tongue playfully hanging out to the side slightly, suggesting a friendly disposition. The idyllic setting is filled with the vibrant greens of lush grass and the soft colors of wildflowers speckled throughout, creating a peaceful harmony between the dog and its natural surroundings.",
    # 描述一只红色犬在秋季森林中的场景
    "A vibrant red-colored dog of medium build stands attentively in an autumn forest setting. Its fur is a deep, rich red, reminiscent of autumn leaves, contrasting with its bright, intelligent eyes, a clear sky blue. The dog's ears perk up, and its tail wags slightly as it looks off into the distance, its posture suggesting alertness and curiosity. Golden sunlight filters through the canopy of russet and gold leaves above, casting dappled light onto the forest floor and the glossy coat of the canine, creating a serene and heartwarming scene.",
]

# CSS 样式定义
css = """
#col-container {
    margin: 0 auto;  # 设置外边距为 0，使容器居中
    max-width: 640px;  # 设置容器最大宽度为 640 像素
}
"""

# 使用 Gradio 创建块式界面
with gr.Blocks(css=css) as demo:
    # 设置触发器，当运行按钮点击或提示提交时调用推断函数
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,  # 指定调用的函数
        inputs=[prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],  # 输入参数
        outputs=[result, seed],  # 输出结果和种子
    )

# 启动 Gradio 应用并处理请求
demo.queue().launch()
```