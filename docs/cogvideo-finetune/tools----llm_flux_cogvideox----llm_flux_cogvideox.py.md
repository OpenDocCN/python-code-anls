# `.\cogvideo-finetune\tools\llm_flux_cogvideox\llm_flux_cogvideox.py`

```py
# 文档字符串，提供项目原始实验代码的链接和使用说明
"""
The original experimental code for this project can be found at:

https://gist.github.com/a-r-r-o-w/d070cce059ab4ceab3a9f289ff83c69c

By using this code, description prompts will be generated through a local large language model, and images will be
generated using the black-forest-labs/FLUX.1-dev model, followed by video generation via CogVideoX.
The entire process utilizes open-source solutions, without the need for any API keys.

You can use the generate.sh file in the same folder to automate running this code
for batch generation of videos and images.

bash generate.sh

"""

# 导入命令行参数解析库
import argparse
# 导入垃圾回收库
import gc
# 导入JSON处理库
import json
# 导入操作系统功能库
import os
# 导入路径操作库
import pathlib
# 导入随机数生成库
import random
# 导入类型提示功能
from typing import Any, Dict

# 从transformers库导入自动标记器
from transformers import AutoTokenizer

# 设置环境变量，指定TORCH_LOGS的日志内容
os.environ["TORCH_LOGS"] = "+dynamo,recompiles,graph_breaks"
# 设置环境变量，开启TORCHDYNAMO的详细输出
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# 导入numpy库
import numpy as np
# 导入PyTorch库
import torch
# 导入transformers库
import transformers
# 从diffusers库导入视频生成相关的管道和调度器
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, DiffusionPipeline
# 从diffusers库导入日志记录工具
from diffusers.utils.logging import get_logger
# 从diffusers库导入视频导出工具
from diffusers.utils import export_to_video

# 设置PyTorch的浮点数乘法精度为高
torch.set_float32_matmul_precision("high")

# 获取日志记录器实例
logger = get_logger(__name__)

# 定义系统提示字符串，指导生成视频描述的任务
SYSTEM_PROMPT = """
You are part of a team of people that create videos using generative models. You use a video-generation model that can generate a video about anything you describe.

For example, if you respond with "A beautiful morning in the woods with the sun peaking through the trees", the video generation model will create a video of exactly as described. You task is to summarize the descriptions of videos provided to by users, and create details prompts to feed into the generative model.

There are a few rules to follow:
- You will only ever output a single video description per request.
- If the user mentions to summarize the prompt in [X] words, make sure to not exceed the limit.

You responses should just be the video generation prompt. Here are examples:
- “A lone figure stands on a city rooftop at night, gazing up at the full moon. The moon glows brightly, casting a gentle light over the quiet cityscape. Below, the windows of countless homes shine with warm lights, creating a contrast between the bustling life below and the peaceful solitude above. The scene captures the essence of the Mid-Autumn Festival, where despite the distance, the figure feels connected to loved ones through the shared beauty of the moonlit sky.”
- "A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting."
# 包含一个描述场景的字符串，描述了一位街头艺术家和他的创作
- "A street artist, clad in a worn-out denim jacket and a colorful banana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall"
""".strip()

# 定义用户提示的模板，要求生成视频生成模型的提示，限制字数
USER_PROMPT = """
Could you generate a prompt for a video generation model? 
Please limit the prompt to [{0}] words.
""".strip()


# 定义一个获取命令行参数的函数
def get_args():
    # 创建命令行解析器实例
    parser = argparse.ArgumentParser()
    # 添加命令行参数：视频数量，类型为整数，默认值为5
    parser.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of unique videos you would like to generate."
    )
    # 添加命令行参数：模型路径，类型为字符串，默认值为指定的模型路径
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX-5B",
        help="The path of Image2Video CogVideoX-5B",
    )
    # 添加命令行参数：标题生成模型ID，类型为字符串，默认值为指定的模型ID
    parser.add_argument(
        "--caption_generator_model_id",
        type=str,
        default="THUDM/glm-4-9b-chat",
        help="Caption generation model. default GLM-4-9B",
    )
    # 添加命令行参数：标题生成模型缓存目录，类型为字符串，默认值为None
    parser.add_argument(
        "--caption_generator_cache_dir",
        type=str,
        default=None,
        help="Cache directory for caption generation model."
    )
    # 添加命令行参数：图像生成模型ID，类型为字符串，默认值为指定的模型ID
    parser.add_argument(
        "--image_generator_model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Image generation model."
    )
    # 添加命令行参数：图像生成模型缓存目录，类型为字符串，默认值为None
    parser.add_argument(
        "--image_generator_cache_dir",
        type=str,
        default=None,
        help="Cache directory for image generation model."
    )
    # 添加命令行参数：图像生成推理步骤数量，类型为整数，默认值为50
    parser.add_argument(
        "--image_generator_num_inference_steps",
        type=int,
        default=50,
        help="Caption generation model."
    )
    # 添加命令行参数：引导比例，类型为浮点数，默认值为7
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7,
        help="Guidance scale to be use for generation."
    )
    # 添加命令行参数：是否使用动态CFG，动作类型为布尔值，默认值为False
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        help="Whether or not to use cosine dynamic guidance for generation [Recommended].",
    )
    # 添加命令行参数：输出目录，类型为字符串，默认值为"outputs/"
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/",
        help="Location where generated images and videos should be stored.",
    )
    # 添加命令行参数：是否编译转换器，动作类型为布尔值，默认值为False
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether or not to compile the transformer of image and video generators."
    )
    # 添加命令行参数：是否启用VAE平铺，动作类型为布尔值，默认值为False
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Whether or not to use VAE tiling when encoding/decoding."
    )
    # 添加命令行参数：随机种子，类型为整数，默认值为42
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )
    # 解析命令行参数并返回结果
    return parser.parse_args()


# 定义一个重置内存的函数
def reset_memory():
    # 垃圾回收器收集所有未使用的对象
    gc.collect()
    # 清空CUDA的缓存
    torch.cuda.empty_cache()
    # 重置CUDA的峰值内存统计信息
    torch.cuda.reset_peak_memory_stats()
    # 重置CUDA的累积内存统计信息
    torch.cuda.reset_accumulated_memory_stats()


# 使用无梯度计算的上下文定义主函数
@torch.no_grad()
def main(args: Dict[str, Any]) -> None:
    # 将输出目录转换为路径对象
    output_dir = pathlib.Path(args.output_dir)
    # 如果输出目录不存在，则创建该目录
    os.makedirs(output_dir.as_posix(), exist_ok=True)

    # 设置随机种子以保证结果可重现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 为所有 GPU 设备设置随机种子，以确保结果可重现
    torch.cuda.manual_seed_all(args.seed)

    # 重置内存，以清理之前的计算图和变量
    reset_memory()
    # 从预训练模型中加载分词器，允许信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(args.caption_generator_model_id, trust_remote_code=True)
    # 创建文本生成管道，使用指定的模型，并配置相关参数
    caption_generator = transformers.pipeline(
        "text-generation",  # 设置任务为文本生成
        model=args.caption_generator_model_id,  # 使用指定的模型ID
        device_map="auto",  # 自动分配设备（CPU/GPU）
        model_kwargs={  # 模型的其他参数配置
            "local_files_only": True,  # 仅使用本地文件
            "cache_dir": args.caption_generator_cache_dir,  # 设置缓存目录
            "torch_dtype": torch.bfloat16,  # 设置张量的数据类型为 bfloat16
        },
        trust_remote_code=True,  # 允许信任远程代码
        tokenizer=tokenizer  # 使用加载的分词器
    )

    # 初始化用于存储生成的标题的列表
    captions = []
    # 遍历指定数量的视频
    for i in range(args.num_videos):
        # 随机选择生成标题的字数
        num_words = random.choice([50, 75, 100])
        # 格式化用户提示，以包含字数信息
        user_prompt = USER_PROMPT.format(num_words)

        # 创建包含系统和用户消息的列表
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},  # 系统消息
            {"role": "user", "content": user_prompt},  # 用户消息
        ]

        # 生成标题，限制新生成的标记数
        outputs = caption_generator(messages, max_new_tokens=226)
        # 提取生成的文本内容
        caption = outputs[0]["generated_text"][-1]["content"]
        # 如果标题以引号开始和结束，去除引号
        if caption.startswith("\"") and caption.endswith("\""):
            caption = caption[1:-1]
        # 将生成的标题添加到列表中
        captions.append(caption)
        # 记录生成的标题
        logger.info(f"Generated caption: {caption}")

    # 将生成的标题保存到 JSON 文件中
    with open(output_dir / "captions.json", "w") as file:
        json.dump(captions, file)  # 将标题列表写入 JSON 文件

    # 删除标题生成器以释放内存
    del caption_generator
    # 重置内存
    reset_memory()

    # 从预训练模型加载图像生成器
    image_generator = DiffusionPipeline.from_pretrained(
        args.image_generator_model_id,  # 使用指定的图像生成模型ID
        cache_dir=args.image_generator_cache_dir,  # 设置缓存目录
        torch_dtype=torch.bfloat16  # 设置张量的数据类型为 bfloat16
    )
    # 将图像生成器移动到 GPU
    image_generator.to("cuda")

    # 如果编译选项被启用，则编译图像生成器的转换器
    if args.compile:
        image_generator.transformer = torch.compile(image_generator.transformer, mode="max-autotune", fullgraph=True)

    # 如果启用 VAE 瓦片功能，则允许图像生成器的 VAE 使用瓦片
    if args.enable_vae_tiling:
        image_generator.vae.enable_tiling()

    # 初始化用于存储生成的图像的列表
    images = []
    # 遍历生成的标题并生成对应的图像
    for index, caption in enumerate(captions):
        # 使用图像生成器生成图像，指定相关参数
        image = image_generator(
            prompt=caption,  # 使用标题作为提示
            height=480,  # 设置生成图像的高度
            width=720,  # 设置生成图像的宽度
            num_inference_steps=args.image_generator_num_inference_steps,  # 设置推理步骤数量
            guidance_scale=3.5,  # 设置指导比例
        ).images[0]  # 获取生成的图像

        # 处理标题以创建合法的文件名
        filename = caption[:25].replace(".", "_").replace("'", "_").replace('"', "_").replace(",", "_")
        # 保存生成的图像到指定目录
        image.save(output_dir / f"{index}_{filename}.png")
        # 将生成的图像添加到列表中
        images.append(image)

    # 删除图像生成器以释放内存
    del image_generator
    # 重置内存
    reset_memory()

    # 从预训练模型加载视频生成器
    video_generator = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to("cuda")  # 移动到 GPU

    # 设置视频生成器的调度器
    video_generator.scheduler = CogVideoXDPMScheduler.from_config(
        video_generator.scheduler.config,  # 使用当前调度器的配置
        timestep_spacing="trailing"  # 设置时间步间隔为 trailing
    )

    # 如果编译选项被启用，则编译视频生成器的转换器
    if args.compile:
        video_generator.transformer = torch.compile(video_generator.transformer, mode="max-autotune", fullgraph=True)

    # 如果启用 VAE 瓦片功能，则允许视频生成器的 VAE 使用瓦片
    if args.enable_vae_tiling:
        video_generator.vae.enable_tiling()

    # 创建随机数生成器并设置种子
    generator = torch.Generator().manual_seed(args.seed)  # 确保随机结果可重现
    # 遍历 captions 和 images 的组合，获取索引及对应的描述和图像
        for index, (caption, image) in enumerate(zip(captions, images)):
            # 调用视频生成器，生成视频帧
            video = video_generator(
                # 设置生成视频的图像和描述
                image=image,
                prompt=caption,
                # 指定视频的高度和宽度
                height=480,
                width=720,
                # 设置生成的帧数和推理步骤
                num_frames=49,
                num_inference_steps=50,
                # 设置引导比例和动态配置选项
                guidance_scale=args.guidance_scale,
                use_dynamic_cfg=args.use_dynamic_cfg,
                # 提供随机数生成器
                generator=generator,
            ).frames[0]  # 获取生成的视频的第一帧
            # 格式化文件名，限制为前25个字符并替换特殊字符
            filename = caption[:25].replace(".", "_").replace("'", "_").replace('"', "_").replace(",", "_")
            # 导出生成的视频到指定目录，命名为索引加文件名
            export_to_video(video, output_dir / f"{index}_{filename}.mp4", fps=8)  # 设置每秒帧数为8
# 判断当前模块是否是主程序入口
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 调用主函数，并传入获取的参数
    main(args)
```