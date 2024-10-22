# `.\cogvideo-finetune\tools\load_cogvideox_lora.py`

```py
# 版权声明，表明版权所有者及其保留权利
# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# 根据 Apache License, Version 2.0 进行授权
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件以“原样”基础分发，不提供任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取有关权限和限制的具体信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学模块以进行数学计算
import math 
# 导入随机模块以生成随机数
import random 
# 导入时间模块以进行时间相关操作
import time
# 从 diffusers.utils 导入导出视频的功能
from diffusers.utils import export_to_video
# 从 diffusers.image_processor 导入 VAE 图像处理器
from diffusers.image_processor import VaeImageProcessor
# 导入日期和时间处理的模块
from datetime import datetime, timedelta
# 从 diffusers 导入多个类以供后续使用
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
# 导入操作系统模块以进行系统级操作
import os
# 导入 PyTorch 库以进行深度学习
import torch
# 导入参数解析模块以处理命令行参数
import argparse

# 根据是否有可用的 GPU 设定设备类型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 定义获取命令行参数的函数
def get_args():
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加预训练模型路径参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    # 添加 LoRA 权重路径参数
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        default=None,
        required=True,
        help="Path to lora weights.",
    )
    # 添加 LoRA 权重的秩参数
    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="""LoRA weights have a rank parameter, with the default for 2B trans set at 128 and 5B trans set at 256. 
        This part is used to calculate the value for lora_scale, which is by default divided by the alpha value, 
        used for stable learning and to prevent underflow. In the SAT training framework,
        alpha is set to 1 by default. The higher the rank, the better the expressive capability,
        but it requires more memory and training time. Increasing this number blindly isn't always better.
        The formula for lora_scale is: lora_r / alpha.
        """,
    )
    # 添加 LoRA 权重的 alpha 参数
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=1,
        help="""LoRA weights have a rank parameter, with the default for 2B trans set at 128 and 5B trans set at 256. 
        This part is used to calculate the value for lora_scale, which is by default divided by the alpha value, 
        used for stable learning and to prevent underflow. In the SAT training framework,
        alpha is set to 1 by default. The higher the rank, the better the expressive capability,
        but it requires more memory and training time. Increasing this number blindly isn't always better.
        The formula for lora_scale is: lora_r / alpha.
        """,
    )
    # 添加用于生成内容的提示参数
    parser.add_argument(
        "--prompt",
        type=str,
        help="prompt",
    )
    # 向解析器添加一个名为 output_dir 的参数
        parser.add_argument(
            "--output_dir",  # 参数的名称
            type=str,  # 参数类型为字符串
            default="output",  # 默认值为 "output"
            help="The output directory where the model predictions and checkpoints will be written.",  # 参数的帮助说明
        )
    # 解析命令行参数并返回结果
        return parser.parse_args()
# 如果该脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 获取命令行参数
    args = get_args()
    # 从预训练模型创建视频处理管道，并将其移动到指定设备
    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    # 加载 LoRA 权重，指定权重文件名和适配器名称
    pipe.load_lora_weights(args.lora_weights_path,  weight_name="pytorch_lora_weights.safetensors", adapter_name="cogvideox-lora")
    # pipe.fuse_lora(lora_scale=args.lora_alpha/args.lora_r, ['transformer'])  # 注释掉的代码，用于融合 LoRA 权重
    # 计算 LoRA 缩放因子
    lora_scaling=args.lora_alpha/args.lora_r
    # 设置适配器及其对应的缩放因子
    pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

    # 根据调度器配置创建视频调度器
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 创建输出目录，如果不存在则自动创建
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成视频帧，设置相关参数
    latents = pipe(
        prompt=args.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        output_type="pt",
        guidance_scale=3.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).frames
    # 获取生成的帧的批量大小
    batch_size = latents.shape[0]
    # 初始化一个列表，用于存储视频帧
    batch_video_frames = []
    # 遍历每一帧，处理并转换为 PIL 图像
    for batch_idx in range(batch_size):
        pt_image = latents[batch_idx]
        # 将当前帧的各个通道堆叠成一个张量
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])

        # 将 PyTorch 图像转换为 NumPy 格式
        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        # 将 NumPy 图像转换为 PIL 图像
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        # 将处理后的 PIL 图像添加到帧列表中
        batch_video_frames.append(image_pil)

    # 获取当前时间戳，用于视频文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 构造视频文件路径
    video_path = f"{args.output_dir}/{timestamp}.mp4"
    # 创建视频文件目录（如果不存在）
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # 选择第一组帧作为视频内容
    tensor = batch_video_frames[0]
    # 计算帧率，假设每 6 帧为 1 秒
    fps=math.ceil((len(batch_video_frames[0]) - 1) / 6)

    # 将处理后的帧导出为视频文件
    export_to_video(tensor, video_path, fps=fps)
```