# `.\cogvideo-finetune\inference\cli_demo.py`

```py
# 脚本说明：演示如何使用 CogVideoX 模型通过 Hugging Face `diffusers` 管道生成视频
"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:


$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"


Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""

# 导入命令行参数解析模块
import argparse
# 从 typing 模块导入 Literal 类型，用于限制参数值
from typing import Literal

# 导入 PyTorch 库
import torch
# 从 diffusers 模块导入所需的类
from diffusers import (
    CogVideoXPipeline,  # 导入 CogVideoX 管道用于视频生成
    CogVideoXDDIMScheduler,  # 导入 DDIM 调度器
    CogVideoXDPMScheduler,  # 导入 DPMS 调度器
    CogVideoXImageToVideoPipeline,  # 导入图像到视频的管道
    CogVideoXVideoToVideoPipeline,  # 导入视频到视频的管道
)

# 从 diffusers.utils 模块导入辅助函数
from diffusers.utils import export_to_video, load_image, load_video


# 定义生成视频的函数，接受多个参数
def generate_video(
    prompt: str,  # 视频描述
    model_path: str,  # 预训练模型的路径
    lora_path: str = None,  # LoRA 权重的路径（可选）
    lora_rank: int = 128,  # LoRA 权重的秩
    output_path: str = "./output.mp4",  # 生成视频的保存路径
    image_or_video_path: str = "",  # 输入图像或视频的路径
    num_inference_steps: int = 50,  # 推理过程中的步骤数
    guidance_scale: float = 6.0,  # 指导尺度
    num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量
    dtype: torch.dtype = torch.bfloat16,  # 计算的数据类型
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # 生成类型限制
    seed: int = 42,  # 可重复性的种子
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1. 加载预训练的 CogVideoX 管道，使用指定的精度（bfloat16）。
    # 添加 device_map="balanced" 到 from_pretrained 函数中，移除 enable_model_cpu_offload() 函数以使用多 GPU。

    # 初始化图像变量
    image = None
    # 初始化视频变量
    video = None
    # 根据生成类型选择合适的管道
    if generate_type == "i2v":
        # 从预训练模型路径加载图像到视频管道，指定数据类型
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        # 加载输入的图像
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        # 从预训练模型路径加载文本到视频管道，指定数据类型
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        # 从预训练模型路径加载视频到视频管道，指定数据类型
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        # 加载输入的视频
        video = load_video(image_or_video_path)

    # 如果使用 lora，添加以下代码
    if lora_path:
        # 加载 lora 权重，指定权重文件和适配器名称
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        # 融合 lora 权重，设置缩放比例
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    # 2. 设置调度器
    # 可以更改为 `CogVideoXDPMScheduler` 或 `CogVideoXDDIMScheduler`
    # 推荐使用 `CogVideoXDDIMScheduler` 适用于 CogVideoX-2B
    # 使用 `CogVideoXDPMScheduler` 适用于 CogVideoX-5B / CogVideoX-5B-I2V

    # 使用 DDIM 调度器设置
    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    # 使用 DPM 调度器设置
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. 为模型启用 CPU 卸载
    # 如果有多个 GPU 或足够的 GPU 内存（如 H100），可以关闭此选项以减少推理时间
    # 并启用到 CUDA 设备
    # pipe.to("cuda")

    # 启用顺序 CPU 卸载
    pipe.enable_sequential_cpu_offload()

    # 启用 VAE 切片
    pipe.vae.enable_slicing()
    # 启用 VAE 平铺
    pipe.vae.enable_tiling()

    # 4. 根据提示生成视频帧
    # `num_frames` 是要生成的帧数
    # 默认值适用于 6 秒视频和 8 fps，并会加 1 帧作为第一帧，总共 49 帧
    if generate_type == "i2v":
        # 调用管道生成视频
        video_generate = pipe(
            prompt=prompt,  # 用于生成视频的提示文本
            image=image,  # 用作视频背景的图像路径
            num_videos_per_prompt=num_videos_per_prompt,  # 每个提示生成的视频数量
            num_inference_steps=num_inference_steps,  # 推理步骤数量
            num_frames=49,  # 要生成的帧数，版本 `0.30.3` 及之后更改为 49
            use_dynamic_cfg=True,  # 此 ID 用于 DPM 调度器，DDIM 调度器应为 False
            guidance_scale=guidance_scale,  # 指导尺度
            generator=torch.Generator().manual_seed(seed),  # 设置种子以确保可重复性
        ).frames[0]  # 获取生成的第一个视频帧
    elif generate_type == "t2v":
        # 调用管道生成视频
        video_generate = pipe(
            prompt=prompt,  # 用于生成视频的提示文本
            num_videos_per_prompt=num_videos_per_prompt,  # 每个提示生成的视频数量
            num_inference_steps=num_inference_steps,  # 推理步骤数量
            num_frames=49,  # 要生成的帧数
            use_dynamic_cfg=True,  # 此 ID 用于 DPM 调度器，DDIM 调度器应为 False
            guidance_scale=guidance_scale,  # 指导尺度
            generator=torch.Generator().manual_seed(seed),  # 设置种子以确保可重复性
        ).frames[0]  # 获取生成的第一个视频帧
    else:  # 如果前面的条件不满足，执行以下代码
        # 调用管道函数生成视频
        video_generate = pipe(
            prompt=prompt,  # 传入生成视频所需的提示文本
            video=video,  # 用作视频背景的文件路径
            num_videos_per_prompt=num_videos_per_prompt,  # 每个提示生成的视频数量
            num_inference_steps=num_inference_steps,  # 推理步骤数量
            # num_frames=49,  # 可选参数：生成的帧数，当前被注释
            use_dynamic_cfg=True,  # 启用动态配置
            guidance_scale=guidance_scale,  # 控制生成内容的引导程度
            generator=torch.Generator().manual_seed(seed),  # 设置随机数生成器的种子，以确保结果可复现
        ).frames[0]  # 获取生成的视频帧中的第一帧
    # 将生成的帧导出为视频文件，fps必须设置为8以符合原视频要求
    export_to_video(video_generate, output_path, fps=8)  # 调用导出函数，传入生成的视频和输出路径
# 仅在直接运行此脚本时执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器，描述程序功能
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    # 添加视频描述的命令行参数，必需
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    # 添加背景图片路径的命令行参数，默认值为 None
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    # 添加预训练模型路径的命令行参数，默认值为指定模型
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    # 添加 LoRA 权重路径的命令行参数，默认值为 None
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    # 添加 LoRA 权重秩的命令行参数，默认值为 128
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    # 添加生成视频保存路径的命令行参数，默认值为当前目录下的 output.mp4
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    # 添加无分类器引导的缩放比例的命令行参数，默认值为 6.0
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    # 添加推理过程中的步骤数的命令行参数，默认值为 50
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    # 添加每个提示生成视频数量的命令行参数，默认值为 1
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    # 添加视频生成类型的命令行参数，默认值为 't2v'
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    # 添加计算使用的数据类型的命令行参数，默认值为 'bfloat16'
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    # 添加用于重现性的随机种子的命令行参数，默认值为 42
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    # 解析命令行参数并将结果存储在 args 中
    args = parser.parse_args()
    # 根据指定的数据类型设置相应的 PyTorch 数据类型
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # 调用生成视频的函数，传递解析后的参数
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
    )
```