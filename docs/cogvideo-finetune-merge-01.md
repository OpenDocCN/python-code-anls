# CogVideo & CogVideoX 微调代码源码解析（二）



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

# `.\cogvideo-finetune\inference\cli_demo_quantization.py`

```py
# 该脚本演示如何使用 CogVideoX 通过文本提示生成视频，并使用量化功能。

# 注意事项：
# 必须从源代码安装 `torchao`，`torch`，`diffusers`，`accelerate` 库以使用量化功能。
# 仅支持 H100 或更高版本的 NVIDIA GPU 进行 FP-8 量化。
# 所有量化方案必须在 NVIDIA GPU 上使用。

# 运行脚本的示例命令：
# python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-2b --quantization_scheme fp8 --dtype float16
# python cli_demo_quantization.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --quantization_scheme fp8 --dtype bfloat16

import argparse  # 导入命令行参数解析库
import os  # 导入操作系统相关的功能库
import torch  # 导入 PyTorch 库
import torch._dynamo  # 导入 PyTorch 动态编译库
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline, CogVideoXDPMScheduler  # 从 diffusers 库导入多个模型和调度器
from diffusers.utils import export_to_video  # 导入用于导出视频的工具
from transformers import T5EncoderModel  # 从 transformers 库导入 T5 编码器模型
from torchao.quantization import quantize_, int8_weight_only  # 导入量化相关函数
from torchao.float8.inference import ActivationCasting, QuantConfig, quantize_to_float8  # 导入 FP8 量化相关工具

os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"  # 设置环境变量以启用 Torch 日志
torch._dynamo.config.suppress_errors = True  # 配置动态编译以抑制错误
torch.set_float32_matmul_precision("high")  # 设置浮点32矩阵乘法的精度为高
torch._inductor.config.conv_1x1_as_mm = True  # 配置 1x1 卷积为矩阵乘法
torch._inductor.config.coordinate_descent_tuning = True  # 启用坐标下降调优
torch._inductor.config.epilogue_fusion = False  # 禁用后处理融合
torch._inductor.config.coordinate_descent_check_all_directions = True  # 检查所有方向的坐标下降

def quantize_model(part, quantization_scheme):  # 定义量化模型的函数
    if quantization_scheme == "int8":  # 如果量化方案为 int8
        quantize_(part, int8_weight_only())  # 对模型进行 int8 量化
    elif quantization_scheme == "fp8":  # 如果量化方案为 fp8
        quantize_to_float8(part, QuantConfig(ActivationCasting.DYNAMIC))  # 对模型进行 fp8 量化
    return part  # 返回量化后的模型

def generate_video(  # 定义生成视频的函数
    prompt: str,  # 视频描述的文本提示
    model_path: str,  # 预训练模型的路径
    output_path: str = "./output.mp4",  # 生成视频的保存路径，默认为 ./output.mp4
    num_inference_steps: int = 50,  # 推理过程的步骤数，更多步骤可能导致更高质量
    guidance_scale: float = 6.0,  # 无分类器引导的尺度，较高的值可以更好地对齐提示
    num_videos_per_prompt: int = 1,  # 每个提示生成的视频数量
    quantization_scheme: str = "fp8",  # 使用的量化方案（'int8'，'fp8'）
    dtype: torch.dtype = torch.bfloat16,  # 计算的数据类型（默认为 torch.bfloat16）
):
    """
    根据给定提示生成视频并保存到指定路径。

    参数：
    - prompt (str): 要生成视频的描述。
    - model_path (str): 要使用的预训练模型的路径。
    - output_path (str): 生成的视频保存路径。
    - num_inference_steps (int): 推理过程的步骤数。更多步骤可以获得更好的质量。
    - guidance_scale (float): 无分类器引导的尺度。较高的值可能导致更好地对齐提示。
    - num_videos_per_prompt (int): 每个提示生成的视频数量。
    - quantization_scheme (str): 要使用的量化方案（'int8'，'fp8'）。
    - dtype (torch.dtype): 计算的数据类型（默认为 torch.bfloat16）。
    """

    text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)  # 从预训练模型加载文本编码器
    # 对文本编码器进行量化处理，以减少模型的内存占用和加速推理
    text_encoder = quantize_model(part=text_encoder, quantization_scheme=quantization_scheme)
    # 从预训练模型加载 3D Transformer 模型，指定模型路径和数据类型
    transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    # 对 Transformer 模型进行量化处理
    transformer = quantize_model(part=transformer, quantization_scheme=quantization_scheme)
    # 从预训练模型加载 VAE（变分自编码器），指定模型路径和数据类型
    vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
    # 对 VAE 进行量化处理
    vae = quantize_model(part=vae, quantization_scheme=quantization_scheme)
    # 创建视频生成管道，传入预训练的组件和数据类型
    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=dtype,
    )
    # 使用调度器配置初始化调度器，设置时间步间距为 "trailing"
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 使用 compile 会加快推理速度。第一次推理约需 30 分钟进行编译。
    # pipe.transformer.to(memory_format=torch.channels_last)

    # 对于 FP8 应该移除 CPU 卸载功能
    pipe.enable_model_cpu_offload()

    # 这对于 FP8 和 INT8 不是必要的，应移除此行
    # pipe.enable_sequential_cpu_offload()
    # 启用 VAE 的切片功能，以便进行高效处理
    pipe.vae.enable_slicing()
    # 启用 VAE 的平铺功能，以支持生成视频的高效处理
    pipe.vae.enable_tiling()
    # 调用管道生成视频，指定提示、每个提示的视频数量、推理步骤、帧数等参数
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    # 将生成的视频导出到指定路径，设置每秒帧数为 8
    export_to_video(video, output_path, fps=8)
# 当脚本被直接执行时，以下代码将会被运行
if __name__ == "__main__":
    # 创建一个参数解析器，提供脚本的功能描述
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    # 添加一个必需的参数，用于输入视频描述
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    # 添加一个可选参数，指定预训练模型的路径，默认为指定的路径
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    # 添加一个可选参数，指定生成视频的输出路径，默认为当前目录下的 output.mp4
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    # 添加一个可选参数，指定推理过程中的步骤数量，默认为 50
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    # 添加一个可选参数，指定无分类器引导的比例，默认为 6.0
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    # 添加一个可选参数，指定每个提示生成的视频数量，默认为 1
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    # 添加一个可选参数，指定计算的数据类型，默认为 bfloat16
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16', 'bfloat16')"
    )
    # 添加一个可选参数，指定量化方案，默认为 bf16，选择范围为 int8 和 fp8
    parser.add_argument(
        "--quantization_scheme",
        type=str,
        default="bf16",
        choices=["int8", "fp8"],
        help="The quantization scheme to use (int8, fp8)",
    )

    # 解析命令行参数并将结果存储在 args 中
    args = parser.parse_args()
    # 根据输入的数据类型设置 PyTorch 数据类型，默认为 bfloat16
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # 调用生成视频的函数，传入所有解析后的参数
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        quantization_scheme=args.quantization_scheme,
        dtype=dtype,
    )
```

# `.\cogvideo-finetune\inference\cli_vae_demo.py`

```py
"""
此脚本旨在演示如何使用 CogVideoX-2b VAE 模型进行视频编码和解码。
它允许将视频编码为潜在表示，解码回视频，或顺序执行这两项操作。
在运行脚本之前，请确保克隆了 CogVideoX Hugging Face 模型仓库，并将
`{your local diffusers path}` 参数设置为克隆仓库的路径。

命令 1：编码视频
使用 CogVideoX-5b VAE 模型编码位于 ../resources/videos/1.mp4 的视频。
内存使用量：编码时大约需要 ~18GB 的 GPU 内存。

如果您没有足够的 GPU 内存，我们在资源文件夹中提供了一个预编码的张量文件（encoded.pt），
您仍然可以运行解码命令。

$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode encode

命令 2：解码视频

将存储在 encoded.pt 中的潜在表示解码回视频。
内存使用量：解码时大约需要 ~4GB 的 GPU 内存。
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --encoded_path ./encoded.pt --mode decode

命令 3：编码和解码视频
编码位于 ../resources/videos/1.mp4 的视频，然后立即解码。
内存使用量：编码 需要 34GB + 解码需要 19GB（顺序执行）。
$ python cli_vae_demo.py --model_path {your local diffusers path}/CogVideoX-2b/vae/ --video_path ../resources/videos/1.mp4 --mode both
"""

# 导入 argparse 模块用于处理命令行参数
import argparse
# 导入 torch 库用于深度学习相关的操作
import torch
# 导入 imageio 库用于读取视频文件
import imageio
# 从 diffusers 库导入 AutoencoderKLCogVideoX 模型
from diffusers import AutoencoderKLCogVideoX
# 从 torchvision 库导入 transforms，用于数据转换
from torchvision import transforms
# 导入 numpy 库用于数值计算
import numpy as np


def encode_video(model_path, video_path, dtype, device):
    """
    加载预训练的 AutoencoderKLCogVideoX 模型并编码视频帧。

    参数：
    - model_path (str): 预训练模型的路径。
    - video_path (str): 视频文件的路径。
    - dtype (torch.dtype): 计算所用的数据类型。
    - device (str): 计算所用的设备（例如，"cuda" 或 "cpu"）。

    返回：
    - torch.Tensor: 编码后的视频帧。
    """

    # 从指定路径加载预训练的模型，并将其移动到指定设备
    model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)

    # 启用切片功能以优化内存使用
    model.enable_slicing()
    # 启用平铺功能以处理大图像
    model.enable_tiling()

    # 使用 ffmpeg 读取视频文件
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    # 将视频的每一帧转换为张量并存储在列表中
    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    # 关闭视频读取器
    video_reader.close()

    # 将帧列表转换为张量，调整维度，并将其移动到指定设备和数据类型
    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)

    # 在不计算梯度的上下文中进行编码
    with torch.no_grad():
        # 使用模型编码帧，并从中获取样本
        encoded_frames = model.encode(frames_tensor)[0].sample()
    # 返回编码后的帧
    return encoded_frames


def decode_video(model_path, encoded_tensor_path, dtype, device):
    """
    加载预训练的 AutoencoderKLCogVideoX 模型并解码编码的视频帧。

    参数：
    - model_path (str): 预训练模型的路径。
    - encoded_tensor_path (str): 编码张量文件的路径。
    # dtype 参数指定计算时使用的数据类型
        - dtype (torch.dtype): The data type for computation.
        # device 参数指定用于计算的设备（例如，“cuda”或“cpu”）
        - device (str): The device to use for computation (e.g., "cuda" or "cpu").
    
        # 返回解码后的视频帧
        Returns:
        - torch.Tensor: The decoded video frames.
        """
        # 从预训练模型加载 AutoencoderKLCogVideoX，并将其转移到指定设备上，设置数据类型
        model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=dtype).to(device)
        # 从指定路径加载编码的张量，并将其转移到设备和指定的数据类型上
        encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
        # 在不计算梯度的上下文中解码编码的帧
        with torch.no_grad():
            # 调用模型解码编码的帧，并获取解码后的样本
            decoded_frames = model.decode(encoded_frames).sample
        # 返回解码后的帧
        return decoded_frames
# 定义一个函数，用于保存视频帧到视频文件
def save_video(tensor, output_path):
    """
    保存视频帧到视频文件。

    参数：
    - tensor (torch.Tensor): 视频帧的张量。
    - output_path (str): 输出视频的保存路径。
    """
    # 将张量转换为浮点32位类型
    tensor = tensor.to(dtype=torch.float32)
    # 将张量的第一个维度去掉，重新排列维度，并转移到 CPU，再转换为 NumPy 数组
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    # 将帧的值裁剪到 0 到 1 之间，并乘以 255 以转换为像素值
    frames = np.clip(frames, 0, 1) * 255
    # 将帧的数据类型转换为无符号 8 位整数
    frames = frames.astype(np.uint8)
    # 创建一个视频写入对象，设置输出路径和帧率为 8
    writer = imageio.get_writer(output_path + "/output.mp4", fps=8)
    # 遍历每一帧，将其添加到视频写入对象中
    for frame in frames:
        writer.append_data(frame)
    # 关闭视频写入对象，完成写入
    writer.close()


# 如果当前脚本是主程序，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器，用于处理命令行参数
    parser = argparse.ArgumentParser(description="CogVideoX encode/decode demo")
    # 添加一个参数，用于指定模型的路径
    parser.add_argument("--model_path", type=str, required=True, help="The path to the CogVideoX model")
    # 添加一个参数，用于指定视频文件的路径（用于编码）
    parser.add_argument("--video_path", type=str, help="The path to the video file (for encoding)")
    # 添加一个参数，用于指定编码的张量文件的路径（用于解码）
    parser.add_argument("--encoded_path", type=str, help="The path to the encoded tensor file (for decoding)")
    # 添加一个参数，用于指定输出文件的保存路径，默认为当前目录
    parser.add_argument("--output_path", type=str, default=".", help="The path to save the output file")
    # 添加一个参数，指定模式：编码、解码或两者
    parser.add_argument(
        "--mode", type=str, choices=["encode", "decode", "both"], required=True, help="Mode: encode, decode, or both"
    )
    # 添加一个参数，指定计算的数据类型，默认为 'bfloat16'
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    # 添加一个参数，指定用于计算的设备，默认为 'cuda'
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to use for computation (e.g., 'cuda' or 'cpu')"
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据指定的设备创建一个设备对象
    device = torch.device(args.device)
    # 根据指定的数据类型设置数据类型，默认为 bfloat16
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # 根据模式选择编码、解码或两者的操作
    if args.mode == "encode":
        # 确保提供了视频路径用于编码
        assert args.video_path, "Video path must be provided for encoding."
        # 调用编码函数，将视频编码为张量
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        # 将编码后的张量保存到指定路径
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        # 打印完成编码的消息
        print(f"Finished encoding the video to a tensor, save it to a file at {encoded_output}/encoded.pt")
    elif args.mode == "decode":
        # 确保提供了编码张量的路径用于解码
        assert args.encoded_path, "Encoded tensor path must be provided for decoding."
        # 调用解码函数，将编码的张量解码为视频帧
        decoded_output = decode_video(args.model_path, args.encoded_path, dtype, device)
        # 调用保存视频的函数，将解码后的输出保存为视频文件
        save_video(decoded_output, args.output_path)
        # 打印完成解码的消息
        print(f"Finished decoding the video and saved it to a file at {args.output_path}/output.mp4")
    elif args.mode == "both":
        # 确保提供了视频路径用于编码
        assert args.video_path, "Video path must be provided for encoding."
        # 调用编码函数，将视频编码为张量
        encoded_output = encode_video(args.model_path, args.video_path, dtype, device)
        # 将编码后的张量保存到指定路径
        torch.save(encoded_output, args.output_path + "/encoded.pt")
        # 调用解码函数，将保存的张量解码为视频帧
        decoded_output = decode_video(args.model_path, args.output_path + "/encoded.pt", dtype, device)
        # 调用保存视频的函数，将解码后的输出保存为视频文件
        save_video(decoded_output, args.output_path)
```

# `.\cogvideo-finetune\inference\convert_demo.py`

```py
"""
该CogVideoX模型旨在根据详细且高度描述性的提示生成高质量的视频。
当提供精细的、细致的提示时，模型表现最佳，这能提高视频生成的质量。
该脚本旨在帮助将简单的用户输入转换为适合CogVideoX的详细提示。
它可以处理文本到视频（t2v）和图像到视频（i2v）的转换。

- 对于文本到视频，只需提供提示。
- 对于图像到视频，提供图像文件的路径和可选的用户输入。
图像将被编码并作为请求的一部分发送给Azure OpenAI。

### 如何运行：
运行脚本进行**文本到视频**：
    $ python convert_demo.py --prompt "一个女孩骑自行车。" --type "t2v"

运行脚本进行**图像到视频**：
    $ python convert_demo.py --prompt "猫在跑" --type "i2v" --image_path "/path/to/your/image.jpg"
"""

# 导入argparse库以处理命令行参数
import argparse
# 从openai库导入OpenAI和AzureOpenAI类
from openai import OpenAI, AzureOpenAI
# 导入base64库以进行数据编码
import base64
# 从mimetypes库导入guess_type函数以推测文件类型
from mimetypes import guess_type

# 定义文本到视频的系统提示
sys_prompt_t2v = """您是一个创建视频的机器人团队的一部分。您与一个助手机器人合作，助手会绘制您所说的方括号中的任何内容。

例如，输出“一个阳光穿过树木的美丽清晨”将触发您的伙伴机器人输出如描述的森林早晨的视频。您将被希望创建详细、精彩视频的人所提示。完成此任务的方法是将他们的简短提示转化为极其详细和描述性的内容。
需要遵循一些规则：

您每次用户请求只能输出一个视频描述。

当请求修改时，您不应简单地将描述变得更长。您应重构整个描述，以整合建议。
有时用户不想要修改，而是希望得到一个新图像。在这种情况下，您应忽略与用户的先前对话。

视频描述必须与以下示例的单词数量相同。多余的单词将被忽略。
"""

# 定义图像到视频的系统提示
sys_prompt_i2v = """
**目标**：**根据输入图像和用户输入给出高度描述性的视频说明。** 作为专家，深入分析图像，运用丰富的创造力和细致的思考。在描述图像的细节时，包含适当的动态信息，以确保视频说明包含合理的动作和情节。如果用户输入不为空，则说明应根据用户的输入进行扩展。

**注意**：输入图像是视频的第一帧，输出视频说明应描述从当前图像开始的运动。用户输入是可选的，可以为空。

**注意**：不要包含相机转场！！！不要包含画面切换！！！不要包含视角转换！！！

**回答风格**：
# 定义将图像文件转换为 URL 的函数
def image_to_url(image_path):
    # 根据图像路径猜测其 MIME 类型，第二个返回值忽略
    mime_type, _ = guess_type(image_path)
    # 如果无法猜测 MIME 类型，则设置为通用二进制流类型
    if mime_type is None:
        mime_type = "application/octet-stream"
    # 以二进制模式打开图像文件
    with open(image_path, "rb") as image_file:
        # 读取图像文件内容并进行 Base64 编码，解码为 UTF-8 格式
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    # 返回格式化的 Base64 数据 URL 字符串
    return f"data:{mime_type};base64,{base64_encoded_data}"


# 定义将提示转换为可用于模型推理的格式的函数
def convert_prompt(prompt: str, retry_times: int = 3, type: str = "t2v", image_path: str = None):
    """
    将提示转换为可用于模型推理的格式
    """

    # 创建 OpenAI 客户端实例
    client = OpenAI()
    ## 如果使用 Azure OpenAI，请取消注释下面一行并注释上面一行
    # client = AzureOpenAI(
    #     api_key="",
    #     api_version="",
    #     azure_endpoint=""
    # )

    # 去除提示字符串两端的空白
    text = prompt.strip()
    # 返回未处理的提示
    return prompt


# 如果当前脚本是主程序
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加提示参数，类型为字符串，必需
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to convert")
    # 添加重试次数参数，类型为整数，默认值为 3
    parser.add_argument("--retry_times", type=int, default=3, help="Number of times to retry the conversion")
    # 添加转换类型参数，类型为字符串，默认值为 "t2v"
    parser.add_argument("--type", type=str, default="t2v", help="Type of conversion (t2v or i2v)")
    # 添加图像路径参数，类型为字符串，默认值为 None
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image file")
    # 解析命令行参数并存储在 args 中
    args = parser.parse_args()

    # 调用 convert_prompt 函数进行提示转换
    converted_prompt = convert_prompt(args.prompt, args.retry_times, args.type, args.image_path)
    # 打印转换后的提示
    print(converted_prompt)
```

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

---
title: CogVideoX-5B
emoji: 🎥
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.42.0
suggested_hardware: a10g-large
suggested_storage: large
app_port: 7860
app_file: app.py
models:
  - THUDM/CogVideoX-5b
tags:
  - cogvideox
  - video-generation
  - thudm
short_description: Text-to-Video
disable_embedding: false
---

# Gradio Composite Demo

This Gradio demo integrates the CogVideoX-5B model, allowing you to perform video inference directly in your browser. It
supports features like UpScale, RIFE, and other functionalities.

## Environment Setup

Set the following environment variables in your system:

+ OPENAI_API_KEY = your_api_key
+ OPENAI_BASE_URL= your_base_url
+ GRADIO_TEMP_DIR= gradio_tmp

## Installation

```py
pip install -r requirements.txt 
```

## Running the code

```py
python gradio_web_demo.py
```




# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet.py`

```py
# 从当前模块导入所有内容
from .refine import *


# 定义转置卷积函数，创建反卷积层
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个包含转置卷积和PReLU激活的顺序容器
    return nn.Sequential(
        # 创建转置卷积层，设置输入输出通道、卷积核大小、步幅和填充
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        # 添加PReLU激活层
        nn.PReLU(out_planes),
    )


# 定义卷积函数，创建卷积层
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个包含卷积层和PReLU激活的顺序容器
    return nn.Sequential(
        # 创建卷积层，设置输入输出通道、卷积核大小、步幅、填充、扩张和偏置
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        # 添加PReLU激活层
        nn.PReLU(out_planes),
    )


# 定义IFBlock类，继承自nn.Module
class IFBlock(nn.Module):
    # 初始化方法，设置输入通道和常量c
    def __init__(self, in_planes, c=64):
        # 调用父类构造函数
        super(IFBlock, self).__init__()
        # 定义初始卷积序列
        self.conv0 = nn.Sequential(
            # 创建第一个卷积层
            conv(in_planes, c // 2, 3, 2, 1),
            # 创建第二个卷积层
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义卷积块，包含多个卷积层
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 定义最后的转置卷积层
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    # 前向传播方法，定义数据流动
    def forward(self, x, flow, scale):
        # 如果scale不等于1，则进行上采样
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        # 如果flow不为None，则进行上采样并与x拼接
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        # 经过初始卷积序列
        x = self.conv0(x)
        # 经过卷积块，并与输入相加实现残差连接
        x = self.convblock(x) + x
        # 经过最后的转置卷积层
        tmp = self.lastconv(x)
        # 对输出进行上采样
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        # 分离出flow和mask
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        # 返回flow和mask
        return flow, mask


# 定义IFNet类，继承自nn.Module
class IFNet(nn.Module):
    # 初始化方法，定义多个IFBlock
    def __init__(self):
        # 调用父类构造函数
        super(IFNet, self).__init__()
        # 创建第一个IFBlock，输入通道为6，c=240
        self.block0 = IFBlock(6, c=240)
        # 创建第二个IFBlock，输入通道为13+4，c=150
        self.block1 = IFBlock(13 + 4, c=150)
        # 创建第三个IFBlock，输入通道为13+4，c=90
        self.block2 = IFBlock(13 + 4, c=90)
        # 创建教师网络的IFBlock，输入通道为16+4，c=90
        self.block_tea = IFBlock(16 + 4, c=90)
        # 创建上下文网络
        self.contextnet = Contextnet()
        # 创建UNet网络
        self.unet = Unet()
    # 前向传播函数，接受输入图像和其他参数
        def forward(self, x, scale=[4, 2, 1], timestep=0.5):
            # 将输入图像分为三部分：前景图像、背景图像和真实图像
            img0 = x[:, :3]
            img1 = x[:, 3:6]
            gt = x[:, 6:]  # 在推理时，gt 为 None
            flow_list = []  # 用于存储流信息的列表
            merged = []  # 用于存储合并结果的列表
            mask_list = []  # 用于存储掩膜信息的列表
            warped_img0 = img0  # 初始化扭曲后的前景图像
            warped_img1 = img1  # 初始化扭曲后的背景图像
            flow = None  # 初始化流为 None
            loss_distill = 0  # 初始化蒸馏损失
            stu = [self.block0, self.block1, self.block2]  # 学生模型的块列表
            for i in range(3):  # 遍历三个模型块
                if flow != None:  # 如果流信息不为 None
                    # 合并图像和流信息，进行前向传播以获取流和掩膜
                    flow_d, mask_d = stu[i](
                        torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i]
                    )
                    flow = flow + flow_d  # 更新流信息
                    mask = mask + mask_d  # 更新掩膜信息
                else:  # 如果流为 None
                    # 仅使用图像进行前向传播，获取流和掩膜
                    flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
                mask_list.append(torch.sigmoid(mask))  # 应用sigmoid函数并存储掩膜
                flow_list.append(flow)  # 存储流信息
                # 根据流信息扭曲图像
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                merged_student = (warped_img0, warped_img1)  # 合并扭曲后的图像
                merged.append(merged_student)  # 存储合并结果
            if gt.shape[1] == 3:  # 如果真实图像的通道数为3
                # 进行教师模型的前向传播
                flow_d, mask_d = self.block_tea(
                    torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
                )
                flow_teacher = flow + flow_d  # 更新教师流信息
                warped_img0_teacher = warp(img0, flow_teacher[:, :2])  # 扭曲教师图像
                warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])  # 扭曲教师图像
                mask_teacher = torch.sigmoid(mask + mask_d)  # 更新教师掩膜
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)  # 合并教师图像
            else:  # 如果真实图像不为3通道
                flow_teacher = None  # 教师流信息为 None
                merged_teacher = None  # 教师合并结果为 None
            for i in range(3):  # 遍历三个模型块
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])  # 根据掩膜合并图像
                if gt.shape[1] == 3:  # 如果真实图像的通道数为3
                    # 计算损失掩膜
                    loss_mask = (
                        ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01)
                        .float()
                        .detach()
                    )
                    # 累加蒸馏损失
                    loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
            c0 = self.contextnet(img0, flow[:, :2])  # 计算上下文信息
            c1 = self.contextnet(img1, flow[:, 2:4])  # 计算上下文信息
            # 使用 U-Net 进行图像处理
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1  # 处理 U-Net 输出结果
            merged[2] = torch.clamp(merged[2] + res, 0, 1)  # 限制合并结果在 [0, 1] 范围内
            # 返回流列表、掩膜列表、合并结果、教师流信息和教师合并结果，以及蒸馏损失
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet_2R.py`

```py
# 从相对路径引入 refine_2R 模块的所有内容
from .refine_2R import *


# 定义反卷积层，输入和输出通道数、卷积核大小、步幅和填充
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个序列，包含反卷积层和 PReLU 激活函数
    return nn.Sequential(
        # 创建反卷积层，指定输入通道数、输出通道数、卷积核大小、步幅和填充
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        # 添加 PReLU 激活函数，输出通道数为 out_planes
        nn.PReLU(out_planes),
    )


# 定义卷积层，输入和输出通道数、卷积核大小、步幅、填充和扩张
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个序列，包含卷积层和 PReLU 激活函数
    return nn.Sequential(
        nn.Conv2d(
            # 输入通道数
            in_planes,
            # 输出通道数
            out_planes,
            # 卷积核大小
            kernel_size=kernel_size,
            # 步幅
            stride=stride,
            # 填充
            padding=padding,
            # 扩张
            dilation=dilation,
            # 启用偏置项
            bias=True,
        ),
        # 添加 PReLU 激活函数，输出通道数为 out_planes
        nn.PReLU(out_planes),
    )


# 定义 IFBlock 类，继承自 nn.Module
class IFBlock(nn.Module):
    # 初始化方法，定义输入通道数和常量 c
    def __init__(self, in_planes, c=64):
        # 调用父类构造函数
        super(IFBlock, self).__init__()
        # 定义第一个卷积序列
        self.conv0 = nn.Sequential(
            # 第一个卷积，输入通道为 in_planes，输出通道为 c // 2
            conv(in_planes, c // 2, 3, 1, 1),
            # 第二个卷积，输入通道为 c // 2，输出通道为 c
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义主卷积块，由多个卷积层组成
        self.convblock = nn.Sequential(
            # 重复调用卷积函数，输入输出通道均为 c
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 定义最后的反卷积层，输入通道为 c，输出通道为 5
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    # 前向传播方法，接受输入 x、光流 flow 和缩放 scale
    def forward(self, x, flow, scale):
        # 如果缩放不为 1，则对 x 进行上采样
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        # 如果光流不为空，则对 flow 进行上采样并与 x 连接
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            # 将 x 和 flow 在通道维度上连接
            x = torch.cat((x, flow), 1)
        # 通过 conv0 处理 x
        x = self.conv0(x)
        # 通过 convblock 处理 x，并与原始 x 相加
        x = self.convblock(x) + x
        # 通过最后的反卷积层处理 x
        tmp = self.lastconv(x)
        # 对 tmp 进行上采样
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        # 从 tmp 中提取 flow，缩放回原始尺寸
        flow = tmp[:, :4] * scale
        # 提取 mask，形状为 (batch_size, 1, H, W)
        mask = tmp[:, 4:5]
        # 返回光流和掩码
        return flow, mask


# 定义 IFNet 类，继承自 nn.Module
class IFNet(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类构造函数
        super(IFNet, self).__init__()
        # 定义多个 IFBlock 实例，输入通道数和常量 c
        self.block0 = IFBlock(6, c=240)
        self.block1 = IFBlock(13 + 4, c=150)
        self.block2 = IFBlock(13 + 4, c=90)
        self.block_tea = IFBlock(16 + 4, c=90)
        # 创建上下文网络和 U-Net 实例
        self.contextnet = Contextnet()
        self.unet = Unet()
    # 定义前向传播函数，接受输入张量 x，缩放比例和时间步长
    def forward(self, x, scale=[4, 2, 1], timestep=0.5):
        # 从输入 x 中提取第一张图像的 RGB 通道
        img0 = x[:, :3]
        # 从输入 x 中提取第二张图像的 RGB 通道
        img1 = x[:, 3:6]
        # 从输入 x 中提取地面真值，推理时该值为 None
        gt = x[:, 6:]  # In inference time, gt is None
        # 初始化流列表
        flow_list = []
        # 初始化合并结果列表
        merged = []
        # 初始化掩膜列表
        mask_list = []
        # 初始化扭曲的第一张图像为原始图像
        warped_img0 = img0
        # 初始化扭曲的第二张图像为原始图像
        warped_img1 = img1
        # 初始化流为 None
        flow = None
        # 初始化蒸馏损失
        loss_distill = 0
        # 获取模型的不同块
        stu = [self.block0, self.block1, self.block2]
        # 循环处理三个块
        for i in range(3):
            # 如果流不为 None，则进行流的更新
            if flow != None:
                # 从当前块中计算流和掩膜
                flow_d, mask_d = stu[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i]
                )
                # 更新流
                flow = flow + flow_d
                # 更新掩膜
                mask = mask + mask_d
            else:
                # 对于第一轮，流为 None，直接计算流和掩膜
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            # 将当前掩膜应用 sigmoid 函数，添加到掩膜列表
            mask_list.append(torch.sigmoid(mask))
            # 将当前流添加到流列表
            flow_list.append(flow)
            # 基于流扭曲第一张图像
            warped_img0 = warp(img0, flow[:, :2])
            # 基于流扭曲第二张图像
            warped_img1 = warp(img1, flow[:, 2:4])
            # 合并扭曲后的图像
            merged_student = (warped_img0, warped_img1)
            # 添加合并结果到合并列表
            merged.append(merged_student)
        # 如果地面真值的通道数为 3
        if gt.shape[1] == 3:
            # 从教师模型计算流和掩膜
            flow_d, mask_d = self.block_tea(
                torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
            )
            # 更新教师流
            flow_teacher = flow + flow_d
            # 基于教师流扭曲第一张图像
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            # 基于教师流扭曲第二张图像
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            # 更新教师掩膜
            mask_teacher = torch.sigmoid(mask + mask_d)
            # 根据教师掩膜合并图像
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            # 如果没有地面真值，则教师流和合并教师图像为 None
            flow_teacher = None
            merged_teacher = None
        # 循环处理三个块
        for i in range(3):
            # 根据掩膜合并当前图像
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            # 如果地面真值的通道数为 3
            if gt.shape[1] == 3:
                # 计算损失掩膜，判断当前合并结果是否优于教师合并结果
                loss_mask = (
                    ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01)
                    .float()
                    .detach()
                )
                # 更新蒸馏损失
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        # 通过上下文网络计算第一张图像的上下文特征
        c0 = self.contextnet(img0, flow[:, :2])
        # 通过上下文网络计算第二张图像的上下文特征
        c1 = self.contextnet(img1, flow[:, 2:4])
        # 通过 U-Net 计算临时结果
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        # 处理临时结果，准备返回
        res = tmp[:, :3] * 2 - 1
        # 更新合并结果，确保在 [0, 1] 范围内
        merged[2] = torch.clamp(merged[2] + res, 0, 1)
        # 返回流列表、第二个掩膜、合并结果、教师流、教师合并结果和蒸馏损失
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet_HDv3.py`

```py
# 导入 PyTorch 和其神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 从同一目录导入 warp 函数
from .warplayer import warp

# 检查是否可用 CUDA，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义卷积层的构造函数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        # 创建一个卷积层
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        # 使用 PReLU 激活函数
        nn.PReLU(out_planes),
    )


# 定义卷积层加批量归一化的构造函数
def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        # 创建卷积层，不使用偏置
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        # 添加批量归一化层
        nn.BatchNorm2d(out_planes),
        # 使用 PReLU 激活函数
        nn.PReLU(out_planes),
    )


# 定义 IFBlock 类，继承自 nn.Module
class IFBlock(nn.Module):
    # 初始化方法，接受输入通道数和常数 c
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        # 定义第一个卷积序列
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义多个卷积块
        self.convblock0 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock1 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock2 = nn.Sequential(conv(c, c), conv(c, c))
        self.convblock3 = nn.Sequential(conv(c, c), conv(c, c))
        # 定义反卷积层，恢复特征图尺寸
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 4, 4, 2, 1),
        )
        # 定义另一个反卷积层，输出单通道特征
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.PReLU(c // 2),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
        )

    # 前向传播方法
    def forward(self, x, flow, scale=1):
        # 调整输入图像尺寸
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        # 调整光流尺寸并缩放
        flow = (
            F.interpolate(
                flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
            )
            * 1.0
            / scale
        )
        # 连接 x 和 flow，经过卷积处理
        feat = self.conv0(torch.cat((x, flow), 1))
        # 通过多个卷积块进行特征增强
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat
        # 处理 flow
        flow = self.conv1(feat)
        # 处理 mask
        mask = self.conv2(feat)
        # 恢复 flow 的尺寸并缩放
        flow = (
            F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            * scale
        )
        # 恢复 mask 的尺寸
        mask = F.interpolate(
            mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False
        )
        # 返回光流和掩码
        return flow, mask


# 定义 IFNet 类，继承自 nn.Module
class IFNet(nn.Module):
    # 初始化 IFNet 类的构造函数
        def __init__(self):
            # 调用父类的构造函数
            super(IFNet, self).__init__()
            # 创建第一个 IFBlock，输入通道为 7 + 4，参数 c 设置为 90
            self.block0 = IFBlock(7 + 4, c=90)
            # 创建第二个 IFBlock，输入通道为 7 + 4，参数 c 设置为 90
            self.block1 = IFBlock(7 + 4, c=90)
            # 创建第三个 IFBlock，输入通道为 7 + 4，参数 c 设置为 90
            self.block2 = IFBlock(7 + 4, c=90)
            # 创建第四个 IFBlock，输入通道为 10 + 4，参数 c 设置为 90
            self.block_tea = IFBlock(10 + 4, c=90)
            # 上下文网络的实例化（被注释掉）
            # self.contextnet = Contextnet()
            # UNet 的实例化（被注释掉）
            # self.unet = Unet()
    
        # 前向传播函数，处理输入 x 和缩放列表，训练标志为 False
        def forward(self, x, scale_list=[4, 2, 1], training=False):
            # 如果不是训练模式
            if training == False:
                # 获取通道数，假设输入有两个部分
                channel = x.shape[1] // 2
                # 将前半部分赋值给 img0
                img0 = x[:, :channel]
                # 将后半部分赋值给 img1
                img1 = x[:, channel:]
            # 初始化流列表
            flow_list = []
            # 初始化合并列表
            merged = []
            # 初始化掩码列表
            mask_list = []
            # 将 img0 赋值给 warped_img0
            warped_img0 = img0
            # 将 img1 赋值给 warped_img1
            warped_img1 = img1
            # 创建一个与 x 的前四个通道相同的流，初始化为零
            flow = (x[:, :4]).detach() * 0
            # 创建一个与 x 的第一个通道相同的掩码，初始化为零
            mask = (x[:, :1]).detach() * 0
            # 初始化约束损失为零
            loss_cons = 0
            # 创建一个包含 block0、block1 和 block2 的列表
            block = [self.block0, self.block1, self.block2]
            # 循环三次，处理三个块
            for i in range(3):
                # 通过块处理图像和流，获取 f0 和 m0
                f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
                # 通过块处理逆向图像和流，获取 f1 和 m1
                f1, m1 = block[i](
                    torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1),
                    torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                    scale=scale_list[i],
                )
                # 更新流，添加平均值
                flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                # 更新掩码，添加平均值
                mask = mask + (m0 + (-m1)) / 2
                # 将掩码添加到掩码列表
                mask_list.append(mask)
                # 将流添加到流列表
                flow_list.append(flow)
                # 对 img0 进行光流变形，更新 warped_img0
                warped_img0 = warp(img0, flow[:, :2])
                # 对 img1 进行光流变形，更新 warped_img1
                warped_img1 = warp(img1, flow[:, 2:4])
                # 将变形后的图像添加到合并列表
                merged.append((warped_img0, warped_img1))
            """
            # 计算上下文特征 c0（被注释掉）
            c0 = self.contextnet(img0, flow[:, :2])
            # 计算上下文特征 c1（被注释掉）
            c1 = self.contextnet(img1, flow[:, 2:4])
            # 通过 UNet 计算临时结果（被注释掉）
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            # 处理临时结果，得到最终结果 res（被注释掉）
            res = tmp[:, 1:4] * 2 - 1
            """
            # 对每个掩码应用 sigmoid 函数
            for i in range(3):
                mask_list[i] = torch.sigmoid(mask_list[i])
                # 合并 warped_img0 和 warped_img1，应用当前掩码
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
                # 进行范围限制（被注释掉）
                # merged[i] = torch.clamp(merged[i] + res, 0, 1)
            # 返回流列表、最后的掩码和合并后的图像
            return flow_list, mask_list[2], merged
```

# `.\cogvideo-finetune\inference\gradio_composite_demo\rife\IFNet_m.py`

```py
# 从当前包中导入 refine 模块的所有内容
from .refine import *


# 定义反卷积层的构造函数
def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    # 返回一个序列，包括一个反卷积层和一个 PReLU 激活层
    return nn.Sequential(
        # 定义反卷积层，输入通道数、输出通道数、卷积核大小、步幅和填充
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        # 定义 PReLU 激活函数，输出通道数
        nn.PReLU(out_planes),
    )


# 定义卷积层的构造函数
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    # 返回一个序列，包括一个卷积层和一个 PReLU 激活层
    return nn.Sequential(
        # 定义卷积层，输入通道数、输出通道数、卷积核大小、步幅、填充、扩张和偏置
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        # 定义 PReLU 激活函数，输出通道数
        nn.PReLU(out_planes),
    )


# 定义 IFBlock 类，继承自 nn.Module
class IFBlock(nn.Module):
    # 初始化方法，定义输入通道和常量 c 的值
    def __init__(self, in_planes, c=64):
        # 调用父类初始化方法
        super(IFBlock, self).__init__()
        # 定义第一个卷积模块，包含两层卷积
        self.conv0 = nn.Sequential(
            # 第一层卷积，输入通道为 in_planes，输出通道为 c // 2
            conv(in_planes, c // 2, 3, 2, 1),
            # 第二层卷积，输入通道为 c // 2，输出通道为 c
            conv(c // 2, c, 3, 2, 1),
        )
        # 定义卷积块，包含多层卷积
        self.convblock = nn.Sequential(
            # 逐层定义卷积，均为输入通道 c，输出通道 c
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        # 定义最后的反卷积层，输出通道为 5
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    # 前向传播方法，处理输入 x、流量 flow 和缩放比例 scale
    def forward(self, x, flow, scale):
        # 如果缩放比例不为 1，调整输入 x 的大小
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        # 如果 flow 不为 None，调整 flow 的大小
        if flow != None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            # 将 x 和 flow 在通道维度上进行拼接
            x = torch.cat((x, flow), 1)
        # 通过卷积模块 conv0 处理 x
        x = self.conv0(x)
        # 通过卷积块处理 x，并与原始 x 相加
        x = self.convblock(x) + x
        # 通过最后的反卷积层处理 x，得到 tmp
        tmp = self.lastconv(x)
        # 调整 tmp 的大小以匹配原始缩放
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        # 从 tmp 中提取 flow，并进行缩放
        flow = tmp[:, :4] * scale * 2
        # 提取掩码
        mask = tmp[:, 4:5]
        # 返回 flow 和掩码
        return flow, mask


# 定义 IFNet_m 类，继承自 nn.Module
class IFNet_m(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类初始化方法
        super(IFNet_m, self).__init__()
        # 定义多个 IFBlock 实例，输入通道数和常量 c 的值不同
        self.block0 = IFBlock(6 + 1, c=240)
        self.block1 = IFBlock(13 + 4 + 1, c=150)
        self.block2 = IFBlock(13 + 4 + 1, c=90)
        self.block_tea = IFBlock(16 + 4 + 1, c=90)
        # 定义上下文网络
        self.contextnet = Contextnet()
        # 定义 U-Net 网络
        self.unet = Unet()
    # 定义前向传播函数，接受输入x、缩放比例、时间步长和是否返回流的标志
        def forward(self, x, scale=[4, 2, 1], timestep=0.5, returnflow=False):
            # 计算时间步长，使用x的第一个通道并设置为默认值
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
            # 获取输入x的前三个通道作为img0
            img0 = x[:, :3]
            # 获取输入x的第四到第六个通道作为img1
            img1 = x[:, 3:6]
            # 获取输入x的其余部分作为gt，在推理时gt为None
            gt = x[:, 6:]  # In inference time, gt is None
            # 初始化流列表和合并结果列表
            flow_list = []
            merged = []
            mask_list = []
            # 将img0和img1赋值给扭曲后的图像
            warped_img0 = img0
            warped_img1 = img1
            # 初始化流和蒸馏损失
            flow = None
            loss_distill = 0
            # 定义包含多个网络模块的列表
            stu = [self.block0, self.block1, self.block2]
            # 对于每个模块，进行三次循环
            for i in range(3):
                # 如果已有流，则进行流和掩码的计算
                if flow != None:
                    flow_d, mask_d = stu[i](
                        # 拼接输入，包含img0、img1、时间步长、扭曲后的图像和掩码
                        torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1), flow, scale=scale[i]
                    )
                    # 更新流和掩码
                    flow = flow + flow_d
                    mask = mask + mask_d
                else:
                    # 第一次计算流和掩码
                    flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None, scale=scale[i])
                # 将掩码经过sigmoid激活后加入掩码列表
                mask_list.append(torch.sigmoid(mask))
                # 将流加入流列表
                flow_list.append(flow)
                # 使用流对img0和img1进行扭曲
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                # 合并扭曲后的图像
                merged_student = (warped_img0, warped_img1)
                merged.append(merged_student)
            # 如果gt的通道数为3，则进行教师网络的计算
            if gt.shape[1] == 3:
                flow_d, mask_d = self.block_tea(
                    # 拼接输入，包括img0、img1、时间步长、扭曲后的图像、掩码和gt
                    torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
                )
                # 更新教师网络的流
                flow_teacher = flow + flow_d
                # 扭曲教师网络的图像
                warped_img0_teacher = warp(img0, flow_teacher[:, :2])
                warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
                # 计算教师网络的掩码
                mask_teacher = torch.sigmoid(mask + mask_d)
                # 合并教师网络的结果
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
            else:
                # 如果没有gt，教师网络的流和合并结果为None
                flow_teacher = None
                merged_teacher = None
            # 对于每个模块，合并结果并计算损失
            for i in range(3):
                merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
                # 如果gt的通道数为3，计算蒸馏损失
                if gt.shape[1] == 3:
                    loss_mask = (
                        # 判断合并结果的绝对误差是否大于教师网络的误差加0.01
                        ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01)
                        .float()
                        .detach()
                    )
                    # 累加蒸馏损失
                    loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
            # 根据返回流的标志决定返回内容
            if returnflow:
                return flow
            else:
                # 使用上下文网络对图像进行处理
                c0 = self.contextnet(img0, flow[:, :2])
                c1 = self.contextnet(img1, flow[:, 2:4])
                # 使用U-Net生成最终结果
                tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
                # 调整结果范围
                res = tmp[:, :3] * 2 - 1
                # 更新合并结果
                merged[2] = torch.clamp(merged[2] + res, 0, 1)
            # 返回流列表、掩码列表、合并结果、教师流、教师合并结果和蒸馏损失
            return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
```