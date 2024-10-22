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