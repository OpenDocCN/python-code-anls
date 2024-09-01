# `.\flux\src\flux\cli.py`

```py
# 导入操作系统相关模块
import os
# 导入正则表达式模块
import re
# 导入时间模块
import time
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 glob 模块导入 iglob 函数，用于文件名模式匹配
from glob import iglob

# 导入 PyTorch 库
import torch
# 从 einops 模块导入 rearrange 函数，用于张量重排
from einops import rearrange
# 从 fire 模块导入 Fire 类，用于命令行接口
from fire import Fire
# 从 PIL 模块导入 ExifTags 和 Image，用于处理图片和元数据
from PIL import ExifTags, Image

# 从 flux.sampling 模块导入采样相关函数
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
# 从 flux.util 模块导入实用工具函数
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
# 从 transformers 模块导入 pipeline，用于加载预训练模型
from transformers import pipeline

# 设置 NSFW（不适宜工作）内容的阈值
NSFW_THRESHOLD = 0.85

# 定义一个数据类，用于存储采样选项
@dataclass
class SamplingOptions:
    # 用户提示文本
    prompt: str
    # 图像宽度
    width: int
    # 图像高度
    height: int
    # 生成图像的步骤数量
    num_steps: int
    # 引导强度
    guidance: float
    # 随机种子，可选
    seed: int | None

# 解析用户输入的提示，并根据选项更新 SamplingOptions
def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    # 提示用户输入下一个提示
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    # 使用说明文本
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )
    # 循环读取用户输入，直到输入不以斜杠开头
    while (prompt := input(user_question)).startswith("/"):
        # 处理以 "/w" 开头的命令，设置宽度
        if prompt.startswith("/w"):
            # 如果命令中没有空格，提示无效命令并继续
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            # 解析命令中的宽度值并设置为16的倍数
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            # 打印设置的宽度和高度，以及总像素数
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        # 处理以 "/h" 开头的命令，设置高度
        elif prompt.startswith("/h"):
            # 如果命令中没有空格，提示无效命令并继续
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            # 解析命令中的高度值并设置为16的倍数
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            # 打印设置的宽度和高度，以及总像素数
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height *options.width/1e6:.2f}MP)"
            )
        # 处理以 "/g" 开头的命令，设置指导值
        elif prompt.startswith("/g"):
            # 如果命令中没有空格，提示无效命令并继续
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            # 解析命令中的指导值
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            # 打印设置的指导值
            print(f"Setting guidance to {options.guidance}")
        # 处理以 "/s" 开头的命令，设置种子值
        elif prompt.startswith("/s"):
            # 如果命令中没有空格，提示无效命令并继续
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            # 解析命令中的种子值
            _, seed = prompt.split()
            options.seed = int(seed)
            # 打印设置的种子值
            print(f"Setting seed to {options.seed}")
        # 处理以 "/n" 开头的命令，设置步骤数
        elif prompt.startswith("/n"):
            # 如果命令中没有空格，提示无效命令并继续
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            # 解析命令中的步骤数
            _, steps = prompt.split()
            options.num_steps = int(steps)
            # 打印设置的步骤数
            print(f"Setting seed to {options.num_steps}")
        # 处理以 "/q" 开头的命令，退出循环
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            # 如果命令不以已知前缀开头，提示无效命令并显示用法
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    # 如果输入不为空，将其设置为提示
    if prompt != "":
        options.prompt = prompt
    # 返回更新后的选项对象
    return options
@torch.inference_mode()
def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    ),
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 3.5,
    offload: bool = False,
    output_dir: str = "output",
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    # Initialize an NSFW image classification pipeline with the specified model and device
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    # Check if the specified model name is valid
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    # Set the PyTorch device based on the provided device string
    torch_device = torch.device(device)
    # Determine the number of sampling steps based on the model name
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # Adjust height and width to be multiples of 16 for compatibility
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    # Construct the output file path and handle directory and index management
    output_name = os.path.join(output_dir, "img_{idx}.jpg")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # Initialize components for the sampling process
    t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(name, device="cpu" if offload else torch_device)
    ae = load_ae(name, device="cpu" if offload else torch_device)

    # Create a random number generator and sampling options
    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    # If loop mode is enabled, adjust the options based on the prompt
    if loop:
        opts = parse_prompt(opts)
    # 当 opts 不为 None 时持续循环
    while opts is not None:
        # 如果 opts 中没有种子，则生成一个新的种子
        if opts.seed is None:
            opts.seed = rng.seed()
        # 打印生成过程的种子和提示
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        # 记录当前时间以计算生成时间
        t0 = time.perf_counter()

        # 准备输入噪声数据
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        # 将种子置为 None 以防止重复使用
        opts.seed = None
        # 如果需要将模型移至 CPU，清理 CUDA 缓存，并将模型移动到指定设备
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        # 准备输入数据，包括将 T5 和 CLIP 模型的输出、噪声以及提示整理成输入
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        # 获取时间步的调度
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        # 如果需要将模型移至 CPU，清理 CUDA 缓存，并将模型移动到 GPU
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # 对初始噪声进行去噪处理
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # 如果需要将模型移至 CPU，清理 CUDA 缓存，并将自动编码器的解码器移至当前设备
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # 将潜在变量解码到像素空间
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)
        # 记录解码处理时间
        t1 = time.perf_counter()

        # 格式化输出文件名
        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        # 将图像数据带入 PIL 格式并保存
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        # 从 numpy 数组创建 PIL 图像对象
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        # 进行 NSFW 内容检测
        nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
        
        # 如果 NSFW 分数低于阈值，则保存图像及其 EXIF 元数据
        if nsfw_score < NSFW_THRESHOLD:
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
            exif_data[ExifTags.Base.Make] = "Black Forest Labs"
            exif_data[ExifTags.Base.Model] = name
            if add_sampling_metadata:
                exif_data[ExifTags.Base.ImageDescription] = prompt
            img.save(fn, exif=exif_data, quality=95, subsampling=0)
            # 增加图像索引
            idx += 1
        else:
            print("Your generated image may contain NSFW content.")

        # 如果设置了循环，则解析新的提示并继续，否则退出循环
        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        else:
            opts = None
# 定义主函数
def app():
    # 使用 Fire 库将 main 函数作为命令行接口
    Fire(main)


# 检查是否为主模块运行
if __name__ == "__main__":
    # 调用 app 函数
    app()
```