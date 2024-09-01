# `.\flux\src\flux\util.py`

```py
# 导入操作系统模块
import os
# 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass

# 导入 PyTorch 库，用于张量操作和深度学习
import torch
# 从 einops 库导入 rearrange 函数，用于重排列和转换张量
from einops import rearrange
# 从 huggingface_hub 库导入 hf_hub_download 函数，用于下载模型文件
from huggingface_hub import hf_hub_download
# 从 imwatermark 库导入 WatermarkEncoder 类，用于在图像中嵌入水印
from imwatermark import WatermarkEncoder
# 从 safetensors 库导入 load_file 函数，并重命名为 load_sft，用于加载安全张量文件
from safetensors.torch import load_file as load_sft

# 从 flux.model 模块导入 Flux 类和 FluxParams 类，用于模型定义和参数配置
from flux.model import Flux, FluxParams
# 从 flux.modules.autoencoder 模块导入 AutoEncoder 类和 AutoEncoderParams 类，用于自动编码器定义和参数配置
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
# 从 flux.modules.conditioner 模块导入 HFEmbedder 类，用于条件嵌入
from flux.modules.conditioner import HFEmbedder


# 定义一个数据类 ModelSpec，用于保存模型的各种规格和参数
@dataclass
class ModelSpec:
    # 定义模型参数
    params: FluxParams
    # 定义自动编码器参数
    ae_params: AutoEncoderParams
    # 定义检查点路径（可以为 None）
    ckpt_path: str | None
    # 定义自动编码器路径（可以为 None）
    ae_path: str | None
    # 定义模型仓库 ID（可以为 None）
    repo_id: str | None
    # 定义流文件仓库 ID（可以为 None）
    repo_flow: str | None
    # 定义自动编码器仓库 ID（可以为 None）
    repo_ae: str | None


# 定义配置字典 configs，包含不同模型的规格
configs = {
    # 配置 "flux-dev" 模型的规格
    "flux-dev": ModelSpec(
        # 设置模型仓库 ID
        repo_id="black-forest-labs/FLUX.1-dev",
        # 设置流文件仓库 ID
        repo_flow="flux1-dev.safetensors",
        # 设置自动编码器仓库 ID
        repo_ae="ae.safetensors",
        # 从环境变量获取检查点路径
        ckpt_path=os.getenv("FLUX_DEV"),
        # 设置 Flux 模型参数
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        # 从环境变量获取自动编码器路径
        ae_path=os.getenv("AE"),
        # 设置自动编码器参数
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    # 配置 "flux-schnell" 模型的规格
    "flux-schnell": ModelSpec(
        # 设置模型仓库 ID
        repo_id="black-forest-labs/FLUX.1-schnell",
        # 设置流文件仓库 ID
        repo_flow="flux1-schnell.safetensors",
        # 设置自动编码器仓库 ID
        repo_ae="ae.safetensors",
        # 从环境变量获取检查点路径
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        # 设置 Flux 模型参数
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        # 从环境变量获取自动编码器路径
        ae_path=os.getenv("AE"),
        # 设置自动编码器参数
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

# 定义函数 print_load_warning，用于打印加载警告信息
def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    # 如果缺少的键和意外的键都存在，则分别打印它们的数量和列表
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    # 如果只有缺少的键存在，则打印它们的数量和列表
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    # 如果意外的键数量大于0
        elif len(unexpected) > 0:
            # 打印意外的键数量和它们的列表
            print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
# 定义加载模型的函数，指定模型名称、设备和是否从 HF 下载
def load_flow_model(name: str, device: str | torch.device = "cuda", hf_download: bool = True):
    # 打印初始化模型的消息
    print("Init model")
    # 获取配置文件中的检查点路径
    ckpt_path = configs[name].ckpt_path
    # 如果检查点路径为空且需要从 HF 下载
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        # 从 HF 下载模型文件
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    # 根据是否有检查点路径选择设备
    with torch.device("meta" if ckpt_path is not None else device):
        # 初始化模型并设置数据类型为 bfloat16
        model = Flux(configs[name].params).to(torch.bfloat16)

    # 如果有检查点路径，加载模型状态
    if ckpt_path is not None:
        print("Loading checkpoint")
        # 加载检查点并转为字符串设备
        sd = load_sft(ckpt_path, device=str(device))
        # 加载状态字典，并检查缺失或意外的参数
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    # 返回模型
    return model


# 定义加载 T5 模型的函数，指定设备和最大序列长度
def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # 创建 HFEmbedder 对象，使用 T5 模型并设置最大序列长度和数据类型
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)


# 定义加载 CLIP 模型的函数，指定设备
def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    # 创建 HFEmbedder 对象，使用 CLIP 模型并设置最大序列长度和数据类型
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)


# 定义加载自动编码器的函数，指定名称、设备和是否从 HF 下载
def load_ae(name: str, device: str | torch.device = "cuda", hf_download: bool = True) -> AutoEncoder:
    # 获取配置文件中的自动编码器路径
    ckpt_path = configs[name].ae_path
    # 如果路径为空且需要从 HF 下载
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        # 从 HF 下载自动编码器文件
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # 打印初始化自动编码器的消息
    print("Init AE")
    # 根据是否有检查点路径选择设备
    with torch.device("meta" if ckpt_path is not None else device):
        # 初始化自动编码器
        ae = AutoEncoder(configs[name].ae_params)

    # 如果有检查点路径，加载自动编码器状态
    if ckpt_path is not None:
        # 加载检查点并转为字符串设备
        sd = load_sft(ckpt_path, device=str(device))
        # 加载状态字典，并检查缺失或意外的参数
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    # 返回自动编码器
    return ae


# 定义水印嵌入器类
class WatermarkEmbedder:
    def __init__(self, watermark):
        # 初始化水印和比特位数
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        # 初始化水印编码器
        self.encoder = WatermarkEncoder()
        # 设置水印比特数据
        self.encoder.set_watermark("bits", self.watermark)
    # 定义一个可调用对象的 `__call__` 方法，用于给输入图像添加预定义的水印
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Adds a predefined watermark to the input image
    
        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]
    
        Returns:
            same as input but watermarked
        """
        # 将图像的像素值从范围 [-1, 1] 线性映射到 [0, 1]
        image = 0.5 * image + 0.5
        # 检查图像张量的形状是否是 4 维 (即 batch size 和通道数)
        squeeze = len(image.shape) == 4
        if squeeze:
            # 如果是 4 维，给图像增加一个额外的维度，变成 5 维
            image = image[None, ...]
        # 获取图像的 batch size
        n = image.shape[0]
        # 将图像从 torch 张量转换为 numpy 数组，并调整形状和通道顺序
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        # 遍历每张图像，为每张图像应用水印编码
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        # 将图像从 numpy 数组转换回 torch 张量，恢复原始的形状和设备
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            image.device
        )
        # 将图像的像素值从 [0, 255] 归一化到 [0, 1]
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            # 如果之前添加了额外的维度，则将其移除，恢复原始形状
            image = image[0]
        # 将图像的像素值从 [0, 1] 转换回 [-1, 1] 范围
        image = 2 * image - 1
        # 返回处理后的图像
        return image
# 固定的 48 位消息，随机选择的
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] 将 x 转换为二进制字符串（去掉前缀 '0b'），然后用 int 将每一位转换为 0 或 1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
# 使用提取的位创建 WatermarkEmbedder 对象
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
```