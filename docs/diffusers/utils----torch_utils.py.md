# `.\diffusers\utils\torch_utils.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（"许可证"）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件
# 根据许可证分发，按“原样”基础提供，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 请参见许可证以了解管理权限和
# 限制的具体条款。
"""
PyTorch 实用工具：与 PyTorch 相关的实用工具
"""

from typing import List, Optional, Tuple, Union  # 导入用于类型注释的类

from . import logging  # 从当前包中导入 logging 模块
from .import_utils import is_torch_available, is_torch_version  # 导入检查 PyTorch 可用性和版本的工具


if is_torch_available():  # 如果 PyTorch 可用
    import torch  # 导入 PyTorch 库
    from torch.fft import fftn, fftshift, ifftn, ifftshift  # 从 PyTorch 的 FFT 模块导入相关函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，命名为模块名

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph  # 尝试导入允许在图中运行的功能
except (ImportError, ModuleNotFoundError):  # 捕获导入错误
    def maybe_allow_in_graph(cls):  # 定义一个备选函数
        return cls  # 返回原类


def randn_tensor(  # 定义生成随机张量的函数
    shape: Union[Tuple, List],  # 输入参数：张量形状，可以是元组或列表
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,  # 可选生成器，用于生成随机数
    device: Optional["torch.device"] = None,  # 可选参数，指定张量所在设备
    dtype: Optional["torch.dtype"] = None,  # 可选参数，指定张量数据类型
    layout: Optional["torch.layout"] = None,  # 可选参数，指定张量布局
):
    """一个帮助函数，用于在所需的 `device` 上创建随机张量，具有所需的 `dtype`。
    当传入生成器列表时，可以单独为每个批次大小设置种子。如果传入 CPU 生成器，张量
    始终在 CPU 上创建。
    """
    # 默认创建张量的设备为传入的设备
    rand_device = device  
    batch_size = shape[0]  # 批大小为形状的第一个元素

    layout = layout or torch.strided  # 如果未指定布局，默认为 strided
    device = device or torch.device("cpu")  # 如果未指定设备，默认为 CPU

    if generator is not None:  # 如果提供了生成器
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type  # 获取生成器的设备类型
        if gen_device_type != device.type and gen_device_type == "cpu":  # 如果生成器在 CPU 上，但期望的设备不同
            rand_device = "cpu"  # 设置随机设备为 CPU
            if device != "mps":  # 如果目标设备不是 MPS
                logger.info(  # 记录信息日志
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":  # 如果生成器在 CUDA 上，但目标设备不同
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")  # 抛出值错误

    # 确保生成器列表长度为 1 时被视为非列表
    if isinstance(generator, list) and len(generator) == 1:  # 如果生成器是列表且长度为 1
        generator = generator[0]  # 将其转换为非列表形式
    # 检查 generator 是否为列表
        if isinstance(generator, list):
            # 调整形状以适应生成的潜在变量
            shape = (1,) + shape[1:]
            # 生成多个随机潜在变量，使用各自的生成器
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            # 将生成的潜在变量沿第0维连接，并转换到目标设备
            latents = torch.cat(latents, dim=0).to(device)
        else:
            # 使用单个生成器生成随机潜在变量并转换到目标设备
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
    
        # 返回生成的潜在变量
        return latents
# 检查模块是否使用 torch.compile() 编译
def is_compiled_module(module) -> bool:
    # 检查 PyTorch 版本是否小于 2.0.0 或者模块中没有 "_dynamo" 属性
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        # 返回 False，表示未编译
        return False
    # 返回模块是否为 OptimizedModule 类型
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


# 进行 Fourier 过滤，作为 FreeU 方法的一部分
def fourier_filter(x_in: "torch.Tensor", threshold: int, scale: int) -> "torch.Tensor":
    # 输入 x_in 为张量，返回经过 Fourier 过滤的张量
    x = x_in
    # 解构输入张量的形状为批量大小、通道数、高度和宽度
    B, C, H, W = x.shape

    # 非 2 的幂次图像必须转换为 float32 类型
    if (W & (W - 1)) != 0 or (H & (H - 1)) != 0:
        # 将输入张量转换为 float32 类型
        x = x.to(dtype=torch.float32)

    # 执行快速傅里叶变换（FFT）
    x_freq = fftn(x, dim=(-2, -1))
    # 将频域数据进行移位操作
    x_freq = fftshift(x_freq, dim=(-2, -1))

    # 解构频域张量的形状
    B, C, H, W = x_freq.shape
    # 创建与输入张量相同形状的全 1 掩码
    mask = torch.ones((B, C, H, W), device=x.device)

    # 计算中心行和列
    crow, ccol = H // 2, W // 2
    # 设置掩码的中心区域的值为 scale
    mask[..., crow - threshold : crow + threshold, ccol - threshold : ccol + threshold] = scale
    # 频域数据与掩码相乘，应用掩码
    x_freq = x_freq * mask

    # 进行逆快速傅里叶变换（IFFT）
    x_freq = ifftshift(x_freq, dim=(-2, -1))
    # 获取频域数据的实部作为滤波结果
    x_filtered = ifftn(x_freq, dim=(-2, -1)).real

    # 返回结果张量，转换为与输入相同的数据类型
    return x_filtered.to(dtype=x_in.dtype)


# 应用 FreeU 机制，基于输入的分辨率索引和隐藏状态
def apply_freeu(
    resolution_idx: int, hidden_states: "torch.Tensor", res_hidden_states: "torch.Tensor", **freeu_kwargs
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    # 检查分辨率索引是否为 0
    if resolution_idx == 0:
        # 计算隐藏状态的一半通道数
        num_half_channels = hidden_states.shape[1] // 2
        # 将隐藏状态前一半的通道乘以 b1 缩放因子
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b1"]
        # 对残差隐藏状态应用 Fourier 过滤
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s1"])
    # 检查分辨率索引是否为 1
    if resolution_idx == 1:
        # 计算隐藏状态的一半通道数
        num_half_channels = hidden_states.shape[1] // 2
        # 将隐藏状态前一半的通道乘以 b2 缩放因子
        hidden_states[:, :num_half_channels] = hidden_states[:, :num_half_channels] * freeu_kwargs["b2"]
        # 对残差隐藏状态应用 Fourier 过滤
        res_hidden_states = fourier_filter(res_hidden_states, threshold=1, scale=freeu_kwargs["s2"])

    # 返回处理后的隐藏状态和残差隐藏状态
    return hidden_states, res_hidden_states
```