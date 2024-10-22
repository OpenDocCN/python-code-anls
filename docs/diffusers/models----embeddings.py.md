# `.\diffusers\models\embeddings.py`

```py
# 版权信息，指明版权归 HuggingFace 团队所有
# 本文件根据 Apache License, Version 2.0 授权
# 使用此文件需遵循许可条款
# 可在此网址获取许可信息
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非有书面协议，否则软件按“原样”提供，不作任何明示或暗示的保证
# 查看许可证以了解特定权限和限制
import math  # 导入数学库以进行数学运算
from typing import List, Optional, Tuple, Union  # 导入类型提示以便于类型注解

import numpy as np  # 导入 NumPy 以进行数组和数值运算
import torch  # 导入 PyTorch 以进行张量操作
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from torch import nn  # 导入 PyTorch 的神经网络模块

from ..utils import deprecate  # 从 utils 模块导入 deprecate 函数
from .activations import FP32SiLU, get_activation  # 从 activations 模块导入激活函数
from .attention_processor import Attention  # 从 attention_processor 模块导入 Attention 类


def get_timestep_embedding(  # 定义获取时间步嵌入的函数
    timesteps: torch.Tensor,  # 输入参数，表示时间步的张量
    embedding_dim: int,  # 输入参数，表示输出嵌入的维度
    flip_sin_to_cos: bool = False,  # 输入参数，决定嵌入的顺序
    downscale_freq_shift: float = 1,  # 输入参数，控制频率维度之间的差异
    scale: float = 1,  # 输入参数，应用于嵌入的缩放因子
    max_period: int = 10000,  # 输入参数，控制嵌入的最大频率
):
    """
    该实现与去噪扩散概率模型中的实现相匹配：创建正弦时间步嵌入。

    参数
        timesteps (torch.Tensor):
            一个一维张量，N个索引，每个批次元素一个。这些可以是分数。
        embedding_dim (int):
            输出的维度。
        flip_sin_to_cos (bool):
            嵌入顺序是否应为 `cos, sin`（如果为 True）或 `sin, cos`（如果为 False）
        downscale_freq_shift (float):
            控制维度之间频率的差异
        scale (float):
            应用于嵌入的缩放因子。
        max_period (int):
            控制嵌入的最大频率
    返回
        torch.Tensor: 一个 [N x dim] 的位置嵌入张量。
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"  # 确保时间步是一个一维数组

    half_dim = embedding_dim // 2  # 计算嵌入维度的一半
    exponent = -math.log(max_period) * torch.arange(  # 计算指数
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device  # 创建从 0 到 half_dim 的范围
    )
    exponent = exponent / (half_dim - downscale_freq_shift)  # 根据下缩放频率差异调整指数

    emb = torch.exp(exponent)  # 计算指数的幂，得到嵌入基础
    emb = timesteps[:, None].float() * emb[None, :]  # 将时间步与嵌入基础相乘

    # 缩放嵌入
    emb = scale * emb  # 应用缩放因子到嵌入

    # 拼接正弦和余弦嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # 在最后一个维度上拼接正弦和余弦嵌入

    # 翻转正弦和余弦嵌入
    if flip_sin_to_cos:  # 如果需要翻转
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)  # 重新排列嵌入

    # 零填充
    if embedding_dim % 2 == 1:  # 如果嵌入维度是奇数
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  # 在最后一维上进行零填充
    return emb  # 返回计算得到的嵌入


def get_3d_sincos_pos_embed(  # 定义获取三维正弦余弦位置嵌入的函数
    embed_dim: int,  # 输入参数，表示嵌入的维度
    spatial_size: Union[int, Tuple[int, int]],  # 输入参数，表示空间大小，可以是单个整数或元组
    temporal_size: int,  # 输入参数，表示时间大小
    spatial_interpolation_scale: float = 1.0,  # 输入参数，空间插值缩放因子
    temporal_interpolation_scale: float = 1.0,  # 输入参数，时间插值缩放因子
) -> np.ndarray:  # 函数返回值为 NumPy 数组
    r"""
    # 参数说明
    Args:
        embed_dim (`int`): 嵌入维度
        spatial_size (`int` or `Tuple[int, int]`): 空间大小
        temporal_size (`int`): 时间大小
        spatial_interpolation_scale (`float`, defaults to 1.0): 空间插值缩放因子，默认为1.0
        temporal_interpolation_scale (`float`, defaults to 1.0): 时间插值缩放因子，默认为1.0
    """
    # 检查嵌入维度是否可被4整除，不符合则抛出异常
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    # 如果空间大小是整数，则将其转为元组形式
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    # 计算空间嵌入维度
    embed_dim_spatial = 3 * embed_dim // 4
    # 计算时间嵌入维度
    embed_dim_temporal = embed_dim // 4

    # 1. 空间
    # 生成纵向坐标网格，并按空间插值缩放因子进行归一化
    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    # 生成横向坐标网格，并按空间插值缩放因子进行归一化
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    # 创建网格，横向坐标在前
    grid = np.meshgrid(grid_w, grid_h)
    # 将网格堆叠为新的数组
    grid = np.stack(grid, axis=0)

    # 重塑网格形状以符合后续计算
    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    # 从网格中获取二维正弦余弦位置嵌入
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 2. 时间
    # 生成时间坐标网格，并按时间插值缩放因子进行归一化
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    # 从时间网格中获取一维正弦余弦位置嵌入
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 3. 连接
    # 扩展空间位置嵌入的维度以适应时间维度
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    # 重复空间嵌入以匹配时间大小
    pos_embed_spatial = np.repeat(pos_embed_spatial, temporal_size, axis=0)  # [T, H*W, D // 4 * 3]

    # 扩展时间位置嵌入的维度以适应空间维度
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    # 重复时间嵌入以匹配空间大小
    pos_embed_temporal = np.repeat(pos_embed_temporal, spatial_size[0] * spatial_size[1], axis=1)  # [T, H*W, D // 4]

    # 连接时间和空间嵌入，形成最终位置嵌入
    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)  # [T, H*W, D]
    # 返回最终位置嵌入
    return pos_embed
# 定义获取2D正弦余弦位置嵌入的函数
def get_2d_sincos_pos_embed(
    # 嵌入维度，网格大小，是否使用类标记，额外标记数量，插值缩放因子，基础大小
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: 网格的高度和宽度为整数返回: pos_embed: [grid_size*grid_size, embed_dim] 或
    [1+grid_size*grid_size, embed_dim] （有或没有类标记）
    """
    # 如果 grid_size 是整数，则将其转换为元组
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    # 计算网格高度的归一化值
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    # 计算网格宽度的归一化值
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    # 创建网格，宽度在前
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格沿新轴堆叠
    grid = np.stack(grid, axis=0)

    # 重新塑形网格以适配后续计算
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    # 从网格获取2D正弦余弦位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # 如果需要类标记并且有额外标记，则在前面添加零向量
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    # 返回位置嵌入
    return pos_embed


# 定义从网格获取2D正弦余弦位置嵌入的辅助函数
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    # 如果嵌入维度不是偶数，则引发错误
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # 使用一半的维度来编码网格高度
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # 使用一半的维度来编码网格宽度
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # 将高度和宽度的嵌入拼接在一起
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    # 返回合并后的嵌入
    return emb


# 定义从位置获取一维正弦余弦位置嵌入的辅助函数
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: 每个位置的输出维度 pos: 待编码的位置列表: 大小 (M,) out: (M, D)
    """
    # 如果嵌入维度不是偶数，则引发错误
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # 计算嵌入频率
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    # 将位置重塑为一维数组
    pos = pos.reshape(-1)  # (M,)
    # 计算位置和频率的外积
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # 计算正弦值
    emb_sin = np.sin(out)  # (M, D/2)
    # 计算余弦值
    emb_cos = np.cos(out)  # (M, D/2)

    # 将正弦和余弦的嵌入拼接在一起
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    # 返回合并后的嵌入
    return emb


# 定义 PatchEmbed 类，用于将2D图像转换为补丁嵌入
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for SD3 cropping."""

    # 初始化函数，设置嵌入的相关参数
    def __init__(
        # 图像高度，宽度，补丁大小，输入通道数，嵌入维度，是否使用层归一化，是否展平，是否使用偏置，插值缩放因子，位置嵌入类型，位置嵌入最大大小
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # 用于 SD3 裁剪
    ):
        # 调用父类的构造函数
        super().__init__()

        # 计算补丁的数量，基于高度和宽度
        num_patches = (height // patch_size) * (width // patch_size)
        # 保存是否扁平化的标志
        self.flatten = flatten
        # 保存是否使用层归一化的标志
        self.layer_norm = layer_norm
        # 保存位置嵌入的最大大小
        self.pos_embed_max_size = pos_embed_max_size

        # 创建一个卷积层，用于将输入通道映射到嵌入维度
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        # 如果使用层归一化，则初始化层归一化对象
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            # 如果不使用层归一化，设置为 None
            self.norm = None

        # 保存补丁大小
        self.patch_size = patch_size
        # 计算高度和宽度对应的补丁数量
        self.height, self.width = height // patch_size, width // patch_size
        # 保存基础尺寸
        self.base_size = height // patch_size
        # 保存插值缩放因子
        self.interpolation_scale = interpolation_scale

        # 基于最大尺寸或默认值计算位置嵌入的网格大小
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        # 如果没有指定位置嵌入类型，设置为 None
        if pos_embed_type is None:
            self.pos_embed = None
        # 如果位置嵌入类型为 "sincos"，则生成相应的嵌入
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            # 持久化标志：如果有最大位置嵌入大小，则为 True
            persistent = True if pos_embed_max_size else False
            # 将位置嵌入注册为缓冲区，并转换为浮点类型
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)
        else:
            # 抛出异常，表示不支持的嵌入类型
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """裁剪位置嵌入以兼容 SD3。"""
        # 如果未设置最大位置嵌入大小，则抛出异常
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        # 计算补丁高度和宽度
        height = height // self.patch_size
        width = width // self.patch_size
        # 检查高度是否超过最大嵌入大小
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        # 检查宽度是否超过最大嵌入大小
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        # 计算裁剪的顶部和左侧位置
        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        # 重塑位置嵌入以适应空间维度
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        # 裁剪位置嵌入
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        # 重塑裁剪后的嵌入为合适的形状
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        # 返回裁剪后的位置嵌入
        return spatial_pos_embed
    # 前向传播函数，接收潜在变量作为输入
    def forward(self, latent):
        # 如果存在最大位置嵌入大小，则获取潜在变量的高度和宽度
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            # 否则根据补丁大小计算高度和宽度
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
    
        # 将潜在变量通过投影层进行变换
        latent = self.proj(latent)
        # 如果需要展平，则将潜在变量展平并转置
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # 如果需要进行层归一化，则对潜在变量进行归一化处理
        if self.layer_norm:
            latent = self.norm(latent)
        # 如果没有位置嵌入，则直接返回潜在变量
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # 根据需要插值或裁剪位置嵌入
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            # 如果高度或宽度不匹配，则生成新的位置嵌入
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                )
                # 将生成的位置嵌入转换为张量并移动到潜在变量的设备上
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                # 否则使用已有的位置嵌入
                pos_embed = self.pos_embed
    
        # 返回潜在变量与位置嵌入的和，并转换为潜在变量的类型
        return (latent + pos_embed).to(latent.dtype)
# 定义一个名为 LuminaPatchEmbed 的类，继承自 nn.Module
class LuminaPatchEmbed(nn.Module):
    """2D Image to Patch Embedding with support for Lumina-T2X"""

    # 初始化方法，设置补丁大小、输入通道、嵌入维度及偏置选项
    def __init__(self, patch_size=2, in_channels=4, embed_dim=768, bias=True):
        # 调用父类的初始化方法
        super().__init__()
        # 保存补丁大小
        self.patch_size = patch_size
        # 创建一个线性层用于投影输入数据到嵌入空间
        self.proj = nn.Linear(
            # 线性层输入特征数：补丁大小的平方乘以输入通道数
            in_features=patch_size * patch_size * in_channels,
            # 线性层输出特征数：嵌入维度
            out_features=embed_dim,
            # 是否使用偏置项
            bias=bias,
        )

    # 前向传播方法，定义如何处理输入数据
    def forward(self, x, freqs_cis):
        """
        Patchifies and embeds the input tensor(s).

        Args:
            x (List[torch.Tensor] | torch.Tensor): The input tensor(s) to be patchified and embedded.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], torch.Tensor]: A tuple containing the patchified
            and embedded tensor(s), the mask indicating the valid patches, the original image size(s), and the
            frequency tensor(s).
        """
        # 将频率张量移到输入张量所在的设备上
        freqs_cis = freqs_cis.to(x[0].device)
        # 获取补丁的高度和宽度
        patch_height = patch_width = self.patch_size
        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size, channel, height, width = x.size()
        # 计算高度和宽度的补丁数
        height_tokens, width_tokens = height // patch_height, width // patch_width

        # 重新排列输入张量的形状，以便进行补丁划分
        x = x.view(batch_size, channel, height_tokens, patch_height, width_tokens, patch_width).permute(
            # 重新排列维度顺序
            0, 2, 4, 1, 3, 5
        )
        # 将补丁维度展平
        x = x.flatten(3)
        # 应用线性层进行投影
        x = self.proj(x)
        # 再次展平张量，使其符合后续处理的形状
        x = x.flatten(1, 2)

        # 创建一个全为 1 的掩码，表示所有补丁都是有效的
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.int32, device=x.device)

        # 返回嵌入的张量、掩码、原始图像尺寸和频率张量
        return (
            x,
            mask,
            # 重复原始图像的高度和宽度信息，以匹配批量大小
            [(height, width)] * batch_size,
            # 选择频率张量的相关部分，并进行展平和添加一个维度
            freqs_cis[:height_tokens, :width_tokens].flatten(0, 1).unsqueeze(0),
        )


# 定义一个名为 CogVideoXPatchEmbed 的类，继承自 nn.Module
class CogVideoXPatchEmbed(nn.Module):
    # 初始化方法，设置多个参数以定义补丁嵌入层
    def __init__(
        self,
        # 补丁大小，默认为 2
        patch_size: int = 2,
        # 输入通道数，默认为 16
        in_channels: int = 16,
        # 嵌入维度，默认为 1920
        embed_dim: int = 1920,
        # 文本嵌入维度，默认为 4096
        text_embed_dim: int = 4096,
        # 是否使用偏置项，默认为 True
        bias: bool = True,
        # 采样宽度，默认为 90
        sample_width: int = 90,
        # 采样高度，默认为 60
        sample_height: int = 60,
        # 采样帧数，默认为 49
        sample_frames: int = 49,
        # 时间压缩比例，默认为 4
        temporal_compression_ratio: int = 4,
        # 最大文本序列长度，默认为 226
        max_text_seq_length: int = 226,
        # 空间插值缩放，默认为 1.875
        spatial_interpolation_scale: float = 1.875,
        # 时间插值缩放，默认为 1.0
        temporal_interpolation_scale: float = 1.0,
        # 是否使用位置嵌入，默认为 True
        use_positional_embeddings: bool = True,
        # 是否使用学习到的位置嵌入，默认为 True
        use_learned_positional_embeddings: bool = True,
    # 定义一个无返回值的构造函数
        ) -> None:
            # 调用父类的构造函数
            super().__init__()
    
            # 设置补丁大小
            self.patch_size = patch_size
            # 设置嵌入维度
            self.embed_dim = embed_dim
            # 设置样本高度
            self.sample_height = sample_height
            # 设置样本宽度
            self.sample_width = sample_width
            # 设置样本帧数
            self.sample_frames = sample_frames
            # 设置时间压缩比
            self.temporal_compression_ratio = temporal_compression_ratio
            # 设置最大文本序列长度
            self.max_text_seq_length = max_text_seq_length
            # 设置空间插值缩放因子
            self.spatial_interpolation_scale = spatial_interpolation_scale
            # 设置时间插值缩放因子
            self.temporal_interpolation_scale = temporal_interpolation_scale
            # 设置是否使用位置嵌入
            self.use_positional_embeddings = use_positional_embeddings
            # 设置是否使用学习的位置信息嵌入
            self.use_learned_positional_embeddings = use_learned_positional_embeddings
    
            # 创建卷积层，用于将输入通道映射到嵌入维度
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
            )
            # 创建线性层，将文本嵌入映射到嵌入维度
            self.text_proj = nn.Linear(text_embed_dim, embed_dim)
    
            # 如果使用位置嵌入或学习的位置信息嵌入
            if use_positional_embeddings or use_learned_positional_embeddings:
                # 持久化设置为是否使用学习的位置信息嵌入
                persistent = use_learned_positional_embeddings
                # 获取位置嵌入
                pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
                # 注册位置嵌入缓冲区
                self.register_buffer("pos_embedding", pos_embedding, persistent=persistent)
    
        # 定义获取位置嵌入的方法
        def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> torch.Tensor:
            # 计算后补丁高度
            post_patch_height = sample_height // self.patch_size
            # 计算后补丁宽度
            post_patch_width = sample_width // self.patch_size
            # 计算后时间压缩帧数
            post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
            # 计算补丁数量
            num_patches = post_patch_height * post_patch_width * post_time_compression_frames
    
            # 获取三维正弦余弦位置嵌入
            pos_embedding = get_3d_sincos_pos_embed(
                self.embed_dim,
                (post_patch_width, post_patch_height),
                post_time_compression_frames,
                self.spatial_interpolation_scale,
                self.temporal_interpolation_scale,
            )
            # 将位置嵌入转换为张量并展平
            pos_embedding = torch.from_numpy(pos_embedding).flatten(0, 1)
            # 创建联合位置嵌入的零张量
            joint_pos_embedding = torch.zeros(
                1, self.max_text_seq_length + num_patches, self.embed_dim, requires_grad=False
            )
            # 将位置嵌入复制到联合位置嵌入中
            joint_pos_embedding.data[:, self.max_text_seq_length :].copy_(pos_embedding)
    
            # 返回联合位置嵌入
            return joint_pos_embedding
    # 定义前向传播函数，接收文本和图像的嵌入张量
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                输入的文本嵌入，预期形状: (batch_size, seq_length, embedding_dim)。
            image_embeds (`torch.Tensor`):
                输入的图像嵌入，预期形状: (batch_size, num_frames, channels, height, width)。
        """
        # 对文本嵌入进行投影处理
        text_embeds = self.text_proj(text_embeds)

        # 解构图像嵌入的形状为 batch, num_frames, channels, height, width
        batch, num_frames, channels, height, width = image_embeds.shape
        # 将图像嵌入重新塑形为 (-1, channels, height, width)
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        # 对图像嵌入进行投影处理
        image_embeds = self.proj(image_embeds)
        # 将投影后的图像嵌入恢复成原来的 batch 和 num_frames 结构
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        # 将图像嵌入展平，并转置维度，变为 [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        # 再次展平图像嵌入，使其变为 [batch, num_frames x height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        # 将文本嵌入和图像嵌入在维度1上进行拼接
        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        # 检查是否使用位置嵌入
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            # 如果使用学习位置嵌入且当前的宽度或高度与样本不匹配，则抛出错误
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            # 计算预压缩时间帧的数量
            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            # 检查样本的高度、宽度和帧数是否与预期不符
            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                # 获取位置嵌入
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
                # 将位置嵌入转移到嵌入的设备和数据类型
                pos_embedding = pos_embedding.to(embeds.device, dtype=embeds.dtype)
            else:
                # 使用已存储的位置嵌入
                pos_embedding = self.pos_embedding

            # 将位置嵌入添加到最终的嵌入中
            embeds = embeds + pos_embedding

        # 返回最终的嵌入张量
        return embeds
# 定义一个函数，生成具有三维结构的视频标记的相对位置嵌入
def get_3d_rotary_pos_embed(
    # 嵌入维度大小，对应隐藏层的大小
    embed_dim, 
    # 裁剪的左上和右下坐标
    crops_coords, 
    # 空间位置嵌入的网格大小（高度，宽度）
    grid_size, 
    # 时间维度的大小
    temporal_size, 
    # 频率计算的缩放因子
    theta: int = 10000, 
    # 如果为真，分别返回实部和虚部，否则返回复数
    use_real: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    # 解包裁剪坐标，分别获取开始和结束坐标
    start, stop = crops_coords
    # 在给定范围内生成高度的均匀网格
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    # 在给定范围内生成宽度的均匀网格
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    # 生成时间维度的均匀网格
    grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)

    # 为每个轴计算维度
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # 计算时间频率
    freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))
    # 将时间网格转换为张量并设为浮点型
    grid_t = torch.from_numpy(grid_t).float()
    # 计算频率与时间网格的乘积
    freqs_t = torch.einsum("n , f -> n f", grid_t, freqs_t)
    # 在最后一个维度上重复频率
    freqs_t = freqs_t.repeat_interleave(2, dim=-1)

    # 计算高度和宽度的空间频率
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2).float() / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2).float() / dim_w))
    # 将高度网格转换为张量并设为浮点型
    grid_h = torch.from_numpy(grid_h).float()
    # 将宽度网格转换为张量并设为浮点型
    grid_w = torch.from_numpy(grid_w).float()
    # 计算频率与高度网格的乘积
    freqs_h = torch.einsum("n , f -> n f", grid_h, freqs_h)
    # 计算频率与宽度网格的乘积
    freqs_w = torch.einsum("n , f -> n f", grid_w, freqs_w)
    # 在最后一个维度上重复高度频率
    freqs_h = freqs_h.repeat_interleave(2, dim=-1)
    # 在最后一个维度上重复宽度频率
    freqs_w = freqs_w.repeat_interleave(2, dim=-1)

    # 在指定维度上广播并连接张量
    # 定义广播函数，接受一个张量列表和一个维度参数，默认为-1
        def broadcast(tensors, dim=-1):
            # 获取张量的数量
            num_tensors = len(tensors)
            # 收集所有张量的维度数量，形成集合以去重
            shape_lens = {len(t.shape) for t in tensors}
            # 确保所有张量的维度数量相同
            assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
            # 获取张量的维度数量
            shape_len = list(shape_lens)[0]
            # 处理负维度，将其转换为正维度
            dim = (dim + shape_len) if dim < 0 else dim
            # 获取所有张量的维度元组，便于后续操作
            dims = list(zip(*(list(t.shape) for t in tensors)))
            # 找到可以扩展的维度，排除当前操作的维度
            expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
            # 确保可扩展维度的大小相同，或者只有两个不同的值
            assert all(
                [*(len(set(t[1])) <= 2 for t in expandable_dims)]
            ), "invalid dimensions for broadcastable concatenation"
            # 获取每个可扩展维度的最大值
            max_dims = [(t[0], max(t[1])) for t in expandable_dims]
            # 为每个最大维度生成一个形状，重复num_tensors次
            expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
            # 将当前维度的值插入到扩展维度列表中
            expanded_dims.insert(dim, (dim, dims[dim]))
            # 生成可扩展形状，供张量扩展使用
            expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
            # 扩展每个张量到其可扩展的形状
            tensors = [t[0].expand(*t[1]) for t in zip(tensors, expandable_shapes)]
            # 在指定维度上连接所有张量
            return torch.cat(tensors, dim=dim)
    
        # 调用广播函数，将三个频率张量组合在一起，按最后一个维度连接
        freqs = broadcast((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
    
        # 获取组合后频率张量的形状参数
        t, h, w, d = freqs.shape
        # 将频率张量重塑为二维形状
        freqs = freqs.view(t * h * w, d)
    
        # 生成正弦和余弦分量
        sin = freqs.sin()  # 计算正弦值
        cos = freqs.cos()  # 计算余弦值
    
        # 根据使用的标志返回不同的结果
        if use_real:
            # 如果使用真实值，返回余弦和正弦
            return cos, sin
        else:
            # 否则，计算复数形式的频率
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成极坐标复数
            return freqs_cis  # 返回复数形式的频率
# 获取带有 2D 结构的旋转位置嵌入
def get_2d_rotary_pos_embed(embed_dim, crops_coords, grid_size, use_real=True):
    # 解包裁剪区域的左上和右下坐标
    start, stop = crops_coords
    # 生成从起始到结束的高度线性空间，数量为 grid_size[0]
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    # 生成从起始到结束的宽度线性空间，数量为 grid_size[1]
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    # 创建网格，宽度先行
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # 将网格堆叠成形状为 [2, W, H]
    grid = np.stack(grid, axis=0)  # [2, W, H]

    # 调整网格形状以便后续计算
    grid = grid.reshape([2, 1, *grid.shape[1:]])
    # 从网格计算 2D 旋转位置嵌入
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    # 返回位置嵌入
    return pos_embed


# 从网格计算 2D 旋转位置嵌入
def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    # 确保嵌入维度可以被 4 整除
    assert embed_dim % 4 == 0

    # 使用一半的维度编码高度
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)
    # 使用一半的维度编码宽度
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)

    # 如果使用真实部分，则返回余弦和正弦部分
    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
        return cos, sin
    else:
        # 否则，合并嵌入
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


# 获取用于光线的 2D 旋转位置嵌入
def get_2d_rotary_pos_embed_lumina(embed_dim, len_h, len_w, linear_factor=1.0, ntk_factor=1.0):
    # 确保嵌入维度可以被 4 整除
    assert embed_dim % 4 == 0

    # 计算高度的旋转位置嵌入
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, len_h, linear_factor=linear_factor, ntk_factor=ntk_factor
    )  # (H, D/4)
    # 计算宽度的旋转位置嵌入
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, len_w, linear_factor=linear_factor, ntk_factor=ntk_factor
    )  # (W, D/4)
    # 调整高度嵌入形状并重复以匹配宽度
    emb_h = emb_h.view(len_h, 1, embed_dim // 4, 1).repeat(1, len_w, 1, 1)  # (H, W, D/4, 1)
    # 调整宽度嵌入形状并重复以匹配高度
    emb_w = emb_w.view(1, len_w, embed_dim // 4, 1).repeat(len_h, 1, 1, 1)  # (H, W, D/4, 1)

    # 合并嵌入并展平
    emb = torch.cat([emb_h, emb_w], dim=-1).flatten(2)  # (H, W, D/2)
    # 返回结果
    return emb


# 获取 1D 旋转位置嵌入
def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
):
    # 预计算给定维度的复数指数的频率张量
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    # 该函数返回包含复数值的张量，数据类型为 complex64，用于频率计算。
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    # 参数说明
    Args:
        dim (`int`): 频率张量的维度。
        pos (`np.ndarray` or `int`): 频率张量的位置索引。可以是数组或标量
        theta (`float`, *optional*, defaults to 10000.0):
            频率计算的缩放因子。默认为 10000.0。
        use_real (`bool`, *optional*):
            如果为 True，则分别返回实部和虚部。否则，返回复数。
        linear_factor (`float`, *optional*, defaults to 1.0):
            上下文外推的缩放因子。默认为 1.0。
        ntk_factor (`float`, *optional*, defaults to 1.0):
            NTK-Aware RoPE 的缩放因子。默认为 1.0。
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            如果为 `True` 且 use_real 为真，实部和虚部各自与自身交错以达到 dim。
            否则，它们将被拼接在一起。
    Returns:
        `torch.Tensor`: 预计算的频率张量，包含复指数。形状为 [S, D/2]
    """
    # 确保 dim 是偶数
    assert dim % 2 == 0

    # 如果 pos 是整数，则生成一个从 0 到 pos-1 的数组
    if isinstance(pos, int):
        pos = np.arange(pos)
    # 使用 ntk_factor 缩放 theta
    theta = theta * ntk_factor
    # 计算频率，生成维度为 [D/2] 的张量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) / linear_factor  # [D/2]
    # 将 pos 转换为张量，并移动到 freqs 的设备上
    t = torch.from_numpy(pos).to(freqs.device)  # type: ignore  # [S]
    # 计算外积，生成形状为 [S, D/2] 的张量
    freqs = torch.outer(t, freqs).float()  # type: ignore   # [S, D/2]
    # 如果需要返回实部且选择交错输出
    if use_real and repeat_interleave_real:
        # 计算 cos 值并交错，生成形状为 [S, D] 的张量
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        # 计算 sin 值并交错，生成形状为 [S, D] 的张量
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        # 返回实部和虚部
        return freqs_cos, freqs_sin
    # 如果只需要返回实部
    elif use_real:
        # 拼接 cos 值，生成形状为 [S, D] 的张量
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [S, D]
        # 拼接 sin 值，生成形状为 [S, D] 的张量
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # [S, D]
        # 返回实部和虚部
        return freqs_cos, freqs_sin
    # 如果需要返回复数
    else:
        # 使用极坐标形式生成复数，形状为 [S, D/2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        # 返回复数张量
        return freqs_cis
# 定义应用旋转嵌入的函数，输入为张量和频率张量
def apply_rotary_emb(
    # 输入张量，形状为[B, H, S, D]
    x: torch.Tensor,
    # 预计算的频率张量，可以是单个张量或元组
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    # 是否使用实数部分的标志
    use_real: bool = True,
    # 实数部分解绑定维度
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转嵌入到输入张量，使用给定的频率张量。该函数将旋转嵌入应用于给定的查询或键 'x' 张量，
    使用提供的频率张量 'freqs_cis'。输入张量被重塑为复数形式，频率张量被重塑为兼容广播的形状。
    返回的张量包含旋转嵌入，并作为实数张量返回。

    参数:
        x (`torch.Tensor`):
            应用旋转嵌入的查询或键张量。 [B, H, S, D] xk (torch.Tensor): 要应用的键张量
        freqs_cis (`Tuple[torch.Tensor]`): 用于复数指数的预计算频率张量。 ([S, D], [S, D],)

    返回:
        Tuple[torch.Tensor, torch.Tensor]: 修改后的查询张量和带有旋转嵌入的键张量的元组。
    """
    # 检查是否使用实数部分
    if use_real:
        # 从频率张量中提取余弦和正弦值，形状为[S, D]
        cos, sin = freqs_cis  # [S, D]
        # 为广播增加两个维度
        cos = cos[None, None]
        sin = sin[None, None]
        # 将余弦和正弦值移到输入张量相同的设备上
        cos, sin = cos.to(x.device), sin.to(x.device)

        # 根据解绑定维度的设置处理输入张量
        if use_real_unbind_dim == -1:
            # 用于例如 Lumina 的情况
            # 重塑张量为复数形式，并解绑定为实部和虚部
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            # 旋转输入张量
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # 用于例如 Stable Audio 的情况
            # 重塑张量为复数形式，并解绑定为实部和虚部
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            # 旋转输入张量
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            # 如果解绑定维度不在预期范围内，则抛出错误
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        # 计算输出张量，将余弦和旋转张量结合
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        # 返回处理后的输出张量
        return out
    else:
        # 将输入张量重塑为复数形式
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # 增加一个维度以便与频率张量进行广播
        freqs_cis = freqs_cis.unsqueeze(2)
        # 计算输出张量，将旋转张量与频率张量相乘并重塑为实数
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        # 返回输出张量，保持与输入张量相同的类型
        return x_out.type_as(x)


# 定义时间步嵌入类，继承自 nn.Module
class TimestepEmbedding(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        # 输入通道数
        in_channels: int,
        # 时间嵌入维度
        time_embed_dim: int,
        # 激活函数类型，默认为 "silu"
        act_fn: str = "silu",
        # 输出维度，默认为 None
        out_dim: int = None,
        # 后续激活函数类型，默认为 None
        post_act_fn: Optional[str] = None,
        # 条件投影维度，默认为 None
        cond_proj_dim=None,
        # 样本投影偏置，默认为 True
        sample_proj_bias=True,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个线性变换层，将输入通道数映射到时间嵌入维度
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        # 如果条件投影维度不为 None，创建条件线性变换层；否则设置为 None
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        # 获取激活函数
        self.act = get_activation(act_fn)

        # 如果输出维度不为 None，设置时间嵌入输出维度；否则使用时间嵌入维度
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        
        # 创建第二个线性变换层，将时间嵌入维度映射到输出维度
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        # 如果后激活函数为 None，设置后激活为 None；否则获取后激活函数
        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    # 定义前向传播函数
    def forward(self, sample, condition=None):
        # 如果条件不为 None，将条件投影添加到样本上
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        
        # 通过第一个线性层处理样本
        sample = self.linear_1(sample)

        # 如果激活函数不为 None，应用激活函数
        if self.act is not None:
            sample = self.act(sample)

        # 通过第二个线性层处理样本
        sample = self.linear_2(sample)

        # 如果后激活函数不为 None，应用后激活函数
        if self.post_act is not None:
            sample = self.post_act(sample)
        
        # 返回处理后的样本
        return sample
# 定义时间步数类，继承自 nn.Module
class Timesteps(nn.Module):
    # 初始化函数，设置参数并调用父类构造函数
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()  # 调用父类的构造函数
        self.num_channels = num_channels  # 设置通道数
        self.flip_sin_to_cos = flip_sin_to_cos  # 是否翻转正弦和余弦
        self.downscale_freq_shift = downscale_freq_shift  # 频率移位因子
        self.scale = scale  # 缩放因子

    # 前向传播函数，处理时间步
    def forward(self, timesteps):
        # 获取时间步嵌入
        t_emb = get_timestep_embedding(
            timesteps,  # 输入时间步
            self.num_channels,  # 通道数
            flip_sin_to_cos=self.flip_sin_to_cos,  # 是否翻转
            downscale_freq_shift=self.downscale_freq_shift,  # 频率移位
            scale=self.scale,  # 缩放因子
        )
        return t_emb  # 返回时间步嵌入


# 定义高斯傅里叶投影类，继承自 nn.Module
class GaussianFourierProjection(nn.Module):
    """高斯傅里叶嵌入，用于噪声水平。"""

    # 初始化函数，设置参数并调用父类构造函数
    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()  # 调用父类构造函数
        # 创建权重参数，并设定不可训练
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log = log  # 是否取对数
        self.flip_sin_to_cos = flip_sin_to_cos  # 是否翻转正弦和余弦

        # 如果需要将 W 设置为 weight
        if set_W_to_weight:
            # 将权重删除，随后创建新的参数 W
            del self.weight
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)  # 创建新的权重
            self.weight = self.W  # 将 weight 指向 W
            del self.W  # 删除 W

    # 前向传播函数，处理输入 x
    def forward(self, x):
        if self.log:  # 如果需要对数
            x = torch.log(x)  # 取输入的对数

        # 进行傅里叶投影计算
        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi

        # 根据设置翻转正弦和余弦
        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # 先余弦后正弦
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # 先正弦后余弦
        return out  # 返回投影结果


# 定义正弦位置嵌入类，继承自 nn.Module
class SinusoidalPositionalEmbedding(nn.Module):
    """将位置信息应用于嵌入序列。

    接收形状为 (batch_size, seq_length, embed_dim) 的嵌入序列并添加位置嵌入。
    
    参数：
        embed_dim: (int): 位置嵌入的维度。
        max_seq_length: 最大序列长度以应用位置嵌入
    """

    # 初始化函数，设置参数并调用父类构造函数
    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()  # 调用父类构造函数
        position = torch.arange(max_seq_length).unsqueeze(1)  # 创建位置索引
        # 计算位置嵌入的分母
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_seq_length, embed_dim)  # 初始化位置嵌入张量
        # 对偶数索引应用正弦函数
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # 对奇数索引应用余弦函数
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 注册位置嵌入为缓冲区

    # 前向传播函数，处理输入 x
    def forward(self, x):
        _, seq_length, _ = x.shape  # 获取输入的序列长度
        x = x + self.pe[:, :seq_length]  # 将位置嵌入加到输入上
        return x  # 返回结果


# 定义图像位置嵌入类，继承自 nn.Module
class ImagePositionalEmbeddings(nn.Module):
    """
    将潜在图像类转换为向量嵌入。将向量嵌入与潜在空间的高度和宽度位置嵌入相加。

    有关更多细节，请参见 dall-e 论文的图 10: https://arxiv.org/abs/2102.12092

    对于 VQ-diffusion：
    # 输出的向量嵌入将作为变换器的输入。
    Output vector embeddings are used as input for the transformer.

    # 注意，变换器的向量嵌入与 VQVAE 的向量嵌入不同。
    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    # 参数说明：
    Args:
        # 潜在像素嵌入的数量。
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        # 潜在图像的高度，即高度嵌入的数量。
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        # 潜在图像的宽度，即宽度嵌入的数量。
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        # 生成的向量嵌入的维度，用于潜在像素、高度和宽度嵌入。
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    # 初始化方法，设置嵌入层及相关参数
    def __init__(
        self,
        num_embed: int,  # 嵌入数量
        height: int,     # 图像高度
        width: int,      # 图像宽度
        embed_dim: int,  # 嵌入维度
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 保存高度参数
        self.height = height
        # 保存宽度参数
        self.width = width
        # 保存嵌入数量参数
        self.num_embed = num_embed
        # 保存嵌入维度参数
        self.embed_dim = embed_dim

        # 创建潜在像素嵌入层
        self.emb = nn.Embedding(self.num_embed, embed_dim)
        # 创建高度嵌入层
        self.height_emb = nn.Embedding(self.height, embed_dim)
        # 创建宽度嵌入层
        self.width_emb = nn.Embedding(self.width, embed_dim)

    # 前向传播方法
    def forward(self, index):
        # 获取潜在像素的嵌入
        emb = self.emb(index)

        # 获取高度嵌入，并生成一个 (1, H) 的张量
        height_emb = self.height_emb(torch.arange(self.height, device=index.device).view(1, self.height))

        # 将高度嵌入的维度扩展为 (1, H, 1, D)
        height_emb = height_emb.unsqueeze(2)

        # 获取宽度嵌入，并生成一个 (1, W) 的张量
        width_emb = self.width_emb(torch.arange(self.width, device=index.device).view(1, self.width))

        # 将宽度嵌入的维度扩展为 (1, 1, W, D)
        width_emb = width_emb.unsqueeze(1)

        # 将高度和宽度嵌入相加以获得位置嵌入
        pos_emb = height_emb + width_emb

        # 将位置嵌入的形状变为 (1, L, D)，其中 L = H * W
        pos_emb = pos_emb.view(1, self.height * self.width, -1)

        # 将位置嵌入与潜在像素嵌入相加
        emb = emb + pos_emb[:, : emb.shape[1], :]

        # 返回最终的嵌入
        return emb
# 定义一个用于嵌入类标签的模型，处理无分类器引导的标签丢弃
class LabelEmbedding(nn.Module):
    """
    嵌入类标签为向量表示，同时处理分类器自由引导的标签丢弃。

    参数：
        num_classes (`int`): 类别数量。
        hidden_size (`int`): 向量嵌入的大小。
        dropout_prob (`float`): 丢弃标签的概率。
    """

    # 初始化方法，定义类的基本属性
    def __init__(self, num_classes, hidden_size, dropout_prob):
        # 调用父类初始化方法
        super().__init__()
        # 判断是否使用分类器自由引导嵌入
        use_cfg_embedding = dropout_prob > 0
        # 创建嵌入表，将类别数加上是否使用引导的布尔值
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # 保存类别数量
        self.num_classes = num_classes
        # 保存丢弃概率
        self.dropout_prob = dropout_prob

    # 定义标签丢弃方法
    def token_drop(self, labels, force_drop_ids=None):
        """
        丢弃标签以启用无分类器引导。
        """
        # 如果没有强制丢弃的 ID
        if force_drop_ids is None:
            # 随机生成丢弃 ID
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            # 根据强制丢弃的 ID 创建丢弃 ID 张量
            drop_ids = torch.tensor(force_drop_ids == 1)
        # 将丢弃的标签替换为类别数量
        labels = torch.where(drop_ids, self.num_classes, labels)
        # 返回处理后的标签
        return labels

    # 定义前向传播方法
    def forward(self, labels: torch.LongTensor, force_drop_ids=None):
        # 判断是否使用丢弃
        use_dropout = self.dropout_prob > 0
        # 如果处于训练状态且使用丢弃或存在强制丢弃 ID
        if (self.training and use_dropout) or (force_drop_ids is not None):
            # 调用标签丢弃方法
            labels = self.token_drop(labels, force_drop_ids)
        # 从嵌入表中获取标签的嵌入向量
        embeddings = self.embedding_table(labels)
        # 返回嵌入向量
        return embeddings


# 定义文本图像投影模型
class TextImageProjection(nn.Module):
    # 初始化方法，定义模型的基本属性
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 保存图像文本嵌入的数量
        self.num_image_text_embeds = num_image_text_embeds
        # 定义图像嵌入层，将图像嵌入维度映射到图像文本嵌入维度
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        # 定义文本投影层，将文本嵌入维度映射到交叉注意力维度
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    # 定义前向传播方法
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        # 获取批次大小
        batch_size = text_embeds.shape[0]

        # 图像处理
        # 从图像嵌入获取图像文本嵌入
        image_text_embeds = self.image_embeds(image_embeds)
        # 重塑图像文本嵌入的形状
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # 文本处理
        # 将文本嵌入映射到交叉注意力维度
        text_embeds = self.text_proj(text_embeds)

        # 连接图像文本嵌入和文本嵌入并返回
        return torch.cat([image_text_embeds, text_embeds], dim=1)


# 定义图像投影模型
class ImageProjection(nn.Module):
    # 初始化方法，定义模型的基本属性
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
    ):
        # 调用父类初始化方法
        super().__init__()

        # 保存图像文本嵌入的数量
        self.num_image_text_embeds = num_image_text_embeds
        # 定义图像嵌入层，将图像嵌入维度映射到图像文本嵌入维度
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        # 定义层归一化层
        self.norm = nn.LayerNorm(cross_attention_dim)
    # 定义前向传播函数，接受图像嵌入作为输入
        def forward(self, image_embeds: torch.Tensor):
            # 获取输入的批大小
            batch_size = image_embeds.shape[0]
    
            # 处理图像嵌入
            image_embeds = self.image_embeds(image_embeds)
            # 重新调整图像嵌入的形状，以适应后续处理
            image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
            # 对图像嵌入进行归一化处理
            image_embeds = self.norm(image_embeds)
            # 返回处理后的图像嵌入
            return image_embeds
# 定义一个IPAdapterFullImageProjection类，继承自nn.Module
class IPAdapterFullImageProjection(nn.Module):
    # 初始化函数，接受图像嵌入维度和交叉注意力维度
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024):
        # 调用父类构造函数
        super().__init__()
        # 从attention模块导入FeedForward类
        from .attention import FeedForward

        # 创建FeedForward层，输入和输出维度分别为image_embed_dim和cross_attention_dim
        self.ff = FeedForward(image_embed_dim, cross_attention_dim, mult=1, activation_fn="gelu")
        # 创建层归一化层，归一化维度为cross_attention_dim
        self.norm = nn.LayerNorm(cross_attention_dim)

    # 前向传播函数，接收图像嵌入
    def forward(self, image_embeds: torch.Tensor):
        # 返回归一化的前馈层输出
        return self.norm(self.ff(image_embeds))


# 定义一个IPAdapterFaceIDImageProjection类，继承自nn.Module
class IPAdapterFaceIDImageProjection(nn.Module):
    # 初始化函数，接受图像嵌入维度、交叉注意力维度、乘数和标记数量
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024, mult=1, num_tokens=1):
        # 调用父类构造函数
        super().__init__()
        # 从attention模块导入FeedForward类
        from .attention import FeedForward

        # 存储标记数量和交叉注意力维度
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        # 创建FeedForward层，输出维度为cross_attention_dim * num_tokens
        self.ff = FeedForward(image_embed_dim, cross_attention_dim * num_tokens, mult=mult, activation_fn="gelu")
        # 创建层归一化层，归一化维度为cross_attention_dim
        self.norm = nn.LayerNorm(cross_attention_dim)

    # 前向传播函数，接收图像嵌入
    def forward(self, image_embeds: torch.Tensor):
        # 通过前馈层处理图像嵌入
        x = self.ff(image_embeds)
        # 将输出重塑为(num_samples, num_tokens, cross_attention_dim)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        # 返回归一化的输出
        return self.norm(x)


# 定义一个CombinedTimestepLabelEmbeddings类，继承自nn.Module
class CombinedTimestepLabelEmbeddings(nn.Module):
    # 初始化函数，接受类别数量、嵌入维度和类别丢弃概率
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        # 调用父类构造函数
        super().__init__()

        # 创建时间投影层，通道数为256，启用翻转正弦到余弦
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        # 创建时间嵌入层，输入通道数为256，时间嵌入维度为embedding_dim
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # 创建类别嵌入层，接受类别数量、嵌入维度和丢弃概率
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    # 前向传播函数，接收时间步和类别标签
    def forward(self, timestep, class_labels, hidden_dtype=None):
        # 处理时间步并获得投影结果
        timesteps_proj = self.time_proj(timestep)
        # 对投影结果进行时间嵌入
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        # 对类别标签进行嵌入
        class_labels = self.class_embedder(class_labels)  # (N, D)

        # 将时间嵌入和类别嵌入相加，得到条件信息
        conditioning = timesteps_emb + class_labels  # (N, D)

        # 返回条件信息
        return conditioning


# 定义一个CombinedTimestepTextProjEmbeddings类，继承自nn.Module
class CombinedTimestepTextProjEmbeddings(nn.Module):
    # 初始化函数，接受嵌入维度和池化投影维度
    def __init__(self, embedding_dim, pooled_projection_dim):
        # 调用父类构造函数
        super().__init__()

        # 创建时间投影层，通道数为256，禁用翻转正弦到余弦
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # 创建时间嵌入层，输入通道数为256，时间嵌入维度为embedding_dim
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # 创建文本嵌入层，接受池化投影维度和嵌入维度
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    # 前向传播函数，接收时间步和池化投影
    def forward(self, timestep, pooled_projection):
        # 处理时间步并获得投影结果
        timesteps_proj = self.time_proj(timestep)
        # 对投影结果进行时间嵌入
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        # 对池化投影进行嵌入
        pooled_projections = self.text_embedder(pooled_projection)

        # 将时间嵌入和池化投影相加，得到条件信息
        conditioning = timesteps_emb + pooled_projections

        # 返回条件信息
        return conditioning


# 定义一个CombinedTimestepGuidanceTextProjEmbeddings类，继承自nn.Module
class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    # 初始化类，设置嵌入维度和池化投影维度
    def __init__(self, embedding_dim, pooled_projection_dim):
        # 调用父类的初始化方法
        super().__init__()
    
        # 创建时间投影对象，设置通道数和其他参数
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # 创建时间嵌入对象，输入通道数和嵌入维度
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # 创建引导嵌入对象，输入通道数和嵌入维度
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # 创建文本嵌入对象，设置池化投影维度和嵌入维度，以及激活函数
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")
    
    # 前向传播函数，接受时间步、引导信息和池化投影作为输入
    def forward(self, timestep, guidance, pooled_projection):
        # 对时间步进行投影
        timesteps_proj = self.time_proj(timestep)
        # 将投影后的时间步转换为嵌入，保持与池化投影相同的数据类型
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)
    
        # 对引导信息进行投影
        guidance_proj = self.time_proj(guidance)
        # 将投影后的引导信息转换为嵌入，保持与池化投影相同的数据类型
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))  # (N, D)
    
        # 合并时间步嵌入和引导嵌入
        time_guidance_emb = timesteps_emb + guidance_emb
    
        # 对池化投影进行文本嵌入
        pooled_projections = self.text_embedder(pooled_projection)
        # 合并时间引导嵌入和池化投影嵌入
        conditioning = time_guidance_emb + pooled_projections
    
        # 返回最终的条件输出
        return conditioning
# 从 nn.Module 继承，定义 HunyuanDiTAttentionPool 类
class HunyuanDiTAttentionPool(nn.Module):
    # 从指定 GitHub 地址复制的代码

    # 初始化方法，接收空间维度、嵌入维度、头数和可选输出维度
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化位置嵌入参数，尺寸为 (spacial_dim + 1, embed_dim)，并进行缩放
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim**0.5)
        # 创建线性层，用于键的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        # 创建线性层，用于查询的投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # 创建线性层，用于值的投影
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 创建线性层，用于输出的投影，使用输出维度或嵌入维度
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        # 保存头数
        self.num_heads = num_heads

    # 前向传播方法，接收输入 x
    def forward(self, x):
        # 转换输入的维度，从 NLC 变为 LNC
        x = x.permute(1, 0, 2)  # NLC -> LNC
        # 在输入的开头添加平均值，形成新的维度 (L+1)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        # 添加位置嵌入，保持输入的数据类型
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        # 调用多头注意力机制进行处理
        x, _ = F.multi_head_attention_forward(
            query=x[:1],  # 查询只使用第一个位置
            key=x,  # 键使用整个输入
            value=x,  # 值也使用整个输入
            embed_dim_to_check=x.shape[-1],  # 检查嵌入维度
            num_heads=self.num_heads,  # 使用的头数
            q_proj_weight=self.q_proj.weight,  # 查询权重
            k_proj_weight=self.k_proj.weight,  # 键权重
            v_proj_weight=self.v_proj.weight,  # 值权重
            in_proj_weight=None,  # 输入权重设为 None
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),  # 合并偏置
            bias_k=None,  # 键的偏置设为 None
            bias_v=None,  # 值的偏置设为 None
            add_zero_attn=False,  # 不添加零注意力
            dropout_p=0,  # 不使用 dropout
            out_proj_weight=self.c_proj.weight,  # 输出权重
            out_proj_bias=self.c_proj.bias,  # 输出偏置
            use_separate_proj_weight=True,  # 使用单独的权重
            training=self.training,  # 使用当前训练状态
            need_weights=False,  # 不需要权重
        )
        # 返回结果，去掉第一维
        return x.squeeze(0)


# 定义 HunyuanCombinedTimestepTextSizeStyleEmbedding 类
class HunyuanCombinedTimestepTextSizeStyleEmbedding(nn.Module):
    # 初始化方法，接收嵌入维度、池化投影维度、序列长度、交叉注意力维度和样式条件标志
    def __init__(
        self,
        embedding_dim,
        pooled_projection_dim=1024,
        seq_len=256,
        cross_attention_dim=2048,
        use_style_cond_and_image_meta_size=True,
    ):
        # 初始化父类
        super().__init__()

        # 创建时间步投影，用于处理时间信息
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # 创建时间步嵌入，用于将时间信息嵌入到模型中
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # 创建大小投影，用于处理图像大小信息
        self.size_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)

        # 初始化注意力池化层，用于聚合序列信息
        self.pooler = HunyuanDiTAttentionPool(
            seq_len, cross_attention_dim, num_heads=8, output_dim=pooled_projection_dim
        )

        # 使用默认学习的嵌入层以便将来扩展
        self.use_style_cond_and_image_meta_size = use_style_cond_and_image_meta_size
        # 如果使用风格条件和图像元大小
        if use_style_cond_and_image_meta_size:
            # 创建风格嵌入层，用于风格信息的嵌入
            self.style_embedder = nn.Embedding(1, embedding_dim)
            # 计算额外输入维度
            extra_in_dim = 256 * 6 + embedding_dim + pooled_projection_dim
        else:
            # 如果不使用风格条件，设置额外输入维度为池化投影维度
            extra_in_dim = pooled_projection_dim

        # 创建额外嵌入层，用于将额外条件信息嵌入
        self.extra_embedder = PixArtAlphaTextProjection(
            in_features=extra_in_dim,
            hidden_size=embedding_dim * 4,
            out_features=embedding_dim,
            act_fn="silu_fp32",
        )

    # 定义前向传播方法
    def forward(self, timestep, encoder_hidden_states, image_meta_size, style, hidden_dtype=None):
        # 计算时间步投影
        timesteps_proj = self.time_proj(timestep)
        # 嵌入时间步信息
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, 256)

        # 额外条件1: 文本信息的池化投影
        pooled_projections = self.pooler(encoder_hidden_states)  # (N, 1024)

        # 如果使用风格条件和图像元大小
        if self.use_style_cond_and_image_meta_size:
            # 额外条件2: 图像元大小嵌入
            image_meta_size = self.size_proj(image_meta_size.view(-1))
            image_meta_size = image_meta_size.to(dtype=hidden_dtype)
            image_meta_size = image_meta_size.view(-1, 6 * 256)  # (N, 1536)

            # 额外条件3: 风格嵌入
            style_embedding = self.style_embedder(style)  # (N, embedding_dim)

            # 将所有额外向量拼接在一起
            extra_cond = torch.cat([pooled_projections, image_meta_size, style_embedding], dim=1)
        else:
            # 只使用池化投影
            extra_cond = torch.cat([pooled_projections], dim=1)

        # 计算条件信息
        conditioning = timesteps_emb + self.extra_embedder(extra_cond)  # [B, D]

        # 返回条件信息
        return conditioning
# 定义一个结合时间步和标题嵌入的模型
class LuminaCombinedTimestepCaptionEmbedding(nn.Module):
    # 初始化方法，设置隐藏层大小、交叉注意力维度和频率嵌入大小
    def __init__(self, hidden_size=4096, cross_attention_dim=2048, frequency_embedding_size=256):
        # 调用父类构造函数
        super().__init__()
        # 创建时间步投影对象，包含频率嵌入参数
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )
        # 创建时间步嵌入器，输入通道和时间嵌入维度
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)
        # 创建标题嵌入器，包含层归一化和线性变换
        self.caption_embedder = nn.Sequential(
            nn.LayerNorm(cross_attention_dim),
            nn.Linear(
                cross_attention_dim,
                hidden_size,
                bias=True,
            ),
        )

    # 前向传播方法，接收时间步、标题特征和标题掩码
    def forward(self, timestep, caption_feat, caption_mask):
        # 时间步嵌入:
        # 通过时间步投影处理时间步
        time_freq = self.time_proj(timestep)
        # 获取时间嵌入，转换为合适的数据类型
        time_embed = self.timestep_embedder(time_freq.to(dtype=self.timestep_embedder.linear_1.weight.dtype))

        # 标题条件嵌入:
        # 将标题掩码转换为浮点数并增加一个维度
        caption_mask_float = caption_mask.float().unsqueeze(-1)
        # 对标题特征进行池化处理，考虑掩码
        caption_feats_pool = (caption_feat * caption_mask_float).sum(dim=1) / caption_mask_float.sum(dim=1)
        # 将池化结果转换为标题特征的类型
        caption_feats_pool = caption_feats_pool.to(caption_feat)
        # 通过标题嵌入器生成标题嵌入
        caption_embed = self.caption_embedder(caption_feats_pool)

        # 将时间嵌入和标题嵌入相加，生成条件嵌入
        conditioning = time_embed + caption_embed

        # 返回条件嵌入
        return conditioning


# 定义文本时间嵌入的模型
class TextTimeEmbedding(nn.Module):
    # 初始化方法，设置编码器维度、时间嵌入维度和头数
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        # 调用父类构造函数
        super().__init__()
        # 创建层归一化对象
        self.norm1 = nn.LayerNorm(encoder_dim)
        # 创建注意力池化对象
        self.pool = AttentionPooling(num_heads, encoder_dim)
        # 创建线性变换，映射编码器维度到时间嵌入维度
        self.proj = nn.Linear(encoder_dim, time_embed_dim)
        # 创建另一个层归一化对象
        self.norm2 = nn.LayerNorm(time_embed_dim)

    # 前向传播方法，接收隐藏状态
    def forward(self, hidden_states):
        # 对隐藏状态进行第一层归一化
        hidden_states = self.norm1(hidden_states)
        # 通过注意力池化处理隐藏状态
        hidden_states = self.pool(hidden_states)
        # 通过线性变换生成时间嵌入
        hidden_states = self.proj(hidden_states)
        # 对时间嵌入进行第二层归一化
        hidden_states = self.norm2(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义文本和图像时间嵌入的模型
class TextImageTimeEmbedding(nn.Module):
    # 初始化方法，设置文本和图像嵌入维度及时间嵌入维度
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        # 调用父类构造函数
        super().__init__()
        # 创建线性变换，将文本嵌入映射到时间嵌入维度
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)
        # 创建层归一化对象，用于文本嵌入
        self.text_norm = nn.LayerNorm(time_embed_dim)
        # 创建线性变换，将图像嵌入映射到时间嵌入维度
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)

    # 前向传播方法，接收文本和图像嵌入
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        # 文本嵌入
        # 通过线性变换处理文本嵌入
        time_text_embeds = self.text_proj(text_embeds)
        # 对文本嵌入进行归一化处理
        time_text_embeds = self.text_norm(time_text_embeds)

        # 图像嵌入
        # 通过线性变换处理图像嵌入
        time_image_embeds = self.image_proj(image_embeds)

        # 返回图像嵌入和文本嵌入的和
        return time_image_embeds + time_text_embeds


# 定义图像时间嵌入的模型
class ImageTimeEmbedding(nn.Module):
    # 初始化方法，设置图像嵌入维度和时间嵌入维度
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        # 调用父类构造函数
        super().__init__()
        # 创建线性变换，将图像嵌入映射到时间嵌入维度
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        # 创建层归一化对象，用于图像嵌入
        self.image_norm = nn.LayerNorm(time_embed_dim)
    # 定义前向传播方法，接受图像嵌入作为输入
        def forward(self, image_embeds: torch.Tensor):
            # 对输入的图像嵌入进行投影，转换为新的表示
            time_image_embeds = self.image_proj(image_embeds)
            # 对投影后的图像嵌入进行归一化处理
            time_image_embeds = self.image_norm(time_image_embeds)
            # 返回处理后的图像嵌入
            return time_image_embeds
# 定义一个名为 ImageHintTimeEmbedding 的神经网络模块，继承自 nn.Module
class ImageHintTimeEmbedding(nn.Module):
    # 初始化方法，接受图像嵌入维度和时间嵌入维度
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个线性层，将图像嵌入维度映射到时间嵌入维度
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        # 定义层归一化，用于归一化时间嵌入
        self.image_norm = nn.LayerNorm(time_embed_dim)
        # 定义一个序列容器，包含多个卷积层和激活函数
        self.input_hint_block = nn.Sequential(
            # 第一个卷积层，将输入的三个通道映射到16个通道，使用3x3卷积，填充1
            nn.Conv2d(3, 16, 3, padding=1),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 第二个卷积层，保持16个通道，使用3x3卷积，填充1
            nn.Conv2d(16, 16, 3, padding=1),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 第三个卷积层，将16个通道映射到32个通道，使用3x3卷积，填充1，步幅为2
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 第四个卷积层，保持32个通道，使用3x3卷积，填充1
            nn.Conv2d(32, 32, 3, padding=1),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 第五个卷积层，将32个通道映射到96个通道，使用3x3卷积，填充1，步幅为2
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 第六个卷积层，保持96个通道，使用3x3卷积，填充1
            nn.Conv2d(96, 96, 3, padding=1),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 第七个卷积层，将96个通道映射到256个通道，使用3x3卷积，填充1，步幅为2
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            # 使用 SiLU 激活函数
            nn.SiLU(),
            # 最后一个卷积层，将256个通道映射到4个通道，使用3x3卷积，填充1
            nn.Conv2d(256, 4, 3, padding=1),
        )

    # 前向传播方法，接受图像嵌入和提示
    def forward(self, image_embeds: torch.Tensor, hint: torch.Tensor):
        # 将图像嵌入通过线性层转换为时间图像嵌入
        time_image_embeds = self.image_proj(image_embeds)
        # 对时间图像嵌入进行层归一化
        time_image_embeds = self.image_norm(time_image_embeds)
        # 将提示通过输入提示块处理
        hint = self.input_hint_block(hint)
        # 返回时间图像嵌入和处理后的提示
        return time_image_embeds, hint


# 定义一个名为 AttentionPooling 的神经网络模块，继承自 nn.Module
class AttentionPooling(nn.Module):
    # 初始化方法，接受头数、嵌入维度和数据类型
    def __init__(self, num_heads, embed_dim, dtype=None):
        # 调用父类的初始化方法
        super().__init__()
        # 保存数据类型
        self.dtype = dtype
        # 定义位置嵌入参数，随机初始化并进行归一化
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
        # 定义键的线性映射层
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        # 定义查询的线性映射层
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        # 定义值的线性映射层
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        # 保存头的数量
        self.num_heads = num_heads
        # 计算每个头的维度
        self.dim_per_head = embed_dim // self.num_heads
    # 前向传播方法，接收输入 x
    def forward(self, x):
        # 获取输入的批量大小、序列长度和宽度
        bs, length, width = x.size()
    
        # 定义内部形状转换函数
        def shape(x):
            # 将输入形状从 (bs, length, width) 转换为 (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # 转置维度，将形状改为 (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # 重塑形状为 (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # 转置维度，将形状改为 (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            # 返回处理后的张量
            return x
    
        # 计算类标记并加上位置嵌入，保持维度
        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        # 将类标记与输入 x 连接，形状变为 (bs, length+1, width)
        x = torch.cat([class_token, x], dim=1)  
    
        # 处理类标记以得到查询向量，形状为 (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # 处理输入 x 以得到键向量，形状为 (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        # 处理输入 x 以得到值向量
        v = shape(self.v_proj(x))
    
        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        # 计算权重，使用爱因斯坦求和约定
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # 使用 f16 时更稳定
        # 对权重应用 softmax，返回相同数据类型
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    
        # 计算注意力输出，形状为 (bs*n_heads, dim_per_head, class_token_length)
        a = torch.einsum("bts,bcs->bct", weight, v)
    
        # 将注意力输出重塑为 (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)
    
        # 返回类标记的输出
        return a[:, 0, :]  # cls_token
# 定义一个从边界框获取傅里叶嵌入的函数
def get_fourier_embeds_from_boundingbox(embed_dim, box):
    """
    参数:
        embed_dim: 整数，嵌入的维度
        box: 3D张量 [B x N x 4]，表示GLIGEN管道的边界框
    返回:
        [B x N x embed_dim]的张量，包含位置嵌入
    """

    # 获取批次大小和边界框数量
    batch_size, num_boxes = box.shape[:2]

    # 计算傅里叶嵌入的基础频率
    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    # 将频率调整为与输入框相同的设备和数据类型
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    # 将频率与边界框进行扩展，生成嵌入
    emb = emb * box.unsqueeze(-1)

    # 计算嵌入的正弦和余弦值
    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    # 调整维度顺序并展平嵌入
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, embed_dim * 2 * 4)

    # 返回计算得到的嵌入
    return emb


# 定义GLIGEN文本边界框投影的类
class GLIGENTextBoundingboxProjection(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(self, positive_len, out_dim, feature_type="text-only", fourier_freqs=8):
        super().__init__()
        # 保存正样本长度
        self.positive_len = positive_len
        # 保存输出维度
        self.out_dim = out_dim

        # 保存傅里叶嵌入的维度
        self.fourier_embedder_dim = fourier_freqs
        # 计算位置嵌入的维度
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        # 如果输出维度是元组，则取第一个元素
        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        # 如果特征类型为文本专用，构建相应的线性层
        if feature_type == "text-only":
            self.linears = nn.Sequential(
                # 第一个线性层，输入维度为正样本长度加位置维度
                nn.Linear(self.positive_len + self.position_dim, 512),
                # 应用SiLU激活函数
                nn.SiLU(),
                # 第二个线性层
                nn.Linear(512, 512),
                # 应用SiLU激活函数
                nn.SiLU(),
                # 输出层
                nn.Linear(512, out_dim),
            )
            # 初始化一个全零的正样本特征参数
            self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))

        # 如果特征类型为文本和图像
        elif feature_type == "text-image":
            # 为文本特征构建线性层
            self.linears_text = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, out_dim),
            )
            # 为图像特征构建线性层
            self.linears_image = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 512),
                nn.SiLU(),
                nn.Linear(512, out_dim),
            )
            # 初始化全零的文本特征参数
            self.null_text_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
            # 初始化全零的图像特征参数
            self.null_image_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))

        # 初始化全零的位置特征参数
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    # 定义前向传播方法
    def forward(
        self,
        boxes,
        masks,
        positive_embeddings=None,
        phrases_masks=None,
        image_masks=None,
        phrases_embeddings=None,
        image_embeddings=None,
    ):
        # 在最后一个维度上扩展 masks 的维度
        masks = masks.unsqueeze(-1)

        # 从边界框生成四维嵌入，可能包含填充作为占位符
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes)  # B*N*4 -> B*N*C

        # 学习的空嵌入，用于填充的占位符
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # 用学习的空嵌入替换填充部分
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        # 如果存在仅文本信息的正嵌入
        if positive_embeddings is not None:
            # 学习的空嵌入，用于填充的占位符
            positive_null = self.null_positive_feature.view(1, 1, -1)

            # 用学习的空嵌入替换填充部分
            positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null

            # 将正嵌入和 xyxy 嵌入拼接后通过线性层得到对象
            objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))

        # 如果存在文本和图像信息
        else:
            # 在最后一个维度上扩展短语和图像的 masks
            phrases_masks = phrases_masks.unsqueeze(-1)
            image_masks = image_masks.unsqueeze(-1)

            # 学习的空嵌入，用于文本和图像的填充占位符
            text_null = self.null_text_feature.view(1, 1, -1)
            image_null = self.null_image_feature.view(1, 1, -1)

            # 用学习的空嵌入替换填充部分
            phrases_embeddings = phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
            image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null

            # 分别通过线性层处理文本和图像嵌入，得到对象
            objs_text = self.linears_text(torch.cat([phrases_embeddings, xyxy_embedding], dim=-1))
            objs_image = self.linears_image(torch.cat([image_embeddings, xyxy_embedding], dim=-1))
            # 将文本和图像的对象拼接
            objs = torch.cat([objs_text, objs_image], dim=1)

        # 返回最终的对象结果
        return objs
# 定义一个名为 PixArtAlphaCombinedTimestepSizeEmbeddings 的神经网络模块，继承自 nn.Module
class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    用于 PixArt-Alpha。

    参考文献:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    # 初始化方法，接受嵌入维度、尺寸嵌入维度和是否使用额外条件的标志
    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        # 调用父类构造函数
        super().__init__()

        # 设置输出维度为尺寸嵌入维度
        self.outdim = size_emb_dim
        # 创建时间步投影模块，指定通道数和其他参数
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        # 创建时间步嵌入模块，指定输入通道和时间嵌入维度
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        # 记录是否使用额外条件
        self.use_additional_conditions = use_additional_conditions
        # 如果使用额外条件，初始化相关的投影和嵌入模块
        if use_additional_conditions:
            # 创建额外条件的时间步投影模块
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            # 创建分辨率嵌入模块
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            # 创建宽高比嵌入模块
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    # 前向传播方法，处理时间步、分辨率、宽高比等输入
    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        # 对时间步进行投影
        timesteps_proj = self.time_proj(timestep)
        # 嵌入时间步投影，并转换数据类型
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        # 如果使用额外条件，处理分辨率和宽高比的嵌入
        if self.use_additional_conditions:
            # 对分辨率进行额外条件投影，并转换数据类型
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            # 嵌入分辨率
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            # 对宽高比进行额外条件投影，并转换数据类型
            aspect_ratio_emb = self.additional_condition_proj(aspect_ratio.flatten()).to(hidden_dtype)
            # 嵌入宽高比
            aspect_ratio_emb = self.aspect_ratio_embedder(aspect_ratio_emb).reshape(batch_size, -1)
            # 计算条件，包含时间步嵌入和分辨率、宽高比的组合
            conditioning = timesteps_emb + torch.cat([resolution_emb, aspect_ratio_emb], dim=1)
        else:
            # 如果不使用额外条件，直接使用时间步嵌入
            conditioning = timesteps_emb

        # 返回条件结果
        return conditioning


# 定义一个名为 PixArtAlphaTextProjection 的神经网络模块，继承自 nn.Module
class PixArtAlphaTextProjection(nn.Module):
    """
    投影标题嵌入。还处理分类无关指导的 dropout。

    修改自 https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    # 初始化方法，接受输入特征、隐藏层大小、输出特征和激活函数类型
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定输出特征，设置为隐藏层大小
        if out_features is None:
            out_features = hidden_size
        # 创建第一个线性层，指定输入和输出特征，启用偏置
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        # 根据指定的激活函数类型初始化相应的激活函数
        if act_fn == "gelu_tanh":
            # 使用 GELU 激活函数，近似为 tanh
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            # 使用 SiLU 激活函数
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            # 使用 FP32 SiLU 激活函数
            self.act_1 = FP32SiLU()
        else:
            # 如果激活函数不在预设范围内，抛出错误
            raise ValueError(f"Unknown activation function: {act_fn}")
        # 创建第二个线性层，指定输入和输出特征，启用偏置
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)
    # 定义前向传播方法，接收输入的 caption
        def forward(self, caption):
            # 将 caption 通过第一个线性层进行变换，得到隐藏状态
            hidden_states = self.linear_1(caption)
            # 对隐藏状态应用激活函数，增加非线性
            hidden_states = self.act_1(hidden_states)
            # 将经过激活的隐藏状态通过第二个线性层进行变换
            hidden_states = self.linear_2(hidden_states)
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个名为 IPAdapterPlusImageProjectionBlock 的神经网络模块，继承自 nn.Module
class IPAdapterPlusImageProjectionBlock(nn.Module):
    # 初始化方法，接收多个参数并设置默认值
    def __init__(
        self,
        embed_dims: int = 768,  # 嵌入维度，默认值为 768
        dim_head: int = 64,  # 注意力头的维度，默认值为 64
        heads: int = 16,  # 并行注意力头的数量，默认值为 16
        ffn_ratio: float = 4,  # 前馈网络扩展比例，默认值为 4
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 从 attention 模块导入 FeedForward 类
        from .attention import FeedForward

        # 创建一个 LayerNorm 层用于输入的标准化
        self.ln0 = nn.LayerNorm(embed_dims)
        # 创建另一个 LayerNorm 层用于 latents 的标准化
        self.ln1 = nn.LayerNorm(embed_dims)
        # 创建注意力层，使用嵌入维度、头维度和头数量
        self.attn = Attention(
            query_dim=embed_dims,  # 查询维度
            dim_head=dim_head,  # 注意力头维度
            heads=heads,  # 注意力头数量
            out_bias=False,  # 不使用偏置
        )
        # 创建一个顺序模块，包括 LayerNorm 和 FeedForward
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dims),  # 对嵌入进行标准化
            FeedForward(embed_dims, embed_dims, activation_fn="gelu", mult=ffn_ratio, bias=False),  # 前馈网络
        )

    # 定义前向传播方法
    def forward(self, x, latents, residual):
        # 对输入 x 进行标准化
        encoder_hidden_states = self.ln0(x)
        # 对 latents 进行标准化
        latents = self.ln1(latents)
        # 将 encoder_hidden_states 和 latents 在最后一个维度上拼接
        encoder_hidden_states = torch.cat([encoder_hidden_states, latents], dim=-2)
        # 通过注意力层计算新的 latents，并加上残差连接
        latents = self.attn(latents, encoder_hidden_states) + residual
        # 通过前馈网络处理 latents，并加上自身
        latents = self.ff(latents) + latents
        # 返回处理后的 latents
        return latents


# 定义一个名为 IPAdapterPlusImageProjection 的神经网络模块，继承自 nn.Module
class IPAdapterPlusImageProjection(nn.Module):
    """Resampler of IP-Adapter Plus.

    Args:
        embed_dims (int): The feature dimension. Defaults to 768. output_dims (int): The number of output channels,
        that is the same
            number of the channels in the `unet.config.cross_attention_dim`. Defaults to 1024.
        hidden_dims (int):
            The number of hidden channels. Defaults to 1280. depth (int): The number of blocks. Defaults
        to 8. dim_head (int): The number of head channels. Defaults to 64. heads (int): Parallel attention heads.
        Defaults to 16. num_queries (int):
            The number of queries. Defaults to 8. ffn_ratio (float): The expansion ratio
        of feedforward network hidden
            layer channels. Defaults to 4.
    """

    # 初始化方法，接收多个参数并设置默认值
    def __init__(
        self,
        embed_dims: int = 768,  # 嵌入维度，默认值为 768
        output_dims: int = 1024,  # 输出维度，默认值为 1024
        hidden_dims: int = 1280,  # 隐藏层维度，默认值为 1280
        depth: int = 4,  # 模块的深度，默认值为 4
        dim_head: int = 64,  # 注意力头的维度，默认值为 64
        heads: int = 16,  # 注意力头的数量，默认值为 16
        num_queries: int = 8,  # 查询的数量，默认值为 8
        ffn_ratio: float = 4,  # 前馈网络的扩展比例，默认值为 4
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个可训练的参数 latents，初始化为正态分布
        self.latents = nn.Parameter(torch.randn(1, num_queries, hidden_dims) / hidden_dims**0.5)

        # 创建输入投影层，将嵌入维度映射到隐藏层维度
        self.proj_in = nn.Linear(embed_dims, hidden_dims)

        # 创建输出投影层，将隐藏层维度映射到输出维度
        self.proj_out = nn.Linear(hidden_dims, output_dims)
        # 创建输出的标准化层
        self.norm_out = nn.LayerNorm(output_dims)

        # 创建多个 IPAdapterPlusImageProjectionBlock 实例，组成层列表
        self.layers = nn.ModuleList(
            [IPAdapterPlusImageProjectionBlock(hidden_dims, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )
    # 定义前向传播方法，接受输入张量并返回输出张量
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input Tensor.  # 输入张量
        Returns:
            torch.Tensor: Output Tensor.  # 输出张量
        """
        # 根据输入张量的批次大小，重复 latents 张量以匹配形状
        latents = self.latents.repeat(x.size(0), 1, 1)

        # 将输入张量通过第一层投影
        x = self.proj_in(x)

        # 遍历所有层，进行前向传播
        for block in self.layers:
            # 保存当前的 latents 作为残差
            residual = latents
            # 更新 latents，通过当前层处理输入和残差
            latents = block(x, latents, residual)

        # 将最终的 latents 通过输出投影
        latents = self.proj_out(latents)
        # 返回经过归一化处理的输出张量
        return self.norm_out(latents)
# 定义 IPAdapterFaceIDPlusImageProjection 类，继承自 nn.Module
class IPAdapterFaceIDPlusImageProjection(nn.Module):
    """FacePerceiverResampler of IP-Adapter Plus.

    Args:
        embed_dims (int): 特征维度，默认为 768。output_dims (int): 输出通道数，和 `unet.config.cross_attention_dim` 中的通道数相同
            默认值为 1024。hidden_dims (int): 隐藏通道数，默认为 1280。depth (int): 块的数量，默认为 8。dim_head (int): 头通道数，默认为 64。heads (int): 并行注意力头数，默认为 16。num_tokens (int): 标记的数量。num_queries (int): 查询的数量，默认为 8。ffn_ratio (float): 前馈网络隐藏层通道的扩展比，默认为 4。
        ffproj_ratio (float): 前馈网络隐藏层通道的扩展比（用于 ID 嵌入），默认为 4。
    """

    # 初始化函数，定义各种参数及其默认值
    def __init__(
        self,
        embed_dims: int = 768,
        output_dims: int = 768,
        hidden_dims: int = 1280,
        id_embeddings_dim: int = 512,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 16,
        num_tokens: int = 4,
        num_queries: int = 8,
        ffn_ratio: float = 4,
        ffproj_ratio: int = 2,
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 从模块中导入 FeedForward 类
        from .attention import FeedForward

        # 设置类属性 num_tokens 为输入的 num_tokens 参数
        self.num_tokens = num_tokens
        # 设置嵌入维度为输入的 embed_dims 参数
        self.embed_dim = embed_dims
        # 初始化 clip_embeds 为 None
        self.clip_embeds = None
        # 设置 shortcut 为 False
        self.shortcut = False
        # 设置 shortcut_scale 为 1.0
        self.shortcut_scale = 1.0

        # 创建前馈网络的投影层，使用 gelu 激活函数
        self.proj = FeedForward(id_embeddings_dim, embed_dims * num_tokens, activation_fn="gelu", mult=ffproj_ratio)
        # 创建层归一化层，用于嵌入维度
        self.norm = nn.LayerNorm(embed_dims)

        # 创建输入投影层，将隐藏维度映射到嵌入维度
        self.proj_in = nn.Linear(hidden_dims, embed_dims)

        # 创建输出投影层，将嵌入维度映射到输出维度
        self.proj_out = nn.Linear(embed_dims, output_dims)
        # 创建输出归一化层，用于输出维度
        self.norm_out = nn.LayerNorm(output_dims)

        # 创建一个模块列表，包含多个 IPAdapterPlusImageProjectionBlock 实例
        self.layers = nn.ModuleList(
            [IPAdapterPlusImageProjectionBlock(embed_dims, dim_head, heads, ffn_ratio) for _ in range(depth)]
        )
    # 定义前向传播函数，接收 ID 嵌入并返回输出张量
    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        """Forward pass.
    
        Args:
            id_embeds (torch.Tensor): Input Tensor (ID embeds).
        Returns:
            torch.Tensor: Output Tensor.
        """
        # 将输入张量转换为与 clip_embeds 相同的数据类型
        id_embeds = id_embeds.to(self.clip_embeds.dtype)
        # 对 ID 嵌入进行线性变换
        id_embeds = self.proj(id_embeds)
        # 将 ID 嵌入重塑为三维张量，形状为 (batch_size, num_tokens, embed_dim)
        id_embeds = id_embeds.reshape(-1, self.num_tokens, self.embed_dim)
        # 对重塑后的 ID 嵌入进行归一化处理
        id_embeds = self.norm(id_embeds)
        # 将处理后的 ID 嵌入赋值给 latents
        latents = id_embeds
    
        # 对 clip_embeds 进行线性变换
        clip_embeds = self.proj_in(self.clip_embeds)
        # 将 clip_embeds 重塑为三维张量，形状为 (batch_size, channels, height, width)
        x = clip_embeds.reshape(-1, clip_embeds.shape[2], clip_embeds.shape[3])
    
        # 遍历网络层进行处理
        for block in self.layers:
            # 保存当前的 latents 以便后续残差连接
            residual = latents
            # 通过当前块处理 x 和 latents，并返回新的 latents
            latents = block(x, latents, residual)
    
        # 对处理后的 latents 进行线性变换
        latents = self.proj_out(latents)
        # 对最终的 latents 进行归一化处理
        out = self.norm_out(latents)
        # 如果开启了 shortcut，则将 ID 嵌入与处理结果结合
        if self.shortcut:
            out = id_embeds + self.shortcut_scale * out
        # 返回最终的输出张量
        return out
# 定义一个多输入适配器图像投影类，继承自 nn.Module
class MultiIPAdapterImageProjection(nn.Module):
    # 初始化方法，接收一个层的列表或元组，用于图像投影
    def __init__(self, IPAdapterImageProjectionLayers: Union[List[nn.Module], Tuple[nn.Module]]):
        # 调用父类构造函数
        super().__init__()
        # 将输入的层转换为 nn.ModuleList 以便后续使用
        self.image_projection_layers = nn.ModuleList(IPAdapterImageProjectionLayers)

    # 前向传播方法，接受图像嵌入作为输入
    def forward(self, image_embeds: List[torch.Tensor]):
        # 初始化一个空列表，用于存储投影后的图像嵌入
        projected_image_embeds = []

        # 当前接受的 `image_embeds` 可以是：
        #  1. 一个张量（已弃用），形状为 [batch_size, embed_dim] 或 [batch_size, sequence_length, embed_dim]
        #  2. 一个包含 `n` 个张量的列表，其中 `n` 为适配器的数量，每个张量的形状可以是 [batch_size, num_images, embed_dim] 或 [batch_size, num_images, sequence_length, embed_dim]
        if not isinstance(image_embeds, list):
            # 构建弃用警告信息
            deprecation_message = (
                "You have passed a tensor as `image_embeds`.This is deprecated and will be removed in a future release."
                " Please make sure to update your script to pass `image_embeds` as a list of tensors to suppress this warning."
            )
            # 调用 deprecate 函数发出弃用警告
            deprecate("image_embeds not a list", "1.0.0", deprecation_message, standard_warn=False)
            # 将输入的张量扩展为单元素列表
            image_embeds = [image_embeds.unsqueeze(1)]

        # 检查图像嵌入的数量是否与投影层的数量一致
        if len(image_embeds) != len(self.image_projection_layers):
            # 如果不一致，抛出值错误
            raise ValueError(
                f"image_embeds must have the same length as image_projection_layers, got {len(image_embeds)} and {len(self.image_projection_layers)}"
            )

        # 遍历每个图像嵌入和对应的投影层
        for image_embed, image_projection_layer in zip(image_embeds, self.image_projection_layers):
            # 获取当前图像嵌入的批次大小和图像数量
            batch_size, num_images = image_embed.shape[0], image_embed.shape[1]
            # 将图像嵌入重塑为新形状，以便进行投影
            image_embed = image_embed.reshape((batch_size * num_images,) + image_embed.shape[2:])
            # 将重塑后的图像嵌入通过当前的投影层进行处理
            image_embed = image_projection_layer(image_embed)
            # 将投影后的图像嵌入重塑回原来的形状
            image_embed = image_embed.reshape((batch_size, num_images) + image_embed.shape[1:])

            # 将处理后的图像嵌入添加到列表中
            projected_image_embeds.append(image_embed)

        # 返回所有投影后的图像嵌入
        return projected_image_embeds
```