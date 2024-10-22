# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\finite_scalar_quantization.py`

```py
# Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
# 代码来自 Jax 版本的附录 A.1

from typing import List, Optional  # 导入 List 和 Optional 类型以便进行类型注释

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.nn import Module  # 从 nn 模块中导入 Module 基类
from torch import Tensor, int32  # 导入 Tensor 和 int32 类型
from torch.cuda.amp import autocast  # 导入自动混合精度的上下文管理器

from einops import rearrange, pack, unpack  # 从 einops 库导入重排、打包和解包函数

# helper functions

def exists(v):  # 定义一个函数检查变量 v 是否存在
    return v is not None  # 返回 v 是否不为 None

def default(*args):  # 定义一个函数以返回第一个存在的参数
    for arg in args:  # 遍历所有参数
        if exists(arg):  # 如果参数存在
            return arg  # 返回该参数
    return None  # 如果没有参数存在，返回 None

def pack_one(t, pattern):  # 定义一个函数以打包一个张量 t
    return pack([t], pattern)  # 将 t 放入列表中并按照模式打包

def unpack_one(t, ps, pattern):  # 定义一个函数以解包一个张量 t
    return unpack(t, ps, pattern)[0]  # 解包 t，返回第一个解包结果

# tensor helpers

def round_ste(z: Tensor) -> Tensor:  # 定义一个函数以进行带有直通梯度的四舍五入
    """Round with straight through gradients."""  # 函数说明
    zhat = z.round()  # 对 z 进行四舍五入
    return z + (zhat - z).detach()  # 返回 z 加上 zhat 和 z 的差的梯度不跟随版本

# main class

class FSQ(Module):  # 定义 FSQ 类，继承自 Module
    def __init__(  # 定义初始化方法
        self,
        levels: List[int],  # 量化级别的列表
        dim: Optional[int] = None,  # 输入维度，可选
        num_codebooks=1,  # 码本数量，默认为 1
        keep_num_codebooks_dim: Optional[bool] = None,  # 是否保留码本维度的可选参数
        scale: Optional[float] = None,  # 缩放因子，可选
    ):
        super().__init__()  # 调用父类的初始化方法
        _levels = torch.tensor(levels, dtype=int32)  # 将 levels 转换为 int32 类型的张量
        self.register_buffer("_levels", _levels, persistent=False)  # 注册不持久化的缓冲区

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)  # 计算基础张量
        self.register_buffer("_basis", _basis, persistent=False)  # 注册不持久化的缓冲区

        self.scale = scale  # 保存缩放因子

        codebook_dim = len(levels)  # 计算码本维度
        self.codebook_dim = codebook_dim  # 保存码本维度

        effective_codebook_dim = codebook_dim * num_codebooks  # 计算有效码本维度
        self.num_codebooks = num_codebooks  # 保存码本数量
        self.effective_codebook_dim = effective_codebook_dim  # 保存有效码本维度

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)  # 确定是否保留码本维度
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)  # 确保规则有效
        self.keep_num_codebooks_dim = keep_num_codebooks_dim  # 保存是否保留的标志

        self.dim = default(dim, len(_levels) * num_codebooks)  # 设置输入维度

        has_projections = self.dim != effective_codebook_dim  # 检查是否需要投影
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()  # 输入投影层
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()  # 输出投影层
        self.has_projections = has_projections  # 保存是否有投影的标志

        self.codebook_size = self._levels.prod().item()  # 计算码本大小

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)  # 获取隐式码本
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)  # 注册隐式码本缓冲区

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:  # 定义边界函数
        """Bound `z`, an array of shape (..., d)."""  # 函数说明
        half_l = (self._levels - 1) * (1 + eps) / 2  # 计算半边界
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)  # 计算偏移量
        shift = (offset / half_l).atanh()  # 计算偏移量的反双曲正切
        return (z + shift).tanh() * half_l - offset  # 返回边界调整后的结果
    # 定义量化函数，将输入张量 z 进行量化，并返回量化后的 zhat，形状与 z 相同
    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # 对输入 z 进行边界处理后，执行四舍五入量化
        quantized = round_ste(self.bound(z))
        # 计算半宽度，用于重新归一化到 [-1, 1] 的范围
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        # 返回量化后的值，归一化到 [-1, 1] 范围
        return quantized / half_width

    # 定义缩放和偏移函数，将归一化的 zhat 转换为相应的范围
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        # 计算半宽度
        half_width = self._levels // 2
        # 将归一化的 zhat 进行缩放和偏移，返回相应值
        return (zhat_normalized * half_width) + half_width

    # 定义反缩放和偏移函数，将 zhat 转换回归一化值
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        # 计算半宽度
        half_width = self._levels // 2
        # 进行反缩放和偏移，返回归一化值
        return (zhat - half_width) / half_width

    # 定义将代码转换为索引的函数
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        # 确保 zhat 的最后一维与代码本的维度匹配
        assert zhat.shape[-1] == self.codebook_dim
        # 将 zhat 进行缩放和偏移
        zhat = self._scale_and_shift(zhat)
        # 计算索引并转换为整数类型
        return (zhat * self._basis).sum(dim=-1).to(int32)

    # 定义将索引转换为代码的函数
    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""
        # 检查输入是否为图像或视频
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        # 调整索引的形状以增加一个维度
        indices = rearrange(indices, "... -> ... 1")
        # 计算非中心化的代码
        codes_non_centered = (indices // self._basis) % self._levels
        # 将非中心化代码进行反缩放和偏移
        codes = self._scale_and_shift_inverse(codes_non_centered)

        # 如果需要保留代码本维度，则调整代码形状
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        # 如果需要，进行投影操作
        if project_out:
            codes = self.project_out(codes)

        # 如果是图像或视频，调整代码的形状
        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        # 返回处理后的代码
        return codes

    # 定义前向传播函数
    @autocast(enabled=False)
    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        # 检查输入是否为图像或视频
        is_img_or_video = z.ndim >= 4

        # 将图像或视频标准化为 (batch, seq, dimension)
        if is_img_or_video:
            # 调整 z 的形状
            z = rearrange(z, "b d ... -> b ... d")
            # 打包 z 以获取适当的维度信息
            z, ps = pack_one(z, "b * d")

        # 确保 z 的最后一维与期望的维度匹配
        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        # 将输入 z 进行投影
        z = self.project_in(z)

        # 调整 z 的形状以便于后续处理
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # 对 z 进行量化，得到代码
        codes = self.quantize(z)
        # 将代码转换为索引
        indices = self.codes_to_indices(codes)

        # 调整代码的形状
        codes = rearrange(codes, "b n c d -> b n (c d)")

        # 对代码进行投影以生成输出
        out = self.project_out(codes)

        # 重新构造图像或视频的维度
        if is_img_or_video:
            # 解包输出以恢复原始形状
            out = unpack_one(out, ps, "b * d")
            # 调整输出的形状
            out = rearrange(out, "b ... d -> b d ...")

            # 解包索引以恢复原始形状
            indices = unpack_one(indices, ps, "b * c")

        # 如果不保留代码本维度，则调整索引的形状
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # 返回最终的输出和索引
        return out, indices
```