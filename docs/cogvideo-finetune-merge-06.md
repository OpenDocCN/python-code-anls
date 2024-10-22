# CogVideo & CogVideoX 微调代码源码解析（七）



# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\base.py`

```py
# 导入抽象方法和类型注解
from abc import abstractmethod
# 导入任意类型和元组类型
from typing import Any, Tuple

# 导入 PyTorch 和功能模块
import torch
import torch.nn.functional as F
# 导入神经网络模块
from torch import nn


# 定义一个抽象正则化器类，继承自 nn.Module
class AbstractRegularizer(nn.Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

    # 定义前向传播方法，接受一个张量并返回张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 抛出未实现错误，强制子类实现该方法
        raise NotImplementedError()

    # 定义获取可训练参数的抽象方法
    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        # 抛出未实现错误，强制子类实现该方法
        raise NotImplementedError()


# 定义身份正则化器类，继承自 AbstractRegularizer
class IdentityRegularizer(AbstractRegularizer):
    # 实现前向传播方法，接受一个张量并返回该张量和空字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        return z, dict()

    # 实现获取可训练参数的方法，返回一个生成器
    def get_trainable_parameters(self) -> Any:
        # 生成器不返回任何值
        yield from ()


# 定义测量困惑度的函数，接受预测索引和质心数量作为参数
def measure_perplexity(predicted_indices: torch.Tensor, num_centroids: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # 评估集群的困惑度，当困惑度等于嵌入数量时，所有集群的使用是完全均匀的
    # 将预测索引转化为独热编码并重塑为二维张量
    encodings = F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    # 计算每个质心的平均概率
    avg_probs = encodings.mean(0)
    # 计算困惑度
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    # 计算使用的集群数量
    cluster_use = torch.sum(avg_probs > 0)
    # 返回困惑度和集群使用数量
    return perplexity, cluster_use
```

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

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\lookup_free_quantization.py`

```py
# 文档字符串：查找自由量化的说明和论文链接
"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

# 从数学库导入对数和向上取整的函数
from math import log2, ceil
# 从 collections 导入命名元组，用于创建简单的类
from collections import namedtuple

# 导入 PyTorch 库及其模块
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.cuda.amp import autocast

# 导入 einops 库，用于张量操作
from einops import rearrange, reduce, pack, unpack

# 常量定义

# 创建一个命名元组，用于存储量化结果及其索引和熵辅助损失
Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])

# 创建一个命名元组，用于存储损失分解的信息
LossBreakdown = namedtuple("LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"])

# 辅助函数

# 检查变量是否存在（非 None）
def exists(v):
    return v is not None

# 返回第一个非 None 的参数
def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

# 将一个张量按模式打包
def pack_one(t, pattern):
    return pack([t], pattern)

# 按模式解包张量，返回第一个解包的元素
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 熵相关函数

# 计算张量的对数，确保最小值不小于 eps
def log(t, eps=1e-5):
    return t.clamp(min=eps).log()

# 计算概率分布的熵
def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# 类定义

# 定义 LFQ 类，继承自 PyTorch 的 Module
class LFQ(Module):
    # 初始化函数，设置多个参数
    def __init__(
        self,
        *,
        dim=None,  # 量化维度
        codebook_size=None,  # 码本大小
        entropy_loss_weight=0.1,  # 熵损失的权重
        commitment_loss_weight=0.25,  # 承诺损失的权重
        diversity_gamma=1.0,  # 多样性控制参数
        straight_through_activation=nn.Identity(),  # 直通激活函数
        num_codebooks=1,  # 码本数量
        keep_num_codebooks_dim=None,  # 保持码本维度
        codebook_scale=1.0,  # 码本缩放因子，残差 LFQ 每层缩小 2 倍
        frac_per_sample_entropy=1.0,  # 每个样本熵的比例，若小于 1 则随机使用部分概率
    ):
        # 调用父类构造函数初始化
        super().__init__()

        # 一些断言验证

        # 确保至少指定 dim 或 codebook_size
        assert exists(dim) or exists(codebook_size), "either dim or codebook_size must be specified for LFQ"
        # 确保 codebook_size 是 2 的幂，若指定了
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        # 若未指定 codebook_size，则使用默认值 2 的 dim 次方
        codebook_size = default(codebook_size, lambda: 2**dim)
        # 计算 codebook 的维度
        codebook_dim = int(log2(codebook_size))

        # 计算总的 codebook 维度
        codebook_dims = codebook_dim * num_codebooks
        # 若未指定 dim，则使用 codebook_dims
        dim = default(dim, codebook_dims)

        # 检查是否存在投影
        has_projections = dim != codebook_dims
        # 根据是否有投影选择线性层或身份映射
        self.project_in = nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        # 存储是否有投影的布尔值
        self.has_projections = has_projections

        # 保存维度和相关参数
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        # 处理保持 codebook 维度的默认值
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        # 确保在多 codebook 时要保持维度
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # 直通激活函数
        self.activation = straight_through_activation

        # 与熵辅助损失相关的权重

        # 确保熵比例在合理范围内
        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy

        # 保存熵损失的权重和其他参数
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # 代码簿缩放因子
        self.codebook_scale = codebook_scale

        # 承诺损失的权重
        self.commitment_loss_weight = commitment_loss_weight

        # 用于推理时没有辅助损失的情况

        # 注册一个掩码，用于后续计算
        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        # 注册一个零张量，非持久化
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # 代码的初始化

        # 创建所有可能的代码
        all_codes = torch.arange(codebook_size)
        # 通过掩码生成二进制位的浮点表示
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        # 将二进制位转换为代码簿
        codebook = self.bits_to_codes(bits)

        # 注册代码簿，非持久化
        self.register_buffer("codebook", codebook, persistent=False)

    # 将位转换为代码的方法
    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # dtype 属性的定义
    @property
    def dtype(self):
        return self.codebook.dtype
    # 将索引转换为代码，返回相应的编码
        def indices_to_codes(self, indices, project_out=True):
            # 检查索引是否为图像或视频数据，判断维度
            is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
    
            # 如果不保留代码本维度，将索引重排列为增加一个维度
            if not self.keep_num_codebooks_dim:
                indices = rearrange(indices, "... -> ... 1")
    
            # 将索引转换为代码，生成-1或1的位
            bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)
    
            # 将位转换为编码
            codes = self.bits_to_codes(bits)
    
            # 将编码重排列为合并维度
            codes = rearrange(codes, "... c d -> ... (c d)")
    
            # 判断是否将编码投影回原始维度
            if project_out:
                codes = self.project_out(codes)
    
            # 将编码重排列回原始形状
            if is_img_or_video:
                codes = rearrange(codes, "b ... d -> b d ...")
    
            # 返回最终的编码
            return codes
    
        # 禁用自动混合精度
        @autocast(enabled=False)
        def forward(
            self,
            x,
            inv_temperature=100.0,
            return_loss_breakdown=False,
            mask=None,
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\quantize.py`

```py
# 导入 logging 模块以便进行日志记录
import logging
# 从 abc 模块导入 abstractmethod，用于定义抽象方法
from abc import abstractmethod
# 从 typing 模块导入多种类型提示
from typing import Dict, Iterator, Literal, Optional, Tuple, Union

# 导入 numpy 库并命名为 np
import numpy as np
# 导入 PyTorch 库及其子模块
import torch
import torch.nn as nn
import torch.nn.functional as F
# 从 einops 导入 rearrange 函数，用于重排张量
from einops import rearrange
# 从 torch 导入 einsum 函数，用于张量操作
from torch import einsum

# 从同一包中导入 AbstractRegularizer 类和 measure_perplexity 函数
from .base import AbstractRegularizer, measure_perplexity

# 创建一个 logger 实例，用于当前模块的日志记录
logpy = logging.getLogger(__name__)


# 定义一个抽象量化器类，继承自 AbstractRegularizer
class AbstractQuantizer(AbstractRegularizer):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 在初始化时定义这些属性
        # shape (N,) 表示该张量的形状为一维
        self.used: Optional[torch.Tensor]  # 定义已使用的张量，可能为 None
        self.re_embed: int  # 定义重嵌入的整数值
        self.unknown_index: Union[Literal["random"], int]  # 定义未知索引，可能为随机或整数

    # 将输入索引映射到已使用的索引
    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        # 确保已定义 used 索引
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape  # 获取输入索引的形状
        assert len(ishape) > 1  # 确保输入维度大于 1
        inds = inds.reshape(ishape[0], -1)  # 重塑输入索引为二维
        used = self.used.to(inds)  # 将 used 张量移动到与 inds 相同的设备
        match = (inds[:, :, None] == used[None, None, ...]).long()  # 计算索引匹配情况
        new = match.argmax(-1)  # 找到每个匹配的最大索引
        unknown = match.sum(2) < 1  # 标记未知索引
        # 如果未知索引为随机，则随机生成新的索引
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index  # 将未知索引设置为指定的未知索引
        return new.reshape(ishape)  # 返回重塑后的新索引

    # 将输入索引映射回所有索引
    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        # 确保已定义 used 索引
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape  # 获取输入索引的形状
        assert len(ishape) > 1  # 确保输入维度大于 1
        inds = inds.reshape(ishape[0], -1)  # 重塑输入索引为二维
        used = self.used.to(inds)  # 将 used 张量移动到与 inds 相同的设备
        # 如果重嵌入数量大于已使用数量，则处理额外的令牌
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # 将超出范围的索引设置为零
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)  # 根据输入索引收集数据
        return back.reshape(ishape)  # 返回重塑后的数据

    # 定义抽象方法以获取编码表条目
    @abstractmethod
    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        raise NotImplementedError()  # 抛出未实现错误

    # 获取可训练参数的迭代器
    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.parameters()  # 生成模型参数


# 定义 Gumbel 量化器类，继承自 AbstractQuantizer
class GumbelQuantizer(AbstractQuantizer):
    """
    credit to @karpathy:
    https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    # 初始化方法
    def __init__(
        self,
        num_hiddens: int,  # 隐藏层单元数
        embedding_dim: int,  # 嵌入维度
        n_embed: int,  # 嵌入数量
        straight_through: bool = True,  # 是否使用直通梯度
        kl_weight: float = 5e-4,  # KL 散度的权重
        temp_init: float = 1.0,  # 初始化温度
        remap: Optional[str] = None,  # 可选的重映射方式
        unknown_index: str = "random",  # 未知索引的默认值为随机
        loss_key: str = "loss/vq",  # 损失键的默认值
    # 定义一个返回 None 的方法
        ) -> None:
            # 调用父类的构造函数
            super().__init__()
    
            # 保存损失的关键字
            self.loss_key = loss_key
            # 设置嵌入维度
            self.embedding_dim = embedding_dim
            # 设置嵌入数量
            self.n_embed = n_embed
    
            # 设置是否使用直通估计
            self.straight_through = straight_through
            # 初始化温度参数
            self.temperature = temp_init
            # 设置 KL 散度权重
            self.kl_weight = kl_weight
    
            # 创建一个 2D 卷积层，输入通道为 num_hiddens，输出通道为 n_embed，卷积核大小为 1
            self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
            # 创建嵌入层，嵌入数量为 n_embed，嵌入维度为 embedding_dim
            self.embed = nn.Embedding(n_embed, embedding_dim)
    
            # 保存重映射文件路径
            self.remap = remap
            # 如果提供了重映射
            if self.remap is not None:
                # 从重映射文件中加载使用的索引，并将其注册为缓冲区
                self.register_buffer("used", torch.tensor(np.load(self.remap)))
                # 设置重嵌入数量为使用的索引的数量
                self.re_embed = self.used.shape[0]
            else:
                # 如果未提供重映射，则使用全部嵌入数量
                self.used = None
                self.re_embed = n_embed
            # 如果未知索引设置为 "extra"
            if unknown_index == "extra":
                # 将未知索引设置为重嵌入数量，并增加一个
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            else:
                # 断言未知索引必须为 "random"、"extra" 或整数
                assert unknown_index == "random" or isinstance(
                    unknown_index, int
                ), "unknown index needs to be 'random', 'extra' or any integer"
                # 设置未知索引
                self.unknown_index = unknown_index  # "random" or "extra" or integer
            # 如果提供了重映射，则记录相关信息
            if self.remap is not None:
                logpy.info(
                    f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                    f"Using {self.unknown_index} for unknown indices."
                )
    
        # 定义前向传播方法，接收输入张量 z 和可选的温度参数
        def forward(
            self, z: torch.Tensor, temp: Optional[float] = None, return_logits: bool = False
        ) -> Tuple[torch.Tensor, Dict]:
            # 在评估模式下强制 hard=True，因为必须进行量化。
            # 实际上，始终为真似乎也有效
            hard = self.straight_through if self.training else True
            # 设置温度，如果未提供，则使用默认值
            temp = self.temperature if temp is None else temp
            # 初始化输出字典
            out_dict = {}
            # 通过卷积层计算 logits
            logits = self.proj(z)
            # 如果提供了重映射
            if self.remap is not None:
                # 创建与 logits 同样形状的全零张量
                full_zeros = torch.zeros_like(logits)
                # 仅保留使用的 logits
                logits = logits[:, self.used, ...]
    
            # 使用 Gumbel-Softmax 函数生成软 one-hot 编码
            soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
            # 如果提供了重映射
            if self.remap is not None:
                # 将未使用的条目设置为零
                full_zeros[:, self.used, ...] = soft_one_hot
                soft_one_hot = full_zeros
            # 根据软 one-hot 编码和嵌入权重计算量化的 z
            z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)
    
            # 计算 KL 散度损失
            qy = F.softmax(logits, dim=1)
            # 计算散度并取平均
            diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
            # 将散度损失存储到输出字典中
            out_dict[self.loss_key] = diff
    
            # 计算 soft_one_hot 编码的最大索引
            ind = soft_one_hot.argmax(dim=1)
            # 如果提供了重映射，将索引转换为使用的索引
            if self.remap is not None:
                ind = self.remap_to_used(ind)
    
            # 如果需要返回 logits，则将其存储到输出字典中
            if return_logits:
                out_dict["logits"] = logits
    
            # 返回量化的 z 和输出字典
            return z_q, out_dict
    # 获取代码本条目的方法，根据给定的索引和形状
    def get_codebook_entry(self, indices, shape):
        # TODO: 当前形状参数尚不可选
        b, h, w, c = shape  # 解包形状参数，获取批次、身高、宽度和通道数
        # 确保索引的总数与给定的形状匹配
        assert b * h * w == indices.shape[0]
        # 重排列索引，将其形状调整为 (b, h, w)
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        # 如果存在重映射，则将索引映射到所有可能的值
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        # 将索引转换为独热编码，调整维度顺序并转换为浮点数
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        # 通过爱因斯坦求和约定计算最终的量化表示 z_q
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        # 返回量化后的表示
        return z_q
# 定义向量量化类，继承自抽象量化器
class VectorQuantizer(AbstractQuantizer):
    """
    ____________________________________________
    VQ-VAE 的离散化瓶颈部分。
    输入:
    - n_e : 嵌入的数量
    - e_dim : 嵌入的维度
    - beta : 在损失项中使用的承诺成本,
        beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    # 初始化方法，定义类的参数
    def __init__(
        self,
        n_e: int,  # 嵌入的数量
        e_dim: int,  # 嵌入的维度
        beta: float = 0.25,  # 默认承诺成本
        remap: Optional[str] = None,  # 可选的重映射文件路径
        unknown_index: str = "random",  # 未知索引的处理方式
        sane_index_shape: bool = False,  # 是否保持合理的索引形状
        log_perplexity: bool = False,  # 是否记录困惑度
        embedding_weight_norm: bool = False,  # 是否使用权重归一化
        loss_key: str = "loss/vq",  # 损失的键
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存嵌入数量
        self.n_e = n_e
        # 保存嵌入维度
        self.e_dim = e_dim
        # 保存承诺成本
        self.beta = beta
        # 保存损失键
        self.loss_key = loss_key

        # 如果不使用权重归一化
        if not embedding_weight_norm:
            # 初始化嵌入层，权重范围均匀分布
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            # 使用权重归一化的嵌入层
            self.embedding = torch.nn.utils.weight_norm(nn.Embedding(self.n_e, self.e_dim), dim=1)

        # 保存重映射参数
        self.remap = remap
        # 如果指定了重映射
        if self.remap is not None:
            # 从重映射文件中加载已使用的索引
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 设置重新嵌入的数量
            self.re_embed = self.used.shape[0]
        else:
            # 否则未使用的索引为 None
            self.used = None
            # 重新嵌入的数量为 n_e
            self.re_embed = n_e
        # 如果未知索引是 "extra"
        if unknown_index == "extra":
            # 设置未知索引为重新嵌入的数量
            self.unknown_index = self.re_embed
            # 重新嵌入的数量加一
            self.re_embed = self.re_embed + 1
        else:
            # 确保未知索引是 "random"、"extra" 或整数
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            # 保存未知索引的值
            self.unknown_index = unknown_index  # "random" 或 "extra" 或整数
        # 如果指定了重映射，记录信息
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

        # 保存是否保持合理的索引形状的标志
        self.sane_index_shape = sane_index_shape
        # 保存是否记录困惑度的标志
        self.log_perplexity = log_perplexity

    # 前向传播方法，定义输入的处理
    def forward(
        self,
        z: torch.Tensor,  # 输入张量
    ) -> Tuple[torch.Tensor, Dict]:  # 定义返回类型为元组，包含一个张量和一个字典
        do_reshape = z.ndim == 4  # 检查 z 的维度是否为 4，决定是否需要重塑
        if do_reshape:  # 如果 z 是 4 维的
            # reshape z -> (batch, height, width, channel) and flatten  # 重塑 z 的维度为 (batch, height, width, channel) 并扁平化
            z = rearrange(z, "b c h w -> b h w c").contiguous()  # 重新排列 z 的维度，并保证内存连续性

        else:  # 如果 z 不是 4 维的
            assert z.ndim < 4, "No reshaping strategy for inputs > 4 dimensions defined"  # 断言 z 的维度小于 4
            z = z.contiguous()  # 确保 z 的内存是连续的

        z_flattened = z.view(-1, self.e_dim)  # 将 z 重塑为 (batch_size, e_dim) 的形状
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z  # 计算 z 到嵌入 e_j 的距离

        d = (  # 计算每个 z 到嵌入的距离
            torch.sum(z_flattened**2, dim=1, keepdim=True)  # 计算 z_flattened 的平方和
            + torch.sum(self.embedding.weight**2, dim=1)  # 计算嵌入权重的平方和
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))  # 计算 z_flattened 与嵌入的内积
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # 找到每个 z 到嵌入距离的最小索引
        z_q = self.embedding(min_encoding_indices).view(z.shape)  # 根据最小索引获取量化的嵌入，并重塑为原始 z 的形状
        loss_dict = {}  # 初始化损失字典
        if self.log_perplexity:  # 如果需要记录困惑度
            perplexity, cluster_usage = measure_perplexity(min_encoding_indices.detach(), self.n_e)  # 计算困惑度和集群使用情况
            loss_dict.update({"perplexity": perplexity, "cluster_usage": cluster_usage})  # 更新损失字典

        # compute loss for embedding  # 计算嵌入的损失
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)  # 计算损失值
        loss_dict[self.loss_key] = loss  # 将损失添加到损失字典中

        # preserve gradients  # 保留梯度
        z_q = z + (z_q - z).detach()  # 将量化的 z_q 与原始 z 结合，保留梯度

        # reshape back to match original input shape  # 重新调整形状以匹配原始输入形状
        if do_reshape:  # 如果之前进行了重塑
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()  # 将 z_q 的维度调整回 (batch, channel, height, width)

        if self.remap is not None:  # 如果需要重映射
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)  # 添加批次维度
            min_encoding_indices = self.remap_to_used(min_encoding_indices)  # 对最小编码索引进行重映射
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # 扁平化为一维

        if self.sane_index_shape:  # 如果索引形状正常
            if do_reshape:  # 如果之前进行了重塑
                min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])  # 将索引重塑为与 z_q 形状相匹配
            else:  # 如果没有重塑
                min_encoding_indices = rearrange(min_encoding_indices, "(b s) 1 -> b s", b=z_q.shape[0])  # 重新排列为 (batch, size)

        loss_dict["min_encoding_indices"] = min_encoding_indices  # 将最小编码索引添加到损失字典中

        return z_q, loss_dict  # 返回量化的 z_q 和损失字典

    def get_codebook_entry(self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:  # 定义方法获取代码本条目
        # shape specifying (batch, height, width, channel)  # shape 指定 (batch, height, width, channel)
        if self.remap is not None:  # 如果需要重映射
            assert shape is not None, "Need to give shape for remap"  # 断言必须提供形状以进行重映射
            indices = indices.reshape(shape[0], -1)  # 添加批次维度
            indices = self.unmap_to_all(indices)  # 对索引进行反向映射
            indices = indices.reshape(-1)  # 再次扁平化

        # get quantized latent vectors  # 获取量化的潜在向量
        z_q = self.embedding(indices)  # 根据索引获取嵌入

        if shape is not None:  # 如果提供了形状
            z_q = z_q.view(shape)  # 将 z_q 重塑为指定的形状
            # reshape back to match original input shape  # 重新调整形状以匹配原始输入形状
            z_q = z_q.permute(0, 3, 1, 2).contiguous()  # 调整维度顺序并确保内存连续

        return z_q  # 返回量化后的 z_q
# 定义一个名为 EmbeddingEMA 的神经网络模块，继承自 nn.Module
class EmbeddingEMA(nn.Module):
    # 初始化函数，接收令牌数量、码本维度、衰减因子和小常数作为参数
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        # 调用父类的初始化方法
        super().__init__()
        # 设置衰减因子
        self.decay = decay
        # 设置小常数以避免除零
        self.eps = eps
        # 生成一个随机的权重矩阵，形状为 (num_tokens, codebook_dim)
        weight = torch.randn(num_tokens, codebook_dim)
        # 将权重定义为不可训练的参数
        self.weight = nn.Parameter(weight, requires_grad=False)
        # 初始化集群大小为零，定义为不可训练的参数
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        # 复制权重并将其定义为不可训练的参数，用于存储嵌入平均值
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        # 设置更新标志为真
        self.update = True

    # 前向传播函数，接收嵌入 ID 并返回对应的嵌入向量
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    # 更新集群大小的指数移动平均
    def cluster_size_ema_update(self, new_cluster_size):
        # 按衰减因子更新当前集群大小
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    # 更新嵌入平均值的指数移动平均
    def embed_avg_ema_update(self, new_embed_avg):
        # 按衰减因子更新当前嵌入平均值
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    # 更新权重，基于平滑的集群大小
    def weight_update(self, num_tokens):
        # 计算集群大小的总和
        n = self.cluster_size.sum()
        # 计算平滑的集群大小，以避免除零错误
        smoothed_cluster_size = (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        # 用平滑的集群大小对嵌入平均值进行归一化
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # 用归一化的嵌入更新权重
        self.weight.data.copy_(embed_normalized)


# 定义一个名为 EMAVectorQuantizer 的抽象量化器类，继承自 AbstractQuantizer
class EMAVectorQuantizer(AbstractQuantizer):
    # 初始化函数，接收嵌入数量、嵌入维度、β、衰减因子、小常数和其他参数
    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置码本维度
        self.codebook_dim = embedding_dim
        # 设置令牌数量
        self.num_tokens = n_embed
        # 设置 β 值
        self.beta = beta
        # 设置损失键
        self.loss_key = loss_key

        # 初始化嵌入 EMA 模块
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        # 处理重映射参数
        self.remap = remap
        if self.remap is not None:
            # 如果提供了重映射路径，则加载并注册重映射后的索引
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            # 重新嵌入的数量为重映射后索引的形状
            self.re_embed = self.used.shape[0]
        else:
            # 如果没有重映射，则用原令牌数量初始化
            self.used = None
            self.re_embed = n_embed
        # 处理未知索引
        if unknown_index == "extra":
            # 如果未知索引为 "extra"，则更新重新嵌入数量
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            # 确保未知索引为有效类型
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            # 设置未知索引为提供的值
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        # 如果存在重映射，则记录重映射信息
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
    # 定义前向传播函数，接受一个张量 z，返回量化后的张量和字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 将 z 的形状调整为 (batch, height, width, channel) 并扁平化
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c h w -> b h w c")  # 调整 z 的维度顺序
        z_flattened = z.reshape(-1, self.codebook_dim)  # 将 z 扁平化为二维张量

        # 计算 z 与嵌入 e_j 的距离 (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)  # 计算 z 的平方和
            + self.embedding.weight.pow(2).sum(dim=1)  # 计算嵌入的平方和
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)  # 计算 z 和嵌入的点积
        )  # 'n d -> d n'

        # 找到每个 z 的最小距离对应的编码索引
        encoding_indices = torch.argmin(d, dim=1)

        # 根据编码索引获取量化后的 z，并调整形状以匹配原始 z
        z_q = self.embedding(encoding_indices).view(z.shape)  
        # 对编码进行独热编码，转换为与 z 相同的数据类型
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)  
        # 计算编码的平均概率
        avg_probs = torch.mean(encodings, dim=0)  
        # 计算困惑度
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))  

        # 如果处于训练状态且允许更新嵌入
        if self.training and self.embedding.update:
            # 更新 EMA 聚类大小
            encodings_sum = encodings.sum(0)  
            self.embedding.cluster_size_ema_update(encodings_sum)  # 更新聚类大小的 EMA
            # 更新 EMA 嵌入平均值
            embed_sum = encodings.transpose(0, 1) @ z_flattened  # 计算加权和
            self.embedding.embed_avg_ema_update(embed_sum)  # 更新嵌入平均值的 EMA
            # 规范化嵌入平均值并更新权重
            self.embedding.weight_update(self.num_tokens)  

        # 计算嵌入的损失
        loss = self.beta * F.mse_loss(z_q.detach(), z)  # 计算量化 z 与原 z 的均方误差损失

        # 保留梯度
        z_q = z + (z_q - z).detach()  # 使用 z 的值加上 z_q 的变化，但不计算梯度

        # 将 z_q 的形状调整回原始输入的形状
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b h w c -> b c h w")  # 恢复到原始的维度顺序

        # 创建一个字典以返回损失和其他信息
        out_dict = {
            self.loss_key: loss,  # 将损失放入字典
            "encodings": encodings,  # 包含独热编码
            "encoding_indices": encoding_indices,  # 包含编码索引
            "perplexity": perplexity,  # 包含困惑度
        }

        # 返回量化后的 z 和输出字典
        return z_q, out_dict  
# 定义一个带有输入投影的向量量化类，继承自 VectorQuantizer
class VectorQuantizerWithInputProjection(VectorQuantizer):
    # 初始化方法，接受输入维度、编码数量、码本维度等参数
    def __init__(
        self,
        input_dim: int,  # 输入数据的维度
        n_codes: int,  # 编码数量
        codebook_dim: int,  # 码本的维度
        beta: float = 1.0,  # 调整项的超参数，默认值为1.0
        output_dim: Optional[int] = None,  # 输出维度，可选
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法
        super().__init__(n_codes, codebook_dim, beta, **kwargs)
        # 创建输入投影层，将输入维度映射到码本维度
        self.proj_in = nn.Linear(input_dim, codebook_dim)
        # 设置输出维度属性
        self.output_dim = output_dim
        # 如果指定了输出维度，则创建输出投影层
        if output_dim is not None:
            self.proj_out = nn.Linear(codebook_dim, output_dim)
        else:
            # 如果没有指定输出维度，则使用恒等映射
            self.proj_out = nn.Identity()

    # 前向传播方法，接受输入张量并返回量化结果和损失字典
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        rearr = False  # 初始化重排列标志
        in_shape = z.shape  # 获取输入张量的形状

        # 如果输入张量的维度大于3，则进行重排列
        if z.ndim > 3:
            rearr = self.output_dim is not None  # 检查是否需要重排列
            # 将输入张量从 (batch, channels, ...) 转换为 (batch, ..., channels)
            z = rearrange(z, "b c ... -> b (...) c")
        # 将输入张量投影到码本维度
        z = self.proj_in(z)
        # 调用父类的前向方法进行量化，获得量化结果和损失字典
        z_q, loss_dict = super().forward(z)

        # 将量化结果通过输出投影层
        z_q = self.proj_out(z_q)
        # 如果需要重排列，根据输入形状调整输出张量
        if rearr:
            # 如果输入维度为4，重排列为 (batch, channels, height, width)
            if len(in_shape) == 4:
                z_q = rearrange(z_q, "b (h w) c -> b c h w ", w=in_shape[-1])
            # 如果输入维度为5，重排列为 (batch, channels, time, height, width)
            elif len(in_shape) == 5:
                z_q = rearrange(z_q, "b (t h w) c -> b c t h w ", w=in_shape[-1], h=in_shape[-2])
            else:
                # 如果输入维度不支持重排列，则抛出异常
                raise NotImplementedError(f"rearranging not available for {len(in_shape)}-dimensional input.")

        # 返回量化结果和损失字典
        return z_q, loss_dict
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\regularizers\__init__.py`

```py
# 导入抽象方法装饰器和类型注解
from abc import abstractmethod
# 导入任意类型和元组类型注解
from typing import Any, Tuple

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从自定义模块导入对角高斯分布
from ....modules.distributions.distributions import DiagonalGaussianDistribution
# 从基类模块导入抽象正则化器
from .base import AbstractRegularizer


# 定义对角高斯正则化器类，继承自抽象正则化器
class DiagonalGaussianRegularizer(AbstractRegularizer):
    # 初始化方法，接收一个布尔值参数，默认值为 True
    def __init__(self, sample: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 设置实例属性 sample
        self.sample = sample

    # 定义获取可训练参数的方法，返回任意类型
    def get_trainable_parameters(self) -> Any:
        # 生成一个空的可迭代对象
        yield from ()

    # 定义前向传播方法，接收一个张量，返回一个元组
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 创建一个空字典，用于存储日志信息
        log = dict()
        # 创建一个对角高斯分布实例，基于输入张量 z
        posterior = DiagonalGaussianDistribution(z)
        # 如果 sample 为 True，进行采样
        if self.sample:
            z = posterior.sample()
        # 否则，使用模式值
        else:
            z = posterior.mode()
        # 计算 KL 散度损失
        kl_loss = posterior.kl()
        # 对 KL 损失求和并平均
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # 将 KL 损失添加到日志字典中
        log["kl_loss"] = kl_loss
        # 返回处理后的张量和日志字典
        return z, log
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\temporal_ae.py`

```py
# 从 typing 模块导入 Callable, Iterable, Union 类型注解
from typing import Callable, Iterable, Union

# 导入 PyTorch 库
import torch
# 从 einops 导入 rearrange 和 repeat 函数，用于张量重排和重复
from einops import rearrange, repeat

# 从自定义模块中导入所需的类和变量
from sgm.modules.diffusionmodules.model import (
    # 检查 XFORMERS 库是否可用
    XFORMERS_IS_AVAILABLE,
    # 导入注意力块、解码器和其他模块
    AttnBlock,
    Decoder,
    MemoryEfficientAttnBlock,
    ResnetBlock,
)
# 从 openaimodel 模块导入 ResBlock 和时间步嵌入函数
from sgm.modules.diffusionmodules.openaimodel import ResBlock, timestep_embedding
# 从 video_attention 模块导入视频变换块
from sgm.modules.video_attention import VideoTransformerBlock
# 从 util 模块导入 partialclass 函数
from sgm.util import partialclass


# 定义一个新的类 VideoResBlock，继承自 ResnetBlock
class VideoResBlock(ResnetBlock):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        out_channels,  # 输出通道数
        *args,  # 额外参数
        dropout=0.0,  # dropout 概率
        video_kernel_size=3,  # 视频卷积核大小
        alpha=0.0,  # 混合因子
        merge_strategy="learned",  # 合并策略
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类构造函数进行初始化
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        # 如果未指定 video_kernel_size，则默认设置
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        # 创建时间堆栈，使用 ResBlock
        self.time_stack = ResBlock(
            channels=out_channels,  # 通道数
            emb_channels=0,  # 嵌入通道数
            dropout=dropout,  # dropout 概率
            dims=3,  # 数据维度
            use_scale_shift_norm=False,  # 是否使用缩放平移归一化
            use_conv=False,  # 是否使用卷积
            up=False,  # 是否向上采样
            down=False,  # 是否向下采样
            kernel_size=video_kernel_size,  # 卷积核大小
            use_checkpoint=False,  # 是否使用检查点
            skip_t_emb=True,  # 是否跳过时间嵌入
        )

        # 设置合并策略
        self.merge_strategy = merge_strategy
        # 根据合并策略注册混合因子
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))  # 固定混合因子
        elif self.merge_strategy == "learned":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))  # 学习混合因子
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")  # 抛出未知合并策略错误

    # 获取 alpha 值的函数
    def get_alpha(self, bs):
        # 根据合并策略返回混合因子
        if self.merge_strategy == "fixed":
            return self.mix_factor  # 固定策略
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)  # 学习策略
        else:
            raise NotImplementedError()  # 抛出未实现错误

    # 前向传播函数
    def forward(self, x, temb, skip_video=False, timesteps=None):
        # 如果未提供时间步，则使用类中的 timesteps
        if timesteps is None:
            timesteps = self.timesteps

        # 获取输入张量的形状
        b, c, h, w = x.shape

        # 调用父类的前向传播方法
        x = super().forward(x, temb)

        # 如果不跳过视频处理
        if not skip_video:
            # 重排张量，将其调整为视频格式
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            # 重排当前张量
            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            # 通过时间堆栈进行处理
            x = self.time_stack(x, temb)

            # 获取 alpha 值
            alpha = self.get_alpha(bs=b // timesteps)
            # 按比例混合两个张量
            x = alpha * x + (1.0 - alpha) * x_mix

            # 再次重排张量
            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x  # 返回处理后的张量


# 定义一个新的类 AE3DConv，继承自 torch.nn.Conv2d
class AE3DConv(torch.nn.Conv2d):
    # 初始化方法，设置输入和输出通道及卷积核大小等参数
        def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
            # 调用父类的初始化方法，传递输入和输出通道及其他参数
            super().__init__(in_channels, out_channels, *args, **kwargs)
            # 检查 video_kernel_size 是否为可迭代对象
            if isinstance(video_kernel_size, Iterable):
                # 如果是可迭代对象，计算每个核的填充大小
                padding = [int(k // 2) for k in video_kernel_size]
            else:
                # 否则，计算单个核的填充大小
                padding = int(video_kernel_size // 2)
    
            # 创建一个 3D 卷积层，用于处理视频数据
            self.time_mix_conv = torch.nn.Conv3d(
                in_channels=out_channels,  # 输入通道数
                out_channels=out_channels,  # 输出通道数
                kernel_size=video_kernel_size,  # 卷积核大小
                padding=padding,  # 填充大小
            )
    
        # 前向传播方法，处理输入数据并返回输出
        def forward(self, input, timesteps, skip_video=False):
            # 调用父类的前向传播方法，处理输入数据
            x = super().forward(input)
            # 如果跳过视频处理，直接返回处理后的数据
            if skip_video:
                return x
            # 调整张量形状，以适应 3D 卷积的输入要求
            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
            # 通过 3D 卷积层处理数据
            x = self.time_mix_conv(x)
            # 调整输出张量的形状，返回到原始格式
            return rearrange(x, "b c t h w -> (b t) c h w")
# 定义一个视频块类，继承自注意力块基类
class VideoBlock(AttnBlock):
    # 初始化函数，接收输入通道数、混合因子和合并策略
    def __init__(self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"):
        # 调用基类初始化函数
        super().__init__(in_channels)
        # 创建视频转换块，使用单头注意力机制
        self.time_mix_block = VideoTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            checkpoint=False,
            ff_in=True,
            attn_mode="softmax",
        )

        # 计算时间嵌入维度
        time_embed_dim = self.in_channels * 4
        # 构建视频时间嵌入的神经网络结构
        self.video_time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, self.in_channels),
        )

        # 设置合并策略
        self.merge_strategy = merge_strategy
        # 根据合并策略注册混合因子为缓冲区
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        # 注册混合因子为可学习参数
        elif self.merge_strategy == "learned":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        # 如果合并策略未知，抛出错误
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    # 前向传播函数，接收输入、时间步和跳过视频标志
    def forward(self, x, timesteps, skip_video=False):
        # 如果跳过视频，调用基类的前向传播
        if skip_video:
            return super().forward(x)

        # 保存输入数据
        x_in = x
        # 进行注意力计算
        x = self.attention(x)
        # 获取输出的高度和宽度
        h, w = x.shape[2:]
        # 重新排列数据形状
        x = rearrange(x, "b c h w -> b (h w) c")

        # 初始化混合输入
        x_mix = x
        # 创建时间步的序列
        num_frames = torch.arange(timesteps, device=x.device)
        # 重复时间步以匹配批大小
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        # 重新排列时间步
        num_frames = rearrange(num_frames, "b t -> (b t)")
        # 生成时间嵌入
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        # 计算时间嵌入的输出
        emb = self.video_time_embed(t_emb)  # b, n_channels
        # 增加一个维度
        emb = emb[:, None, :]
        # 将时间嵌入与输入混合
        x_mix = x_mix + emb

        # 获取当前的混合因子
        alpha = self.get_alpha()
        # 进行时间混合块计算
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        # 根据混合因子合并输入和混合输出
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        # 重新排列输出形状
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        # 投影到输出空间
        x = self.proj_out(x)

        # 返回输入与输出的和
        return x_in + x

    # 获取当前的混合因子
    def get_alpha(
        self,
    ):
        # 如果合并策略是固定，返回固定的混合因子
        if self.merge_strategy == "fixed":
            return self.mix_factor
        # 如果合并策略是学习的，返回经过 sigmoid 函数处理的混合因子
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        # 如果合并策略未知，抛出错误
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")


# 定义一个内存高效的视频块类，继承自内存高效注意力块
class MemoryEfficientVideoBlock(MemoryEfficientAttnBlock):
    # 初始化类，设置输入通道、混合因子和合并策略
        def __init__(self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"):
            # 调用父类构造函数，传递输入通道数
            super().__init__(in_channels)
            # 创建视频变换块，设置相关参数
            self.time_mix_block = VideoTransformerBlock(
                dim=in_channels,
                n_heads=1,
                d_head=in_channels,
                checkpoint=False,
                ff_in=True,
                attn_mode="softmax-xformers",
            )
    
            # 计算时间嵌入维度
            time_embed_dim = self.in_channels * 4
            # 定义时间嵌入序列，包含两层线性变换和SiLU激活
            self.video_time_embed = torch.nn.Sequential(
                torch.nn.Linear(self.in_channels, time_embed_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(time_embed_dim, self.in_channels),
            )
    
            # 保存合并策略
            self.merge_strategy = merge_strategy
            # 如果合并策略是固定，则注册混合因子为缓冲区
            if self.merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([alpha]))
            # 如果合并策略是学习的，则注册混合因子为可学习参数
            elif self.merge_strategy == "learned":
                self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
            # 否则抛出错误
            else:
                raise ValueError(f"unknown merge strategy {self.merge_strategy}")
    
        # 前向传播函数
        def forward(self, x, timesteps, skip_time_block=False):
            # 如果跳过时间块，调用父类的前向传播
            if skip_time_block:
                return super().forward(x)
    
            # 保存输入数据
            x_in = x
            # 应用注意力机制
            x = self.attention(x)
            # 获取输出的高度和宽度
            h, w = x.shape[2:]
            # 重排张量以便于处理
            x = rearrange(x, "b c h w -> b (h w) c")
    
            # 初始化混合输入
            x_mix = x
            # 创建时间帧的张量
            num_frames = torch.arange(timesteps, device=x.device)
            # 重复时间帧以匹配批次大小
            num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
            # 重排张量
            num_frames = rearrange(num_frames, "b t -> (b t)")
            # 获取时间嵌入
            t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
            # 应用视频时间嵌入
            emb = self.video_time_embed(t_emb)  # b, n_channels
            # 在第二维插入新的维度
            emb = emb[:, None, :]
            # 将嵌入加到混合输入
            x_mix = x_mix + emb
    
            # 获取混合因子
            alpha = self.get_alpha()
            # 应用时间混合块
            x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
            # 根据alpha进行混合
            x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge
    
            # 重新排列张量以恢复原始维度
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            # 应用输出投影
            x = self.proj_out(x)
    
            # 返回输入与输出的和
            return x_in + x
    
        # 获取混合因子的函数
        def get_alpha(
            self,
        ):
            # 如果合并策略是固定，则返回混合因子
            if self.merge_strategy == "fixed":
                return self.mix_factor
            # 如果合并策略是学习的，则返回经过sigmoid处理的混合因子
            elif self.merge_strategy == "learned":
                return torch.sigmoid(self.mix_factor)
            # 否则抛出未实现错误
            else:
                raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")
# 创建时空注意力机制的函数，接受多个参数以配置注意力类型和其他设置
def make_time_attn(
    in_channels,  # 输入通道数
    attn_type="vanilla",  # 注意力类型，默认为'vanilla'
    attn_kwargs=None,  # 额外的注意力参数，默认为None
    alpha: float = 0,  # 参数alpha，默认为0
    merge_strategy: str = "learned",  # 合并策略，默认为'learned'
):
    # 检查注意力类型是否在支持的选项中
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
    ], f"attn_type {attn_type} not supported for spatio-temporal attention"
    # 打印当前创建的注意力类型及其输入通道数
    print(f"making spatial and temporal attention of type '{attn_type}' with {in_channels} in_channels")
    # 如果不支持xformers，且当前注意力类型为'vanilla-xformers'，则回退到'vanilla'
    if not XFORMERS_IS_AVAILABLE and attn_type == "vanilla-xformers":
        print(
            f"Attention mode '{attn_type}' is not available. Falling back to vanilla attention. "
            f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
        )
        attn_type = "vanilla"

    # 如果注意力类型为'vanilla'，则返回部分类VideoBlock
    if attn_type == "vanilla":
        assert attn_kwargs is None  # 确保没有提供额外的参数
        return partialclass(VideoBlock, in_channels, alpha=alpha, merge_strategy=merge_strategy)
    # 如果注意力类型为'vanilla-xformers'，则返回部分类MemoryEfficientVideoBlock
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return partialclass(
            MemoryEfficientVideoBlock,
            in_channels,
            alpha=alpha,
            merge_strategy=merge_strategy,
        )
    else:
        return NotImplementedError()  # 如果不支持的类型，返回未实现错误


# 自定义的卷积层包装器，继承自torch.nn.Conv2d
class Conv2DWrapper(torch.nn.Conv2d):
    # 前向传播方法，调用父类的前向传播
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


# 视频解码器类，继承自Decoder
class VideoDecoder(Decoder):
    available_time_modes = ["all", "conv-only", "attn-only"]  # 可用的时间模式列表

    # 初始化方法，设置视频解码器的各种参数
    def __init__(
        self,
        *args,
        video_kernel_size: Union[int, list] = 3,  # 视频卷积核大小，默认为3
        alpha: float = 0.0,  # alpha参数，默认为0.0
        merge_strategy: str = "learned",  # 合并策略，默认为'learned'
        time_mode: str = "conv-only",  # 时间模式，默认为'conv-only'
        **kwargs,
    ):
        self.video_kernel_size = video_kernel_size  # 设置视频卷积核大小
        self.alpha = alpha  # 设置alpha参数
        self.merge_strategy = merge_strategy  # 设置合并策略
        self.time_mode = time_mode  # 设置时间模式
        # 确保时间模式在可用选项内
        assert (
            self.time_mode in self.available_time_modes
        ), f"time_mode parameter has to be in {self.available_time_modes}"
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法

    # 获取最后一层的权重，支持跳过时间混合的选项
    def get_last_layer(self, skip_time_mix=False, **kwargs):
        # 如果时间模式为'attn-only'，则抛出未实现错误
        if self.time_mode == "attn-only":
            raise NotImplementedError("TODO")
        else:
            # 返回适当的权重，基于跳过时间混合的选项
            return self.conv_out.time_mix_conv.weight if not skip_time_mix else self.conv_out.weight

    # 创建注意力机制的方法
    def _make_attn(self) -> Callable:
        # 根据时间模式返回适当的部分类
        if self.time_mode not in ["conv-only", "only-last-conv"]:
            return partialclass(
                make_time_attn,
                alpha=self.alpha,
                merge_strategy=self.merge_strategy,
            )
        else:
            return super()._make_attn()  # 否则调用父类的方法

    # 创建卷积层的方法
    def _make_conv(self) -> Callable:
        # 根据时间模式返回适当的部分类或卷积包装器
        if self.time_mode != "attn-only":
            return partialclass(AE3DConv, video_kernel_size=self.video_kernel_size)
        else:
            return Conv2DWrapper  # 返回卷积包装器
    # 定义一个私有方法，用于创建残差块，返回一个可调用对象
        def _make_resblock(self) -> Callable:
            # 检查当前的时间模式是否不在指定的两种模式中
            if self.time_mode not in ["attn-only", "only-last-conv"]:
                # 返回一个部分应用的类，用于创建 VideoResBlock 实例
                return partialclass(
                    VideoResBlock,
                    video_kernel_size=self.video_kernel_size,
                    alpha=self.alpha,
                    merge_strategy=self.merge_strategy,
                )
            else:
                # 否则，调用父类的方法以创建残差块
                return super()._make_resblock()
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_dec_3d.py`

```py
# pytorch_diffusion + derived encoder decoder
# 导入所需的库
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .movq_enc_3d import CausalConv3d, Upsample3D, DownSample3D


# 将输入转换为元组，确保其长度为指定值
def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


# 检查一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0


# 判断一个数是否为奇数
def is_odd(n):
    return not divisible_by(n, 2)


# 获取时间步的嵌入向量
def get_timestep_embedding(timesteps, embedding_dim):
    """
    此函数与 Denoising Diffusion Probabilistic Models 中的实现匹配：
    来自 Fairseq。
    构建正弦嵌入向量。
    此实现与 tensor2tensor 中的实现匹配，但与 "Attention Is All You Need" 中第 3.5 节的描述略有不同。
    """
    # 确保时间步的维度为 1
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2  # 计算嵌入维度的一半
    emb = math.log(10000) / (half_dim - 1)  # 计算对数因子
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  # 生成指数衰减的嵌入
    emb = emb.to(device=timesteps.device)  # 将嵌入移动到时间步的设备上
    emb = timesteps.float()[:, None] * emb[None, :]  # 扩展时间步并计算嵌入
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # 计算正弦和余弦值
    if embedding_dim % 2 == 1:  # 如果嵌入维度为奇数，则进行零填充
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb  # 返回嵌入


# 定义非线性激活函数，使用 Swish 激活函数
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


# 定义三维空间归一化模块
class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,  # 特征通道数
        zq_channels,  # 嵌入通道数
        norm_layer=nn.GroupNorm,  # 归一化层类型
        freeze_norm_layer=False,  # 是否冻结归一化层的参数
        add_conv=False,  # 是否添加卷积层
        pad_mode="constant",  # 填充模式
        **norm_layer_params,  # 归一化层的其他参数
    ):
        super().__init__()  # 调用父类构造函数
        # 初始化归一化层
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:  # 如果需要冻结归一化层
            for p in self.norm_layer.parameters:  # 遍历所有参数
                p.requires_grad = False  # 不更新参数
        self.add_conv = add_conv  # 保存是否添加卷积层的标志
        if self.add_conv:  # 如果添加卷积层
            # 创建三维因果卷积层
            self.conv = CausalConv3d(zq_channels, zq_channels, kernel_size=3, pad_mode=pad_mode)
        # 创建用于特征和嵌入的卷积层
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)

    # 前向传播函数
    def forward(self, f, zq):
        if zq.shape[2] > 1:  # 如果嵌入的时间步数大于1
            # 将特征拆分为第一时间步和剩余部分
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]  # 获取尺寸
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]  # 拆分嵌入
            # 使用最近邻插值调整 zq_first 的尺寸
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
            # 使用最近邻插值调整 zq_rest 的尺寸
            zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            zq = torch.cat([zq_first, zq_rest], dim=2)  # 合并嵌入
        else:  # 如果时间步数为1
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")  # 调整 zq 尺寸
        if self.add_conv:  # 如果添加卷积层
            zq = self.conv(zq)  # 通过卷积层处理 zq
        norm_f = self.norm_layer(f)  # 对特征进行归一化
        # 计算新的特征值
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f  # 返回新的特征值


# 定义 Normalize3D 类的起始部分（未完成）
def Normalize3D(in_channels, zq_ch, add_conv):
    # 返回一个三维空间归一化层的实例
        return SpatialNorm3D(
            # 输入通道数
            in_channels,
            # 量化通道数
            zq_ch,
            # 归一化层使用的类型，这里使用的是分组归一化
            norm_layer=nn.GroupNorm,
            # 是否冻结归一化层的参数，这里设置为不冻结
            freeze_norm_layer=False,
            # 是否添加卷积层，使用传入的参数
            add_conv=add_conv,
            # 归一化的组数
            num_groups=32,
            # 防止除零的极小值
            eps=1e-6,
            # 是否使用仿射变换，这里设置为使用
            affine=True,
        )
# 定义一个三维残差块类，继承自 nn.Module
class ResnetBlock3D(nn.Module):
    # 初始化方法，接收多种参数以配置该块
    def __init__(
        self,
        *,
        in_channels,  # 输入通道数
        out_channels=None,  # 输出通道数（可选）
        conv_shortcut=False,  # 是否使用卷积快捷连接
        dropout,  # dropout 比例
        temb_channels=512,  # 时间嵌入通道数
        zq_ch=None,  # zq 相关通道数（可选）
        add_conv=False,  # 是否添加卷积层
        pad_mode="constant",  # 填充模式
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 确定输出通道数，若未指定则等于输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存是否使用卷积快捷连接的标志
        self.use_conv_shortcut = conv_shortcut

        # 初始化第一个归一化层
        self.norm1 = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # 初始化第一个因果卷积层
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 若时间嵌入通道数大于0，初始化时间嵌入投影层
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 初始化第二个归一化层
        self.norm2 = Normalize3D(out_channels, zq_ch, add_conv=add_conv)
        # 初始化 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # 初始化第二个因果卷积层
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 如果输入和输出通道数不相等
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷连接，则初始化对应的卷积层
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            # 否则，初始化 1x1 卷积层作为快捷连接
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x, temb, zq):
        # 将输入赋值给 h
        h = x
        # 对 h 进行归一化
        h = self.norm1(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过第一个卷积层
        h = self.conv1(h)

        # 如果时间嵌入不为 None，则将其投影到 h 上
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        # 对 h 进行第二次归一化
        h = self.norm2(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 应用 dropout
        h = self.dropout(h)
        # 通过第二个卷积层
        h = self.conv2(h)

        # 如果输入和输出通道数不相等
        if self.in_channels != self.out_channels:
            # 如果使用卷积快捷连接
            if self.use_conv_shortcut:
                # 将输入 x 通过卷积快捷连接
                x = self.conv_shortcut(x)
            # 否则使用 1x1 卷积
            else:
                x = self.nin_shortcut(x)

        # 返回输入和 h 的和
        return x + h


# 定义一个二维注意力块类，继承自 nn.Module
class AttnBlock2D(nn.Module):
    # 初始化方法，接收输入通道数、zq 通道数和是否添加卷积的标志
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化归一化层
        self.norm = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # 初始化查询、键、值卷积层，均为 1x1 卷积
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 初始化输出卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    # 前向传播函数，接受输入 x 和查询 zq
    def forward(self, x, zq):
        # 将输入 x 赋值给 h_
        h_ = x
        # 对 h_ 进行归一化处理，使用查询 zq
        h_ = self.norm(h_, zq)
    
        # 获取 h_ 的时间步长 t
        t = h_.shape[2]
        # 重排 h_ 的维度，将时间步和批次维度合并
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")
    
        # 计算查询、键和值
        q = self.q(h_)  # 计算查询
        k = self.k(h_)  # 计算键
        v = self.v(h_)  # 计算值
    
        # 计算注意力
        b, c, h, w = q.shape  # 解包 q 的形状信息
        q = q.reshape(b, c, h * w)  # 将 q 重塑为 (b, c, hw)
        q = q.permute(0, 2, 1)  # 变换维度顺序为 (b, hw, c)
        k = k.reshape(b, c, h * w)  # 将 k 重塑为 (b, c, hw)
        # 计算 q 和 k 的批量矩阵乘法，得到注意力权重 w_
        w_ = torch.bmm(q, k)  # 计算注意力权重
        w_ = w_ * (int(c) ** (-0.5))  # 对权重进行缩放
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 对最后一维进行 softmax
    
        # 根据注意力权重对值进行加权
        v = v.reshape(b, c, h * w)  # 将 v 重塑为 (b, c, hw)
        w_ = w_.permute(0, 2, 1)  # 变换维度顺序为 (b, hw, hw)
        # 计算加权和，得到输出特征 h_
        h_ = torch.bmm(v, w_)  # 计算输出
        h_ = h_.reshape(b, c, h, w)  # 将 h_ 重塑回 (b, c, h, w)
    
        # 对 h_ 进行投影
        h_ = self.proj_out(h_)
    
        # 将 h_ 的维度重排回原来的形状
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)
    
        # 返回输入 x 和输出 h_ 的和
        return x + h_
# 定义一个名为 MOVQDecoder3D 的神经网络模块类，继承自 nn.Module
class MOVQDecoder3D(nn.Module):
    # 初始化方法，接受多个参数来配置解码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力机制的分辨率
        dropout=0.0,  # dropout 概率，用于防止过拟合
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入的通道数
        resolution,  # 输入图像的分辨率
        z_channels,  # 噪声通道数
        give_pre_end=False,  # 是否给出预处理结束标志
        zq_ch=None,  # 量化通道数
        add_conv=False,  # 是否添加卷积层
        pad_mode="first",  # 填充模式，默认为 'first'
        temporal_compress_times=4,  # 时间压缩的倍数
        **ignorekwargs,  # 其他未指定的关键字参数
    ):
        # 调用父类构造函数
        super().__init__()
        # 初始化通道数
        self.ch = ch
        # 初始化时间嵌入通道数为0
        self.temb_ch = 0
        # 获取分辨率数量
        self.num_resolutions = len(ch_mult)
        # 获取残差块数量
        self.num_res_blocks = num_res_blocks
        # 设置分辨率
        self.resolution = resolution
        # 设置输入通道数
        self.in_channels = in_channels
        # 设置是否在前向传播后给出结束标志
        self.give_pre_end = give_pre_end

        # 计算 temporal_compress_times 的 log2 值
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # 如果 zq_ch 为 None，则使用 z_channels
        if zq_ch is None:
            zq_ch = z_channels

        # 计算当前块的输入通道数
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # 计算当前分辨率
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # 设置 z 的形状
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # 创建输入卷积层
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, pad_mode=pad_mode)

        # 创建中间模块
        self.mid = nn.Module()
        # 添加第一个残差块
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )

        # 添加第二个残差块
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )

        # 创建上采样模块
        self.up = nn.ModuleList()
        # 从最高分辨率开始遍历
        for i_level in reversed(range(self.num_resolutions)):
            # 创建块和注意力模块的容器
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 计算当前块的输出通道数
            block_out = ch * ch_mult[i_level]
            # 添加指定数量的残差块
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        pad_mode=pad_mode,
                    )
                )
                # 更新输入通道数
                block_in = block_out
                # 如果当前分辨率在注意力分辨率列表中，添加注意力块
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in, zq_ch, add_conv=add_conv))
            # 创建上采样模块
            up = nn.Module()
            up.block = block
            up.attn = attn
            # 如果不是最底层，进行上采样配置
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=False)
                else:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=True)
                # 更新当前分辨率为原来的两倍
                curr_res = curr_res * 2
            # 将上采样模块插入到列表开头以保持顺序
            self.up.insert(0, up)  # prepend to get consistent order

        # 创建输出归一化层
        self.norm_out = Normalize3D(block_in, zq_ch, add_conv=add_conv)
        # 创建输出卷积层
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, pad_mode=pad_mode)
    # 定义前向传播方法，接受输入 z 和可选参数 use_cp
    def forward(self, z, use_cp=False):
        # 保存输入 z 的形状以便后续使用
        self.last_z_shape = z.shape
    
        # 定义时间步嵌入变量，初始为 None
        temb = None
    
        # 获取 z 的时间步数（即第三维的大小）
        t = z.shape[2]
        # 将 z 赋值给 zq，用于后续计算
    
        zq = z
        # 通过输入卷积层处理 z
        h = self.conv_in(z)
    
        # 中间处理阶段
        # 使用第一个中间块处理 h，传入 temb 和 zq
        h = self.mid.block_1(h, temb, zq)
        # h = self.mid.attn_1(h, zq)  # 注释掉的注意力层
        # 使用第二个中间块处理 h，传入 temb 和 zq
        h = self.mid.block_2(h, temb, zq)
    
        # 上采样阶段
        # 反向遍历每个分辨率级别
        for i_level in reversed(range(self.num_resolutions)):
            # 遍历每个残差块
            for i_block in range(self.num_res_blocks + 1):
                # 在当前上采样级别中处理 h，传入 temb 和 zq
                h = self.up[i_level].block[i_block](h, temb, zq)
                # 如果当前级别有注意力层，则对 h 进行注意力处理
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果当前不是最后一个级别，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)
    
        # 结束阶段
        # 如果给定了预结束标志，则直接返回 h
        if self.give_pre_end:
            return h
    
        # 对 h 进行归一化处理，传入 zq
        h = self.norm_out(h, zq)
        # 对 h 应用非线性激活函数
        h = nonlinearity(h)
        # 通过输出卷积层处理 h
        h = self.conv_out(h)
        # 返回最终的 h
        return h
    
    # 获取最后一层的卷积权重
    def get_last_layer(self):
        return self.conv_out.conv.weight
# 定义一个新的3D解码器类，继承自 nn.Module
class NewDecoder3D(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道数的倍增因子
        num_res_blocks,  # 残差块的数量
        attn_resolutions,  # 注意力分辨率
        dropout=0.0,  # dropout比率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入通道数
        resolution,  # 输入的分辨率
        z_channels,  # 噪声通道数
        give_pre_end=False,  # 是否返回预结束的输出
        zq_ch=None,  # 可选的量化通道数
        add_conv=False,  # 是否添加额外的卷积层
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩次数
        post_quant_conv=False,  # 是否使用量化后的卷积
        **ignorekwargs,  # 其他忽略的参数
    ):
    def forward(self, z):
        # 断言输入的形状与预期的 z_shape 一致（已注释）
        # self.last_z_shape = z.shape  # 保存最后的输入形状
        self.last_z_shape = z.shape

        # 时间步嵌入初始化为 None
        temb = None

        # 获取输入 z 的时间步长
        t = z.shape[2]
        # 将 z 赋值给 zq 以备后续使用
        zq = z
        # 如果存在后量化卷积，则对 z 进行处理
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        # 对 z 进行初始卷积
        h = self.conv_in(z)

        # 中间处理阶段
        h = self.mid.block_1(h, temb, zq)  # 通过第一个中间块处理 h
        # h = self.mid.attn_1(h, zq)  # (注释掉)可能的注意力机制处理
        h = self.mid.block_2(h, temb, zq)  # 通过第二个中间块处理 h

        # 上采样阶段
        for i_level in reversed(range(self.num_resolutions)):  # 反向遍历每个分辨率级别
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, zq)  # 处理 h
                # 如果有注意力模块，则应用注意力
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果当前级别不是0，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 结束处理阶段
        if self.give_pre_end:  # 如果需要预结束输出，则返回 h
            return h

        # 通过归一化处理输出
        h = self.norm_out(h, zq)
        h = nonlinearity(h)  # 应用非线性激活函数
        h = self.conv_out(h)  # 通过最终卷积层处理 h
        return h  # 返回最终的输出

    # 获取最后一层的权重
    def get_last_layer(self):
        return self.conv_out.conv.weight  # 返回卷积层的权重
```

# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\vqvae\movq_dec_3d_dev.py`

```py
# pytorch_diffusion + derived encoder decoder
import math  # 导入数学库，提供数学函数
import torch  # 导入 PyTorch 库，进行张量计算
import torch.nn as nn  # 导入 nn 模块，构建神经网络
import torch.nn.functional as F  # 导入功能性模块，提供常用操作
import numpy as np  # 导入 NumPy 库，进行数值计算

from beartype import beartype  # 从 beartype 导入 beartype，用于类型检查
from beartype.typing import Union, Tuple, Optional, List  # 导入类型提示
from einops import rearrange  # 从 einops 导入 rearrange，用于重排张量维度

from .movq_enc_3d import CausalConv3d, Upsample3D, DownSample3D  # 从本地模块导入 3D 卷积和上采样、下采样类


def cast_tuple(t, length=1):
    # 如果 t 不是元组，则将其转换为指定长度的元组
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    # 检查 num 是否可以被 den 整除
    return (num % den) == 0


def is_odd(n):
    # 检查 n 是否为奇数
    return not divisible_by(n, 2)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    这个函数构建正弦嵌入，与 Denoising Diffusion Probabilistic Models 的实现匹配：
    来源于 Fairseq。
    构建正弦嵌入。
    与 tensor2tensor 的实现匹配，但与 "Attention Is All You Need" 第 3.5 节中的描述略有不同。
    """
    # 确保 timesteps 是一维的
    assert len(timesteps.shape) == 1

    # 计算嵌入维度的一半
    half_dim = embedding_dim // 2
    # 计算嵌入的基础
    emb = math.log(10000) / (half_dim - 1)
    # 计算正弦嵌入的值
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # 将嵌入移动到 timesteps 的设备上
    emb = emb.to(device=timesteps.device)
    # 计算每个时间步的嵌入
    emb = timesteps.float()[:, None] * emb[None, :]
    # 将正弦和余弦值连接起来
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    # 如果嵌入维度是奇数，进行零填充
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    # 返回最终的嵌入
    return emb


def nonlinearity(x):
    # 使用 Swish 激活函数
    return x * torch.sigmoid(x)


class SpatialNorm3D(nn.Module):
    # 定义一个 3D 空间归一化的类
    def __init__(
        self,
        f_channels,
        zq_channels,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        **norm_layer_params,
    ):
        # 初始化函数，设置参数
        super().__init__()  # 调用父类的初始化函数
        # 创建归一化层
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        # 如果需要冻结归一化层的参数
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:  # 遍历归一化层的参数
                p.requires_grad = False  # 冻结参数不进行更新
        self.add_conv = add_conv  # 是否添加卷积层
        # 如果添加卷积层，创建 causal 卷积层
        if self.add_conv:
            # self.conv = nn.Conv3d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
            self.conv = CausalConv3d(zq_channels, zq_channels, kernel_size=3, pad_mode=pad_mode)
        # 创建一个 1x1 卷积层用于 y 和 b 通道
        # self.conv_y = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        # self.conv_b = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
    # 定义前向传播方法，接收输入 f 和 zq
    def forward(self, f, zq):
        # 如果 zq 的第三维大于 1，表示有多个通道
        if zq.shape[2] > 1:
            # 分割 f 为第一个通道和其余通道
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            # 获取第一个通道和其余通道的尺寸
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            # 分割 zq 为第一个通道和其余通道
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            # 对第一个通道进行最近邻插值调整尺寸
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
            # 对其余通道进行最近邻插值调整尺寸
            zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            # 将调整后的通道合并在一起
            zq = torch.cat([zq_first, zq_rest], dim=2)
        # 如果 zq 只有一个通道，直接调整其尺寸
        else:
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")
        # 如果需要添加卷积层
        if self.add_conv:
            # 对 zq 进行卷积操作
            zq = self.conv(zq)
        # 对 f 应用归一化层
        norm_f = self.norm_layer(f)
        # 计算新的 f，结合卷积后的 zq
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        # 返回新的 f
        return new_f
# 定义一个 3D 归一化函数，接收输入通道、量化通道及是否添加卷积的标志
def Normalize3D(in_channels, zq_ch, add_conv):
    # 调用空间归一化 3D，传入相应参数
    return SpatialNorm3D(
        in_channels,
        zq_ch,
        norm_layer=nn.GroupNorm,  # 使用分组归一化层
        freeze_norm_layer=False,   # 不冻结归一化层
        add_conv=add_conv,         # 是否添加卷积
        num_groups=32,             # 设置组的数量
        eps=1e-6,                  # 设置小常数以防止除零
        affine=True,               # 使用仿射变换
    )


# 定义 3D ResNet 块，继承自 nn.Module
class ResnetBlock3D(nn.Module):
    # 初始化方法，接收多个参数配置
    def __init__(
        self,
        *,
        in_channels,               # 输入通道数
        out_channels=None,         # 输出通道数，可选
        conv_shortcut=False,       # 是否使用卷积短接
        dropout,                   # dropout 比率
        temb_channels=512,         # 时间嵌入通道数
        zq_ch=None,                # 量化通道
        add_conv=False,            # 是否添加卷积
        pad_mode="constant",       # 填充模式
    ):
        super().__init__()  # 调用父类构造函数
        self.in_channels = in_channels  # 设置输入通道数
        out_channels = in_channels if out_channels is None else out_channels  # 设置输出通道数
        self.out_channels = out_channels  # 保存输出通道数
        self.use_conv_shortcut = conv_shortcut  # 保存是否使用卷积短接的标志

        # 创建第一个归一化层
        self.norm1 = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # self.conv1 = torch.nn.Conv3d(in_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        # 使用因果卷积创建第一个卷积层
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if temb_channels > 0:  # 如果时间嵌入通道数大于零
            # 创建时间嵌入投影层
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        # 创建第二个归一化层
        self.norm2 = Normalize3D(out_channels, zq_ch, add_conv=add_conv)
        # 创建 dropout 层
        self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = torch.nn.Conv3d(out_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        # 使用因果卷积创建第二个卷积层
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        # 如果输入和输出通道数不一致
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:  # 如果使用卷积短接
                # self.conv_shortcut = torch.nn.Conv3d(in_channels,
                #                                      out_channels,
                #                                      kernel_size=3,
                #                                      stride=1,
                #                                      padding=1)
                # 使用因果卷积创建短接卷积层
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                # 创建一个 1x1 的卷积层作为短接
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                # self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, pad_mode=pad_mode)
    # 定义前向传播函数，接收输入张量 x、时间嵌入 temb 和 zq
        def forward(self, x, temb, zq):
            # 将输入赋值给 h
            h = x
            # 对 h 应用第一层归一化，使用 zq 作为参数
            h = self.norm1(h, zq)
            # 对 h 应用非线性激活函数
            h = nonlinearity(h)
            # 对 h 应用第一层卷积
            h = self.conv1(h)
    
            # 如果时间嵌入 temb 不为空
            if temb is not None:
                # 将时间嵌入经过非线性处理并进行投影后加到 h
                h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
    
            # 对 h 应用第二层归一化，使用 zq 作为参数
            h = self.norm2(h, zq)
            # 对 h 应用非线性激活函数
            h = nonlinearity(h)
            # 对 h 应用 dropout 操作
            h = self.dropout(h)
            # 对 h 应用第二层卷积
            h = self.conv2(h)
    
            # 如果输入通道数与输出通道数不相等
            if self.in_channels != self.out_channels:
                # 如果使用卷积短路
                if self.use_conv_shortcut:
                    # 对输入 x 应用卷积短路
                    x = self.conv_shortcut(x)
                else:
                    # 对输入 x 应用 NIN 短路
                    x = self.nin_shortcut(x)
    
            # 返回输入 x 与 h 的和
            return x + h
# 定义一个二维注意力块类，继承自 nn.Module
class AttnBlock2D(nn.Module):
    # 初始化方法，设置输入通道数、zq 通道数以及是否添加卷积
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 初始化 3D 归一化层
        self.norm = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # 定义查询（Q）卷积层，kernel_size=1
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义键（K）卷积层，kernel_size=1
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义值（V）卷积层，kernel_size=1
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 定义输出投影卷积层，kernel_size=1
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x, zq):
        # 保存输入数据
        h_ = x
        # 对输入数据进行归一化
        h_ = self.norm(h_, zq)

        # 获取时间步长
        t = h_.shape[2]
        # 重新排列张量维度，将时间步和批次合并
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")

        # 计算查询（Q）、键（K）和值（V）
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q.shape  # 获取批次大小、通道数、高度和宽度
        q = q.reshape(b, c, h * w)  # 重新排列查询张量维度
        q = q.permute(0, 2, 1)  # 交换维度顺序，变为 b, hw, c
        k = k.reshape(b, c, h * w)  # 重新排列键张量维度
        # 计算注意力权重矩阵，使用批量矩阵乘法
        w_ = torch.bmm(q, k)  # b, hw, hw，计算 q 和 k 的点积
        w_ = w_ * (int(c) ** (-0.5))  # 对权重进行缩放
        w_ = torch.nn.functional.softmax(w_, dim=2)  # 对权重进行 softmax 归一化

        # 注意力机制应用于值（V）
        v = v.reshape(b, c, h * w)  # 重新排列值张量维度
        w_ = w_.permute(0, 2, 1)  # 交换权重维度顺序
        # 计算加权值，得到注意力输出
        h_ = torch.bmm(v, w_)  # b, c, hw，计算 v 和权重的点积
        h_ = h_.reshape(b, c, h, w)  # 重新排列输出张量维度

        # 通过输出投影层进行变换
        h_ = self.proj_out(h_)

        # 恢复张量到原来的维度结构
        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)

        # 返回输入和输出的和
        return x + h_


# 定义一个三维 MOVQ 解码器类，继承自 nn.Module
class MOVQDecoder3D(nn.Module):
    # 初始化方法，设置多个参数，包括通道数、分辨率等
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        pad_mode="first",
        temporal_compress_times=4,
        **ignorekwargs,
    # 定义前向传播函数，接受输入 z 和一个可选的 use_cp 参数
        def forward(self, z, use_cp=False):
            # 断言输入 z 的形状与预期的形状一致（此行被注释掉）
            # assert z.shape[1:] == self.z_shape[1:]
            # 保存输入 z 的形状，以备后用
            self.last_z_shape = z.shape
    
            # 初始化时间步嵌入变量
            temb = None
    
            # 获取 z 的时间步长度
            t = z.shape[2]
            # 将 z 赋值给 zq，作为输入块
    
            zq = z
            # 对输入 z 进行卷积操作，得到初步特征 h
            h = self.conv_in(z)
    
            # 中间层处理
            h = self.mid.block_1(h, temb, zq)  # 通过第一个中间块处理特征
            # h = self.mid.attn_1(h, zq)  # 注释掉的注意力机制处理
            h = self.mid.block_2(h, temb, zq)  # 通过第二个中间块处理特征
    
            # 上采样过程
            for i_level in reversed(range(self.num_resolutions)):  # 反向遍历分辨率层级
                for i_block in range(self.num_res_blocks + 1):  # 遍历每个块
                    h = self.up[i_level].block[i_block](h, temb, zq)  # 对当前特征进行块处理
                    if len(self.up[i_level].attn) > 0:  # 如果当前层有注意力机制
                        h = self.up[i_level].attn[i_block](h, zq)  # 进行注意力机制处理
                if i_level != 0:  # 如果不是最后一层
                    h = self.up[i_level].upsample(h)  # 进行上采样
    
            # 结束处理
            if self.give_pre_end:  # 如果需要返回中间结果
                return h
    
            h = self.norm_out(h, zq)  # 对 h 进行规范化处理
            h = nonlinearity(h)  # 应用非线性激活函数
            h = self.conv_out(h)  # 最后卷积操作，输出结果
            return h  # 返回最终结果
    
        # 获取最后一层的权重
        def get_last_layer(self):
            return self.conv_out.conv.weight  # 返回卷积层的权重
# 定义一个新的 3D 解码器类，继承自 nn.Module
class NewDecoder3D(nn.Module):
    # 初始化方法，接收多个参数以配置解码器
    def __init__(
        self,
        *,
        ch,  # 输入通道数
        out_ch,  # 输出通道数
        ch_mult=(1, 2, 4, 8),  # 通道倍增因子
        num_res_blocks,  # 残差块数量
        attn_resolutions,  # 注意力分辨率
        dropout=0.0,  # dropout 比率
        resamp_with_conv=True,  # 是否使用卷积进行重采样
        in_channels,  # 输入通道
        resolution,  # 输入分辨率
        z_channels,  # 噪声通道数
        give_pre_end=False,  # 是否给出预处理结束
        zq_ch=None,  # 可选的量化通道数
        add_conv=False,  # 是否添加额外卷积层
        pad_mode="first",  # 填充模式
        temporal_compress_times=4,  # 时间压缩倍数
        post_quant_conv=False,  # 是否使用后量化卷积
        **ignorekwargs,  # 其他忽略的关键字参数
    ):
        # 初始化父类 nn.Module
        super(NewDecoder3D, self).__init__()

    # 定义前向传播方法，接收输入 z
    def forward(self, z):
        # 断言检查 z 的形状是否与 z_shape 匹配
        # assert z.shape[1:] == self.z_shape[1:]
        # 记录 z 的最后形状
        self.last_z_shape = z.shape

        # 定义时间步嵌入，初始为 None
        temb = None

        # 获取 z 的时间步长
        t = z.shape[2]
        # z 赋值给 zq，作为量化输入

        zq = z
        # 如果定义了后量化卷积，则对 z 进行处理
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        # 对输入 z 进行初始卷积处理
        h = self.conv_in(z)

        # 中间层处理
        h = self.mid.block_1(h, temb, zq)  # 通过第一个中间块
        # h = self.mid.attn_1(h, zq)  # 可选的注意力机制
        h = self.mid.block_2(h, temb, zq)  # 通过第二个中间块

        # 上采样处理
        for i_level in reversed(range(self.num_resolutions)):  # 从高到低分辨率处理
            for i_block in range(self.num_res_blocks + 1):  # 遍历每个残差块
                h = self.up[i_level].block[i_block](h, temb, zq)  # 通过当前上采样块
                # 如果存在注意力模块，则应用
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            # 如果不是最后一层，则进行上采样
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 结束处理
        if self.give_pre_end:  # 如果需要预处理结束，则返回 h
            return h

        # 对输出进行归一化处理
        h = self.norm_out(h, zq)
        # 应用非线性激活函数
        h = nonlinearity(h)
        # 通过最终卷积层得到输出
        h = self.conv_out(h)
        # 返回最终输出
        return h

    # 定义获取最后一层权重的方法
    def get_last_layer(self):
        # 返回最后卷积层的权重
        return self.conv_out.conv.weight
```