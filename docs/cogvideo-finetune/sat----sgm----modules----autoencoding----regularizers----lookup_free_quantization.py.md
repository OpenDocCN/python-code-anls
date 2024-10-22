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