# `.\flux\src\flux\modules\layers.py`

```py
# 导入数学库
import math
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 PyTorch 库
import torch
# 从 einops 库导入 rearrange 函数
from einops import rearrange
# 从 torch 库导入 Tensor 和 nn 模块
from torch import Tensor, nn

# 从 flux.math 模块导入 attention 和 rope 函数
from flux.math import attention, rope


# 定义一个嵌入类，用于处理 N 维数据
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        # 初始化维度、角度和轴维度
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        # 获取输入 Tensor 的最后一维大小
        n_axes = ids.shape[-1]
        # 对每个轴应用 rope 函数并在-3维上连接
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        # 在第1维上增加一个维度
        return emb.unsqueeze(1)


# 定义时间步嵌入函数，创建正弦时间步嵌入
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    创建正弦时间步嵌入。
    :param t: 一维 Tensor，包含每批次元素的索引，可以是小数。
    :param dim: 输出的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个 (N, D) 维的 Tensor，表示位置嵌入。
    """
    # 根据时间因子缩放输入 Tensor
    t = time_factor * t
    # 计算半维度
    half = dim // 2
    # 计算频率
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )
    # 计算嵌入
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # 如果维度是奇数，追加零向量
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    # 如果 t 是浮点类型，将嵌入转换为 t 的类型
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


# 定义一个 MLP 嵌入器类
class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        # 初始化输入层、激活函数和输出层
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # 执行前向传递，经过输入层、激活函数和输出层
        return self.out_layer(self.silu(self.in_layer(x)))


# 定义 RMSNorm 类
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 初始化尺度参数
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        # 将输入转换为浮点数
        x_dtype = x.dtype
        x = x.float()
        # 计算均方根归一化
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        # 应用归一化和尺度参数
        return (x * rrms).to(dtype=x_dtype) * self.scale


# 定义 QKNorm 类
class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 初始化查询和键的归一化
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # 对查询和键进行归一化
        q = self.query_norm(q)
        k = self.key_norm(k)
        # 返回归一化后的查询、键以及原始值
        return q.to(v), k.to(v)


# 定义自注意力机制类
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        # 设置头的数量和每个头的维度
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 初始化查询、键、值线性变换层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 初始化归一化层
        self.norm = QKNorm(head_dim)
        # 初始化投影层
        self.proj = nn.Linear(dim, dim)
    # 前向传播函数，接受输入张量和位置编码，返回处理后的张量
    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        # 将输入张量通过 qkv 层，生成查询、键、值的联合表示
        qkv = self.qkv(x)
        # 重新排列 qkv 张量，将其拆分成查询 (q)、键 (k)、值 (v)，并根据头数 (num_heads) 分组
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        # 对查询、键和值进行归一化处理
        q, k = self.norm(q, k, v)
        # 计算注意力权重并应用于值，得到加权后的输出
        x = attention(q, k, v, pe=pe)
        # 通过 proj 层将注意力结果映射到输出空间
        x = self.proj(x)
        # 返回最终的输出张量
        return x
# 定义一个包含三个张量的结构体 ModulationOut
@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


# 定义一个继承自 nn.Module 的 Modulation 类
class Modulation(nn.Module):
    # 初始化方法，设置维度和是否双倍
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double  # 存储是否为双倍标志
        self.multiplier = 6 if double else 3  # 根据标志设置 multiplier
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)  # 定义线性层

    # 前向传播方法，处理输入张量并返回结果
    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        # 应用激活函数后，进行线性变换，并将结果按 multiplier 切分
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        # 返回切分后的结果，前半部分和后半部分（如果是双倍）
        return (
            ModulationOut(*out[:3]),  # 前三部分
            ModulationOut(*out[3:]) if self.is_double else None,  # 后三部分（如果是双倍）
        )


# 定义一个继承自 nn.Module 的 DoubleStreamBlock 类
class DoubleStreamBlock(nn.Module):
    # 初始化方法，设置隐藏层大小、注意力头数、MLP 比例等
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)  # 计算 MLP 隐藏层维度
        self.num_heads = num_heads  # 存储注意力头数
        self.hidden_size = hidden_size  # 存储隐藏层大小
        self.img_mod = Modulation(hidden_size, double=True)  # 定义图像模调模块
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 定义图像的第一层归一化
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)  # 定义图像的自注意力模块

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 定义图像的第二层归一化
        self.img_mlp = nn.Sequential(  # 定义图像的 MLP 网络
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),  # 第一层线性变换
            nn.GELU(approximate="tanh"),  # 激活函数
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),  # 第二层线性变换
        )

        self.txt_mod = Modulation(hidden_size, double=True)  # 定义文本模调模块
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 定义文本的第一层归一化
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)  # 定义文本的自注意力模块

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # 定义文本的第二层归一化
        self.txt_mlp = nn.Sequential(  # 定义文本的 MLP 网络
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),  # 第一层线性变换
            nn.GELU(approximate="tanh"),  # 激活函数
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),  # 第二层线性变换
        )
    # 前向传播函数，处理图像和文本输入，返回更新后的图像和文本
    def forward(self, img: Tensor
# 定义一个 DiT 模块，其中包含并行的线性层以及调整的调制接口
class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        # 初始化隐藏层维度和注意力头的数量
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        # 计算缩放因子
        self.scale = qk_scale or head_dim**-0.5

        # 计算 MLP 层的隐藏维度
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # 定义用于 QKV 和 MLP 输入的线性层
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # 定义用于投影和 MLP 输出的线性层
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        # 定义归一化层
        self.norm = QKNorm(head_dim)

        # 定义层归一化层
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # 定义激活函数和调制层
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        # 通过调制层计算调制因子
        mod, _ = self.modulation(vec)
        # 对输入进行预归一化并应用调制
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # 将线性层的输出分割为 QKV 和 MLP 输入
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        # 重新排列 QKV 张量，并进行归一化
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # 计算注意力
        attn = attention(q, k, v, pe=pe)
        # 计算 MLP 流中的激活，拼接结果并通过第二个线性层
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # 将原始输入与输出加权和相加
        return x + mod.gate * output


# 定义最后一层的网络模块
class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        # 定义最终的层归一化
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 定义线性层将隐藏维度映射到最终输出通道
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # 定义自适应层归一化调制
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        # 通过调制层计算 shift 和 scale
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        # 归一化输入并应用 shift 和 scale
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        # 通过线性层计算最终输出
        x = self.linear(x)
        return x
```