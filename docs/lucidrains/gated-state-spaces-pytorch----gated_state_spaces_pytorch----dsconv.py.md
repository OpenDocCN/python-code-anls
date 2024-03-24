# `.\lucidrains\gated-state-spaces-pytorch\gated_state_spaces_pytorch\dsconv.py`

```py
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft
from einops import rearrange
from scipy.fftpack import next_fast_len

# functions

# 检查值是否存在
def exists(val):
    return val is not None

# 在张量中添加指定数量的维度
def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

# 使用傅立叶技巧进行 O(N log(N)) 的一维卷积
def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = torch.fft.rfft(x, n = fast_len, dim = dim)
    f_weight = torch.fft.rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = torch.fft.irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# classes

# 高效的深度可分离卷积模块
class EfficientDsConv(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads
    ):
        super().__init__()
        assert (dim % heads) == 0

        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.to_weight = nn.Linear(dim, heads, bias = False)

        # 参数 D
        self.param_D = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)

        # 学习的加权残差
        residual = u * self.param_D

        # dsconv 核取决于序列长度
        K = self.to_weight(x)
        K = torch.flip(K, dims = (1,))

        # 一维卷积傅立叶变换 O(nlog(n))
        u = rearrange(u, '... (h d) -> ... h d', h = self.heads)

        out = conv1d_fft(u, K, dim = -3, weight_dim = -2)

        out = rearrange(out, '... h d -> ... (h d)')

        return out + residual

# 门控深度可分离卷积模块
class GatedDsConv(nn.Module):
    """ Pseudocode 3.2 """
    """ except state spaces replaced with regular learned convolution kernel """

    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_dsconv = 512,
        dim_expansion_factor = 4,
        reverse_seq = False
    ):
        super().__init__()
        assert (dim_dsconv % heads) == 0
        self.reverse_seq = reverse_seq

        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dim_dsconv, bias = False), nn.GELU())

        self.dsconv = EfficientDsConv(dim = dim_dsconv, heads = heads)

        self.to_gate = nn.Linear(dim_dsconv, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        if self.reverse_seq:
            x = torch.flip(x, dims = (1,))

        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.dsconv(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        out = out + residual

        if self.reverse_seq:
            out = torch.flip(out, dims = (1,))

        return out

# 门控深度可分离卷积 LM
class GatedDsConvLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_dsconv = 512,
        max_seq_len = 2048,
        dim_expansion_factor = 4,
    ):  
        # 初始化函数，继承父类的初始化方法
        super().__init__()
        # 创建一个嵌入层，用于将输入的 token 转换为指定维度的向量表示
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 设置最大序列长度
        self.max_seq_len = max_seq_len

        # 创建一个空的神经网络层列表
        self.layers = nn.ModuleList([])
        # 根据深度循环创建 GatedDsConv 层，并添加到神经网络层列表中
        for _ in range(depth):
            self.layers.append(
                GatedDsConv(
                    dim = dim,
                    heads = heads,
                    dim_dsconv = dim_dsconv,
                    dim_expansion_factor = dim_expansion_factor
                )
            )

        # 创建一个线性层，用于将输出的向量转换为预测的 token
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(self, x, labels = None):
        # 断言输入的序列长度不超过最大序列长度
        assert x.shape[1] <= self.max_seq_len

        # 将输入的 token 转换为向量表示
        x = self.token_emb(x)

        # 遍历神经网络层列表，依次对输入进行处理
        for dsconv in self.layers:
            x = dsconv(x)

        # 将处理后的向量转换为预测的 token
        logits = self.to_logits(x)

        # 如果没有提供标签，则直接返回预测结果
        if not exists(labels):
            return logits

        # 重新排列预测结果的维度，以便计算交叉熵损失
        logits = rearrange(logits, 'b n c -> b c n')
        # 计算交叉熵损失并返回
        return F.cross_entropy(logits, labels)
```