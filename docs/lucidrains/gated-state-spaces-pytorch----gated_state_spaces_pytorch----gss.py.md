# `.\lucidrains\gated-state-spaces-pytorch\gated_state_spaces_pytorch\gss.py`

```
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.fft import rfft, irfft

from einops import rearrange

# functions

# 检查值是否存在
def exists(val):
    return val is not None

# classes

# 定义 DSS 类
class DSS(nn.Module):
    def __init__(
        self,
        *,
        dim,
        kernel_N = 512,
        dss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Lambda

        # 初始化 Lambda 的实部参数
        self.Lambda_real = nn.Parameter(torch.randn(kernel_N))
        # 初始化 Lambda 的虚部参数
        self.Lambda_imag = nn.Parameter(torch.randn(kernel_N))

        # C

        # 初始化 C 的实部参数
        self.C_real = nn.Parameter(torch.randn(dim, kernel_N))
        # 初始化 C 的虚部参数
        self.C_imag = nn.Parameter(torch.randn(dim, kernel_N))

        # params D

        # 初始化参数 D
        self.param_D = nn.Parameter(torch.randn(dim))

        # 是否对 Lambda 的虚部进行指数运算
        self.dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp

    def forward(self, x):
        """
        einstein notation:
        b - batch
        l - sequence length
        d - dimension
        """

        device, seq_len = x.device, x.shape[1]
        u = self.norm(x)

        # learned weighted residual

        # 计算加权残差
        residual = u * self.param_D

        # derive simple dss kernel

        # 计算简单的 DSS 核
        Lambda_imag = self.Lambda_imag.exp() if self.dss_kernel_lambda_imag_exp else self.Lambda_imag

        Lambda = -self.Lambda_real.exp() + 1j * Lambda_imag
        C = self.C_real + 1j * self.C_imag

        arange = torch.arange(seq_len, device = device)

        S = (rearrange(Lambda, 'n -> n 1') * rearrange(arange, 'l -> 1 l')).exp()
        C = C * (Lambda.exp() - 1) / Lambda

        K = einsum('h n, n l -> l h', C, S).real

        # conv1d fft O(nlog(n))

        u_f = rfft(u, n = seq_len * 2, dim = -2)
        K_f = rfft(K, n = seq_len * 2, dim = -2)

        y = irfft(u_f * K_f, seq_len * 2, dim = -2)[..., :seq_len, :]

        return y + residual

# 定义 GSS 类
class GSS(nn.Module):
    """ Pseudocode 3.2 """

    def __init__(
        self,
        *,
        dim,
        dim_expansion_factor = 4,
        dss_kernel_N = 512,
        dss_kernel_H = 256,
        reverse_seq = False,
        dss_kernel_lambda_imag_exp = True
    ):
        super().__init__()
        self.reverse_seq = reverse_seq
        self.norm = nn.LayerNorm(dim)

        dim_hidden = int(dim_expansion_factor * dim)
        self.to_u = nn.Sequential(nn.Linear(dim, dim_hidden, bias = False), nn.GELU())
        self.to_v = nn.Sequential(nn.Linear(dim, dss_kernel_H, bias = False), nn.GELU())

        self.dss = DSS(dim = dss_kernel_H, kernel_N = dss_kernel_N, dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp)

        self.to_gate = nn.Linear(dss_kernel_H, dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        if self.reverse_seq:
            x = torch.flip(x, dims = (1,))

        residual, x = x.clone(), self.norm(x)

        u = self.to_u(x)
        v = self.to_v(x)

        v = self.dss(v)

        uc = self.to_gate(v)
        out = self.to_out(uc * u)

        out = out + residual

        if self.reverse_seq:
            out = torch.flip(out, dims = (1,))

        return out

# Gated State Spaces LM

# 定义 GatedStateSpacesLM 类
class GatedStateSpacesLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_expansion_factor = 4,
        dss_kernel_N = 512,
        dss_kernel_H = 256,
        dss_kernel_lambda_imag_exp = True
    # 初始化函数，继承父类的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个嵌入层，用于将输入的标记转换为向量表示
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 创建一个空的模块列表，用于存储多个 GSS 模块
        self.layers = nn.ModuleList([])
        # 循环创建 depth 次 GSS 模块，并添加到模块列表中
        for _ in range(depth):
            self.layers.append(
                GSS(
                    dim = dim,
                    dss_kernel_H = dss_kernel_H,
                    dss_kernel_N = dss_kernel_N,
                    dim_expansion_factor = dim_expansion_factor,
                    dss_kernel_lambda_imag_exp = dss_kernel_lambda_imag_exp
                )
            )

        # 创建一个线性层，用于将模型输出的向量转换为预测的标记
        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    # 前向传播函数，接收输入 x 和标签 labels
    def forward(self, x, labels = None):
        # 将输入的标记转换为向量表示
        x = self.token_emb(x)

        # 遍历模块列表中的每个 GSS 模块，依次对输入进行处理
        for gss in self.layers:
            x = gss(x)

        # 将处理后的向量转换为预测的标记
        logits = self.to_logits(x)

        # 如果没有提供标签，则直接返回预测结果
        if not exists(labels):
            return logits

        # 重新排列 logits 的维度，以适应交叉熵损失函数的输入要求
        logits = rearrange(logits, 'b n c -> b c n')
        # 计算交叉熵损失并返回
        return F.cross_entropy(logits, labels)
```