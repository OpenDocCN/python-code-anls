# `.\lucidrains\llama-qrlhf\llama_qrlhf\llama.py`

```py
import torch  # 导入 PyTorch 库
from torch.nn import Module, ModuleList  # 导入 PyTorch 中的 Module 和 ModuleList
from torch import nn, einsum, Tensor  # 导入 PyTorch 中的 nn、einsum 和 Tensor
import torch.nn.functional as F  # 导入 PyTorch 中的 nn.functional，并使用别名 F

from einops import rearrange, reduce  # 导入 einops 库中的 rearrange 和 reduce 函数
from einops.layers.torch import Rearrange  # 从 einops 库中导入 torch 版的 Rearrange 模块

# helpers

def exists(v):  # 定义一个函数 exists，用于判断变量是否存在
    return v is not None  # 返回变量是否不为 None 的布尔值

# norm

class RMSNorm(Module):  # 定义一个 RMSNorm 类，继承自 Module
    def __init__(self, dim):  # 初始化方法，接收维度参数 dim
        super().__init__()  # 调用父类的初始化方法
        self.scale = dim ** 0.5  # 计算缩放因子
        self.gamma = nn.Parameter(torch.ones(dim))  # 创建一个可学习的参数 gamma

    def forward(self, x):  # 前向传播方法，接收输入 x
        return F.normalize(x, dim=-1) * self.scale * self.gamma  # 对输入 x 进行归一化处理并乘以缩放因子和 gamma

# rotary

class RotaryEmbedding(Module):  # 定义一个 RotaryEmbedding 类，继承自 Module
    def __init__(self, dim, theta=10000):  # 初始化方法，接收维度参数 dim 和 theta，默认值为 10000
        super().__init__()  # 调用父类的初始化方法
        inv_freq = theta ** -(torch.arange(0, dim, 2).float() / dim)  # 计算频率的倒数
        self.register_buffer('inv_freq', inv_freq)  # 将频率的倒数注册为缓冲张量

    def forward(self, seq_len, device):  # 前向传播方法，接收序列长度和设备信息
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)  # 生成序列长度张量 t
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)  # 计算频率
        return torch.cat((freqs, freqs), dim=-1)  # 拼接频率张量并返回

def rotate_half(x):  # 定义一个函数 rotate_half，用于将输入张量 x 分成两部分并旋转
    x1, x2 = x.chunk(2, dim=-1)  # 将输入张量 x 按照最后一个维度分成两部分
    return torch.cat((-x2, x1), dim=-1)  # 将两部分旋转后拼接并返回

def apply_rotary_pos_emb(pos, t):  # 定义一个函数 apply_rotary_pos_emb，用于应用旋转位置编码
    return t * pos.cos() + rotate_half(t) * pos.sin()  # 返回应用旋转位置编码后的结果

# feedforward

class GEGLU(Module):  # 定义一个 GEGLU 类，继承自 Module
    def forward(self, x):  # 前向传播方法，接收输入 x
        x, gate = x.chunk(2, dim=-1)  # 将输入 x 按照最后一个维度分成两部分
        return F.gelu(gate) * x  # 对其中一部分应用 GELU 激活函数并返回乘积结果

def FeedForward(dim, mult=4):  # 定义一个 FeedForward 函数，用于创建前馈神经网络
    dim_hidden = int(dim * mult * 2 / 3)  # 计算隐藏层维度
    return nn.Sequential(  # 返回一个序列模块
        RMSNorm(dim),  # 添加 RMSNorm 模块
        nn.Linear(dim, dim_hidden * 2),  # 添加线性层
        GEGLU(),  # 添加 GEGLU 模块
        nn.Linear(dim_hidden, dim)  # 添加线性层
    )

# attention

class Attention(Module):  # 定义一个 Attention 类，继承自 Module
    def __init__(  # 初始化方法，接收维度参数 dim 和关键字参数
        self,
        dim,
        *,
        dim_head=64,
        heads=8
    ):
        super().__init__()  # 调用父类的初始化方法
        self.scale = dim_head ** -0.5  # 计算缩放因子
        dim_hidden = dim_head * heads  # 计算隐藏层维度

        self.to_qkv = nn.Sequential(  # 创建一个序列模块
            RMSNorm(dim),  # 添加 RMSNorm 模块
            nn.Linear(dim, dim_hidden * 3, bias=False),  # 添加线性层
            Rearrange('b n (qkv h d) -> qkv b h n d', h=heads, qkv=3)  # 重新排列张量维度
        )

        self.to_out = nn.Sequential(  # 创建一个序列模块
            Rearrange('b h n d -> b n (h d)'),  # 重新排列张量维度
            nn.Linear(dim_hidden, dim, bias=False)  # 添加线性层
        )

    def forward(self, x, rotary_emb=None):  # 前向传播方法，接收输入 x 和旋转位置编码
        q, k, v = self.to_qkv(x)  # 将输入 x 转换为查询、键、值

        if exists(rotary_emb):  # 如果旋转位置编码存在
            q, k = map(lambda t: apply_rotary_pos_emb(rotary_emb, t), (q, k))  # 应用旋转位置编码到查询和键

        q = q * self.scale  # 缩放查询
        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # 计算相似度

        i, j = sim.shape[-2:]  # 获取相似度张量的形状
        causal_mask = torch.ones((i, j), device=x.device, dtype=torch.bool).triu(j - i + 1)  # 创建因果掩码
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)  # 对相似度张量应用掩码

        attn = sim.softmax(dim=-1)  # 对相似度张量进行 softmax 操作

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # 计算加权和

        return self.to_out(out)  # 返回输出结果

# Q head

class DuelingHead(Module):  # 定义一个 DuelingHead 类，继承自 Module
    def __init__(  # 初始化方法，接收关键字参数
        self,
        *,
        dim,
        num_tokens,
        expansion_factor=2,
    ):
        super().__init__()  # 调用父类的初始化方法
        dim_hidden = int(dim * expansion_factor)  # 计算隐藏层维度

        self.stem = nn.Sequential(  # 创建一个序列模块
            nn.Linear(dim, dim_hidden),  # 添加线性层
            nn.SiLU()  # 添加 SiLU 激活函数
        )

        self.to_values = nn.Sequential(  # 创建一个序列模块
            nn.Linear(dim_hidden, 1)  # 添加线性层
        )

        self.to_advantages = nn.Sequential(  # 创建一个序列模块
            nn.Linear(dim_hidden, num_tokens)  # 添加线性层
        )

    def forward(self, x):  # 前向传播方法，接收输入 x
        x = self.stem(x)  # 应用 stem 模块到输入 x

        advantages = self.to_advantages(x)  # 计算优势值
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')  # 计算优势值的平均值

        values = self.to_values(x)  # 计算值函数

        q_values = values + advantages  # 计算 Q 值
        return q_values  # 返回 Q 值

# llama

class Llama(Module):  # 定义一个 Llama 类，继承自 Module
    def __init__(  # 初始化方法，接收关键字参数
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        dueling_q_head=False,
        dueling_q_head_expansion_factor=2
    # 初始化模型，继承父类的初始化方法
    ):
        super().__init__()

        # 创建 token embedding 层，将输入 token 映射为指定维度的向量
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建旋转 embedding 层，用于在注意力机制中引入旋转
        self.rotary_emb = RotaryEmbedding(dim_head)

        # 创建多层 Transformer 模型
        self.layers = ModuleList([])

        # 循环创建指定层数的 Transformer 层
        for _ in range(depth):
            # 每层包含注意力机制和前馈神经网络
            self.layers.append(ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # 创建最终的归一化层
        self.final_norm = RMSNorm(dim)

        # 创建输出层，将模型输出映射为预测的 token
        self.to_logits = nn.Linear(dim, num_tokens)

        # 如果使用 dueling q head，则创建 dueling 头部
        if dueling_q_head:
            self.to_q = DuelingHead(num_tokens = num_tokens, dim = dim, expansion_factor = dueling_q_head_expansion_factor)
        else:
            # 否则创建普通的线性层
            self.to_q = nn.Linear(dim, num_tokens)

    # 模型的前向传播方法
    def forward(
        self,
        x,
        return_q_values = False
    ):
        # 获取输入序列的长度和设备信息
        seq_len, device = x.shape[-1], x.device

        # 对输入序列进行 token embedding
        x = self.token_emb(x)

        # 创建旋转 embedding
        rotary_emb = self.rotary_emb(seq_len, device = device)

        # 遍历每一层 Transformer
        for attn, ff in self.layers:
            # 执行注意力机制和前馈神经网络
            x = attn(x, rotary_emb = rotary_emb) + x
            x = ff(x) + x

        # 对输出进行最终的归一化
        embed = self.final_norm(x)
        # 将归一化后的输出映射为预测的 token
        logits = self.to_logits(embed)

        # 如果需要返回 Q 值，则计算 Q 值并返回
        if not return_q_values:
            return logits

        return logits, self.to_q(embed)
```