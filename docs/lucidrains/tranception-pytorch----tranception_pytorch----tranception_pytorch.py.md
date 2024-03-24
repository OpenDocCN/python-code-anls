# `.\lucidrains\tranception-pytorch\tranception_pytorch\tranception_pytorch.py`

```py
# 导入 math 模块
import math
# 导入 torch 模块
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch 模块中导入 nn、einsum 模块
from torch import nn, einsum
# 从 einops 模块中导入 rearrange 函数
from einops import rearrange
# 从 einops_exts 模块中导入 rearrange_many 函数
from einops_exts import rearrange_many
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 相对位置偏置

# 自定义类 LearnedAlibiPosBias 继承自 nn.Module
class LearnedAlibiPosBias(nn.Module):
    # 初始化函数
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        # 计算斜率并转换为张量
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.slopes = nn.Parameter(slopes)
        # 注册缓冲区 bias
        self.register_buffer('bias', None, persistent = False)

    # 获取相对位置偏置的函数
    def get_bias(self, i, j, device):
        i_arange = torch.arange(i, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    # 静态方法，用于获取斜率
    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    # 前向传播函数
    def forward(self, qk_sim):
        h, i, j, device = *qk_sim.shape[-3:], qk_sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return bias

# 辅助类

# 自定义类 ReluSquared 继承自 nn.Module
class ReluSquared(nn.Module):
    """ found with neural architecture search in Primer paper """
    # 前向传播函数
    def forward(self, x):
        return F.relu(x) ** 2

# 定义 FeedForward 函数
def FeedForward(dim, mult = 4):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        ReluSquared(),
        nn.Linear(hidden_dim, dim)
    )

# 自定义类 DepthwiseConv1d 继承自 nn.Module
class DepthwiseConv1d(nn.Module):
    # 初始化函数
    def __init__(self, dim, kernel_size, causal = True):
        super().__init__()
        assert (kernel_size % 2) == 1

        self.padding = (kernel_size - 1, 0) if causal else (kernel_size // 2, kernel_size // 2)
        self.conv = nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim)

    # 前向传播函数
    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# 自定义类 Attention 继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False,
        ds_conv_kernel_sizes = (0, 3, 5, 7) # heads were grouped into 4 groups and given a depthwise conv after the queries / keys / values projection
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置头数等于卷积核大小的组数，确保头数大于等于组数且头数能被组数整除
        self.groups = len(ds_conv_kernel_sizes)
        assert heads >= self.groups and (heads % self.groups) == 0, f'heads must be greater than {self.groups} and divisible by {self.groups}'

        # 设置缩放因子为头尺寸的负平方根
        self.scale = dim_head ** -0.5
        # 是否使用因果卷积
        self.causal = causal

        self.heads = heads
        self.heads_per_group = heads // self.groups

        inner_dim = heads * dim_head

        # 对输入进行 LayerNorm
        self.norm = nn.LayerNorm(dim)

        # 用 1x1 卷积层将输入转换为查询、键、值
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)

        # 使用不同卷积核大小的深度卷积进行 4 组头的处理
        self.qkv_ds_convs = nn.ModuleList([])

        for _ in range(3): # for queries, keys, values
            ds_convs = nn.ModuleList([])

            for kernel_size in ds_conv_kernel_sizes:
                if kernel_size == 0:
                    ds_convs.append(nn.Identity())
                    continue

                ds_convs.append(DepthwiseConv1d(dim_head * self.heads_per_group, kernel_size, causal = causal))

            self.qkv_ds_convs.append(ds_convs)

        # 为 4 组头学习位置偏置
        self.learned_alibi_pos_biases = nn.ModuleList([LearnedAlibiPosBias(heads = self.heads_per_group) for _ in range(self.groups)])

        # 输出投影
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        device, heads_per_group = x.device, self.heads_per_group

        # 对输入进行 LayerNorm，并重新排列维度
        x = self.norm(x)
        x = rearrange(x, 'b n d -> b d n')

        # 将输入转换为查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        # 重新排列查询、键、值的维度
        q, k, v = rearrange_many((q, k, v), 'b (h d) n -> b h d n', h = self.heads)

        # 对分组头应用因果深度卷积
        def apply_causal_ds_conv_to_grouped_heads(args):
            projs, ds_convs = args
            batch = projs.shape[0]

            projs = rearrange_many(projs.split(heads_per_group, dim = 1), 'b h d n -> b (h d) n')
            conv_out = [fn(t) for fn, t in zip(ds_convs, projs)]
            conv_out = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = heads_per_group), conv_out)
            conv_out = torch.cat(tuple(conv_out), dim = 1)
            return rearrange(conv_out, 'b h d n -> b h n d')

        q, k, v = map(apply_causal_ds_conv_to_grouped_heads, zip((q, k, v), self.qkv_ds_convs))

        # 缩放和计算相似度
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 对 4 组头应用学习的位置偏置
        grouped_sims = sim.split(self.heads // self.groups, dim = 1)
        grouped_sims = [(alibi(sim_group) + sim_group) for alibi, sim_group in zip(self.learned_alibi_pos_biases, grouped_sims)]
        sim = torch.cat(grouped_sims, dim = 1)

        # 因果掩码
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # 注意力机制
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
# 定义一个名为 Tranception 的类，继承自 nn.Module
class Tranception(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        *,
        dim,  # 特征维度
        depth,  # 模型深度
        num_tokens = 21,  # 标记数量，默认为 21
        heads = 8,  # 多头注意力机制中的头数，默认为 8
        dim_head = 64,  # 每个头的维度，默认为 64
        ff_mult = 4,  # FeedForward 层的倍数，默认为 4
        ds_conv_kernel_sizes = (0, 3, 5, 7),  # 下采样卷积的内核大小，默认为 (0, 3, 5, 7)
        causal = True  # 是否使用因果卷积，默认为 True
    ):
        super().__init__()  # 调用父类的初始化函数
        self.token_emb = nn.Embedding(num_tokens, dim)  # 创建一个标记嵌入层

        self.layers = nn.ModuleList([])  # 创建一个空的模块列表
        for _ in range(depth):  # 根据深度循环
            self.layers.append(nn.ModuleList([  # 向模块列表中添加模块列表
                Attention(dim = dim, heads = heads, dim_head = dim_head, ds_conv_kernel_sizes = ds_conv_kernel_sizes, causal = causal),  # 添加注意力层
                FeedForward(dim, mult = ff_mult)  # 添加前馈神经网络层
            ]))

        self.to_logits = nn.Sequential(  # 创建一个序列模块
            nn.LayerNorm(dim),  # 添加层归一化层
            nn.Linear(dim, num_tokens)  # 添加线性层
        )

    # 前向传播函数，接受输入 x 和掩码 mask，默认为 None
    def forward(
        self,
        x,
        mask = None
    ):
        x = self.token_emb(x)  # 将输入 x 通过标记嵌入层

        for attn, ff in self.layers:  # 遍历模块列表中的模块
            x = attn(x) + x  # 执行注意力层并将结果与输入相加
            x = ff(x) + x  # 执行前馈神经网络层并将结果与输入相加

        return self.to_logits(x)  # 返回经过线性层处理后的结果
```