# `.\lucidrains\omninet-pytorch\omninet_pytorch\omninet_pytorch.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块，并重命名为 F
import torch.nn.functional as F

# 从 einops 库中导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 从 einops.layers.torch 库中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 使用 PerformerAttention 作为自注意力机制，因为它有最好的报告数字
from performer_pytorch import SelfAttention as PerformerAttention

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 获取模块所在设备的函数
def get_module_device(module):
    return next(module.parameters()).device

# 查找指定类型模块的函数
def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# 类定义

# 预层归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # 初始化 LayerNorm 归一化层
        self.norm = nn.LayerNorm(dim)
        # 初始化传入的函数
        self.fn = fn

    def forward(self, x, **kwargs):
        # 对输入进行归一化后，再传入函数进行处理
        return self.fn(self.norm(x), **kwargs)

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        # 定义前馈神经网络结构
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        # 前馈神经网络前向传播
        return self.net(x)

# 自注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads =  heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        # 定义 Q、K、V 的线性变换层
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # 定义输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        # 获取输入 x 的形状信息
        b, n, d, h, device = *x.shape, self.heads, x.device
        # 将输入 x 进行 Q、K、V 的线性���换，并分割为 Q、K、V
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # 计算注意力分数
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 定义最大负值
        max_neg_value = -torch.finfo(sim.dtype).max

        # 如果存在 mask，则进行 mask 操作
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            sim.masked_fill_(~mask, max_neg_value)

        # 如果是因果注意力机制，则进行 mask 操作
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = rearrange(causal_mask, 'i j -> () i j')
            sim.masked_fill_(causal_mask, max_neg_value)

        # 计算注意力权重
        attn = sim.softmax(dim = -1)

        # 计算输出
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# 主类

class Omninet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        pool_layer_tokens_every = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        feature_redraw_interval = 1000
    ):
        super().__init__()

        layers = nn.ModuleList([])
        for ind in range(depth):
            num_layers = ind + 1
            should_pool = num_layers % pool_layer_tokens_every

            layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)),
                PerformerAttention(dim = dim, heads= heads, dim_head = dim_head) if should_pool else None
            ]))

        self.layers = layers
        self.pool_num_layers = pool_layer_tokens_every

        # 跟踪重新绘制 Performer 投影矩阵的次数
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    # 修复投影矩阵的函数
    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None
    # 检查是否需要重新绘制投影矩阵
    def check_redraw_projections(self):
        # 如果不处于训练状态，则直接返回
        if not self.training:
            return

        # 如果存在特征重新绘制间隔，并且自上次重新绘制以来的调用次数超过间隔
        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            # 获取模块所在设备
            device = get_module_device(self)

            # 查找所有 FastAttention 模块
            fast_attentions = find_modules(self, FastAttention)
            # 对每个 FastAttention 模块重新绘制投影矩阵
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            # 重置自上次重新绘制以来的调用次数
            self.calls_since_last_redraw.zero_()
            return

        # 自上次重新绘制以来的调用次数加一
        self.calls_since_last_redraw += 1

    # 前向传播函数
    def forward(self, x, mask = None):
        # 检查是否需要重新绘制投影矩阵
        self.check_redraw_projections()
        # 获取池化层数
        pool_num_layers = self.pool_num_layers

        # 初始化隐藏层列表
        hiddens = [x]

        # 遍历每个注意力层、前馈层和高效注意力层
        for attn, ff, efficient_attn in self.layers:
            # 注意力层的输出加上输入，得到新的输出
            x = attn(x, mask = mask) + x
            # 前馈层的输出加上输入，得到新的输出
            x = ff(x) + x

            # 将新的输出添加到隐藏层列表中
            hiddens.append(x)
            # 如果存在高效注意力层
            if exists(efficient_attn):
                # 选择最近的池化层数量的隐藏层
                layers_to_pool = hiddens[-pool_num_layers:]
                num_layers = len(layers_to_pool)

                # 将所有隐藏层的 token 合并成一个张量
                all_tokens = torch.stack(layers_to_pool)
                all_tokens = rearrange(all_tokens, 'l b n d -> b (n l) d')

                # 初始化池化注意力层的掩码
                pool_attn_mask = None
                if exists(mask):
                    pool_attn_mask = repeat(mask, 'b n -> b (n l)', l = num_layers)

                # 对合并的 token 应用高效注意力层
                attended_tokens = efficient_attn(all_tokens, mask = pool_attn_mask)

                # 重新排列输出张量的维度
                attended_tokens = rearrange(attended_tokens, 'b n c -> b c n')
                # 对注意力输出进行最大池化
                pooled_tokens = F.max_pool1d(attended_tokens, kernel_size = num_layers, stride = num_layers)
                # 将池化后的 token 添加到输出中
                x += rearrange(pooled_tokens, 'b c n -> b n c')

        # 返回最终输出
        return x
# 定义一个名为 OmninetCausal 的类，用于处理因果关系的情况，采用轴向注意力层，直到重写线性注意力的 CUDA 内核
class OmninetCausal(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        pool_layer_tokens_every = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()

        # 初始化层位置嵌入参数
        self.layer_pos_emb = nn.Parameter(torch.randn(depth + 1, dim))

        # 初始化层列表
        layers = nn.ModuleList([])
        for ind in range(depth):
            num_layers = ind + 1
            should_pool = num_layers % pool_layer_tokens_every

            # 添加每一层的注意力、前馈和轴向注意力（如果需要池化）到层列表中
            layers.append(nn.ModuleList([
                PreNorm(dim, Attention(causal = True, dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)),
                Attention(dim = dim, heads= heads, dim_head = dim_head) if should_pool else None
            ]))

        self.layers = layers
        self.pool_num_layers = pool_layer_tokens_every

    # 前向传播函数
    def forward(self, x, mask = None):
        pool_num_layers = self.pool_num_layers

        b = x.shape[0]
        pos_embs = rearrange(self.layer_pos_emb, 'n d -> () n d')

        x += pos_embs[:, 0]
        hiddens = [x]

        for ind, (attn, ff, layer_axial_attn) in enumerate(self.layers):

            # 执行注意力层操作
            x = attn(x, mask = mask) + x
            # 执行前馈层操作
            x = ff(x) + x

            x += pos_embs[:, ind + 1]
            hiddens.append(x)

            if exists(layer_axial_attn):
                layers_to_pool = hiddens[-pool_num_layers:]
                num_layers = len(layers_to_pool)

                # 重排层的 tokens，并进行轴向注意力操作
                layer_tokens = rearrange(torch.stack(layers_to_pool), 'l b n d -> (b n) l d')

                attended_tokens = layer_axial_attn(layer_tokens)
                attended_tokens = rearrange(attended_tokens, '(b n) l d -> b n l d', b = b)
                pooled_attended_tokens = attended_tokens.max(dim = -2).values
                x += pooled_attended_tokens

        return x
```