# `.\lucidrains\vit-pytorch\vit_pytorch\cvt.py`

```
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper methods

# 根据条件将字典分组
def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 根据前缀分组并移除前缀
def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items()))
    return kwargs_without_prefix, kwargs

# classes

# 自定义 LayerNorm 类
class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 自定义 FeedForward 类
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 自定义 DepthWiseConv2d 类
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# 自定义 Attention 类
class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

# 自定义 Transformer 类
class Transformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 遍历 self.layers 中的每个元素，每个元素包含一个注意力机制和一个前馈神经网络
        for attn, ff in self.layers:
            # 使用注意力机制处理输入 x，并将结果与原始输入相加
            x = attn(x) + x
            # 使用前馈神经网络处理上一步的结果，并将结果与原始输入相加
            x = ff(x) + x
        # 返回处理后的结果 x
        return x
# 定义一个名为 CvT 的神经网络模型，继承自 nn.Module 类
class CvT(nn.Module):
    # 初始化函数，接收一系列参数
    def __init__(
        self,
        *,
        num_classes,  # 类别数量
        s1_emb_dim = 64,  # s1 阶段的嵌入维度
        s1_emb_kernel = 7,  # s1 阶段的卷积核大小
        s1_emb_stride = 4,  # s1 阶段的卷积步长
        s1_proj_kernel = 3,  # s1 阶段的投影卷积核大小
        s1_kv_proj_stride = 2,  # s1 阶段的键值投影步长
        s1_heads = 1,  # s1 阶段的注意力头数
        s1_depth = 1,  # s1 阶段的深度
        s1_mlp_mult = 4,  # s1 阶段的 MLP 扩展倍数
        s2_emb_dim = 192,  # s2 阶段的嵌入维度
        s2_emb_kernel = 3,  # s2 阶段的卷积核大小
        s2_emb_stride = 2,  # s2 阶段的卷积步长
        s2_proj_kernel = 3,  # s2 阶段的投影卷积核大小
        s2_kv_proj_stride = 2,  # s2 阶段的键值投影步长
        s2_heads = 3,  # s2 阶段的注意力头数
        s2_depth = 2,  # s2 阶段的深度
        s2_mlp_mult = 4,  # s2 阶段的 MLP 扩展倍数
        s3_emb_dim = 384,  # s3 阶段的嵌入维度
        s3_emb_kernel = 3,  # s3 阶段的卷积核大小
        s3_emb_stride = 2,  # s3 阶段的卷积步长
        s3_proj_kernel = 3,  # s3 阶段的投影卷积核大小
        s3_kv_proj_stride = 2,  # s3 阶段的键值投影步长
        s3_heads = 6,  # s3 阶段的注意力头数
        s3_depth = 10,  # s3 阶段的深度
        s3_mlp_mult = 4,  # s3 阶段的 MLP 扩展倍数
        dropout = 0.,  # Dropout 概率
        channels = 3  # 输入通道数
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将参数保存到字典中
        kwargs = dict(locals())

        # 初始化维度为输入通道数
        dim = channels
        # 初始化层列表
        layers = []

        # 遍历 s1、s2、s3 三个阶段
        for prefix in ('s1', 's2', 's3'):
            # 根据前缀分组参数，并从参数字典中移除前缀
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            # 将卷积、LayerNorm 和 Transformer 层添加到层列表中
            layers.append(nn.Sequential(
                nn.Conv2d(dim, config['emb_dim'], kernel_size = config['emb_kernel'], padding = (config['emb_kernel'] // 2), stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'], proj_kernel = config['proj_kernel'], kv_proj_stride = config['kv_proj_stride'], depth = config['depth'], heads = config['heads'], mlp_mult = config['mlp_mult'], dropout = dropout)
            ))

            # 更新维度为当前阶段的嵌入维度
            dim = config['emb_dim']

        # 将所有层组成一个序列
        self.layers = nn.Sequential(*layers)

        # 定义输出层，包括全局平均池化、重排和全连接层
        self.to_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, x):
        # 经过所有层得到特征向量
        latents = self.layers(x)
        # 将特征向量传递给输出层得到预测结果
        return self.to_logits(latents)
```