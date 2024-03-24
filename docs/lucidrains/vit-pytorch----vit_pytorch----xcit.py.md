# `.\lucidrains\vit-pytorch\vit_pytorch\xcit.py`

```
# 从 random 模块中导入 randrange 函数
from random import randrange

# 导入 torch 模块及相关子模块
import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

# 导入 einops 模块及相关函数
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 将张量打包成指定模式的函数
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式的函数
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 对张量进行 L2 归一化的函数
def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# 对神经网络层进行 dropout 处理的函数
def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # 确保至少有一层不被丢弃
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# 类

# LayerScale 类，用于对输入进行缩放
class LayerScale(Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 < depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.fn = fn
        self.scale = nn.Parameter(torch.full((dim,), init_eps))

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

# FeedForward 类，前馈神经网络层
class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Attention 类，注意力机制层
class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        h = self.heads

        x = self.norm(x)
        context = x if not exists(context) else torch.cat((x, context), dim = 1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# XCAttention 类，交叉通道注意力机制层
class XCAttention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 获取头数
        h = self.heads
        # 将输入 x 打包成指定格式，并返回打包后的数据和打包方案 ps
        x, ps = pack_one(x, 'b * d')

        # 对输入 x 进行归一化处理
        x = self.norm(x)
        # 将 x 转换为查询、键、值，并按最后一个维度分割成三部分
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # 将查询、键、值按照指定格式重新排列
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h d n', h=h), (q, k, v))

        # 对查询、键进行 L2 归一化处理
        q, k = map(l2norm, (q, k))

        # 计算注意力矩阵，包括计算相似度、温度调节和注意力计算
        sim = einsum('b h i n, b h j n -> b h i j', q, k) * self.temperature.exp()

        # 进行注意力聚合
        attn = self.attend(sim)
        # 对注意力矩阵进行 dropout 处理
        attn = self.dropout(attn)

        # 根据注意力矩阵和值计算输出
        out = einsum('b h i j, b h j n -> b h i n', attn, v)
        # 将输出按指定格式重新排列
        out = rearrange(out, 'b h d n -> b n (h d)')

        # 将输出解包成原始格式
        out = unpack_one(out, ps, 'b * d')
        # 返回输出结果
        return self.to_out(out)
class LocalPatchInteraction(Module):
    # 定义局部补丁交互模块，继承自 Module 类
    def __init__(self, dim, kernel_size = 3):
        # 初始化函数，接受维度 dim 和卷积核大小 kernel_size，默认为 3
        super().__init__()
        # 调用父类的初始化函数

        assert (kernel_size % 2) == 1
        # 断言卷积核大小为奇数
        padding = kernel_size // 2
        # 计算卷积的填充大小

        self.net = nn.Sequential(
            # 定义神经网络模块
            nn.LayerNorm(dim),
            # 对输入进行层归一化
            Rearrange('b h w c -> b c h w'),
            # 重新排列张量的维度
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            # 二维卷积层
            nn.BatchNorm2d(dim),
            # 对输入进行批归一化
            nn.GELU(),
            # GELU 激活函数
            nn.Conv2d(dim, dim, kernel_size, padding = padding, groups = dim),
            # 二维卷积层
            Rearrange('b c h w -> b h w c'),
            # 重新排列张量的维度
        )

    def forward(self, x):
        # 前向传播函数，接受输入 x
        return self.net(x)
        # 返回经过网络处理后的结果

class Transformer(Module):
    # 定义 Transformer 模块，继承自 Module 类
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        # 初始化函数，接受维度 dim、深度 depth、头数 heads、头维度 dim_head、MLP维度 mlp_dim、dropout率 dropout 和层dropout率 layer_dropout，默认为 0
        super().__init__()
        # 调用父类的初始化函数
        self.layers = ModuleList([])
        # 初始化模块列表

        self.layer_dropout = layer_dropout
        # 设置层dropout率

        for ind in range(depth):
            # 循环遍历深度次数
            layer = ind + 1
            # 计算当前层索引
            self.layers.append(ModuleList([
                # 向模块列表中添加模块列表
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                # 添加注意力机制模块
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
                # 添加前馈神经网络模块
            ]))

    def forward(self, x, context = None):
        # 前向传播函数，接受输入 x 和上下文 context，默认为 None
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)
        # 对模块列表进行层dropout处理

        for attn, ff in layers:
            # 遍历模块列表中的注意力机制和前馈神经网络模块
            x = attn(x, context = context) + x
            # 经过注意力机制处理后与原始输入相加
            x = ff(x) + x
            # 经过前馈神经网络处理后与原始输入相加

        return x
        # 返回处理后的结果

class XCATransformer(Module):
    # 定义 XCAttention Transformer 模块，继承自 Module 类
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size = 3, dropout = 0., layer_dropout = 0.):
        # 初始化函数，接受维度 dim、深度 depth、头数 heads、头维度 dim_head、MLP维度 mlp_dim、局部补丁卷积核大小 local_patch_kernel_size，默认为 3，dropout率 dropout 和层dropout率 layer_dropout，默认为 0
        super().__init__()
        # 调用父类的初始化函数
        self.layers = ModuleList([])
        # 初始化模块列表

        self.layer_dropout = layer_dropout
        # 设置层dropout率

        for ind in range(depth):
            # 循环遍历深度次数
            layer = ind + 1
            # 计算当前层索引
            self.layers.append(ModuleList([
                # 向模块列表中添加模块列表
                LayerScale(dim, XCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = layer),
                # 添加交叉协方差注意力机制模块
                LayerScale(dim, LocalPatchInteraction(dim, local_patch_kernel_size), depth = layer),
                # 添加局部补丁交互模块
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = layer)
                # 添加前馈神经网络模块
            ]))

    def forward(self, x):
        # 前向传播函数，接受输入 x
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)
        # 对模块列表进行层dropout处理

        for cross_covariance_attn, local_patch_interaction, ff in layers:
            # 遍历模块列表中的交叉协方差注意力机制、局部补丁交互和前馈神经网络模块
            x = cross_covariance_attn(x) + x
            # 经过交叉协方差注意力机制处理后与原始输入相加
            x = local_patch_interaction(x) + x
            # 经过局部补丁交互处理后与原始输入相加
            x = ff(x) + x
            # 经过前馈神经网络处理后与原始输入相加

        return x
        # 返回处理后的结果

class XCiT(Module):
    # 定义 XCiT 模块，继承自 Module 类
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        local_patch_kernel_size = 3,
        layer_dropout = 0.
    ):
        # 初始化函数，接受关键字参数 image_size、patch_size、num_classes、dim、depth、cls_depth、heads、mlp_dim、dim_head、dropout、emb_dropout、局部补丁卷积核大小 local_patch_kernel_size，默认为 3，层dropout率 layer_dropout，默认为 0
        super().__init__()
        # 调用父类的初始化函数
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # 断言图像尺寸必须能被补丁大小整除

        num_patches = (image_size // patch_size) ** 2
        # 计算补丁数量
        patch_dim = 3 * patch_size ** 2
        # 计算补丁维度

        self.to_patch_embedding = nn.Sequential(
            # 定义序列模块
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # 重新排列张量的维度
            nn.LayerNorm(patch_dim),
            # 对输入进行层归一化
            nn.Linear(patch_dim, dim),
            # 线性变换
            nn.LayerNorm(dim)
            # 对输入进行层归一化
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 定义位置编码参数
        self.cls_token = nn.Parameter(torch.randn(dim))
        # 定义类别标记参数

        self.dropout = nn.Dropout(emb_dropout)
        # 定义丢弃层

        self.xcit_transformer = XCATransformer(dim, depth, heads, dim_head, mlp_dim, local_patch_kernel_size, dropout, layer_dropout)
        # 定义 XCAttention Transformer 模块

        self.final_norm = nn.LayerNorm(dim)
        # 对最终结果进行层归一化

        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        # 定义 Transformer 模块

        self.mlp_head = nn.Sequential(
            # 定义序列模块
            nn.LayerNorm(dim),
            # 对输入进行层归一化
            nn.Linear(dim, num_classes)
            # 线性变换
        )
        # 定义 MLP 头部模块
    # 前向传播函数，接收输入图像并进行处理
    def forward(self, img):
        # 将输入图像转换为补丁嵌入
        x = self.to_patch_embedding(img)

        # 将嵌入的补丁打包成一个张量
        x, ps = pack_one(x, 'b * d')

        # 获取张量的形状信息
        b, n, _ = x.shape
        # 添加位置嵌入到张量中
        x += self.pos_embedding[:, :n]

        # 解包张量
        x = unpack_one(x, ps, 'b * d')

        # 对张量进行 dropout 操作
        x = self.dropout(x)

        # 使用 XCIT Transformer 处理张量
        x = self.xcit_transformer(x)

        # 对处理后的张量进行最终归一化
        x = self.final_norm(x)

        # 重复生成类别标记 tokens
        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)

        # 重新排列张量的维度
        x = rearrange(x, 'b ... d -> b (...) d')
        # 使用类别标记 tokens 和上下文张量进行类别 Transformer 操作
        cls_tokens = self.cls_transformer(cls_tokens, context = x)

        # 返回 MLP 头部处理后的结果
        return self.mlp_head(cls_tokens[:, 0])
```