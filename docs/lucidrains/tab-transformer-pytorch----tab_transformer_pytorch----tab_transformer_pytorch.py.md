# `.\lucidrains\tab-transformer-pytorch\tab_transformer_pytorch\tab_transformer_pytorch.py`

```py
# 导入 PyTorch 库
import torch
import torch.nn.functional as F
from torch import nn, einsum

# 导入 einops 库中的 rearrange 和 repeat 函数
from einops import rearrange, repeat

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 类定义

# 残差连接模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 预层归一化模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 注意力机制

# GEGLU 激活函数
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# 前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

# 注意力机制模块
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# Transformer 模块
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = x + attn_out
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# 多层感知机模块
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# 主类 TabTransformer
class TabTransformer(nn.Module):
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        *,
        categories,  # 类别特征的数量列表
        num_continuous,  # 连续特征的数量
        dim,  # 模型的维度
        depth,  # Transformer 模型的深度
        heads,  # Transformer 模型的头数
        dim_head = 16,  # 每个头的维度
        dim_out = 1,  # 输出的维度
        mlp_hidden_mults = (4, 2),  # MLP 隐藏层的倍数
        mlp_act = None,  # MLP 的激活函数
        num_special_tokens = 2,  # 特殊标记的数量
        continuous_mean_std = None,  # 连续特征的均值和标准差
        attn_dropout = 0.,  # 注意力机制的 dropout
        ff_dropout = 0.,  # FeedForward 层的 dropout
        use_shared_categ_embed = True,  # 是否使用共享的类别嵌入
        shared_categ_dim_divisor = 8   # 在论文中，他们将维度的 1/8 保留给共享的类别嵌入
    ):
        super().__init__()
        # 断言确保每个类别的数量大于 0
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        # 断言确保类别数量和连续特征数量之和大于 0
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # 与类别相关的计算

        self.num_categories = len(categories)  # 类别的数量
        self.num_unique_categories = sum(categories)  # 所有类别的总数

        # 创建类别嵌入表

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        # 处理共享的类别嵌入

        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # 用于自动偏移唯一类别 id 到类别嵌入表中的正确位置

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # 连续特征

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)

            self.norm = nn.LayerNorm(num_continuous)

        # Transformer 模型

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # MLP 转换为 logits

        input_size = (dim * self.num_categories) + num_continuous

        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)
    # 定义一个前向传播函数，接受类别特征和连续特征作为输入，可选择返回注意力权重
    def forward(self, x_categ, x_cont, return_attn = False):
        # 初始化一个空列表用于存储不同类型特征的输出
        xs = []

        # 检查类别特征的最后一个维度是否与预期的类别数量相同
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        # 如果存在唯一的类别数量大于0
        if self.num_unique_categories > 0:
            # 对类别特征进行偏移处理
            x_categ = x_categ + self.categories_offset

            # 对类别特征进行嵌入处理
            categ_embed = self.category_embed(x_categ)

            # 如果使用共享的类别嵌入
            if self.use_shared_categ_embed:
                # 复制共享的类别嵌入并与类别嵌入拼接
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            # 使用 Transformer 处理类别嵌入特征，可选择返回注意力权重
            x, attns = self.transformer(categ_embed, return_attn = True)

            # 将处理后的类别特征展平
            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)

        # 检查连续特征的第二个维度是否与预期的连续特征数量相同
        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        # 如果连续特征数量大于0
        if self.num_continuous > 0:
            # 如果存在连续特征的均值和标准差
            if exists(self.continuous_mean_std):
                # 分离连续特征的均值和标准差
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                # 对连续特征进行标准化处理
                x_cont = (x_cont - mean) / std

            # 对标准化后的连续特征进行归一化处理
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        # 将处理后的类别特征和连续特征拼接在一起
        x = torch.cat(xs, dim = -1)
        # 使用 MLP 处理拼接后的特征，得到输出 logits

        logits = self.mlp(x)

        # 如果不需要返回注意力权重，则直接返回 logits
        if not return_attn:
            return logits

        # 如果需要返回注意力权重，则同时返回 logits 和注意力权重
        return logits, attns
```