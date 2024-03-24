# `.\lucidrains\tab-transformer-pytorch\tab_transformer_pytorch\ft_transformer.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn 和 einsum 模块
from torch import nn, einsum
# 从 einops 中导入 rearrange 和 repeat 函数

from einops import rearrange, repeat

# feedforward and attention

# 定义 GEGLU 类，继承自 nn.Module
class GEGLU(nn.Module):
    # 前向传播函数
    def forward(self, x):
        # 将输入 x 按照最后一个维度分成两部分
        x, gates = x.chunk(2, dim = -1)
        # 返回 x 乘以 gates 经过 gelu 激活函数的结果
        return x * F.gelu(gates)

# 定义 FeedForward 函数，接受维度 dim、倍数 mult 和 dropout 参数
def FeedForward(dim, mult = 4, dropout = 0.):
    # 返回一个序列模块
    return nn.Sequential(
        # LayerNorm 层
        nn.LayerNorm(dim),
        # 线性变换层，输入维度为 dim，输出维度为 dim * mult * 2
        nn.Linear(dim, dim * mult * 2),
        # GEGLU 层
        GEGLU(),
        # Dropout 层
        nn.Dropout(dropout),
        # 线性变换层，输入维度为 dim * mult，输出维度为 dim
        nn.Linear(dim * mult, dim)
    )

# 定义 Attention 类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数，接受维度 dim、头数 heads、头维度 dim_head 和 dropout 参数
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 头数和头维度的缩放系数
        self.heads = heads
        self.scale = dim_head ** -0.5

        # LayerNorm 层
        self.norm = nn.LayerNorm(dim)

        # 线性变换层，输入维度为 dim，输出维度为 inner_dim * 3
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        # 线性变换层，输入维度为 inner_dim，输出维度为 dim
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数
    def forward(self, x):
        # 头数
        h = self.heads

        # 对输入 x 进行 LayerNorm
        x = self.norm(x)

        # 将输入 x 经过线性变换得到 q、k、v
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # 对 q、k、v 进行维度重排
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # 对 q 进行缩放
        q = q * self.scale

        # 计算注意力矩阵
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 对注意力矩阵进行 softmax
        attn = sim.softmax(dim = -1)
        # 对 softmax 结果进行 dropout
        dropped_attn = self.dropout(attn)

        # 计算输出
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

# 定义 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化函数，接受维度 dim、深度 depth、头数 heads、头维度 dim_head、注意力 dropout 和前馈 dropout 参数
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
        # 初始化层列表
        self.layers = nn.ModuleList([])

        # 循环创建 depth 个层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 注意力层
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                # 前馈层
                FeedForward(dim, dropout = ff_dropout),
            ]))

    # 前向传播函数
    def forward(self, x, return_attn = False):
        # 存储后 softmax 的注意力矩阵
        post_softmax_attns = []

        # 遍历每个层
        for attn, ff in self.layers:
            # 获取注意力层的输出和后 softmax 的注意力矩阵
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            # 更新 x
            x = attn_out + x
            x = ff(x) + x

        # 如果不返回注意力矩阵，则返回 x
        if not return_attn:
            return x

        # 返回 x 和后 softmax 的注意力矩阵
        return x, torch.stack(post_softmax_attns)

# numerical embedder

# 定义 NumericalEmbedder 类，继承自 nn.Module
class NumericalEmbedder(nn.Module):
    # 初始化函数，接受维度 dim 和数值类型数量 num_numerical_types
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        # 定义权重参数和偏置参数
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    # 前向传播函数
    def forward(self, x):
        # 将输入 x 维度重排
        x = rearrange(x, 'b n -> b n 1')
        # 返回加权和偏置后的结果
        return x * self.weights + self.biases

# main class

# 定义 FTTransformer 类，继承自 nn.Module
class FTTransformer(nn.Module):
    # 初始化函数，接受关键字参数 categories、num_continuous、dim、depth、heads、头维度 dim_head、输出维度 dim_out、特殊标记数量 num_special_tokens、注意力 dropout 和前馈 dropout
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        # 调用父类的构造函数
        super().__init__()
        # 断言所有类别的数量必须大于0
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        # 断言类别数量加上连续值的数量不能为0
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        # 计算类别相关的参数
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        # 创建类别嵌入表
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        # 用于自动将唯一类别ID偏移至类别嵌入表中的正确位置
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            # 类别嵌入
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        # 连续值
        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        # 类别标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        # 变换器
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        # 转换为logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset

            x_categ = self.categorical_embeds(x_categ)

            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        # 连接类别和连续值
        x = torch.cat(xs, dim = 1)

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # attend

        # 注意力机制
        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        # 获取类别标记
        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        # 论文中的输出是线性(ReLU(LN(cls)))
        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns
```