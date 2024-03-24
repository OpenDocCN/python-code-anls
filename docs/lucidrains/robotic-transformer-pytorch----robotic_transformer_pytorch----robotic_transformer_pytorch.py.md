# `.\lucidrains\robotic-transformer-pytorch\robotic_transformer_pytorch\robotic_transformer_pytorch.py`

```py
# 导入 torch 库
import torch
# 导入 torch 中的函数库
import torch.nn.functional as F
# 从 torch 中导入 nn, einsum, Tensor
from torch import nn, einsum, Tensor
# 从 typing 中导入 List, Optional, Callable, Tuple
from typing import List, Optional, Callable, Tuple
# 从 beartype 中导入 beartype
from beartype import beartype
# 从 einops 中导入 pack, unpack, repeat, reduce, rearrange
from einops import pack, unpack, repeat, reduce, rearrange
# 从 einops.layers.torch 中导入 Rearrange, Reduce
from einops.layers.torch import Rearrange, Reduce
# 从 functools 中导入 partial
from functools import partial
# 从 classifier_free_guidance_pytorch 中导入 TextConditioner, AttentionTextConditioner, classifier_free_guidance

# helpers

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 default，返回值或默认值
def default(val, d):
    return val if exists(val) else d

# 定义函数 cast_tuple，将值转换为元组
def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# 定义函数 pack_one，将值按照指定模式打包
def pack_one(x, pattern):
    return pack([x], pattern)

# 定义函数 unpack_one，将值按照指定模式解包
def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# sinusoidal positions

# 定义函数 posemb_sincos_1d，生成一维正弦余弦位置编码
def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

# 定义类 Residual，实现残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 定义类 LayerNorm，实现层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 定义类 FeedForward，实现前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, cond_fn = None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)

# MBConv

# 定义类 SqueezeExcitation，实现 MBConv 中的 Squeeze-and-Excitation 模块
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

# 定义类 MBConvResidual，实现 MBConv 中的残差连接
class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

# 定义类 Dropsample，实现随机丢弃采样
class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

# 定义函数 MBConv，实现 MBConv 模块
def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1
    # 定义一个神经网络模型，包括卷积层、批量归一化层、GELU激活函数等
    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),  # 输入通道数为dim_in，输出通道数为hidden_dim的1x1卷积层
        nn.BatchNorm2d(hidden_dim),  # 对隐藏层进行批量归一化
        nn.GELU(),  # GELU激活函数
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),  # 3x3卷积层，带有步长、填充和分组参数
        nn.BatchNorm2d(hidden_dim),  # 对隐藏层进行批量归一化
        nn.GELU(),  # GELU激活函数
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),  # Squeeze-and-Excitation模块
        nn.Conv2d(hidden_dim, dim_out, 1),  # 输入通道数为hidden_dim，输出通道数为dim_out的1x1卷积层
        nn.BatchNorm2d(dim_out)  # 对输出层进行批量归一化
    )

    # 如果输入通道数等于输出通道数且不需要下采样，则添加MBConvResidual模块
    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    # 返回构建好的神经网络模型
    return net
# 定义注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_mem_kv = 4
    ):
        super().__init__()
        # 确保维度可以被头部维度整除
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        # 归一化层
        self.norm = LayerNorm(dim)

        # 头部数量
        self.heads = dim // dim_head
        # 缩放因子
        self.scale = dim_head ** -0.5

        # 查询、键、值的线性变换
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # 记忆键值对
        self.mem_kv = nn.Parameter(torch.randn(2, self.heads, num_mem_kv, dim_head))

        # 注意力机制
        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # 相对位置偏置
        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        # 相对位置索引计算
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        # 注册相对位置索引
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        # 获取输入张量的形状信息
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # 归一化输入张量
        x = self.norm(x)

        # 展平张量
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # 查询、键、值的投影
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # 分割头部
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 缩放
        q = q * self.scale

        # 空值/记忆/注册键值对
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = q.shape[0]),  self.mem_kv)
        num_mem = mk.shape[-2]

        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        # 相似度计算
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 添加位置偏置
        bias = self.rel_pos_bias(self.rel_pos_indices)
        bias = F.pad(bias, (0, 0, num_mem, 0), value = 0.)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # 注意力计算
        attn = self.attend(sim)

        # 聚合
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并头部
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # 合并头部输出
        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

# 定义 MaxViT 模型类
class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
        ):
        # 调用父类的构造函数
        super().__init__()
        # 断言 depth 是元组类型，如果不是则抛出异常
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # 卷积干部

        # 设置卷积干部的维度
        dim_conv_stem = default(dim_conv_stem, dim)

        # 创建卷积干部的序列
        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # 变量

        # 获取深度的阶段数
        num_stages = len(depth)

        # 计算每个阶段的维度
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # 创建模块列表
        self.layers = nn.ModuleList([])

        # 用于高效块-网格式注意力的窗口大小的简写

        w = window_size

        # 遍历各个阶段

        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # 块状注意力
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # 网格式注意力
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp 头部输出

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob = 0.,
        return_embeddings = False
    ):
        # 对输入进行卷积干部处理
        x = self.conv_stem(x)

        # 初始化条件函数
        cond_fns = iter(default(cond_fns, []))

        # 遍历每个阶段
        for stage in self.layers:
            # 获取下一个条件函数
            cond_fn = next(cond_fns, None)

            # 如果条件函数存在，则应用条件函数
            if exists(cond_fn):
                x = cond_fn(x)

            # 应用当前阶段的模块
            x = stage(x)

        # 如果需要返回嵌入向量，则返回嵌入向量
        if return_embeddings:
            return x

        # 返回经过 MLP 头部处理后的结果
        return self.mlp_head(x)
# 定义 TransformerAttention 类，用于实现 Transformer 中的注意力机制
class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None
    ):
        b = x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义 Transformer 类，用于实现 Transformer 模型
@beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
        cond_fns = iter(default(cond_fns, []))

        for attn, ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns, None)) + x
             x = ff(x, cond_fn = next(cond_fns, None)) + x
        return x

# 定义 TokenLearner 类，用于实现 TokenLearner 模块
class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )
    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 将输入 x 打包成指定格式，并返回打包后的数据和打包参数
        x, ps = pack_one(x, '* c h w')
        # 将输入 x 重复多次，改变维度，以适应网络输入要求
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        # 使用网络进行处理
        attn = self.net(x)

        # 重新排列注意力矩阵的维度
        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        # 重新排列输入 x 的维度
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        # 对输入 x 和注意力矩阵进行元素级乘法，并对结果进行降维求均值
        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        # 解包 x，恢复原始维度
        x = unpack_one(x, ps, '* c n')
        # 返回处理后的结果 x
        return x
# Robotic Transformer

# 使用 beartype 装饰器对 RT1 类进行类型检查
@beartype
class RT1(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        vit: MaxViT,  # 接收一个 MaxViT 类型的参数 vit
        num_actions = 11,  # 默认参数，表示动作的数量
        action_bins = 256,  # 默认参数，表示动作的分组数量
        depth = 6,  # 默认参数，表示 Transformer 的深度
        heads = 8,  # 默认参数，表示 Transformer 的头数
        dim_head = 64,  # 默认参数，表示每个头的维度
        token_learner_ff_mult = 2,  # 默认参数，表示 TokenLearner 的前馈倍数
        token_learner_num_layers = 2,  # 默认参数，表示 TokenLearner 的层数
        token_learner_num_output_tokens = 8,  # 默认参数，表示 TokenLearner 的输出 token 数量
        cond_drop_prob = 0.2,  # 默认参数，表示条件丢弃的概率
        use_attn_conditioner = False,  # 默认参数，表示是否使用 AttentionTextConditioner
        conditioner_kwargs: dict = dict()  # 默认参数，表示条件器的其他参数
    ):
        super().__init__()
        self.vit = vit  # 初始化 vit

        self.num_vit_stages = len(vit.cond_hidden_dims)  # 计算 vit 的隐藏维度数量

        # 根据是否使用 AttentionTextConditioner 选择条件器类
        conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner

        # 初始化条件器
        self.conditioner = conditioner_klass(
            hidden_dims = (*tuple(vit.cond_hidden_dims), *((vit.embed_dim,) * depth * 2)),
            hiddens_channel_first = (*((True,) * self.num_vit_stages), *((False,) * depth * 2)),
            cond_drop_prob = cond_drop_prob,
            **conditioner_kwargs
        )

        # 初始化 TokenLearner
        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens  # 记录 TokenLearner 的输出 token 数量

        self.transformer_depth = depth  # 记录 Transformer 的深度

        # 初始化 Transformer
        self.transformer = Transformer(
            dim = vit.embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth
        )

        self.cond_drop_prob = cond_drop_prob  # 记录条件丢弃的概率

        # 初始化输出层
        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, num_actions * action_bins),
            Rearrange('... (a b) -> ... a b', b = action_bins)
        )

    # 嵌入文本信息
    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    # 前向传播函数
    @classifier_free_guidance
    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = 0.
        ):
        # 断言只有 texts 或者 text_embeds 其中一个存在
        assert exists(texts) ^ exists(text_embeds)
        # 根据传入的参数创建条件参数字典
        cond_kwargs = dict(texts = texts, text_embeds = text_embeds)

        # 获取 transformer 的深度和条件丢弃概率
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 获取视频帧数和设备信息
        frames, device = video.shape[2], video.device

        # 调用 conditioner 方法生成条件函数
        cond_fns, _ = self.conditioner(
            **cond_kwargs,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_vit_stages), *((1,) * self.transformer_depth * 2))
        )

        # 将条件函数分为 vit_cond_fns 和 transformer_cond_fns
        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        # 重新排列视频数据的维度
        video = rearrange(video, 'b c f h w -> b f c h w')
        # 打包视频数据
        images, packed_shape = pack_one(video, '* c h w')

        # 使用 vit 模型处理图像数据
        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )

        # 解包 tokens 数据
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        # 使用 token_learner 处理 tokens 数据
        learned_tokens = self.token_learner(tokens)

        # 重新排列 learned_tokens 的维度
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')

        # 生成 causal attention mask
        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # 生成 sinusoidal positional embedding
        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)
        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

        # 进行 attention 操作
        attended_tokens = self.transformer(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = ~attn_mask)

        # 对 attended_tokens 进行池化操作
        pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)

        # 将池化后的结果传入到 logits 模型中
        logits = self.to_logits(pooled)
        return logits
```