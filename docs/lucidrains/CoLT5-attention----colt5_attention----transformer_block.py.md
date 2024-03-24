# `.\lucidrains\CoLT5-attention\colt5_attention\transformer_block.py`

```py
import math
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum

from typing import Tuple, Optional

from local_attention import LocalMHA
from einops import rearrange, repeat, pack, unpack

from colt5_attention.attend import Attend

# helper functions

# 检查变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 检查是否可以被整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 将张量打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包的张量解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor, seq_len

    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    padded_tensor = F.pad(tensor, (*pad_offset, 0, remainder), value=value)
    return padded_tensor, seq_len

# 从张量中按照索引获取数据
def batched_gather(x, indices):
    batch_range = create_batch_range(indices, indices.ndim - 1)
    return x[batch_range, indices]

# 返回输入张量本身
def identity(t):
    return t

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim=-1)

# tensor helpers

# 创建批次范围
def create_batch_range(t, right_pad_dims=1):
    b, device = t.shape[0], t.device
    batch_range = torch.arange(b, device=device)
    pad_dims = ((1,) * right_pad_dims)
    return batch_range.reshape(-1, *pad_dims)

# rotary positional embeddign
# https://arxiv.org/abs/2104.09864

# 旋转位置嵌入
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs

# 旋转张量的一半
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# normalization

# RMS 归一化
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma

# modules

# 前馈神经网络
def FeedForward(dim, mult=4):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Linear(dim_hidden, dim)
    )

# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        use_flash=False,
        prenorm=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_hidden = dim_head * heads

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.attend = Attend(use_flash=use_flash)

        self.to_qkv = nn.Linear(dim, dim_hidden * 3, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim, bias=False)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        # 获取查询、键、值
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # 注意力
        out = self.attend(q, k, v)

        # 合并头部
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        multiply_keys_by_score=False,
        use_flash=False
        # 调用父类的初始化方法
        super().__init__()
        # 初始化头数和头维度的比例
        self.heads = heads
        self.scale = dim_head ** -0.5
        # 计算隐藏层维度
        dim_hidden = dim_head * heads

        # 设置是否使用乘以键的分数
        self.multiply_keys_by_score = multiply_keys_by_score

        # 初始化 RMS 归一化层
        self.norm = RMSNorm(dim)
        # 初始化空键值对参数
        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        # 初始化 Attend 层
        self.attend = Attend(use_flash = use_flash)

        # 初始化将输入转换为查询向量的线性层
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        # 初始化将输入转换为键值对向量的线性层
        self.to_kv = nn.Linear(dim, dim_hidden * 2, bias = False)
        # 初始化将输出转换为隐藏层向量的线性层
        self.to_out = nn.Linear(dim_hidden, dim, bias = False)

    # 前向传播方法
    def forward(
        self,
        x,
        context = None,
        mask = None,
        normalized_scores_kv = None,
        normalized_scores_q = None,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None
        ):
            """
            einops:
            b - batch
            h - heads, or number of heads per route
            r - routing dimension, for routing different sets of key / values - should be more expressive
            n - sequence dimension
            d - head dimension
            i - input model dimension
            """

            # 获取输入张量 x 的 batch 大小和头数
            batch, h = x.shape[0], self.heads

            # 对输入张量 x 进行归一化处理
            x = self.norm(x)

            # 如果存在上下文张量 context，则对其进行归一化处理
            if exists(context):
                context = self.norm(context)

            # 如果不存在上下文张量，则将其设为输入张量 x
            context = default(context, x)

            # 如果上下文张量的维度为 3，则在第二维度上添加一个维度
            if context.ndim == 3:
                context = rearrange(context, 'b n d -> b 1 n d')

            # 如果存在归一化后的得分张量 normalized_scores_kv 且为 torch.Tensor 类型
            if exists(normalized_scores_kv) and isinstance(normalized_scores_kv, torch.Tensor):
                # 如果 normalized_scores_kv 的维度为 2，则在第二维度上添加一个维度
                if normalized_scores_kv.ndim == 2:
                    normalized_scores_kv = rearrange(normalized_scores_kv, 'b n -> b 1 n')

                # 重新排列 normalized_scores_kv 的维度
                normalized_scores_kv = rearrange(normalized_scores_kv, 'b r n -> b r 1 n 1')

            # 获取上下文张量的 key / value 路由数
            num_kv_routes = context.shape[1]

            # 获取查询张量 q
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            # 如果存在归一化后的查询得分张量 normalized_scores_q 且为 torch.Tensor 类型
            if exists(normalized_scores_q) and isinstance(normalized_scores_q, torch.Tensor):
                # 将查询张量 q 乘以归一化后的查询得分张量 normalized_scores_q
                q = q * rearrange(normalized_scores_q, 'b n -> b 1 n 1')

            # 处理 key / value，使用路由维度，在路由之间分配头数
            assert divisible_by(h, num_kv_routes), 'number of heads must be divisible by the number of key / value routes'
            heads_per_route = h // num_kv_routes

            # 重新排列 key / value 权重张量的维度
            kv_weight = rearrange(self.to_kv.weight, '(r h d) i -> r h d i', h = heads_per_route, r = num_kv_routes)

            # 计算 key / value
            kv = einsum('r h d i, b r n i -> b r h n d', kv_weight, context)
            k, v = kv.chunk(2, dim = -1)

            # 如果存在归一化后的 key / value 得分张量
            if exists(normalized_scores_kv):
                # 将 value 乘以归一化后的 key / value 得分张量
                v = v * normalized_scores_kv

                # 如果需要将 key 乘以得分
                if self.multiply_keys_by_score:
                    k = k * normalized_scores_kv

            # 如果存在旋转嵌入
            if exists(rotary_emb):
                q_rotary_emb, k_rotary_emb = rotary_emb
                q = apply_rotary_pos_emb(q_rotary_emb, q)

                # 如果 k_rotary_emb 的维度为 4
                if k_rotary_emb.ndim == 4:
                    k_rotary_emb = repeat(k_rotary_emb, 'b 1 n d -> b r 1 n d', r = k.shape[1])

                k = apply_rotary_pos_emb(k_rotary_emb, k)

            # 合并 key / value 的路由维度和头数
            k, v = map(lambda t: rearrange(t, 'b r h n d -> b (r h) n d'), (k, v))

            # 空 key / value
            nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = batch), self.null_kv)

            # 拼接 key / value
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

            # 掩码
            if exists(mask):
                if mask.ndim == 3:
                    mask = repeat(mask, 'b r j -> b (r h) 1 j', h = heads_per_route)
                else:
                    mask = rearrange(mask, 'b j -> b 1 1 j')

                mask = F.pad(mask, (1, 0), value = True)

            # 注意力
            out = self.attend(q, k, v, mask = mask)

            # 合并头数
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)
# 导入所需的模块和函数
from colt5_attention.coor_descent import coor_descent
# 定义一个命名元组，用于存储路由器返回的结果
RouterReturn = namedtuple('RouterReturn', ['indices', 'scores', 'routed_tokens', 'routed_mask'])

# 定义一个路由器类，实现坐标下降算法
class CoordinateDescentRouter(nn.Module):
    """
    from Wright et al. https://arxiv.org/abs/1502.04759
    then adopted by https://arxiv.org/abs/2211.01267 for multi-vector document retrieval by Qian et al
    finally, used successfully by this paper for routing to heavy branch attention / feedforward
    """

    def __init__(
        self,
        dim,
        straight_through = True,
        n_iters = 20,                   # 使用20次迭代，采用ε-scaling
        fetch_k_ratio = 9 / 8,          # 在论文中，稍微增加k（乘以这个比率）以获得更好的学习效果
        eps = 0.03,                     # 坐标下降的ε值。在最近的一篇论文中，文本使用0.03，语音使用1.0
        eps_decay = 0.7,
        eps_init = 4.,
        num_routing_tokens = 1,
        learned_routing_tokens = False,
        use_triton = False,
        cosine_sim_routing = False,
        cosine_sim_scale = 8,
        route_block_size = None,
        triton_checkpoint_segments = None # 是否将坐标下降重新计算为多个段，使用4和50次迭代，向后加速3倍，牺牲前向和一些内存以保存初始a和b
    ):
        super().__init__()
        assert fetch_k_ratio >= 1.

        self.n_iters = n_iters
        self.fetch_k_ratio = fetch_k_ratio

        self.coor_descent = coor_descent

        # 与ε-scaling相关的超参数

        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_init = eps_init

        if use_triton:
            from colt5_attention.triton_coor_descent import triton_coor_descent
            triton_checkpoint_segments = default(triton_checkpoint_segments, n_iters // 5)
            self.coor_descent = partial(triton_coor_descent, checkpoint_segments = triton_checkpoint_segments)

        self.is_one_routing_token = num_routing_tokens == 1
        self.num_routing_tokens = num_routing_tokens

        self.route_block_size = route_block_size

        self.routing_token = nn.Parameter(torch.randn(num_routing_tokens, dim)) if not learned_routing_tokens else None
        self.straight_through = straight_through

        # 是否使用余弦相似度进行路由

        self.cosine_sim_routing = cosine_sim_routing
        self.cosine_sim_scale = cosine_sim_scale

    # 将路由后的结果还原到原始张量中
    def route_back(self, src, routed_tokens, indices):
        batch_range = create_batch_range(routed_tokens)
        src[batch_range, indices] = routed_tokens
        return src

    # 前向传播函数
    def forward(
        self,
        x,
        *,
        num_tokens,
        mask = None,
        random_route = False,
        routing_tokens = None,
        keep_one_route_dim = False  # 如果只有一个路由，是否保持维度
# 主要类

# 有条件的路由前馈网络
class ConditionalRoutedFeedForward(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4,
        router_straight_through = True, # 确保所有归一化分数为1，仍可微分
        router_kwargs: dict = {},
        use_triton = False
    ):
        super().__init__()
        self.num_heavy_tokens = num_heavy_tokens

        if use_triton:
            router_kwargs = {**router_kwargs, 'use_triton': True}

        # 初始化路由器
        self.router = CoordinateDescentRouter(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        # 初始化轻量级前馈网络和重量级前馈网络
        self.light_ff = FeedForward(dim, light_ff_mult)
        self.heavy_ff = FeedForward(dim, heavy_ff_mult)

    # 前向传播函数
    def forward(
        self,
        x,
        mask = None,
        num_heavy_tokens = None
        ):
        # 获取输入张量的设备信息和重要令牌数量
        device, num_heavy_tokens = x.device, default(num_heavy_tokens, self.num_heavy_tokens)

        # 轻量级前馈网络看到所有令牌（隐藏维度仅为模型维度的1/2）
        light_out = self.light_ff(x)

        # 适当路由令牌到重型分支
        indices, normalized_scores, routed_tokens, _ = self.router(x, num_tokens=num_heavy_tokens, mask=mask)

        # 仅使用路由的令牌进行更重的分支
        routed_tokens_out = self.heavy_ff(routed_tokens) * rearrange(normalized_scores, '... -> ... 1')

        # 将重型前馈分支的输出散回
        if exists(indices):
            heavy_out = torch.zeros_like(x)
            heavy_out = self.router.route_back(heavy_out, routed_tokens_out, indices)
        else:
            heavy_out = routed_tokens_out

        # 将轻量级和重型分支相加并返回结果
        return light_out + heavy_out
class ConditionalRoutedAttention(nn.Module):
    # 定义一个条件路由注意力的类，继承自 nn.Module
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens_q,
        num_heavy_tokens_kv,
        num_routed_kv = 1,
        light_dim_head = 64,
        light_heads = 8,
        light_window_size = 128,        # 每个令牌左右各看 ~ 64 个令牌
        heavy_dim_head = 64,
        heavy_heads = 8,
        router_straight_through = True, # 确保所有归一化分数为 1，仍可微分
        router_kwargs: dict = {},
        multiply_keys_by_score = False,
        multiply_queries_by_score = False,
        use_triton = False,
        use_null_q_tokens = True,
        use_flash_attn = False,
        rotary_emb = False
    ):
        super().__init__()

        if use_triton:
            router_kwargs = {**router_kwargs, 'use_triton': True}

        self.num_heavy_tokens_q = num_heavy_tokens_q
        self.num_heavy_tokens_kv = num_heavy_tokens_kv

        self.multiply_queries_by_score = multiply_queries_by_score

        self.light_attn = LocalMHA(
            dim = dim,
            dim_head = light_dim_head,
            heads = light_heads,
            window_size = light_window_size // 2,
            prenorm = True,
            causal = False,
            use_rotary_pos_emb = False,
            look_backward = 1,
            look_forward = 1
        )

        self.null_q_token = None
        if use_null_q_tokens:
            self.null_q_token = nn.Parameter(torch.randn(dim)) # 为未被路由器选择的查询令牌提供一个学习到的输出嵌入

        self.q_router = CoordinateDescentRouter(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.kv_router = CoordinateDescentRouter(
            dim = dim,
            num_routing_tokens = num_routed_kv,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = heavy_dim_head,
            heads = heavy_heads,
            multiply_keys_by_score = multiply_keys_by_score,
            use_flash = use_flash_attn
        )

        # 旋转嵌入

        self.rotary_emb = RotaryEmbedding(heavy_dim_head) if rotary_emb else None

    def forward(
        self,
        x,
        *,
        num_heavy_tokens_q = None,
        num_heavy_tokens_kv = None,
        mask = None
        ):
        # 解包输入张量的批次大小、序列长度和设备信息
        batch, seq, device = *x.shape[:2], x.device

        # 设置查询和键值中的重要令牌数量，默认为模型中定义的数量
        num_heavy_tokens_q = default(num_heavy_tokens_q, self.num_heavy_tokens_q)
        num_heavy_tokens_kv = default(num_heavy_tokens_kv, self.num_heavy_tokens_kv)

        # 轻量级局部注意力机制查看有限上下文中的所有令牌

        light_out = self.light_attn(x, mask = mask)

        # 适当路由令牌以供重型分支使用

        indices_q, normalized_scores_q, routed_tokens_q, _ = self.q_router(x, num_tokens = num_heavy_tokens_q, mask = mask)
        indices_kv, normalized_scores_kv, routed_tokens_kv, routed_tokens_kv_mask = self.kv_router(x, num_tokens = num_heavy_tokens_kv, mask = mask)

        # 如果指定了旋转嵌入，则获取旋转嵌入

        rotary_emb = None

        if exists(self.rotary_emb):
            seq_rotary_emb = self.rotary_emb(seq)
            q_rotary_emb = rearrange(seq_rotary_emb[indices_q], 'b n d -> b 1 n d') if exists(indices_q) else seq_rotary_emb
            k_rotary_emb = rearrange(seq_rotary_emb[indices_kv], '... n d -> ... 1 n d') if exists(indices_kv) else seq_rotary_emb
            rotary_emb = (q_rotary_emb, k_rotary_emb)

        # 使用仅路由令牌的重型分支

        routed_tokens_out = self.heavy_attn(
            routed_tokens_q,
            mask = routed_tokens_kv_mask,
            context = routed_tokens_kv,
            rotary_emb = rotary_emb,
            normalized_scores_kv = normalized_scores_kv,
            normalized_scores_q = normalized_scores_q if self.multiply_queries_by_score else None
        )

        routed_tokens_out = routed_tokens_out * rearrange(normalized_scores_q, '... -> ... 1')

        # 将重型分支的输出散回

        if exists(indices_q):
            if exists(self.null_q_token):
                heavy_out = rearrange(self.null_q_token, 'd -> 1 1 d')
                heavy_out = heavy_out.expand_as(x).clone()
            else:
                heavy_out = torch.zeros_like(x)

            heavy_out = self.q_router.route_back(heavy_out, routed_tokens_out, indices_q)
        else:
            heavy_out = routed_tokens_out

        # 汇总轻量级和重量级分支的输出

        return light_out + heavy_out
# 定义一个条件路由的图像特征映射注意力模块
class ConditionalRoutedImageAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens_q,
        num_heavy_tokens_kv,
        num_routed_kv = 1,
        light_dim_head = 64,
        light_heads = 8,
        light_window_size = 128,        # 每个令牌左右各看大约 64 个令牌
        heavy_dim_head = 64,
        heavy_heads = 8,
        router_straight_through = True, # 确保所有归一化分数为 1，仍然可微分
        router_kwargs: dict = {},
        multiply_keys_by_score = False,
        multiply_queries_by_score = False,
        use_triton = False,
        use_null_q_tokens = True,
        use_flash_attn = False,
        channel_first = False
    ):
        super().__init__()
        self.channel_first = channel_first

        # 如果使用 Triton，设置 router_kwargs 中的 'use_triton' 为 True
        if use_triton:
            router_kwargs = {**router_kwargs, 'use_triton': True}

        self.num_heavy_tokens_q = num_heavy_tokens_q
        self.num_heavy_tokens_kv = num_heavy_tokens_kv

        self.multiply_queries_by_score = multiply_queries_by_score

        self.light_window_size = light_window_size

        # 创建轻量级自注意力模块
        self.light_attn = SelfAttention(
            dim = dim,
            dim_head = light_dim_head,
            heads = light_heads,
            prenorm = True
        )

        self.null_q_token = None
        # 如果使用空查询令牌，为其创建一个学习到的输出嵌入
        if use_null_q_tokens:
            self.null_q_token = nn.Parameter(torch.randn(dim))

        # 创建查询路由器
        self.q_router = CoordinateDescentRouter(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        # 创建键值路由器
        self.kv_router = CoordinateDescentRouter(
            dim = dim,
            num_routing_tokens = num_routed_kv,
            straight_through = router_straight_through,
            **router_kwargs
        )

        # 创建重量级注意力模块
        self.heavy_attn = Attention(
            dim = dim,
            dim_head = heavy_dim_head,
            heads = heavy_heads,
            multiply_keys_by_score = multiply_keys_by_score,
            use_flash = use_flash_attn
        )

    def forward(
        self,
        x,
        *,
        num_heavy_tokens_q = None,
        num_heavy_tokens_kv = None,
        mask = None
        ):
        # 断言输入张量 x 的维度为 4
        assert x.ndim == 4
        # 获取输入张量 x 的批大小、设备信息、是否通道优先、光窗口大小
        batch, device, channel_first, w = x.shape[0], x.device, self.channel_first, self.light_window_size

        # 如果通道优先，则重新排列张量 x 的维度
        if channel_first:
            x = rearrange(x, 'b d ... -> b ... d')

        # 设置轻量级注意力机制中的重要令牌数量
        num_heavy_tokens_q = default(num_heavy_tokens_q, self.num_heavy_tokens_q)
        num_heavy_tokens_kv = default(num_heavy_tokens_kv, self.num_heavy_tokens_kv)

        # 轻量级局部注意力机制看到有限上下文中的所有令牌

        # 重新排列输入张量 x，以便进行轻量级注意力计算
        light_input = rearrange(x, 'b (h p1) (w p2) d -> b h w (p1 p2) d', p1 = w, p2 = w)
        x, ps = pack_one(light_input, '* n d')

        # 使用轻量级注意力机制计算输出
        light_out = self.light_attn(x)
        light_out = unpack_one(light_out, ps, '* n d')
        light_out = rearrange(light_out, 'b h w (p1 p2) d -> b (h p1) (w p2) d', p1 = w, p2 = w)

        # 为重型分支适当路由令牌

        # 使用查询路由器对输入张量 x 进行路由，获取相关信息
        indices_q, normalized_scores_q, routed_tokens_q, _ = self.q_router(x, num_tokens = num_heavy_tokens_q, mask = mask)
        # 使用键值路由器对输入张量 x 进行路由，获取相关信息
        indices_kv, normalized_scores_kv, routed_tokens_kv, routed_tokens_kv_mask = self.kv_router(x, num_tokens = num_heavy_tokens_kv, mask = mask)

        # 使用仅包含路由令牌的重型注意力机制进行计算

        routed_tokens_out = self.heavy_attn(
            routed_tokens_q,
            mask = routed_tokens_kv_mask,
            context = routed_tokens_kv,
            normalized_scores_kv = normalized_scores_kv,
            normalized_scores_q = normalized_scores_q if self.multiply_queries_by_score else None
        )

        routed_tokens_out = routed_tokens_out * rearrange(normalized_scores_q, '... -> ... 1')

        # 将重型分支的输出散回

        # 如果存在空查询令牌，则使用该令牌进行填充
        if exists(self.null_q_token):
            heavy_out = rearrange(self.null_q_token, 'd -> 1 1 d')
            heavy_out = heavy_out.expand_as(x).clone()
        else:
            heavy_out = torch.zeros_like(x)

        heavy_out = self.q_router.route_back(heavy_out, routed_tokens_out, indices_q)

        heavy_out = unpack_one(heavy_out, ps, '* n d')
        heavy_out = rearrange(heavy_out, 'b h w (p1 p2) d -> b (h p1) (w p2) d', p1 = w, p2 = w)

        # 将轻量级和重型分支的输出相加

        out = light_out + heavy_out

        # 如果通道优先，则重新排列输出张量的维度
        if channel_first:
            out = rearrange(out, 'b ... d -> b d ...')

        # 返回最终输出
        return out
# 定义条件路由的自回归注意力模块
class ConditionalRoutedAutoregressiveAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_heavy_tokens_q,
        num_heavy_tokens_kv,
        num_routed_kv = 1,
        light_dim_head = 64,
        light_heads = 8,
        light_window_size = 128,        # 每个标记左右各看到 ~ 64 个标记
        heavy_window_size = None,
        heavy_dim_head = 64,
        heavy_heads = 8,
        router_straight_through = True, # 确保所有归一化分数为 1，仍可微分
        router_kwargs: dict = {},
        multiply_keys_by_score = False,
        multiply_queries_by_score = False,
        use_triton = False,
        use_null_q_tokens = True,
        use_flash_attn = False,
        rotary_emb = False
    ):
        super().__init__()

        if use_triton:
            router_kwargs = {**router_kwargs, 'use_triton': True}

        self.num_heavy_tokens_q = num_heavy_tokens_q
        self.num_heavy_tokens_kv = num_heavy_tokens_kv

        self.multiply_queries_by_score = multiply_queries_by_score

        self.heavy_window_size = default(heavy_window_size, light_window_size)

        self.light_attn = LocalMHA(
            dim = dim,
            dim_head = light_dim_head,
            heads = light_heads,
            window_size = light_window_size,
            prenorm = True,
            causal = True,
            exact_windowsize = False,
            use_rotary_pos_emb = False
        )

        self.null_q_token = None
        if use_null_q_tokens:
            self.null_q_token = nn.Parameter(torch.randn(dim)) # 为未被路由器选择的查询标记提供一个学习到的输出嵌入

        self.q_router = CoordinateDescentRouter(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.kv_router = CoordinateDescentRouter(
            dim = dim,
            num_routing_tokens = num_routed_kv,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = heavy_dim_head,
            heads = heavy_heads,
            multiply_keys_by_score = multiply_keys_by_score,
            use_flash = use_flash_attn
        )

        # 旋转嵌入

        self.rotary_emb = RotaryEmbedding(heavy_dim_head) if rotary_emb else None

    def forward(
        self,
        x,
        *,
        num_heavy_tokens_q = None,
        num_heavy_tokens_kv = None,
        random_route = False
# 调整条件路由的自注意力以适应交叉注意力

# 定义条件路由的交叉注意力模块
class ConditionalRoutedCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens_q,
        num_tokens_kv,
        num_sets_kv = 1,                # 如果设置大于 1，将路由多组键/值，每组大小为 num_tokens_kv，使用这么多路由标记
        dim_head = 64,
        heads = 8,
        router_straight_through = True, # 确保所有归一化分数为 1，仍可微分
        router_kwargs: dict = {},
        kv_routing_tokens = 1,
        multiply_keys_by_score = False,
        use_triton = False,
        use_null_q_tokens = True,
        use_flash_attn = False,
        route_block_size = None
    ):
        super().__init__()

        if use_triton:
            router_kwargs = {**router_kwargs, 'use_triton': True}

        self.num_tokens_q = num_tokens_q
        self.num_tokens_kv = num_tokens_kv

        self.null_q_token = None
        if use_null_q_tokens:
            self.null_q_token = nn.Parameter(torch.randn(dim)) # 为未被路由器选择的查询标记提供一个学习到的输出嵌入

        self.q_router = CoordinateDescentRouter(
            dim = dim,
            straight_through = router_straight_through,
            **router_kwargs
        )

        self.kv_router = CoordinateDescentRouter(
            dim = dim,
            straight_through = router_straight_through,
            num_routing_tokens = kv_routing_tokens,
            route_block_size = route_block_size,
            **router_kwargs
        )

        self.heavy_attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            multiply_keys_by_score = multiply_keys_by_score,
            use_flash = use_flash_attn
        )

    def forward(
        self,
        x,
        context,
        *,
        num_tokens_q = None,
        num_tokens_kv = None,
        mask = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        # route the queries

        query_length = x.shape[-2]
        num_tokens_q = default(num_tokens_q, self.num_tokens_q)

        indices_q, normalized_scores_q, routed_tokens_q, _ = self.q_router(x, num_tokens = num_tokens_q, mask = mask)

        # route the long contexts

        key_value_length = context.shape[-2]
        num_tokens_kv = default(num_tokens_kv, self.num_tokens_kv)

        routed_tokens_kv = context
        routed_tokens_kv_mask = context_mask
        normalized_scores_kv = None

        should_route_kv = key_value_length > num_tokens_kv

        if should_route_kv:
            indices_kv, normalized_scores_kv, routed_tokens_kv, routed_tokens_kv_mask = self.kv_router(context, num_tokens = num_tokens_kv, mask = context_mask)

        # do the heavier branch with only routed tokens

        routed_tokens_out = self.heavy_attn(
            routed_tokens_q,
            mask = routed_tokens_kv_mask,
            context = routed_tokens_kv,
            normalized_scores_kv = normalized_scores_kv
        )

        if should_route_queries:
            routed_tokens_out = routed_tokens_out * rearrange(normalized_scores_q, '... -> ... 1')

        # early return if queries did not undergo routing

        if not should_route_queries:
            return routed_tokens_out

        # otherwise, scatter back the query outputs

        if exists(self.null_q_token):
            out = rearrange(self.null_q_token, 'd -> 1 1 d')
            out = out.expand_as(x).clone()
        else:
            out = torch.zeros_like(x)

        if exists(indices_q):
            out = self.q_router.route_back(out, routed_tokens_out, indices_q)

        return out
# 定义一个名为 ConditionalRoutedTransformerBlock 的类，继承自 nn.Module
class ConditionalRoutedTransformerBlock(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        *,
        num_heavy_attn_tokens_q,
        num_heavy_attn_tokens_kv,
        num_routed_kv = 1,
        num_heavy_ff_tokens,
        light_dim_head = 64,
        light_heads = 8,
        light_window_size = 128,
        heavy_dim_head = 64,
        heavy_heads = 8,
        light_ff_mult = 0.5,
        heavy_ff_mult = 4,
        router_straight_through = True,
        router_kwargs: dict = {},
        multiply_keys_by_score = False,
        multiply_queries_by_score = False,
        use_triton = False,
        use_null_q_tokens = True,
        use_flash_attn = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 创建 ConditionalRoutedFeedForward 对象并赋值给 self.conditional_ff
        self.conditional_ff = ConditionalRoutedFeedForward(
            dim,
            num_heavy_tokens = num_heavy_ff_tokens,
            light_ff_mult = light_ff_mult,
            heavy_ff_mult = heavy_ff_mult,
            router_straight_through = router_straight_through,
            router_kwargs = router_kwargs,
            use_triton = use_triton
        )

        # 创建 ConditionalRoutedAttention 对象并赋值给 self.conditional_attn
        self.conditional_attn = ConditionalRoutedAttention(
            dim,
            light_dim_head = light_dim_head,
            light_heads = light_heads,
            light_window_size = light_window_size,
            heavy_dim_head = heavy_dim_head,
            heavy_heads = heavy_heads,
            num_heavy_tokens_q = num_heavy_attn_tokens_q,
            num_heavy_tokens_kv = num_heavy_attn_tokens_kv,
            num_routed_kv = num_routed_kv,
            router_straight_through = router_straight_through,
            router_kwargs = router_kwargs,
            multiply_keys_by_score = multiply_keys_by_score,
            multiply_queries_by_score = multiply_queries_by_score,
            use_triton = use_triton,
            use_null_q_tokens = use_null_q_tokens,
            use_flash_attn = use_flash_attn
        )

    # 前向传播函数，接受多个参数
    def forward(
        self,
        x,
        mask = None,
        num_heavy_attn_tokens_q = None,
        num_heavy_attn_tokens_kv = None,
        num_heavy_ff_tokens = None
    ):
        # 调用 self.conditional_attn 进行注意力计算，并将结果与输入 x 相加
        x = self.conditional_attn(x, mask = mask, num_heavy_tokens_q = num_heavy_attn_tokens_q, num_heavy_tokens_kv = num_heavy_attn_tokens_kv) + x
        # 调用 self.conditional_ff 进行前馈计算，并将结果与输入 x 相加
        x = self.conditional_ff(x, mask = mask, num_heavy_tokens = num_heavy_ff_tokens) + x
        # 返回计算结果
        return x
```