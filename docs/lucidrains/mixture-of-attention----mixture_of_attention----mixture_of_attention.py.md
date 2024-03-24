# `.\lucidrains\mixture-of-attention\mixture_of_attention\mixture_of_attention.py`

```
# 导入数学库
import math

# 导入 PyTorch 库
import torch
import torch.nn.functional as F
from torch import Tensor, nn, einsum

# 导入类型提示
from typing import Tuple, Optional

# 导入 einops 库中的函数
from einops import rearrange, repeat, reduce, pack, unpack

# 导入自定义模块
from mixture_of_attention.attend import Attend
from mixture_of_attention.rotary_emb import apply_rotary_pos_emb

from local_attention import LocalMHA

from colt5_attention import CoordinateDescentRouter

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将张量打包成指定模式的形状
def pack_one(t, pattern):
    return pack([t], pattern)

# 将打包后的张量解包成原始形状
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor, seq_len

    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    padded_tensor = F.pad(tensor, (*pad_offset, 0, remainder), value = value)
    return padded_tensor, seq_len

# 归一化

# RMS 归一化模块
class RMSNorm(nn.Module):
    def __init__(self, dim, groups = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(groups, dim, 1))

    def forward(self, x):
        normed = F.normalize(x, dim = -2)
        return normed * self.scale * self.gamma

# 注意力机制

# 注意力模块
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        causal = False,
        groups = 1, # 定义专家的数量
        dropout = 0.,
        flash = False,
        prenorm = False
    ):
        super().__init__()
        self.heads = heads
        self.groups = groups

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.norm = RMSNorm(dim, groups = groups) if prenorm else nn.Identity()
        self.context_norm = RMSNorm(dim_context, groups = groups) if prenorm else nn.Identity()

        self.attend = Attend(
            dropout = dropout,
            causal = causal,
            flash = flash
        )

        # 空键/值，用于防止一行全部被掩码掉

        self.null_kv = nn.Parameter(torch.randn(2, groups, heads, 1, dim_head))

        # 利用卷积组并行处理专家

        self.to_q = nn.Conv1d(dim * groups, dim_inner * groups, 1, bias = False, groups = groups)
        self.to_kv = nn.Conv1d(dim_context * groups, dim_inner * 2 * groups, 1, bias = False, groups = groups)
        self.to_out = nn.Conv1d(dim_inner * groups, dim * groups, 1, bias = False, groups = groups)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        queries_scale = None,
        keys_scale = None,
        values_scale = None,
        output_scale = None,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None
        ):
            """
            einops
            b - batch
            g - groups
            n - sequence
            d - feature dimension
            """
            # 获取输入张量的形状信息
            b, g, h = x.shape[0], self.groups, self.heads

            # 判断是否只有一个专家
            one_expert = x.ndim == 3

            # 如果只有一个专家，则将其维度扩展为4维
            if one_expert:
                assert g == 1
                x = rearrange(x, 'b n d -> b 1 n d')

            # 断言输入张量为4维
            assert x.ndim == 4
            # 断言输入张量的第二维为groups
            assert x.shape[1] == g

            # 将groups折叠到特征维度中，以便通过分组卷积一次处理
            x = rearrange(x, 'b g n d -> b g d n')

            # 处理交叉注意力的上下文
            if exists(context):
                context_one_expert = context.ndim == 3

                if context_one_expert:
                    assert g == 1
                    context = rearrange(context, 'b n d -> b 1 n d')

                assert context.ndim == 4
                assert context.shape[1] == g

                context = rearrange(context, 'b g n d -> b g d n')

            # 如果没有传入context，则使用输入张量x
            context = default(context, x)

            # 处理mask
            if exists(mask):
                if mask.ndim == 2:
                    mask = repeat(mask, 'b n -> (b g) n', g = g)
                elif mask.ndim == 3:
                    mask = rearrange(mask, 'b g n -> (b g) n')

                mask = F.pad(mask, (1, 0), value = True)

            # 如果适用，进行预归一化
            x = self.norm(x)
            context = self.context_norm(context)

            # 将groups折叠到维度中以进行分组卷积
            x, context = map(lambda t: rearrange(t, 'b g d n -> b (g d) n'), (x, context))

            # 获取查询、键、值
            q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = 1))

            # 拆分头部并将groups合并到批次中
            q, k, v = map(lambda t: rearrange(t, 'b (g h d) n -> b g h n d', h = h, g = g), (q, k, v))

            # 旋转嵌入
            if exists(rotary_emb):
                q_rotary_emb, k_rotary_emb = rotary_emb

                if q_rotary_emb.ndim > 2:
                    q_rotary_emb = rearrange(q_rotary_emb, 'b g n d -> b g 1 n d')

                if k_rotary_emb.ndim > 2:
                    k_rotary_emb = rearrange(k_rotary_emb, 'b g n d -> b g 1 n d')

                q = apply_rotary_pos_emb(q_rotary_emb, q)
                k = apply_rotary_pos_emb(k_rotary_emb, k)

            # 如果传入了queries_scale，则给查询加权
            if exists(queries_scale):
                q = q * queries_scale

            # 如果传入了keys_scale，则给键加权
            if exists(keys_scale):
                k = k * keys_scale

            # 如果传入了values_scale，则给值加权
            if exists(values_scale):
                v = v * values_scale

            # 将groups合并到批次中
            q, k, v = map(lambda t: rearrange(t, 'b g ... -> (b g) ...'), (q, k, v))

            # 连接空键/值，以防止一行中所有元素都被屏蔽并节省大量麻烦
            nk, nv = map(lambda t: repeat(t, 'g h 1 d -> (b g) h 1 d', b = b), self.null_kv)

            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

            # 注意力机制
            out = self.attend(q, k, v, mask = mask)

            # 合并头部输出
            out = rearrange(out, '(b g) h n d -> b (g h d) n', g = g)

            out = self.to_out(out)

            out = rearrange(out, 'b (g d) n -> b g n d', g = g)

            # 如果只有一个专家，则将其维度还原为3维
            if one_expert:
                out = rearrange(out, 'b 1 n d -> b n d')

            # 如果传入了output_scale，则给输出加权
            if exists(output_scale):
                out = out * output_scale

            return out
# 定义混合注意力机制的类
class MixtureOfAttention(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        *,
        num_routed_queries,
        num_routed_key_values,
        dim_context = None,
        local_attn = False,
        local_attn_window_size = None,
        num_experts = 2,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_triton = True,
        flash_attn = True,
        prenorm = True,
        average_routed = False,
        **kwargs
    ):
        super().__init__()
        dim_context = default(dim_context, dim)
        self.num_routed_queries = num_routed_queries
        self.num_routed_key_values = num_routed_key_values

        # 如果不是本地注意力，创建一个参数化的空路由令牌
        self.null_routed_token = nn.Parameter(torch.randn(1, 1, dim)) if not local_attn else None

        self.average_routed = average_routed

        self.local_attn = None

        # 如果使用本地注意力，创建本地多头注意力对象
        if local_attn:
            assert exists(local_attn_window_size)
            self.local_attn = LocalMHA(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                prenorm = prenorm,
                window_size = local_attn_window_size
            )

        # 创建查询路由器对象
        self.query_router = CoordinateDescentRouter(
            dim,
            num_routing_tokens = num_experts,
            use_triton = use_triton,
            **kwargs
        )

        # 创建键值路由器对象
        self.key_value_router = CoordinateDescentRouter(
            dim_context,
            num_routing_tokens = num_experts,
            use_triton = use_triton,
            **kwargs
        )

        # 创建注意力对象
        self.attn = Attention(
            dim = dim,
            dim_context = dim_context,
            dim_head = dim_head,
            heads = heads,
            groups = num_experts,
            dropout = dropout,
            flash = flash_attn,
            prenorm = prenorm
        )

    # 返回模型参数所在的设备
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数
    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        num_routed_queries = None,
        num_routed_key_values = None,
        rotary_emb = None
        ):
            # 设置路由查询数量为默认值或者传入的值
            num_routed_queries = default(num_routed_queries, self.num_routed_queries)
            # 设置路由键值对数量为默认值或者传入的值
            num_routed_key_values = default(num_routed_key_values, self.num_routed_key_values)

            # 判断是否进行跨注意力
            is_cross_attn = exists(context)

            # 断言不能同时存在本地注意力和跨注意力
            assert not (exists(self.local_attn) and is_cross_attn), 'cannot do cross attention with local attention (only for self attention)'

            if not is_cross_attn:
                # 如果不是跨注意力，则使用自注意力
                context = x
                context_mask = mask

            # 获取查询索引、查询分数、查询、查询掩码
            query_indices, query_scores, queries, query_mask = self.query_router(x, mask = mask, num_tokens = num_routed_queries, keep_one_route_dim = True)
            query_scores = rearrange(query_scores, 'b g n -> b g n 1')

            # 获取键值索引、键值分数、键值、键值掩码
            kv_indices, key_value_scores, key_values, key_value_mask = self.key_value_router(context, mask = context_mask, num_tokens = num_routed_key_values, keep_one_route_dim = True)
            key_value_scores = rearrange(key_value_scores, 'b g n -> b g 1 n 1')

            # 旋转嵌入

            if exists(rotary_emb):
                assert not is_cross_attn, 'rotary embedding should not be used for cross attending'
                q_rotary_emb = rotary_emb[query_indices] if exists(query_indices) else rotary_emb
                k_rotary_emb = rotary_emb[kv_indices] if exists(kv_indices) else rotary_emb
                rotary_emb = (q_rotary_emb, k_rotary_emb)

            # 注意力计算

            attn_out = self.attn(
                queries,
                rotary_emb = rotary_emb,
                context = key_values,
                mask = key_value_mask,
                values_scale = key_value_scores,
                output_scale = query_scores
            )

            local_out = None
            if exists(self.local_attn):
                local_out = self.local_attn(x, mask = mask)

            need_route_queries = exists(query_indices)

            if not need_route_queries:
                out = attn_out

                if exists(local_out):
                    local_out = rearrange(local_out, 'b n d -> b 1 n d')
                    out = torch.cat((local_out, out), dim = 1)

                out = reduce(attn_out, 'b e n d -> b n d', 'mean')

                if exists(mask):
                    out = out.masked_fill(~mask[..., None], 0.)

                return out

            out = torch.zeros_like(x)
            counts = torch.zeros(x.shape[:-1], device = x.device)

            query_indices = rearrange(query_indices, 'b g n -> b (g n)')
            attn_out = rearrange(attn_out, 'b g n d -> b (g n) d')

            expanded_query_indices = repeat(query_indices, 'b n -> b n d', d = x.shape[-1])

            attn_out_summed = out.scatter_add(1, expanded_query_indices, attn_out)

            ones = torch.ones(attn_out.shape[:-1], device = self.device)

            if exists(query_mask):
                ones = ones * rearrange(query_mask, 'b g n -> b (g n)')

            counts = counts.scatter_add(1, query_indices, ones)
            counts = rearrange(counts, '... -> ... 1')

            has_unrouted = not exists(local_out)

            if not has_unrouted:
                counts = counts + 1
                attn_out_summed = attn_out_summed + local_out
            else:
                not_routed_mask = counts == 0
                attn_out_summed = attn_out_summed.masked_fill(not_routed_mask, 0.)

            out = attn_out_summed

            # 如果需要，进行平均

            if self.average_routed:
                out = out / counts.clamp(min = 1e-5)

            # 对于未路由的位置，使用学习到的路由令牌而不是仅仅是0

            if has_unrouted:
                out = torch.where(
                    not_routed_mask,
                    self.null_routed_token,
                    out,
                )

            if exists(mask):
                out = out.masked_fill(~mask[..., None], 0.)

            return out
# 定义一个混合自回归注意力模型类
class MixtureOfAutoregressiveAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_routed_queries,  # 路由查询的数量
        num_routed_key_values,  # 路由键值对的数量
        local_attn_window_size,  # 本地注意力窗口大小
        routed_window_size = None,  # 路由窗口大小，默认为None
        num_experts = 2,  # 专家数量，默认为2
        dim_head = 64,  # 头维度，默认为64
        heads = 8,  # 头数，默认为8
        dropout = 0.,  # 丢弃率，默认为0
        use_triton = False,  # 是否使用 Triton，默认为False
        flash_attn = True,  # 是否使用 Flash 注意力，默认为True
        prenorm = True,  # 是否使用预归一化，默认为True
        average_routed = False,  # 是否平均路由，默认为False
        **kwargs
    ):
        super().__init__()
        self.num_routed_queries = num_routed_queries  # 初始化路由查询数量
        self.num_routed_key_values = num_routed_key_values  # 初始化路由键值对数量

        self.num_experts = num_experts  # 初始化专家数量
        self.null_tokens = nn.Parameter(torch.randn(num_experts, dim))  # 初始化空令牌

        routed_window_size = default(routed_window_size, local_attn_window_size)  # 设置路由窗口大小为默认值或本地注意力窗口大小

        self.routed_window_size = routed_window_size  # 初始化路由窗口大小
        self.average_routed = average_routed  # 初始化是否平均路由

        # 创建本地多头自注意力模块
        self.local_attn = LocalMHA(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            prenorm = prenorm,
            causal = True,
            window_size = local_attn_window_size
        )

        # 创建查询路由器
        self.query_router = CoordinateDescentRouter(
            dim,
            num_routing_tokens = num_experts,
            use_triton = use_triton,
            **kwargs
        )

        # 创建键值路由器
        self.key_value_router = CoordinateDescentRouter(
            dim,
            num_routing_tokens = num_experts,
            use_triton = use_triton,
            **kwargs
        )

        # 创建注意力模块
        self.attn = Attention(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            groups = num_experts,
            dropout = dropout,
            flash = flash_attn,
            prenorm = prenorm
        )

    # 定义设备属性
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数
    def forward(
        self,
        x,
        rotary_emb = None,
        num_routed_queries = None,
        num_routed_key_values = None
```