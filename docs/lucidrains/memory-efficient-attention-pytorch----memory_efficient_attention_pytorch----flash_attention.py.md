# `.\lucidrains\memory-efficient-attention-pytorch\memory_efficient_attention_pytorch\flash_attention.py`

```py
# 导入数学库和 PyTorch 库
import math
import torch
# 导入 partial 函数
from functools import partial
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum
# 从 torch.autograd.function 模块中导入 Function 类
from torch.autograd.function import Function
# 从 einops 库中导入 rearrange 函数

from einops import rearrange

# 定义常量 EPSILON
EPSILON = 1e-10

# 定义辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# flash attention 前向和后向

# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf

# 定义 FlashAttentionFunction 类，继承自 Function 类
class FlashAttentionFunction(Function):
    # 静态方法，用 @torch.no_grad() 装饰
    @staticmethod
    @torch.no_grad()
    # 前向传播函数，接收参数 q, k, v, mask, causal, q_bucket_size, k_bucket_size
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 1 in the v2 paper """

        # 获取设备信息
        device = q.device
        # 获取最大负值
        max_neg_value = -torch.finfo(q.dtype).max
        # 计算 q 和 k 的长度差
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        # 初始化输出 o，所有行的和和最大值
        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device=device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device=device)

        # 缩放因子
        scale = (q.shape[-1] ** -0.5)

        # 计算行和列的分块数量
        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        # 处理 mask
        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b n -> b 1 1 n')

        if not exists(mask):
            col_masks = (None,) * num_col_tiles
            mask = (col_masks,) * num_row_tiles 
        else:
            mask = ((mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split(q_bucket_size, dim=-2)
            mask = tuple(((row_mask,) * num_col_tiles) if row_mask.shape[-1] == 1 else row_mask.split(k_bucket_size, dim=-1) for row_mask in mask)

        # 按行分块
        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            # 按列分块
            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                row_mask
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                # 计算注意力权重
                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(col_mask):
                    attn_weights.masked_fill_(~col_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_weights = torch.exp(attn_weights - new_row_maxes)

                if exists(col_mask):
                    exp_weights.masked_fill_(~col_mask, 0.)

                block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                oc.mul_(exp_row_max_diff).add_(exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

            oc.div_(row_sums)

        lse = all_row_sums.log() + all_row_maxes

        # 保存参数并返回输出 o
        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    # 静态方法，用 @torch.no_grad() 装饰
    @staticmethod
    @torch.no_grad()
    # 定义一个向后传播函数，实现 v2 论文中的算法 2
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        # 从上下文中获取参数
        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        # 获取计算设备
        device = q.device

        # 获取最大负值
        max_neg_value = -torch.finfo(q.dtype).max
        # 计算 q 和 k 的长度差
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        # 初始化 dq, dk, dv
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # 按照 q_bucket_size 分割 q, o, do, mask, lse, dq
        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        # 遍历每个分割后的行
        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            # 按照 k_bucket_size 分割 k, v, dk, dv, row_mask
            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
                row_mask
            )

            # 遍历每个分割后的列
            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                # 计算注意力权重
                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                # 如果是因果注意力机制，并且 q_start_index 小于 (k_start_index + k_bucket_size - 1)
                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                # 计算概率
                p = torch.exp(attn_weights - lsec)

                # 如果存在列掩码，则将概率中对应位置置零
                if exists(col_mask):
                    p.masked_fill_(~col_mask, 0.)

                # 计算 dv_chunk
                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                # 计算 D 和 ds
                D = (doc * oc).sum(dim = -1, keepdims = True)
                ds = p * scale * (dp - D)

                # 计算 dq_chunk, dk_chunk
                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                # 累加到梯度中
                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        # 返回梯度 dq, dk, dv
        return dq, dk, dv, None, None, None, None
# 主类 FlashAttention，用于实现注意力机制
# 在纯 PyTorch 中实现会比在 CUDA 中实现慢很多
# 用于调试和教育目的

class FlashAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入维度
        heads = 8,  # 头数
        dim_head = 64,  # 每个头的维度
        causal = False,  # 是否使用因果注意力
        q_bucket_size = 512,  # 查询桶大小
        k_bucket_size = 1024  # 键值桶大小
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)  # 查询线性层
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)  # 键值线性层
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出线性层

        # 内存高效的注意力相关参数
        # 可以在前向传播中被覆盖
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        x,  # 输入张量
        context = None,  # 上下文张量
        mask = None,  # 掩码张量
        q_bucket_size = None,  # 查询桶大小
        k_bucket_size = None,  # 键值桶大小
    ):
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)  # 设置查询桶大小
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)  # 设置键值桶大小

        h = self.heads
        context = default(context, x)  # 如果上下文为空，则使用输入张量作为上下文

        q = self.to_q(x)  # 计算查询张量
        k, v = self.to_kv(context).chunk(2, dim = -1)  # 计算键值张量并分割为键和值

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))  # 重排张量形状

        out = FlashAttentionFunction.apply(q, k, v, mask, self.causal, q_bucket_size, k_bucket_size)  # 调用自定义的注意力函数

        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排输出张量形状
        return self.to_out(out)  # 返回输出结果
```