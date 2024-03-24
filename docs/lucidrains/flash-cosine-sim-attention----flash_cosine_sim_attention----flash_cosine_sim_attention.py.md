# `.\lucidrains\flash-cosine-sim-attention\flash_cosine_sim_attention\flash_cosine_sim_attention.py`

```
import os
import math
import importlib
from functools import partial, wraps

import torch
from torch import einsum
import torch.nn.functional as F
from torch.autograd import Function

# 导入版本信息
exec(open(os.path.dirname(os.path.abspath(__file__)) + '/version.py').read())

# 尝试导入 CUDA 扩展
try:
    cuda_pkg = importlib.import_module(__cuda_pkg_name__)

    # 从 CUDA 包中导入函数
    forward = cuda_pkg.forward
    backward = cuda_pkg.backward
    debug = cuda_pkg.debug

except ImportError:
    # 如果导入失败，则打印错误信息
    print('CUDA extension for flash-cosine-sim-attention was not compiled correctly - please run `pip install flash-cosine-sim-attention --force-reinstall --no-cache-dir`')

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 检查是否可以整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# CPU 上的 L2 范数计算
def l2norm_cpu(t):
    eps = 1e-12 if t.dtype == torch.float32 else 1e-3
    norm = t.norm(dim = -1)
    norm_clamped = torch.where(norm > eps, norm, eps)
    return t / norm_clamped[..., None]

# 对输入进行 L2 范数归一化
def l2norm(t):
    if t.data.is_cuda:
        return F.normalize(t, dim = -1)

    return l2norm_cpu(t)

# 对分组进行 L2 范数归一化
def grouped_l2norm(t, groups = 1):
    shape = t.shape
    dim = shape[-1]
    t = t.reshape(*shape[:-1], groups, dim // groups)
    t = l2norm(t)
    return t.reshape(shape)

# 对多个张量进行 L2 范数归一化
def l2norm_tensors(*tensors, groups = 1):
    assert len(tensors) > 0
    dtype = tensors[0].dtype

    fn = partial(grouped_l2norm, groups = groups)

    tensors = tuple(map(fn, tensors))
    tensors = tuple(map(lambda t: t.type(dtype), tensors))
    return tensors

# 原始的余弦相似度注意力机制

# b - batch
# h - heads
# i - src sequence length
# j - target sequence length
# d - feature dimension

def plain_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    attn_bias = None,
    scale = 8,
    groups = 1,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False

):
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    is_merged_batch_heads_query = q.ndim == 3
    single_head_kv = k.ndim == 3

    if is_merged_batch_heads_query:
        assert k.ndim == 3 and v.ndim ==3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'

        attn_bias_batch_dim = True
        q = q[:, None, ...]

    if l2norm_qk:
        q, k = l2norm_tensors(q, k, groups = groups)

    kv_einsum_eq = 'b j d' if single_head_kv else 'b h j d'
    sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
    sim = sim * scale

    if exists(attn_bias):
        attn_bias = attn_bias.unsqueeze(1 if attn_bias_batch_dim else 0)
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if causal:
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, mask_value)

    if exists(mask):
        sim = sim.masked_fill(~mask[:, None, None, :], mask_value)

    attn = sim.softmax(dim = -1)
    out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

    if is_merged_batch_heads_query:
        out = out.squeeze(1)

    return out

# CPU 上的前向传播

def flash_cosine_sim_attention_cpu(
    q, k, v,
    mask,
    attn_bias,
    scale,
    causal,
    attn_bias_batch_dim,
    row_tile_size = 512,
    col_tile_size = 512
):
    needs_backwards = any([exists(t) and t.requires_grad for t in (q, k, v, attn_bias)])

    assert not needs_backwards, 'cpu version does not support backwards'
    assert not (causal and exists(mask)), 'mask should not be supplied if causality is needed'

    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()

    is_merged_batch_heads_query = q.ndim == 3
    single_head_kv = k.ndim == 3

    shape = q.shape
    col_seq_len = k.shape[-2]
    row_seq_len = q.shape[-2]
    seq_len_diff = col_seq_len - row_seq_len
    # 计算行方向上的瓦片数量
    row_tiles = math.ceil(row_seq_len / row_tile_size)
    # 计算列方向上的瓦片数量
    col_tiles = math.ceil(col_seq_len / col_tile_size)
    # 获取数据类型 q 的最小负值
    max_neg_value = -torch.finfo(q.dtype).max

    # 如果合并了批次和头部的查询，则确保键和值也只有3个维度
    if is_merged_batch_heads_query:
        assert k.ndim == 3 and v.ndim ==3, 'if batch and heads are merged for queries, keys and values must also similarly have only 3 dimensions'

        # 在批次维度上添加一个维度
        attn_bias_batch_dim = True
        q = q.unsqueeze(1)

    # 如果存在注意力偏置
    if exists(attn_bias):
        # 在适当的维度上添加一个维度
        attn_bias = attn_bias.unsqueeze(1 if attn_bias_batch_dim else 0)

    # 根据是否为单头键值对，设置矩阵乘法的公式
    kv_einsum_eq = 'b j d' if single_head_kv else 'b h j d'

    # 循环遍历行和列

    # 创建一个与 q 相同形状的全零张量
    o = torch.zeros_like(q)
    # 创建一个与 q 形状除了最后一个维度为1的张量
    l = torch.zeros((*q.shape[:-1], 1))

    # 准备掩码

    # 如果不存在掩码，则创建与列瓦片数量相同数量的 None
    if not exists(mask):
        mask = (None,) * col_tiles
    else:
        # 在适当的维度上添加一个维度，并按列瓦片大小拆分
        mask = mask[:, None, None, :]
        mask = mask.split(col_tile_size, dim = -1)

    # 如果不存在注意力偏置，则创建与行瓦片数量相同数量的 None
    if not exists(attn_bias):
        attn_bias = (None,) * row_tiles
    else:
        # 按行瓦片大小拆分
        attn_bias = attn_bias.split(row_tile_size, dim = -2)

    # 按行瓦片大小拆分 q, o, l 和 attn_bias
    row_splits = zip(
        q.split(row_tile_size, dim = -2),
        o.split(row_tile_size, dim = -2),
        l.split(row_tile_size, dim = -2),
        attn_bias
    )

    # 遍历行拆分
    for ind, (qc, oc, lc, bc) in enumerate(row_splits):
        row_chunk_size = qc.shape[-2]
        q_start_index = ind * row_tile_size + seq_len_diff

        # 如果不存在 bc，则创建与列瓦片数量相同数量的 None
        if not exists(bc):
            bc = (None,) * col_tiles
        else:
            # 按列瓦片大小拆分
            bc = bc.split(col_tile_size, dim = -1)

        # 按列瓦片大小拆分 k, v, mask 和 bc
        col_splits = zip(
            k.split(col_tile_size, dim = -2),
            v.split(col_tile_size, dim = -2),
            mask,
            bc
        )

        # 遍历列拆分
        for k_ind, (kc, vc, maskc, bias) in enumerate(col_splits):
            col_chunk_size = kc.shape[-2]
            k_start_index = k_ind * col_tile_size

            # 如果是因果的，并且 q_start_index 大于等于 (k_start_index + col_tile_size - 1)，则跳过
            if causal and q_start_index >= (k_start_index + col_tile_size - 1):
                continue

            # 计算注意力权重
            attn_weights = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', qc, kc) * scale

            # 如果存在偏置，则加上偏置
            if exists(bias):
                attn_weights += bias

            # 如果存在掩码，则用最大负值填充不满足掩码条件的位置
            if exists(maskc):
                attn_weights.masked_fill_(~maskc, max_neg_value)

            # 如果是因果的，并且 q_start_index 小于 (k_start_index + col_tile_size - 1)
            if causal and q_start_index < (k_start_index + col_tile_size - 1):
                # 创建一个因果掩码
                causal_mask = torch.ones((row_chunk_size, col_chunk_size), dtype = torch.bool).triu(q_start_index - k_start_index + 1)
                attn_weights.masked_fill_(causal_mask, max_neg_value)

            # 计算指数权重
            exp_weights = torch.exp(attn_weights - scale)

            # 如果存在掩码，则用 0 填充不满足掩码条件的位置
            if exists(maskc):
                exp_weights.masked_fill_(~maskc, 0.)

            # 计算指数值
            exp_values = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', exp_weights, vc)

            # 更新输出张量和权重总和张量
            oc.add_(exp_values)
            lc.add_(exp_weights.sum(dim = -1, keepdim = True))
    
    # 对输出张量除以权重总和张量，并返回重塑后的结果
    o.div_(l.clamp(min = 1e-12))
    return o.reshape(shape).type(dtype)
# 主要类

class FlashCosineSimAttention(Function):
    # 前向传播函数
    @staticmethod
    def forward(
        ctx,
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    ):
        # 调用前向传播函数计算输出
        o, inv_l, should_backwards = forward(
            q, k, v,
            mask,
            attn_bias,
            attn_bias_batch_dim,
            scale,
            causal
        )

        # 如果不需要反向传播，则直接返回输出
        if not should_backwards:
            return o

        # 保存需要反向传播的信息
        ctx.should_backwards = should_backwards

        ctx.save_for_backward(o, inv_l, q, k, v, mask, attn_bias)

        ctx.params = (
            scale,
            causal,
            attn_bias_batch_dim
        )

        return o

    # 反向传播函数
    @staticmethod
    def backward(ctx, do):
        assert ctx.should_backwards

        o, inv_l, q, k, v, mask, attn_bias = ctx.saved_tensors

        (
            scale,
            causal,
            attn_bias_batch_dim
        ) = ctx.params

        # 调用反向传播函数计算梯度
        dq, dk, dv, db = backward(
            do, o, inv_l,
            q, k, v,
            mask,
            attn_bias,
            attn_bias_batch_dim,
            scale,
            causal
        )

        return dq, dk, dv, None, db, None, None, None, None, None, None, None, None, None, None

# 使用 CUDA 实现的 FlashCosineSimAttention 类
flash_cosine_sim_attention_cuda = FlashCosineSimAttention.apply

# 包装函数

def flash_cosine_sim_attention(
    q,
    k,
    v,
    mask = None,
    attn_bias = None,
    scale = 8,
    groups = 1,
    causal = False,
    l2norm_qk = True,
    attn_bias_batch_dim = False
):
    # 如果需要对输入进行 L2 归一化，则调用 l2norm_tensors 函数
    if l2norm_qk:
        q, k = l2norm_tensors(q, k, groups = groups)

    # 根据输入是否在 CUDA 上选择使用 CUDA 还是 CPU 实现的函数
    fn = flash_cosine_sim_attention_cuda if q.data.is_cuda else flash_cosine_sim_attention_cpu

    # 调用实现函数计算输出
    o = fn(
        q, k, v,
        mask,
        attn_bias,
        scale,
        causal,
        attn_bias_batch_dim
    )

    return o
```