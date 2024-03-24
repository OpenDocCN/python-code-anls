# `.\lucidrains\memory-efficient-attention-pytorch\memory_efficient_attention_pytorch\cosine_sim_flash_attention.py`

```py
# 导入所需的库
import math
import torch
from functools import partial
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd.function import Function

from einops import rearrange

# 定义常量
EPSILON = 1e-6

# 辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回其值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 对输入张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# FlashAttentionFunction 类，实现了自定义的 PyTorch 函数
class FlashAttentionFunction(Function):
    # 前向传播函数
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, scale, causal, q_bucket_size, k_bucket_size):
        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        k_len = k.shape[-2] # 在余弦相似度注意力中，行和受到键/值序列长度的限制

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)

        # 处理输入的 mask
        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.split(q_bucket_size, dim = -1)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            mask,
            all_row_sums.split(q_bucket_size, dim = -2),
        )

        # 遍历每个分块的行
        for ind, (qc, oc, row_mask, row_sums) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
            )

            # 遍历每个分块的列
            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                # 计算注意力权重
                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                # 如果存在行 mask，则进行填充
                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                # 如果启用因果注意力，并且当前位置不应该看到后续位置的信息，则进行填充
                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                attn_weights -= scale
                exp_weights = torch.exp(attn_weights)

                # 如果存在行 mask，则进行填充
                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.)

                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                oc.add_(exp_values / k_len)
                row_sums.add_(block_row_sums)

        # 保存参数和中间结果，用于反向传播
        ctx.args = (scale, causal, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums)

        # 对输出进行缩放
        o.mul_(k_len / all_row_sums)

        return o

    @staticmethod
    @torch.no_grad()
    # 定义一个反向传播函数，接收上下文和梯度作为参数
    def backward(ctx, do):
        # 解包上下文参数
        scale, causal, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l = ctx.saved_tensors

        # 获取设备信息
        device = q.device

        # 计算最大负值
        max_neg_value = -torch.finfo(q.dtype).max
        # 计算 q 和 k 的长度差
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        # 初始化梯度变量
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # 按照 q_bucket_size 分割张量
        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            l.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        # 遍历分割后的张量
        for ind, (qc, oc, doc, row_mask, lc, dqc) in enumerate(row_splits):
            # 计算 q 的起始索引
            q_start_index = ind * q_bucket_size - qk_len_diff

            # 按照 k_bucket_size 分割张量
            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
            )

            # 遍历分割后的张量
            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                # 计算 k 的起始索引
                k_start_index = k_ind * k_bucket_size

                # 计算注意力权重
                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                # 如果是因果注意力机制，进行掩码处理
                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                # 计算指数化的注意力权重
                exp_attn_weights = torch.exp(attn_weights - scale)

                # 如果存在行掩码，进行填充
                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.)

                # 计算概率
                p = exp_attn_weights / lc

                # 计算 dv_chunk
                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                # 计算 dp
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                # 计算 D
                D = (doc * oc).sum(dim = -1, keepdims = True)
                # 计算 ds
                ds = p * scale * (dp - D)

                # 计算 dq_chunk
                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                # 计算 dk_chunk
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                # 累加梯度
                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        # 返回梯度
        return dq, dk, dv, None, None, None, None, None
# 主类
# 闪光注意力机制用于余弦相似度注意力
# 相对较简单，不再需要担心 softmax 数值稳定性问题，行和受到限制

class FlashAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        scale = 16,
        heads = 8,
        dim_head = 64,
        causal = False,
        q_bucket_size = 512,
        k_bucket_size = 1024
    ):
        super().__init__()
        self.heads = heads

        self.scale = scale
        self.causal = causal

        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # 内存高效的注意力相关参数
        # 可以在前向传播中被覆盖
        self.q_bucket_size = q_bucket_size
        self.k_bucket_size = k_bucket_size

    def forward(
        self,
        x,
        context = None,
        mask = None,
        q_bucket_size = None,
        k_bucket_size = None,
    ):
        q_bucket_size = default(q_bucket_size, self.q_bucket_size)
        k_bucket_size = default(k_bucket_size, self.k_bucket_size)

        h = self.heads
        context = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q, k = map(l2norm, (q, k))

        out = FlashAttentionFunction.apply(q, k, v, mask, self.scale, self.causal, q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```