# `stable-diffusion-webui\modules\sub_quadratic_attention.py`

```
# 导入必要的库
from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple

# 定义一个函数，用于截取张量的部分内容
def narrow_trunc(
    input: Tensor,
    dim: int,
    start: int,
    length: int
) -> Tensor:
    return torch.narrow(input, dim, start, length if input.shape[dim] >= start + length else input.shape[dim] - start)

# 定义一个命名元组，用于存储注意力机制的计算结果
class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor

# 定义一个类，用于汇总注意力机制的计算结果
class SummarizeChunk:
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...

# 定义一个类，用于计算查询向量的注意力机制
class ComputeQueryChunkAttn:
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor: ...

# 定义一个函数，用于汇总注意力机制的计算结果
def _summarize_chunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> AttnChunk:
    # 计算注意力权重
    attn_weights = torch.baddbmm(
        torch.zeros(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    # 获取最大分数
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    # 计算指数权重
    exp_weights = torch.exp(attn_weights - max_score)
    # 根据条件选择不同的计算方式，计算注意力值与数值的乘积
    exp_values = torch.bmm(exp_weights, value) if query.device.type == 'mps' else torch.bmm(exp_weights, value.to(exp_weights.dtype)).to(value.dtype)
    # 去除最大分数的维度，使其变为一维张量
    max_score = max_score.squeeze(-1)
    # 返回经过注意力计算后的结果，包括注意力值、权重之和以及最大分数
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)
# 定义一个函数，用于计算分块注意力机制
def _query_chunk_attention(
    query: Tensor,  # 查询张量
    key: Tensor,    # 键张量
    value: Tensor,  # 值张量
    summarize_chunk: SummarizeChunk,  # 摘要块函数
    kv_chunk_size: int,  # 键值分块大小
) -> Tensor:  # 返回值为张量
    batch_x_heads, k_tokens, k_channels_per_head = key.shape  # 获取键张量的形状信息
    _, _, v_channels_per_head = value.shape  # 获取值张量的形状信息

    # 定义一个内部函数，用于扫描分块
    def chunk_scanner(chunk_idx: int) -> AttnChunk:  # 返回值为注意力块
        key_chunk = narrow_trunc(  # 截取键张量的部分内容
            key,
            1,
            chunk_idx,
            kv_chunk_size
        )
        value_chunk = narrow_trunc(  # 截取值张量的部分内容
            value,
            1,
            chunk_idx,
            kv_chunk_size
        )
        return summarize_chunk(query, key_chunk, value_chunk)  # 调用摘要块函数

    # 使用列表推导式生成分块列表
    chunks: list[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, kv_chunk_size)
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))  # 合并分块内容
    chunk_values, chunk_weights, chunk_max = acc_chunk  # 获取分块值、权重和最大值

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)  # 计算全局最大值
    max_diffs = torch.exp(chunk_max - global_max)  # 计算最大值差异
    chunk_values *= torch.unsqueeze(max_diffs, -1)  # 更新分块值
    chunk_weights *= max_diffs  # 更新分块权重

    all_values = chunk_values.sum(dim=0)  # 汇总所有值
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)  # 汇总所有权重
    return all_values / all_weights  # 返回加权平均值

# TODO: 重构 CrossAttention#get_attention_scores 以与此共享代码
# 定义一个函数，用于计算注意力分数（不使用键值分块）
def _get_attention_scores_no_kv_chunking(
    query: Tensor,  # 查询张量
    key: Tensor,    # 键张量
    value: Tensor,  # 值张量
    scale: float,   # 缩放因子
) -> Tensor:  # 返回值为张量
    attn_scores = torch.baddbmm(  # 执行矩阵相乘并加法操作
        torch.zeros(1, 1, 1, device=query.device, dtype=query.dtype),  # 创建全零张量
        query,
        key.transpose(1,2),  # 转置键张量
        alpha=scale,  # 缩放因子
        beta=0,
    )
    attn_probs = attn_scores.softmax(dim=-1)  # 计算注意力概率
    del attn_scores  # 删除不再需要的张量
    hidden_states_slice = torch.bmm(attn_probs, value) if query.device.type == 'mps' else torch.bmm(attn_probs, value.to(attn_probs.dtype)).to(value.dtype)  # 计算隐藏状态切片
    return hidden_states_slice  # 返回隐藏状态切片

# 定义一个命名元组，用于存储扫描的分块信息
class ScannedChunk(NamedTuple):
    chunk_idx: int  # 分块索引
    attn_chunk: AttnChunk  # 注意力块

# 定义一个函数，用于执行高效的点积注意力计算
def efficient_dot_product_attention(
    query: Tensor,  # 查询张量
    # 定义函数参数 key，表示查询的键值对
    key: Tensor,
    # 定义函数参数 value，表示查询的值
    value: Tensor,
    # 定义函数参数 query_chunk_size，表示查询的块大小，默认为 1024
    query_chunk_size=1024,
    # 定义函数参数 kv_chunk_size，表示键值对的块大小，可选参数，默认为 None
    kv_chunk_size: Optional[int] = None,
    # 定义函数参数 kv_chunk_size_min，表示键值对的最小块大小，可选参数，默认为 None
    kv_chunk_size_min: Optional[int] = None,
    # 定义函数参数 use_checkpoint，表示是否使用检查点，默认为 True
    use_checkpoint=True,
    # 计算给定查询、键和数值的高效点积注意力
    # 这是 https://arxiv.org/abs/2112.05682v2 中提出的高效注意力版本，具有 O(sqrt(n)) 的内存需求
    # 参数：
    #   query: 用于计算注意力的查询，形状为 `[batch * num_heads, tokens, channels_per_head]`
    #   key: 用于计算注意力的键，形状为 `[batch * num_heads, tokens, channels_per_head]`
    #   value: 用于注意力的值，形状为 `[batch * num_heads, tokens, channels_per_head]`
    #   query_chunk_size: int: 查询块大小
    #   kv_chunk_size: Optional[int]: 键/值块大小。如果为 None，则默认为 sqrt(key_tokens)
    #   kv_chunk_size_min: Optional[int]: 键/值最小块大小。仅在 kv_chunk_size 为 None 时考虑。将 `sqrt(key_tokens)` 更改为 `max(sqrt(key_tokens), kv_chunk_size_min)`，以确保我们的块大小不会太小（更小的块 = 更多块 = 较少并发工作）
    #   use_checkpoint: bool: 是否使用检查点（推荐在训练时为 True，在推理时为 False）
    # 返回：
    #   形状为 `[batch * num_heads, query_tokens, channels_per_head]` 的输出
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, k_tokens, _ = key.shape
    scale = q_channels_per_head ** -0.5

    # 计算键/值块大小
    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    # 定义获取查询块的函数
    def get_query_chunk(chunk_idx: int) -> Tensor:
        return narrow_trunc(
            query,
            1,
            chunk_idx,
            min(query_chunk_size, q_tokens)
        )

    # 部分应用 _summarize_chunk 函数，设置比例为 scale
    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    # 如果 use_checkpoint 为 True，则使用检查点函数包装 summarize_chunk
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    # 定义一个函数 compute_query_chunk_attn，根据条件选择不同的计算方式
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale
    ) if k_tokens <= kv_chunk_size else (
        # 当每个查询块只有一个键值对块时的快速路径（这实际上是切片注意力）
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    # 如果查询令牌数量小于等于查询块大小
    if q_tokens <= query_chunk_size:
        # 当只有一个查询块时的快速路径
        return compute_query_chunk_attn(
            query=query,
            key=key,
            value=value,
        )

    # 创建一个与查询形状相同的全零张量
    res = torch.zeros_like(query)
    # 遍历每个查询块
    for i in range(math.ceil(q_tokens / query_chunk_size)):
        # 计算查询块的注意力分数
        attn_scores = compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key=key,
            value=value,
        )

        # 将注意力分数填充到结果张量的相应位置
        res[:, i * query_chunk_size:i * query_chunk_size + attn_scores.shape[1], :] = attn_scores

    # 返回结果张量
    return res
```