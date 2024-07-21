# `.\pytorch\torch\_inductor\fx_passes\fuse_attention.py`

```py
# 设置允许未类型化定义（用于类型检查工具mypy）
mypy: allow-untyped-defs
# 导入模块functools、inspect、logging、math
import functools
import inspect
import logging
import math

# 导入PyTorch模块torch及其attention模块中的sdpa_kernel和SDPBackend
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
# 导入特定路径下的模块counters
from ..._dynamo.utils import counters
# 导入pattern_matcher模块中的函数filter_nodes、fwd_only、gen_register_replacement、joint_fwd_bwd
from ..pattern_matcher import (
    filter_nodes,
    fwd_only,
    gen_register_replacement,
    joint_fwd_bwd,
)

# 设置日志记录器
log = logging.getLogger(__name__)
# 设置torch的aten操作为变量aten
aten = torch.ops.aten

# 根据torch版本的HIP属性条件定义函数_scaled_dot_product_attention
if torch.version.hip:

    def _scaled_dot_product_attention(*args, **kwargs):
        # 使用sdpa_kernel进行注意力计算，支持的后端有MATH和FLASH_ATTENTION
        with sdpa_kernel(backends=[SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
            return aten.scaled_dot_product_attention(*args, **kwargs)

else:
    # 否则使用torch的aten操作中的scaled_dot_product_attention
    _scaled_dot_product_attention = aten.scaled_dot_product_attention


# 定义_sfdp_pattern_1函数，根据query、key、value和inv_scale进行注意力模式计算
def _sfdp_pattern_1(query, key, value, inv_scale):
    return (
        # 计算query与key的点积，然后除以inv_scale，并对结果进行softmax处理，最后与value相乘
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale)
        .softmax(dim=-1)
        .matmul(value)
    )


# 定义_sfdp_replacement_1函数，根据query、key、value和inv_scale调用_scaled_dot_product_attention进行替换操作
def _sfdp_replacement_1(query, key, value, inv_scale):
    # 增加计数器中"inductor"下的"fuse_attention"键值对计数
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


# 定义_sfdp_pattern_2函数，根据query、key、value和scale_factor进行注意力模式计算
def _sfdp_pattern_2(query, key, value, scale_factor):
    return (
        # 计算query与key的点积，然后乘以scale_factor，并对结果进行softmax处理，最后与value相乘
        torch.matmul(query, key.transpose(-2, -1))
        .mul(scale_factor)
        .softmax(dim=-1)
        .matmul(value)
    )


# 定义_sfdp_replacement_2函数，根据query、key、value和scale_factor调用_scaled_dot_product_attention进行替换操作
def _sfdp_replacement_2(query, key, value, scale_factor):
    # 增加计数器中"inductor"下的"fuse_attention"键值对计数
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale_factor,
    )


# 定义_sfdp_pattern_3函数，根据query、key、value、inv_scale_factor和dropout_p进行注意力模式计算
def _sfdp_pattern_3(query, key, value, inv_scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        # 计算query与key的点积，然后除以inv_scale_factor，并对结果进行softmax处理，然后应用dropout，最后与value相乘
        torch.matmul(query, key.transpose(-2, -1))
        .div(inv_scale_factor)
        .softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


# 定义_sfdp_replacement_3函数，根据query、key、value、inv_scale_factor和dropout_p调用_scaled_dot_product_attention进行替换操作
def _sfdp_replacement_3(query, key, value, inv_scale_factor, dropout_p):
    # 增加计数器中"inductor"下的"fuse_attention"键值对计数
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


# 定义_sfdp_pattern_4函数，根据query、key、value、scale_factor和dropout_p进行注意力模式计算
def _sfdp_pattern_4(query, key, value, scale_factor, dropout_p):
    return torch.nn.functional.dropout(
        # 计算query与key的点积，然后乘以scale_factor，并对结果进行softmax处理，然后应用dropout，最后与value相乘
        torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(value)


# 定义_sfdp_replacement_4函数，根据query、key、value、scale_factor和dropout_p调用_scaled_dot_product_attention进行替换操作
def _sfdp_replacement_4(query, key, value, scale_factor, dropout_p):
    # 增加计数器中"inductor"下的"fuse_attention"键值对计数
    counters["inductor"]["fuse_attention"] += 1
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale_factor,
    )
def _sfdp_pattern_5(query, key, value, attn_mask):
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
    )
    # 计算注意力权重，通过 softmax 归一化得到每个位置的注意力权重
    return attn_weight @ value


def _sfdp_replacement_5(query, key, value, attn_mask):
    counters["inductor"]["fuse_attention"] += 1
    # 计数器更新，统计触发注意力融合的次数
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=0.0,
        is_causal=False,
    )


def _sfdp_pattern_6(query, key, value, attn_mask, dropout_p):
    attn_weight = torch.softmax(
        (query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))) + attn_mask, dim=-1
    )
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    # 应用 dropout 操作，以 dropout_p 的概率随机置零注意力权重
    return attn_weight @ value


def _sfdp_replacement_6(query, key, value, attn_mask, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    # 计数器更新，统计触发注意力融合的次数
    return _scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=dropout_p,
        is_causal=False,
    )


def _sfdp_pattern_7(query, key, value, dropout_p):
    # 在实际工作负载中，输入到矩阵乘法的顺序会被排列
    # 导致矩阵乘法扩展为一系列的扩展和克隆调用
    # 我们希望在模式跟踪过程中发生相同的操作
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, True)
    attn_weight = attn_weight.to(torch.float16)
    # 执行注意力权重计算，包括顺序调整和类型转换
    return attn_weight @ v


def _sfdp_replacement_7(query, key, value, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    # 计数器更新，统计触发注意力融合的次数
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=dropout_p,
        is_causal=False,
    )


def _sfdp_pattern_8(query, key, value):
    # 模式 7 的无 dropout 版本
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    div = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    div = div.to(torch.float32)
    attn_weight = torch.softmax(div, dim=-1)
    attn_weight = attn_weight.to(torch.float16)
    # 执行注意力权重计算，包括顺序调整和类型转换
    return attn_weight @ v


def _sfdp_replacement_8(query, key, value):
    counters["inductor"]["fuse_attention"] += 1
    # 计数器更新，统计触发注意力融合的次数
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    # 对输入的value张量进行维度置换，调整顺序为(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
    # 调用注意力机制函数 _scaled_dot_product_attention 进行注意力计算
    # 参数q为查询张量，k为键张量，v为值张量，attn_mask为注意力掩码（此处为None），dropout_p为dropout概率，is_causal表示是否是因果关系
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
def _sfdp_pattern_9(query, key, value, dropout_p):
    """
    计算注意力权重并应用到值向量上，使用了缩放的点积注意力机制。
    """
    q = query.permute(0, 2, 1, 3)  # 将查询张量进行维度置换
    k = key.permute(0, 2, 1, 3)    # 将键张量进行维度置换
    v = value.permute(0, 2, 1, 3)  # 将值张量进行维度置换
    q = q / math.sqrt(q.size(-1))  # 缩放查询张量
    div = q @ k.transpose(-2, -1)  # 计算点积，并进行转置
    div = div.to(torch.float32)    # 将点积结果转换为32位浮点数
    attn_weight = torch.softmax(div, dim=-1)  # 对点积结果进行softmax操作得到注意力权重
    attn_weight = torch.dropout(attn_weight, dropout_p, True)  # 应用dropout操作到注意力权重
    attn_weight = attn_weight.to(torch.float16)  # 将注意力权重转换为16位浮点数
    return attn_weight @ v  # 返回注意力权重与值的乘积结果


def _sfdp_replacement_9(query, key, value, dropout_p):
    """
    替代函数，使用封装的缩放点积注意力函数进行注意力计算。
    """
    counters["inductor"]["fuse_attention"] += 1  # 计数器递增
    q = query.permute(0, 2, 1, 3)  # 将查询张量进行维度置换
    k = key.permute(0, 2, 1, 3)    # 将键张量进行维度置换
    v = value.permute(0, 2, 1, 3)  # 将值张量进行维度置换
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # 注意力掩码，这里为None
        dropout_p=dropout_p,
        is_causal=False,
    )


def _sfdp_pattern_10(query, key, value):
    """
    无dropout版本的_sfdp_pattern_9函数。
    """
    q = query.permute(0, 2, 1, 3)  # 将查询张量进行维度置换
    k = key.permute(0, 2, 1, 3)    # 将键张量进行维度置换
    v = value.permute(0, 2, 1, 3)  # 将值张量进行维度置换
    q = q / math.sqrt(q.size(-1))  # 缩放查询张量
    div = q @ k.transpose(-2, -1)  # 计算点积，并进行转置
    div = div.to(torch.float32)    # 将点积结果转换为32位浮点数
    attn_weight = torch.softmax(div, dim=-1)  # 对点积结果进行softmax操作得到注意力权重
    attn_weight = attn_weight.to(torch.float16)  # 将注意力权重转换为16位浮点数
    return attn_weight @ v  # 返回注意力权重与值的乘积结果


def _sfdp_replacement_10(query, key, value):
    """
    替代函数，使用封装的缩放点积注意力函数进行注意力计算，无dropout版本。
    """
    counters["inductor"]["fuse_attention"] += 1  # 计数器递增
    q = query.permute(0, 2, 1, 3)  # 将查询张量进行维度置换
    k = key.permute(0, 2, 1, 3)    # 将键张量进行维度置换
    v = value.permute(0, 2, 1, 3)  # 将值张量进行维度置换
    return _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,  # 注意力掩码，这里为None
        dropout_p=0.0,
        is_causal=False,
    )


def _sfdp_pattern_11(query, key, value, inv_scale):
    """
    主要针对huggingface模型，使用了自定义的缩放点积注意力机制。
    """
    q = query.permute(0, 2, 1, 3)  # 将查询张量进行维度置换
    k = key.permute(0, 2, 1, 3)    # 将键张量进行维度置换
    v = value.permute(0, 2, 1, 3)  # 将值张量进行维度置换
    return torch.matmul(q, k.transpose(-2, -1)).div(inv_scale).softmax(dim=-1).matmul(v)  # 使用自定义缩放点积注意力机制计算结果


def _sfdp_replacement_11(query, key, value, inv_scale):
    """
    替代函数，使用封装的缩放点积注意力函数进行注意力计算，针对huggingface模型。
    """
    counters["inductor"]["fuse_attention"] += 1  # 计数器递增
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None,  # 注意力掩码，这里为None
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_12(query, key, value, inv_scale_factor, dropout_p):
    """
    使用torch.nn.functional的dropout函数对自定义缩放点积注意力机制结果进行处理。
    """
    q = query.permute(0, 2, 1, 3)  # 将查询张量进行维度置换
    k = key.permute(0, 2, 1, 3)    # 将键张量进行维度置换
    v = value.permute(0, 2, 1, 3)  # 将值张量进行维度置换
    return torch.nn.functional.dropout(
        torch.matmul(q, k.transpose(-2, -1)).div(inv_scale_factor).softmax(dim=-1),
        p=dropout_p,
    ).matmul(v)  # 对自定义缩放点积注意力机制结果应用dropout操作，然后与值向量进行乘积


def _sfdp_replacement_12(query, key, value, inv_scale_factor, dropout_p):
    """
    替代函数，使用封装的缩放点积注意力函数进行注意力计算，同时应用dropout，针对自定义缩放因子。
    """
    counters["inductor"]["fuse_attention"] += 1  # 计数器递增
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None,  # 注意力掩码，这里为None
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale_factor,
    )


def _sfdp_pattern_13(query, key, value, dropout_p):
    """
    待实现的函数，功能暂未说明。
    """
    pass
    # 计算注意力权重，使用query和key的乘积的softmax作为权重矩阵
    attn_weight = torch.bmm(query, key.transpose(1, 2)).softmax(dim=-1)
    # 对注意力权重进行dropout操作，以减少过拟合风险
    attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p)
    # 根据注意力权重对value进行加权求和，得到注意力机制加权后的输出
    return torch.bmm(attn_weight, value)
def _sfdp_replacement_13(query, key, value, dropout_p):
    # 增加计数器中的特定计数值
    counters["inductor"]["fuse_attention"] += 1
    # 调用缩放点积注意力函数，并返回结果（去除扩展的维度）
    return _scaled_dot_product_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        dropout_p=dropout_p,
        scale=1.0,
    ).squeeze(0)


def _sfdp_pattern_14(query, key, value, attn_mask, inv_scale):
    # 用于BertLarge
    # 需要进行排列以在图中创建克隆
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    # 执行注意力计算：计算注意力分数，应用softmax，然后进行加权求和
    return (
        (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask)
        .softmax(dim=-1)
        .matmul(v)
    )


def _sfdp_replacement_14(query, key, value, attn_mask, inv_scale):
    # 增加计数器中的特定计数值
    counters["inductor"]["fuse_attention"] += 1
    # 调用缩放点积注意力函数，同时进行维度转置和类型转换
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=query.dtype),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_15(query, key, value, attn_mask, inv_scale):
    # 用于DistilBert
    # 需要进行排列以在图中创建克隆
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    bs = q.size(0)
    k_len = k.size(-2)
    # 执行注意力计算：计算注意力分数，应用softmax，然后进行加权求和，应用掩码
    scores = q @ k.transpose(-2, -1)
    scores = scores.div(inv_scale)
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    return torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1) @ v


def _sfdp_replacement_15(query, key, value, attn_mask, inv_scale):
    # 增加计数器中的特定计数值
    counters["inductor"]["fuse_attention"] += 1
    bs = query.size(0)
    n_head = query.size(2)
    q_len = query.size(1)
    k_len = key.size(1)
    # 在缩放点积注意力函数中执行掩码的逻辑非操作
    attn_mask = (
        (attn_mask == 1).view((bs, 1, 1, k_len)).expand((bs, n_head, q_len, k_len))
    )
    # 调用缩放点积注意力函数，同时进行维度转置和类型转换
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=torch.bool),
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_16(query, key, value, attn_mask, inv_scale, dropout_p):
    # 用于BertLarge和带有dropout的情况
    # 需要进行排列以在图中创建克隆
    q = query.permute([0, 2, 1, 3])
    k = key.permute([0, 2, 1, 3])
    v = value.permute([0, 2, 1, 3])
    # 执行注意力计算：计算注意力分数，应用softmax，然后进行加权求和，带有dropout
    return (
        torch.nn.functional.dropout(
            (torch.matmul(q, k.transpose(-2, -1)).div(inv_scale) + attn_mask).softmax(
                dim=-1
            ),
            dropout_p,
        )
        .to(dtype=query.dtype)
        .matmul(v)
    )


def _sfdp_replacement_16(query, key, value, attn_mask, inv_scale, dropout_p):
    # 增加计数器中的特定计数值
    counters["inductor"]["fuse_attention"] += 1
    # 调用一个函数进行缩放点积注意力计算，并返回计算结果
    return _scaled_dot_product_attention(
        # 将查询向量按指定维度转置，通常是为了匹配键和值的维度以便进行注意力计算
        query.transpose(1, 2),
        # 将键向量按指定维度转置，通常是为了匹配查询和值的维度以便进行注意力计算
        key.transpose(1, 2),
        # 将值向量按指定维度转置，通常是为了匹配查询和键的维度以便进行注意力计算
        value.transpose(1, 2),
        # 注意力掩码，用于指定需要忽略的位置或者加权的位置
        attn_mask=attn_mask.to(dtype=query.dtype),
        # Dropout 概率，用于在计算注意力之前对输入进行随机失活
        dropout_p=dropout_p,
        # 是否是因果注意力，即只允许当前位置之前的信息影响当前位置
        is_causal=False,
        # 缩放因子，用于缩放点积注意力的输出，通常是点积注意力的维度倒数
        scale=1.0 / inv_scale,
    )
def _sfdp_pattern_17(query, key, value, attn_mask, inv_scale, dropout_p):
    # for DistilBert with dropout
    # 将查询张量按照指定维度顺序重新排列
    q = query.permute([0, 2, 1, 3])
    # 将键张量按照指定维度顺序重新排列
    k = key.permute([0, 2, 1, 3])
    # 将值张量按照指定维度顺序重新排列
    v = value.permute([0, 2, 1, 3])
    # 获取批量大小
    bs = q.size(0)
    # 获取键的长度
    k_len = k.size(-2)
    # 计算注意力分数
    scores = q @ k.transpose(-2, -1)
    # 将注意力分数除以缩放因子
    scores = scores.div(inv_scale)
    # 创建填充值，用于在注意力掩码中标记无效位置
    fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    # 根据注意力掩码生成掩码张量
    attn_mask = (attn_mask == 0).view((bs, 1, 1, k_len)).expand_as(scores)
    # 返回经过注意力权重和值的结果
    return (
        torch.nn.functional.dropout(
            torch.softmax(scores.masked_fill(attn_mask, fill_value), dim=-1), dropout_p
        )
        @ v
    )


def _sfdp_replacement_17(query, key, value, attn_mask, inv_scale, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    # 获取批量大小
    bs = query.size(0)
    # 获取头数
    n_head = query.size(2)
    # 获取查询长度
    q_len = query.size(1)
    # 获取键长度
    k_len = key.size(1)
    # 将注意力掩码按照指定的逻辑反转
    attn_mask = (
        (attn_mask == 1).view((bs, 1, 1, k_len)).expand((bs, n_head, q_len, k_len))
    )
    # 调用封装的函数进行缩放点积注意力计算
    return _scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask.to(dtype=torch.bool),
        dropout_p=dropout_p,
        is_causal=False,
        scale=1.0 / inv_scale,
    )


def _sfdp_pattern_18(query, key, value, causal_mask, dropout_p):
    # for hf_GPT2 with dropout (introduces clone node) for inference
    # it also returns permuted key & value
    # 将查询张量按照指定维度顺序重新排列
    query = query.permute([0, 2, 1, 3])
    # 将键张量按照指定维度顺序重新排列
    key = key.permute([0, 2, 1, 3])
    # 将值张量按照指定维度顺序重新排列
    value = value.permute([0, 2, 1, 3])
    # 计算注意力权重
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    # 计算缩放因子
    inv_scale = torch.full(
        [],
        value.size(-1) ** 0.5,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    # 将注意力权重除以缩放因子
    attn_weights = attn_weights.div(inv_scale)
    # 创建用于有因果掩码的填充值
    causal_mask_value = torch.full(
        (), torch.finfo(query.dtype).min, dtype=query.dtype, device=query.device
    )
    # 根据因果掩码过滤注意力权重
    attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
    # 返回经过注意力权重和值的结果，同时返回重新排列的键和值
    return (
        torch.nn.functional.dropout(attn_weights.softmax(dim=-1), dropout_p).matmul(
            value
        ),
        key,
        value,
    )


def _sfdp_replacement_18(query, key, value, causal_mask, dropout_p):
    counters["inductor"]["fuse_attention"] += 1
    # 将键按照指定维度顺序重新排列
    permuted_key = key.transpose(1, 2)
    # 将值按照指定维度顺序重新排列
    permuted_value = value.transpose(1, 2)
    # 调用封装的函数进行缩放点积注意力计算
    return (
        _scaled_dot_product_attention(
            query.transpose(1, 2),
            permuted_key,
            permuted_value,
            attn_mask=causal_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=1.0 / math.sqrt(value.size(-1)),
        ),
        permuted_key,
        permuted_value,
    )


def _sfdp_pattern_19(query, key, value, causal_mask, attn_mask, dropout_p):
    # for token-classification+gpt2 / text-generation+gpt2
    # This function is currently incomplete and requires additional annotation.
    # 此函数目前未完成，需要额外的注释。
    # 计算注意力权重，使用查询(query)和键(key)的乘积
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    
    # 计算缩放因子的倒数，用于缩放注意力权重
    inv_scale = torch.full(
        [],
        value.size(-1) ** 0.5,  # 缩放因子为值(value)的维度数的平方根倒数
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    
    # 缩放注意力权重
    attn_weights = attn_weights.div(inv_scale)
    
    # 创建一个因果掩码的值，设为查询(query)数据类型的最小值
    causal_mask_value = torch.full(
        (), torch.finfo(query.dtype).min,  # 使用查询(query)数据类型的最小值
        dtype=query.dtype,
        device=query.device,
    )
    
    # 应用因果掩码到注意力权重
    attn_weights = torch.where(causal_mask, attn_weights, causal_mask_value)
    
    # 加上注意力掩码
    attn_weights = attn_weights + attn_mask
    
    # 对注意力权重进行softmax操作，使其归一化
    attn_weights = attn_weights.softmax(dim=-1).type(value.dtype)
    
    # 对归一化后的注意力权重应用dropout，并将结果与值(value)矩阵相乘
    return torch.nn.functional.dropout(attn_weights, dropout_p).matmul(value)
# 增加"inductor"计数器中"fuse_attention"的计数
counters["inductor"]["fuse_attention"] += 1

# 创建一个值为负无穷大的张量，数据类型与query相同，设备与query相同
fill_value = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)

# 使用causal_mask更新attn_mask张量，如果causal_mask为True，则使用attn_mask的值，否则使用fill_value
attn_mask = torch.where(causal_mask, attn_mask, fill_value)

# 调用_scaled_dot_product_attention函数进行缩放点积注意力计算
return _scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
    is_causal=False,
    scale=1.0 / math.sqrt(value.size(-1)),
)

# 检查match中是否包含"query"、"key"和"value"关键字参数
assert all(k in match.kwargs for k in ("query", "key", "value"))

# 获取query、key、value的元数据中的值
query = match.kwargs["query"].meta["val"]
key = match.kwargs["key"].meta["val"]
value = match.kwargs["value"].meta["val"]

# 检查query、key、value的数据类型和设备是否一致，若不一致返回False
if not (query.dtype == key.dtype == value.dtype) or not (
    query.device == key.device == value.device
):
    return False

# 在match的节点中过滤出所有类型为aten.add.Tensor的节点
add_mask_node = filter_nodes(match.nodes, aten.add.Tensor)

# 如果add_mask_node列表不为空
if len(add_mask_node) > 0:
    # 获取第一个add_mask_node节点的第二个参数，通常为attn_mask张量
    attn_mask_node = add_mask_node[0].args[1]

    # 如果attn_mask_node不具有"meta"属性，返回False
    if not hasattr(attn_mask_node, "meta"):
        return False

    # 从attn_mask_node的元数据中获取值作为attn_mask张量
    attn_mask = attn_mask_node.meta["val"]  # type: ignore[union-attr]

    # 确保attn_mask的类型是torch.Tensor，并且其数据类型为query的数据类型或为torch.bool或为torch.float
    if (
        not isinstance(attn_mask, torch.Tensor)
        or not (
            attn_mask.dtype == query.dtype
            or attn_mask.dtype == torch.bool
            or attn_mask.dtype == torch.float
        )
        or query.device != attn_mask.device
    ):
        return False

# 返回True表示通过额外的参数检查
return True

# 定义一个函数_fn，用于执行特定模式的匹配检查
def _sfdp_extra_check(scale_factor_op=None, disable_cuda=False):
    def fn(match):
        # 如果disable_cuda为True，并且"query"关键字在match中，且"query"的设备包含"cuda"，则返回False
        if (
            disable_cuda
            and "query" in match.kwargs
            and "cuda" in str(match.kwargs["query"].meta["val"].device)
        ):
            return False
        
        # 如果scale_factor_op不为None，则从match的节点中过滤出符合scale_factor_op操作的节点列表
        if scale_factor_op is not None:
            scale_factor_node = filter_nodes(match.nodes, scale_factor_op)[0]
            
            # 注意：scale_factor_node的args[1]始终是当前模式的scale_factor
            scale_factor = scale_factor_node.args[1]
            
            # 确保scale_factor为float或int类型
            if not isinstance(scale_factor, (float, int)):
                return False
        
        # 调用_sfdp_params_check函数检查参数匹配
        return _sfdp_params_check(match)

    return fn

# 定义partialize_and_update_signature函数，类似于functools.partial但还更新返回函数的签名
def partialize_and_update_signature(func, **kwargs):
    """
    Equivalent to functools.partial but also updates the signature on returned function
    """
    # 获取原始函数的签名和参数
    original_sig = inspect.signature(func)
    parameters = original_sig.parameters

    # 构建新的参数字典，移除kwargs中的参数
    new_parameters = {
        key: value for key, value in parameters.items() if key not in kwargs
    }
    
    # 构建新的签名对象
    new_sig = inspect.Signature(parameters=list(new_parameters.values()))

    # 使用functools.partial创建一个新的部分应用函数
    partial_func = functools.partial(func, **kwargs)
    # 定义一个包装函数 wrapper，接受任意位置参数和关键字参数
    def wrapper(*args, **kwargs):
        # 调用 partial_func 函数，并传递所有位置参数和关键字参数
        return partial_func(*args, **kwargs)

    # 将 wrapper 函数的 __signature__ 属性设置为 new_sig，类型提示忽略类型检查
    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]
    
    # 将 wrapper 函数的 __name__ 属性设置为 func 函数的名称
    wrapper.__name__ = func.__name__

    # 返回定义好的 wrapper 函数作为装饰器的结果
    return wrapper
# 从本地导入模块中的patterns变量
from .joint_graph import patterns

# 检查当前环境是否支持CUDA，选择合适的设备
if torch.cuda.is_available():
    # 当CUDA可用时，选择设备为CUDA，解决https://github.com/pytorch/pytorch/issues/97894的问题
    device = "cuda"
else:
    # 当CUDA不可用时，选择设备为CPU
    device = "cpu"

# 创建一个functools.partial对象，用于生成一个形状为(2, 4, 8, 16)的空张量，设备由上一步确定，同时需要梯度信息
g_inp = functools.partial(torch.empty, (2, 4, 8, 16), device=device, requires_grad=True)

# 创建一个functools.partial对象，用于生成一个形状为(1, 1, 8, 8)的空张量，设备由上一步确定
b_inp = functools.partial(torch.empty, (1, 1, 8, 8), device=device)

# 创建一个functools.partial对象，用于生成一个形状为(2, 1, 1, 4)的空张量，设备由上一步确定
m_inp = functools.partial(torch.empty, (2, 1, 1, 4), device=device)

# 创建一个functools.partial对象，用于生成一个值为2.0的张量，设备由上一步确定
c_inp = functools.partial(torch.tensor, 2.0, device=device)

# 设置一个字典，包含一个名为'dropout_p'的键值对，值为0.113377，解决https://github.com/pytorch/pytorch/issues/97894的问题
d = {"dropout_p": 0.113377}

# 创建一个functools.partial对象，用于生成一个形状为(1024, 128, 128)的空张量，设备由上一步确定，同时需要梯度信息
g_3d_inp = functools.partial(torch.empty, (1024, 128, 128), device=device, requires_grad=True)

# 创建一个functools.partial对象，用于生成一个形状为(1, 4, 8, 16)的空张量，设备由上一步确定，同时需要梯度信息
g_bs1_inp = functools.partial(torch.empty, (1, 4, 8, 16), device=device, requires_grad=True)

# 创建一个functools.partial对象，用于生成一个形状为(1, 1, 1, 4)的空张量，设备由上一步确定
m_bs1_inp = functools.partial(torch.empty, (1, 1, 1, 4), device=device)

# 使用softmax函数时，如果输入是半精度浮点数(half)，会执行数据类型转换，但是如果是单精度浮点数(float)，则不会进行转换，因此需要为两种情况生成模式
# 这里我们生成两种不同数据类型的模式
```