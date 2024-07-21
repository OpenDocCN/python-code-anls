# `.\pytorch\torch\utils\flop_counter.py`

```
# 设置允许未标注的函数定义，用于类型检查
mypy: allow-untyped-defs

# 导入torch库
import torch
# 从torch.utils._pytree中导入相关函数：tree_map, tree_flatten, tree_unflatten
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
# 从.module_tracker模块中导入ModuleTracker类
from .module_tracker import ModuleTracker
# 导入类型相关的模块：List, Any, Dict, Optional, Union, Tuple, Iterator
from typing import List, Any, Dict, Optional, Union, Tuple, Iterator
# 导入defaultdict类
from collections import defaultdict
# 从torch.utils._python_dispatch中导入TorchDispatchMode类
from torch.utils._python_dispatch import TorchDispatchMode
# 从torch._decomp中导入register_decomposition函数
from torch._decomp import register_decomposition
# 导入prod函数
from math import prod
# 导入wraps函数
from functools import wraps
# 导入warnings模块
import warnings

# 设置模块可导出的公共标识符
__all__ = ["FlopCounterMode", "register_flop_formula"]

# 导入torch的aten操作
aten = torch.ops.aten

# 定义函数get_shape，根据输入对象返回其形状
def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

# 定义一个空的字典，用于记录FLOP计算公式
flop_registry: Dict[Any, Any] = {}

# 定义装饰器函数shape_wrapper，用于将输入参数映射为其形状，并调用原函数
def shape_wrapper(f):
    @wraps(f)
    def nf(*args, out_val=None, **kwargs):
        # 将args、kwargs、out_val中的每个元素都映射为其形状
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out_val))
        # 调用原函数，并传入形状信息
        return f(*args, out_shape=out_shape, **kwargs)
    return nf

# 定义装饰器函数register_flop_formula，用于注册FLOP计算公式
def register_flop_formula(targets, get_raw=False):
    def register_fun(flop_formula):
        if not get_raw:
            # 如果get_raw为False，则将FLOP计算公式包装为shape_wrapper修饰后的版本
            flop_formula = shape_wrapper(flop_formula)
        # 将flop_formula注册到targets中，使用flop_registry作为注册表
        register_decomposition(targets, registry=flop_registry, unsafe=True)(flop_formula)
        return flop_formula

    return register_fun

# 定义计算矩阵乘法FLOP的函数，注册到torch的aten.mm操作上
@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for matmul."""
    # 输入参数应该是一个长度为2的列表，分别表示两个矩阵的形状
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    # 返回矩阵乘法操作的FLOP计算结果
    # 注意：理论上应该是2 * k - 1，这里是简化版本
    return m * n * 2 * k

# 定义计算addmm操作FLOP的函数，注册到torch的aten.addmm操作上
@register_flop_formula(aten.addmm)
def addmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for addmm."""
    # 调用mm_flop函数计算矩阵乘法操作的FLOP
    return mm_flop(a_shape, b_shape)

# 定义计算批矩阵乘法操作FLOP的函数，注册到torch的aten.bmm操作上
@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for the bmm operation."""
    # 输入参数应该是一个长度为2的列表，分别表示两个张量的形状
    b, m, k = a_shape
    b2, k2, n = b_shape
    assert b == b2
    assert k == k2
    # 返回批矩阵乘法操作的FLOP计算结果
    # 注意：理论上应该是2 * k - 1，这里是简化版本
    flop = b * m * n * 2 * k
    return flop

# 定义计算批矩阵乘法加法操作FLOP的函数，注册到torch的aten.baddbmm操作上
@register_flop_formula(aten.baddbmm)
def baddbmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for the baddbmm operation."""
    # 调用bmm_flop函数计算批矩阵乘法操作的FLOP
    return bmm_flop(a_shape, b_shape)

# 定义计算卷积操作FLOP的函数
def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> int:
    """Count flops for convolution.

    Note only multiplication is
    counted. Computation for bias are ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    """
    # 计算卷积操作的FLOP，只计算乘法部分，忽略偏置项的计算
    # 对于转置卷积，计算公式为(x_shape[2:] * prod(w_shape) * batch_size)
    return prod(x_shape[2:]) * prod(w_shape)
    # 获取输入形状的批量大小
    batch_size = x_shape[0]
    # 根据是否是转置卷积选择相应的卷积形状
    conv_shape = (x_shape if transposed else out_shape)[2:]
    # 解构卷积核形状
    c_out, c_in, *filter_size = w_shape

    """
    这里的一般想法是对于常规卷积，对于输出空间维度中的每个点，我们将滤波器与某些内容进行卷积（因此
    `prod(conv_shape) * prod(filter_size)` 次操作）。然后，这个结果乘以
    1. 批量大小，2. 输入和权重通道的叉乘。

    对于转置卷积，不是在*输出*空间维度的每个点，而是在*输入*空间维度的每个点。
    """
    # 注意：我认为这里没有正确考虑填充的情况 :think:
    # 注意：从理论上讲应该是 2 * c_in - 1 来计算 FLOP。
    flop = prod(conv_shape) * prod(filter_size) * batch_size * c_out * c_in * 2
    # 返回计算得到的 FLOP 数量
    return flop
# 注册卷积操作的 FLOP 计算函数，适用于前向传播
@register_flop_formula([aten.convolution, aten._convolution])
def conv_flop(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> int:
    """Count flops for convolution."""
    # 调用 conv_flop_count 函数计算卷积操作的 FLOP
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


# 注册卷积操作的反向传播 FLOP 计算函数
@register_flop_formula(aten.convolution_backward)
def conv_backward_flop(
        grad_out_shape,
        x_shape,
        w_shape,
        _bias,
        _stride,
        _padding,
        _dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
        out_shape) -> int:

    def t(shape):
        return [shape[1], shape[0]] + list(shape[2:])
    # 初始化 FLOP 计数
    flop_count = 0

    """
    Let's say we have a regular 1D conv
    {A, B, C} [inp]
    {i, j} [weight]
    => (conv)
    {Ai + Bj, Bi + Cj} [out]

    And as a reminder, the transposed conv of the above is
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]

    For the backwards of conv, we now have
    {D, E} [grad_out]
    {A, B, C} [inp]
    {i, j} [weight]

    # grad_inp as conv_transpose(grad_out, weight)
    Let's first compute grad_inp. To do so, we can simply look at all the
    multiplications that each element of inp is involved in. For example, A is
    only involved in the first element of the output (and thus only depends upon
    D in grad_out), and C is only involved in the last element of the output
    (and thus only depends upon E in grad_out)

    {Di, Dj + Ei, Ej} [grad_inp]

    Note that this corresponds to the below conv_transpose. This gives us the
    output_mask[0] branch, which is grad_inp.

    {D, E} [inp (grad_out)]
    {i, j} [weight]
    => (conv_transpose)
    {Di, Dj + Ei, Ej} [out (grad_inp)]

    I leave the fact that grad_inp for a transposed conv is just conv(grad_out,
    weight) as an exercise for the reader.

    # grad_weight as conv(inp, grad_out)
    To compute grad_weight, we again look at the terms in the output, which as
    a reminder is:
    => {Ai + Bj, Bi + Cj} [out]
    => {D, E} [grad_out]
    If we manually compute the gradient for the weights, we see it's
    {AD + BE, BD + CE} [grad_weight]

    This corresponds to the below conv
    {A, B, C} [inp]
    {D, E} [weight (grad_out)]
    => (conv)
    {AD + BE, BD + CE} [out (grad_weight)]

    # grad_weight of transposed conv as conv(grad_out, inp)
    As a reminder, the terms of the output of a transposed conv are:
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]
    => {D, E, F, G} [grad_out]

    Manually computing the gradient for the weights, we see it's
    {AD + BE + CF, AE + BF + CG} [grad_weight]

    This corresponds to the below conv
    {D, E, F, G} [inp (grad_out)]
    {A, B, C} [weight (inp)]
    => (conv)
    {AD + BE + CF, AE + BF + CG} [out (grad_weight)]

    For the full backwards formula, there are also some details involving
    transpose of the batch/channel dimensions and groups, but I skip those for
    """
    the sake of brevity (and they're pretty similar to matmul backwards)

    Check [conv backwards decomposition as conv forwards]
    """
    # 根据输出掩码选择相应的操作来计算梯度
    # 如果输出掩码的第一个元素为真，则计算 grad_inp = conv_transpose(grad_out, weight)
    if output_mask[0]:
        # 获取 grad_inp 的形状
        grad_input_shape = get_shape(out_shape[0])
        # 增加相应卷积操作的浮点运算次数到 flop_count
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not transposed)

    # 如果输出掩码的第二个元素为真
    if output_mask[1]:
        # 获取 grad_weight 的形状
        grad_weight_shape = get_shape(out_shape[1])
        # 如果是转置卷积
        if transposed:
            # 计算转置卷积的 grad_weight，即 conv(grad_out, inp)
            flop_count += conv_flop_count(t(grad_out_shape), t(x_shape), t(grad_weight_shape), transposed=False)
        else:
            # 计算普通卷积的 grad_weight，即 conv(inp, grad_out)
            flop_count += conv_flop_count(t(x_shape), t(grad_out_shape), t(grad_weight_shape), transposed=False)

    # 返回最终的浮点运算次数
    return flop_count
# 计算自注意力机制的浮点运算次数（FLOPs）
def sdpa_flop_count(query_shape, key_shape, value_shape):
    """
    Count flops for self-attention.

    NB: We can assume that value_shape == key_shape
    """
    # 解包查询张量的形状信息
    b, h, s_q, d_q = query_shape
    # 解包键张量的形状信息
    _b2, _h2, s_k, _d2 = key_shape
    # 解包值张量的形状信息
    _b3, _h3, _s3, d_v = value_shape
    # 断言确保所有批次（batch）、头数（head）、序列长度和深度维度相匹配
    assert b == _b2 == _b3 and h == _h2 == _h3 and d_q == _d2 and s_k == _s3 and d_q == _d2
    total_flops = 0
    # 计算矩阵乘法的 FLOPs： q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    # 计算矩阵乘法的 FLOPs： scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention, aten._scaled_dot_product_flash_attention])
def sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention."""
    # 注意：这里不考虑因果注意力（causal attention）
    return sdpa_flop_count(query_shape, key_shape, value_shape)


def _unpack_flash_attention_nested_shapes(
    *,
    query,
    key,
    value,
    grad_out=None,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Optional[Tuple[int, ...]]]]:
    """
    Given inputs to a flash_attention_(forward|backward) kernel, this will handle behavior for
    NestedTensor inputs by effectively unbinding the NestedTensor and yielding the shapes for
    each batch element.

    In the case that this isn't a NestedTensor kernel, then it just yields the original shapes.
    """
    if cum_seq_q is not None:
        # 这意味着我们正在处理嵌套的不规则张量查询。
        # 输入的形状为 (sum(sequence len), heads, dimension)
        # 相比之下，非嵌套输入的形状为 (batch, heads, sequence len, dimension)
        # 为了处理这种情况，我们将其转换为形状为 (batch, heads, max_seq_len, dimension)
        # 因此，在这种情况下，FLOPS（浮点运算数）计算是实际 FLOPS 的一个高估。
        assert len(key.shape) == 3
        assert len(value.shape) == 3
        assert grad_out is None or grad_out.shape == query.shape
        # 获取查询张量的形状信息
        _, h_q, d_q = query.shape
        # 获取键张量的形状信息
        _, h_k, d_k = key.shape
        # 获取值张量的形状信息
        _, h_v, d_v = value.shape
        # 确保 cum_seq_q 和 cum_seq_k 非空
        assert cum_seq_q is not None
        assert cum_seq_k is not None
        # 确保 cum_seq_q 和 cum_seq_k 的形状相同
        assert cum_seq_q.shape == cum_seq_k.shape
        # 计算每个查询序列长度和键序列长度
        seq_q_lengths = (cum_seq_q[1:] - cum_seq_q[:-1]).tolist()
        seq_k_lengths = (cum_seq_k[1:] - cum_seq_k[:-1]).tolist()
        # 遍历每对查询序列长度和键序列长度
        for (seq_q_len, seq_k_len) in zip(seq_q_lengths, seq_k_lengths):
            # 构造新的查询张量形状
            new_query_shape = (1, h_q, seq_q_len, d_q)
            # 构造新的键张量形状
            new_key_shape = (1, h_k, seq_k_len, d_k)
            # 构造新的值张量形状
            new_value_shape = (1, h_v, seq_k_len, d_v)
            # 如果存在梯度输出，构造新的梯度输出张量形状
            new_grad_out_shape = new_query_shape if grad_out is not None else None
            # 返回新的张量形状
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape
        # 函数结束
        return

    # 如果 cum_seq_q 为空，则处理非嵌套查询
    yield query.shape, key.shape, value.shape, grad_out.shape if grad_out is not None else None
def _unpack_efficient_attention_nested_shapes(
    *,
    query,
    key,
    value,
    grad_out=None,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Optional[Tuple[int, ...]]]]:
    """
    Given inputs to an efficient_attention_(forward|backward) kernel, this function handles behavior for
    NestedTensor inputs by unbinding them and yielding the shapes for each batch element.

    If the inputs are not NestedTensor, it yields the original shapes.

    Parameters:
    - query: Tensor, input query tensor
    - key: Tensor, input key tensor
    - value: Tensor, input value tensor
    - grad_out: Optional[Tensor], gradient tensor (default: None)
    - cu_seqlens_q: Tensor, cumulative sequence lengths of query tensor
    - cu_seqlens_k: Tensor, cumulative sequence lengths of key tensor
    - max_seqlen_q: int, maximum sequence length of query tensor
    - max_seqlen_k: int, maximum sequence length of key tensor

    Yields:
    - Tuple of shapes for query, key, value, and grad_out (if not None) after unbinding NestedTensors.

    Notes:
    - For NestedTensor inputs, reshapes tensors to match the expected format.
    """
    if cu_seqlens_q is not None:
        # Unlike flash_attention_forward, we get a 4D tensor instead of a 3D tensor for efficient attention.
        #
        # This means we should be dealing with a Nested Jagged Tensor query.
        # The inputs will have shape                  (sum(sequence len), heads, dimension)
        # In comparison, non-Nested inputs have shape (batch, heads, sequence len, dimension)
        # To deal with this, we convert to a shape of (batch, heads, max_seq_len, dimension)
        # So the flops calculation in this case is an overestimate of the actual flops.
        assert len(key.shape) == 4  # Ensure key tensor shape is 4-dimensional
        assert len(value.shape) == 4  # Ensure value tensor shape is 4-dimensional
        assert grad_out is None or grad_out.shape == query.shape  # Ensure grad_out shape matches query shape if provided
        _, _, h_q, d_q = query.shape  # Extract dimensions from query tensor shape
        _, _, h_k, d_k = key.shape  # Extract dimensions from key tensor shape
        _, _, h_v, d_v = value.shape  # Extract dimensions from value tensor shape
        assert cu_seqlens_q is not None  # Ensure cu_seqlens_q tensor is not None
        assert cu_seqlens_k is not None  # Ensure cu_seqlens_k tensor is not None
        assert cu_seqlens_q.shape == cu_seqlens_k.shape  # Ensure shape consistency between cu_seqlens_q and cu_seqlens_k
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()  # Calculate sequence lengths for query
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist()  # Calculate sequence lengths for key
        for len_q, len_k in zip(seqlens_q, seqlens_k):
            new_query_shape = (1, h_q, len_q, d_q)  # Shape after reshaping query tensor
            new_key_shape = (1, h_k, len_k, d_k)  # Shape after reshaping key tensor
            new_value_shape = (1, h_v, len_k, d_v)  # Shape after reshaping value tensor
            new_grad_out_shape = new_query_shape if grad_out is not None else None  # Shape after reshaping grad_out tensor if provided
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape  # Yield reshaped shapes
        return

    yield query.shape, key.shape, value.shape, grad_out.shape if grad_out is not None else None


@register_flop_formula(aten._flash_attention_forward, get_raw=True)
def _flash_attention_forward_flop(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    out_shape=None,
    **kwargs
) -> int:
    """
    Count flops for self-attention.

    Parameters:
    - query: Tensor, input query tensor
    - key: Tensor, input key tensor
    - value: Tensor, input value tensor
    - cum_seq_q: Tensor, cumulative sequence lengths of query tensor
    - cum_seq_k: Tensor, cumulative sequence lengths of key tensor
    - max_q: int, maximum sequence length of query tensor
    - max_k: int, maximum sequence length of key tensor
    - *args: Additional positional arguments
    - out_shape: Optional, output shape (default: None)
    - **kwargs: Additional keyword arguments

    Returns:
    - int, number of flops for the self-attention operation.

    Notes:
    - Does not account for causal attention.
    - If a nested tensor is detected, unpacks individual batch elements and sums the flops per element.
    """
    # NB: We aren't accounting for causal attention here
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    # 计算总的 SDPA（Scaled Dot-Product Attention）的 FLOP（浮点运算次数）。
    # 对于每个元组 sizes 中的每个元素，提取 query_shape、key_shape、value_shape，并调用 sdpa_flop_count 函数计算 FLOP。
    # 将计算得到的 FLOP 累加求和并返回总和。
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )
# 注册一个函数作为计算自注意力（self-attention）前向传播FLOPs的公式，使用aten._efficient_attention_forward作为键，返回原始函数。
# 该函数返回一个整数，表示前向传播操作的FLOPs。
@register_flop_formula(aten._efficient_attention_forward, get_raw=True)
def _efficient_attention_forward_flop(
    query,
    key,
    value,
    bias,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs
) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    # 如果这是一个嵌套张量，我们需要解包每个批次元素，然后对每个批次元素的FLOPs求和
    sizes = _unpack_efficient_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    # 对每个批次元素的大小进行迭代，并累加相应的FLOPs
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


# 计算自注意力（self-attention）后向传播FLOPs的函数，根据梯度输出形状、查询、键、值的形状来返回一个整数表示FLOPs。
def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    total_flops = 0
    # 解包形状，确保批次、头数、序列长度和深度的一致性
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_q == _d2
    assert d_v == _d4 and s_k == _s3 and s_q == _s4
    total_flops = 0
    # 第一步：重新计算分数矩阵。
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))

    # 第二步：通过得分 @ v 操作传播梯度。
    # gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    # scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))

    # 第三步：通过 k @ v 操作传播梯度
    # gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    # q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops


# 注册一个函数作为计算自注意力（self-attention）后向传播FLOPs的公式，使用aten._scaled_dot_product_efficient_attention_backward和aten._scaled_dot_product_flash_attention_backward作为键。
# 返回原始函数，该函数返回一个整数，表示后向传播操作的FLOPs。
@register_flop_formula([aten._scaled_dot_product_efficient_attention_backward, aten._scaled_dot_product_flash_attention_backward])
def sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention backward."""
    return sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)


# 注册一个函数作为计算flash注意力后向传播FLOPs的公式，使用aten._flash_attention_backward作为键。
# 返回原始函数，该函数返回一个整数，表示后向传播操作的FLOPs。
@register_flop_formula(aten._flash_attention_backward, get_raw=True)
def _flash_attention_backward_flop(
    grad_out,
    query,
    key,
    value,
    out,  # 为避免与封装器中创建的out_shape参数冲突，将此参数命名为_out_shape
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    **kwargs,
) -> int:
    # 如果这是一个嵌套的张量，我们解压各个批次元素
    # 然后对每个批次元素的浮点操作次数求和
    shapes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    # 返回所有批次元素的自注意力机制反向传播的浮点操作次数之和
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )
@register_flop_formula(aten._efficient_attention_backward, get_raw=True)
# 注册一个函数作为 _efficient_attention_backward 的 FLOP 公式，从而获取原始计数
def _efficient_attention_backward_flop(
    grad_out,
    query,
    key,
    value,
    bias,
    out,  # 命名为 _out 以避免与包装器中创建的 out 的关键字参数冲突
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs,
) -> int:
    # 如果这是嵌套张量，我们将解包各个批次元素，然后对每个批次元素的 FLOP 进行求和
    shapes = _unpack_efficient_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    # 返回所有批次元素的 FLOP 总数
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


flop_registry = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten._scaled_dot_product_efficient_attention: sdpa_flop,
    aten._scaled_dot_product_flash_attention: sdpa_flop,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_flop,
    aten._flash_attention_forward: _flash_attention_forward_flop,
    aten._efficient_attention_forward: _efficient_attention_forward_flop,
    aten._flash_attention_backward: _flash_attention_backward_flop,
    aten._efficient_attention_backward: _efficient_attention_backward_flop,
}
# 注册不同函数与其对应的 FLOP 计数器，以便后续查询和使用

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# 定义不同数量级的后缀
suffixes = ["", "K", "M", "B", "T"]
# Thanks BingChat!
def get_suffix_str(number):
    # 根据数字的位数确定合适的后缀索引，用于显示大数值的简写形式
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 2) // 3))
    return suffixes[index]

def convert_num_with_suffix(number, suffix):
    index = suffixes.index(suffix)
    # 将数字除以 1000 的指数 index，并保留两位小数
    value = f"{number / 1000 ** index:.3f}"
    # 返回带有后缀的格式化字符串
    return value + suffixes[index]

def convert_to_percent_str(num, denom):
    if denom == 0:
        return "0%"
    # 将分子 num 与分母 denom 转换为百分比形式的字符串，保留两位小数
    return f"{num / denom:.2%}"

def _pytreeify_preserve_structure(f):
    @wraps(f)
    def nf(args):
        flat_args, spec = tree_flatten(args)
        out = f(*flat_args)
        return tree_unflatten(out, spec)

    return nf


class FlopCounterMode(TorchDispatchMode):
    """
    ``FlopCounterMode`` 是一个上下文管理器，用于计算其上下文中的 FLOP 数量。
    """
    def __init__(
            self,
            mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: Optional[Dict[Any, Any]] = None):
        # 初始化实例变量，用于存储每个模块的操作次数
        self.flop_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        # 设置深度参数，控制递归遍历模块的层级
        self.depth = depth
        # 设置是否显示信息的标志
        self.display = display
        # 如果未提供自定义映射，则设为空字典
        if custom_mapping is None:
            custom_mapping = {}
        # 如果传入了 mods 参数，发出警告提示不再需要该参数
        if mods is not None:
            warnings.warn("mods argument is not needed anymore, you can stop passing it", stacklevel=2)
        # 初始化操作注册表，结合默认的注册表和自定义映射进行初始化
        self.flop_registry = {
            **flop_registry,
            **{k: v if getattr(v, "_get_raw", False) else shape_wrapper(v) for k, v in custom_mapping.items()}
        }
        # 初始化模块追踪器
        self.mod_tracker = ModuleTracker()

    def get_total_flops(self) -> int:
        # 返回全局总的浮点操作次数
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        # 返回所有模块的浮点操作次数统计信息
        return {k: dict(v) for k, v in self.flop_counts.items()}
    # 定义一个方法用于获取模块的表格展示，可以选择性地指定展示深度
    def get_table(self, depth=None):
        # 如果未指定深度，则使用对象的默认深度
        if depth is None:
            depth = self.depth
        # 如果默认深度仍然未设置，则使用一个较大的默认值
        if depth is None:
            depth = 999999

        # 导入 tabulate 库并设置保留空白符选项
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        # 定义表头
        header = ["Module", "FLOP", "% Total"]
        # 定义存储表格数据的列表
        values = []
        # 获取全局 FLOP 数量
        global_flops = self.get_total_flops()
        # 获取全局 FLOP 数量的单位后缀字符串
        global_suffix = get_suffix_str(global_flops)
        # 判断全局 FLOP 是否被模块所包含的标志
        is_global_subsumed = False

        # 定义处理单个模块的函数
        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            # 计算当前模块的总 FLOP 数量
            total_flops = sum(self.flop_counts[mod_name].values())

            # 检查当前模块是否包含全局 FLOP
            is_global_subsumed |= total_flops >= global_flops

            # 构建当前模块的数据行
            padding = " " * depth  # 根据深度设置缩进
            values = []
            values.append([
                padding + mod_name,  # 模块名称
                convert_num_with_suffix(total_flops, global_suffix),  # 转换后的 FLOP 数量字符串
                convert_to_percent_str(total_flops, global_flops)  # FLOP 占比字符串
            ])
            # 遍历当前模块下的各个操作项
            for k, v in self.flop_counts[mod_name].items():
                values.append([
                    padding + " - " + str(k),  # 操作项名称
                    convert_num_with_suffix(v, global_suffix),  # 转换后的操作项 FLOP 数量字符串
                    convert_to_percent_str(v, global_flops)  # 操作项 FLOP 占比字符串
                ])
            return values

        # 遍历排序后的模块列表，生成表格数据
        for mod in sorted(self.flop_counts.keys()):
            if mod == 'Global':
                continue
            mod_depth = mod.count(".") + 1  # 计算当前模块的深度
            if mod_depth > depth:  # 如果模块深度超过指定深度，则跳过
                continue

            cur_values = process_mod(mod, mod_depth - 1)  # 处理当前模块
            values.extend(cur_values)  # 将当前模块的数据添加到总数据中

        # 在这里进行一些处理，仅在全局 FLOP 中存在未完全包含的情况下输出“Global”值
        if 'Global' in self.flop_counts and not is_global_subsumed:
            for idx, value in enumerate(values):
                values[idx][0] = " " + values[idx][0]  # 缩进处理

            values = process_mod('Global', 0) + values  # 添加全局 FLOP 数据到表格数据开头

        # 如果表格数据为空，则创建一个包含全局 FLOP 值的空表格
        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        # 使用 tabulate 库生成并返回最终的表格输出字符串
        return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

    # 进入上下文管理器时执行的方法，清空 FLOP 统计数据，并进入模块追踪器的上下文
    def __enter__(self):
        self.flop_counts.clear()  # 清空 FLOP 统计数据
        self.mod_tracker.__enter__()  # 进入模块追踪器的上下文
        super().__enter__()  # 调用父类的 __enter__ 方法
        return self  # 返回对象本身

    # 退出上下文管理器时执行的方法，调用父类的 __exit__ 方法，退出模块追踪器的上下文，
    # 如果需要显示结果，则打印当前对象的表格输出
    def __exit__(self, *args):
        super().__exit__(*args)  # 调用父类的 __exit__ 方法
        self.mod_tracker.__exit__()  # 退出模块追踪器的上下文
        if self.display:
            print(self.get_table(self.depth))  # 如果需要显示结果，则打印表格输出

    # 自定义 Torch 分发方法，调用指定函数并返回计算 FLOP 后的结果
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)  # 调用指定函数
        return self._count_flops(func._overloadpacket, out, args, kwargs)  # 返回计算后的 FLOP 数量
    # 定义一个方法用于计算浮点操作数（FLOPs），接收四个参数：func_packet、out、args 和 kwargs
    def _count_flops(self, func_packet, out, args, kwargs):
        # 如果 func_packet 在 flop_registry 中已经注册过
        if func_packet in self.flop_registry:
            # 获取与 func_packet 相关的计算 FLOPs 的函数
            flop_count_func = self.flop_registry[func_packet]
            # 调用计算 FLOPs 的函数，传入 args 和 kwargs，将结果存入 out，并忽略类型检查
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            # 遍历模块追踪器的父节点集合，更新父节点的 FLOPs 计数
            for par in set(self.mod_tracker.parents):
                self.flop_counts[par][func_packet] += flop_count

        # 返回方法的输出参数 out
        return out
```