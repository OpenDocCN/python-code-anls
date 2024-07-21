# `.\pytorch\torch\nested\__init__.py`

```py
# mypy: allow-untyped-defs
# 引入类型定义
from typing import List, Optional, Tuple, Union

# 引入 PyTorch 库
import torch
import torch.nn.functional as F
from torch import SymInt, Tensor
from torch._C import _add_docstr, _nested  # type: ignore[attr-defined]

# 引入类型别名
from torch.types import _device as Device, _dtype as DType

# 导出的函数名列表
__all__ = [
    "to_padded_tensor",
    "as_nested_tensor",
    "nested_tensor",
    "nested_tensor_from_jagged",
    "narrow",
]

# 嵌套张量构造函数

def as_nested_tensor(
    ts: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
    layout=None
) -> Tensor:
    r"""
    从张量或张量列表/元组构造嵌套张量，并保留自动求导历史。

    如果传入嵌套张量，则直接返回，除非设备、数据类型或布局不同。注意，转换设备/数据类型会导致复制，
    而转换布局目前不被此函数支持。

    如果传入非嵌套张量，则视为一批具有一致大小的组成部分。如果传入的设备/数据类型与输入不同，或者
    输入不是连续的，则会进行复制。否则，将直接使用输入的存储。

    如果提供张量列表，则在构造嵌套张量时始终复制列表中的张量。

    Args:
        ts (Tensor or List[Tensor] or Tuple[Tensor]): 要视为嵌套张量的张量或张量列表/元组，这些张量具有相同的维数

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): 返回的嵌套张量的期望类型。默认情况下，与列表中最左侧张量相同的 :class:`torch.dtype`
        device (:class:`torch.device`, optional): 返回的嵌套张量的期望设备。默认情况下，与列表中最左侧张量相同的 :class:`torch.device`
        layout (:class:`torch.layout`, optional): 返回的嵌套张量的期望布局。只支持步进和不规则布局。默认情况下，为步进布局。

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.as_nested_tensor([a, b])
        >>> nt.is_leaf
        False
        >>> fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)])
        >>> nt.backward(fake_grad)
        >>> a.grad
        tensor([1., 1., 1.])
        >>> b.grad
        tensor([0., 0., 0., 0., 0.])
        >>> c = torch.randn(3, 5, requires_grad=True)
        >>> nt2 = torch.nested.as_nested_tensor(c)
    """
    is_tensor_list = isinstance(ts, (list, tuple)) and all(isinstance(t, Tensor) for t in ts)
    # 检查第一个参数是否为 Tensor 类型且不是 tensor 列表
    # 如果不是 Tensor 类型且不是 tensor 列表，则抛出类型错误异常
    if not isinstance(ts, Tensor) and not is_tensor_list:
        raise TypeError(
            "as_nested_tensor(): Expected first argument to be a tensor or a list / tuple of tensors "
        )
    
    # 如果 is_tensor_list 为 True 并且 ts 不是列表，则将其转换为列表
    if is_tensor_list and not isinstance(ts, list):
        ts = list(ts)

    # 如果 ts 是 Tensor 类型且维度小于 2，则抛出运行时错误异常
    if isinstance(ts, Tensor) and ts.dim() < 2:
        raise RuntimeError("as_nested_tensor(): Expected tensor argument to have dim() > 1")

    # 如果 ts 是 Tensor 类型并且具有嵌套属性
    if isinstance(ts, Tensor) and ts.is_nested:
        # 如果指定的布局与输入的布局相同，则直接返回输入或将输入复制到指定设备和数据类型
        if layout == ts.layout:
            return ts.to(device=device, dtype=dtype)
        else:
            # 否则抛出运行时错误异常，指示不支持嵌套张量布局之间的转换
            raise RuntimeError(
                "as_nested_tensor(): Converting between nested tensor layouts is not supported")

    # 如果未指定布局，则默认使用 torch.strided 布局
    if layout is None:
        layout = torch.strided
    
    # 如果布局为 torch.strided
    if layout == torch.strided:
        # 如果 ts 是 Tensor 类型
        if isinstance(ts, Tensor):
            # 可能需要调用 contiguous() 来获取扁平化视图
            buffer = ts.contiguous().view(-1).to(device=device, dtype=dtype)
            # 创建包含每个张量形状的张量
            nested_sizes = torch.tensor([t.shape for t in ts])
            # 使用内部函数创建嵌套视图
            return torch._nested_view_from_buffer(
                buffer,
                nested_sizes,
                *torch._nested_compute_contiguous_strides_offsets(nested_sizes))
        else:
            # 否则，断言 ts 是列表，并使用内部函数创建嵌套张量
            assert isinstance(ts, list)
            return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    
    # 如果布局为 torch.jagged
    elif layout == torch.jagged:
        # 如果 ts 是 Tensor 类型
        if isinstance(ts, Tensor):
            # 可能需要调用 contiguous() 来获取扁平化视图
            values = ts.contiguous().flatten(0, 1).to(device=device, dtype=dtype)
            # 计算扁平化视图的偏移量
            batch_size = ts.shape[0]
            seq_len = ts.shape[1]
            offsets = torch.arange(0, batch_size * seq_len + 1, seq_len,
                                   device=device, dtype=torch.int64)
            # 导入内部函数，并使用其创建嵌套视图
            from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

            return nested_view_from_values_offsets(values, offsets)
        else:
            # 否则，导入内部函数，并使用其创建 jagged 嵌套张量
            from torch.nested._internal.nested_tensor import jagged_from_list

            assert isinstance(ts, list)
            nt, _ = jagged_from_list(ts, offsets=None, device=device, dtype=dtype)
            return nt
    
    # 如果布局不是 torch.strided 或 torch.jagged，则抛出运行时错误异常
    else:
        raise RuntimeError(f"Specified layout is unsupported for nested tensors: {layout}")
# 添加文档字符串，将 torch.nested.nested_to_padded_tensor 函数连接到 torch._C._nested 内置函数。
to_padded_tensor = _add_docstr(
    _nested.nested_to_padded_tensor,
    r"""
to_padded_tensor(input, padding, output_size=None, out=None) -> Tensor

Returns a new (non-nested) Tensor by padding the :attr:`input` nested tensor.
The leading entries will be filled with the nested data,
while the trailing entries will be padded.

.. warning::

    :func:`to_padded_tensor` always copies the underlying data,
    since the nested and the non-nested tensors differ in memory layout.

Args:
    padding (float): The padding value for the trailing entries.

Keyword args:
    output_size (Tuple[int]): The size of the output tensor.
                              If given, it must be large enough to contain all nested data;
                              else, will infer by taking the max size of each nested sub-tensor along each dimension.
    out (Tensor, optional): the output tensor.

Example::

    >>> nt = torch.nested.nested_tensor([torch.randn((2, 5)), torch.randn((3, 4))])
    nested_tensor([
      tensor([[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276],
              [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995]]),
      tensor([[-1.8546, -0.7194, -0.2918, -0.1846],
              [ 0.2773,  0.8793, -0.5183, -0.6447],
              [ 1.8009,  1.8468, -0.9832, -1.5272]])
    ])
    >>> pt_infer = torch.nested.to_padded_tensor(nt, 0.0)
    tensor([[[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276],
             [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995],
             [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
            [[-1.8546, -0.7194, -0.2918, -0.1846,  0.0000],
             [ 0.2773,  0.8793, -0.5183, -0.6447,  0.0000],
             [ 1.8009,  1.8468, -0.9832, -1.5272,  0.0000]]])
    >>> pt_large = torch.nested.to_padded_tensor(nt, 1.0, (2, 4, 6))
    tensor([[[ 1.6862, -1.1282,  1.1031,  0.0464, -1.3276,  1.0000],
             [-1.9967, -1.0054,  1.8972,  0.9174, -1.4995,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]],
            [[-1.8546, -0.7194, -0.2918, -0.1846,  1.0000,  1.0000],
             [ 0.2773,  0.8793, -0.5183, -0.6447,  1.0000,  1.0000],
             [ 1.8009,  1.8468, -0.9832, -1.5272,  1.0000,  1.0000],
             [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000]]])
    >>> pt_small = torch.nested.to_padded_tensor(nt, 2.0, (2, 2, 2))
    RuntimeError: Value in output_size is less than NestedTensor padded size. Truncation is not supported.

""",
)
    # tensor_list 是一个列表，其中每个元素是张量或可以传递给 torch.tensor 的对象，
    # 列表中的每个元素具有相同的维度。
    # 如果未指定布局，则默认为 strided 布局
    if layout is None:
        layout = torch.strided
    
    # 如果布局为 strided，则使用内部函数 _nested.nested_tensor 创建嵌套张量
    if layout == torch.strided:
        return _nested.nested_tensor(
            tensor_list,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            pin_memory=pin_memory)
    
    # 如果布局为 jagged，则需要将标量列表包装为张量列表
    elif layout == torch.jagged:
        # 将标量列表转换为张量列表（如果不是张量，则转换为张量）
        list_of_tensors = [t if isinstance(t, Tensor) else torch.as_tensor(t) for t in tensor_list]

        # 导入内部函数 jagged_from_list
        from torch.nested._internal.nested_tensor import jagged_from_list

        # 使用 torch.no_grad() 创建 jagged 嵌套张量
        with torch.no_grad():
            nt, _ = jagged_from_list(list_of_tensors, offsets=None, device=device, dtype=dtype)

        # 设置是否需要梯度
        nt.requires_grad_(requires_grad)
        
        # 如果需要固定内存，则将张量固定在内存中
        if pin_memory:
            nt = nt.pin_memory()  # type: ignore[assignment]

        return nt
    
    # 如果布局既不是 strided 也不是 jagged，则抛出运行时错误
    else:
        raise RuntimeError(f"Specified layout is unsupported for nested tensors: {layout}")
"""
SDPA kernels can deal with format easily, resulting in performance improvements.

Args:
    tensor (:class:`torch.Tensor`): a strided tensor, which will be used as the underlying data
        for the nested tensor if using the jagged layout or will be copied for the strided layout.
    dim (int): the dimension where narrow will be applied. Only `dim=1` is supported for the
        jagged layout, while strided supports all dim
    start (Union[int, :class:`torch.Tensor`]): starting element for the narrow operation
    length (Union[int, :class:`torch.Tensor`]): number of elements taken during the narrow op

Keyword arguments:
    layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
        Only strided and jagged layouts are supported. Default: if None, the strided layout.

Example::

    >>> starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    >>> lengths = torch.tensor([3, 2, 2, 1, 5], dtype=torch.int64)
    >>> narrow_base = torch.randn(5, 10, 20)
    >>> nt_narrowed = torch.nested.narrow(narrow_base, 1, starts, lengths, layout=torch.jagged)
    >>> nt_narrowed.is_contiguous()
    False
"""
if not isinstance(start, (int, SymInt, Tensor)):
    raise RuntimeError("start must be an integer or a tensor")

if not isinstance(length, (int, SymInt, Tensor)):
    raise RuntimeError("length must be an integer or a tensor")

if layout == torch.strided:
    if isinstance(start, Tensor) or isinstance(length, Tensor):
        raise RuntimeError("start and length must be integers for the strided layout NT impl")
    # TODO: switch to as_nested_tensor(tensor) when it is available
    nt = as_nested_tensor(torch.unbind(tensor), layout=torch.strided).narrow(dim, start, length)
elif layout == torch.jagged:
    if dim != 1:
        raise RuntimeError("jagged layout only supports dim=1")

    from torch.nested._internal.nested_tensor import jagged_from_tensor_and_lengths

    if isinstance(start, (int, SymInt)):
        start = torch.tensor([start], device=tensor.device, dtype=torch.int64)

    if isinstance(length, (int, SymInt)):
        length = torch.tensor([length], device=tensor.device, dtype=torch.int64)

    nt, _, _ = jagged_from_tensor_and_lengths(tensor, start, length)
else:
    raise RuntimeError(f"Specified layout is unsupported for nested narrow: {layout}")

return nt


def nested_tensor_from_jagged(
    values: Tensor,
    offsets: Optional[Tensor] = None,
    lengths: Optional[Tensor] = None,
    jagged_dim: Optional[int] = None,
) -> Tensor:
    r"""
Constructs a jagged layout nested tensor from the given jagged components. The jagged layout
consists of a required values buffer with the jagged dimension packed into a single dimension.
The offsets / lengths metadata determines how this dimension is split into batch elements
and are expected to be allocated on the same device as the values buffer.
"""
"""
如果没有提供 offsets 参数：
    如果 lengths 参数也未提供：
        抛出运行时错误，要求至少提供 offsets 或 lengths 中的一个。
    否则：
        将 lengths 参数累积求和并填充一个额外的元素作为 offsets，用于内核方便。
        将 lengths 参数置为 None。

如果未提供 jagged_dim 参数：
    将 jagged_dim 参数设为 1。
"""
if offsets is None:
    if lengths is None:
        raise RuntimeError(
            "nested_tensor_from_jagged(): At least one of offsets or lengths is required."
        )
    else:
        # TODO: Truly support offsets=None at some point?
        # For now, just convert lengths -> offsets for kernel convenience
        offsets = F.pad(lengths.cumsum(0), (1, 0))
        lengths = None

if jagged_dim is None:
    jagged_dim = 1
    # 从 torch.nested._internal.nested_tensor 模块导入 nested_view_from_values_offsets_lengths 函数
    from torch.nested._internal.nested_tensor import nested_view_from_values_offsets_lengths
    
    # 调用 nested_view_from_values_offsets_lengths 函数，并传入以下参数：
    # values: 包含嵌套张量数据的列表或张量
    # offsets: 嵌套结构的偏移量列表，指示每个嵌套层级的起始索引
    # lengths: 嵌套结构的长度列表，指示每个嵌套层级的元素数量
    # ragged_idx=jagged_dim: 可选参数，指示是否使用不规则索引来创建视图的维度
    return nested_view_from_values_offsets_lengths(values, offsets, lengths, ragged_idx=jagged_dim)
```