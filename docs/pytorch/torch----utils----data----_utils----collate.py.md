# `.\pytorch\torch\utils\data\_utils\collate.py`

```
# mypy: allow-untyped-defs
# 包含了由 _BaseDataLoaderIter 工作线程使用的方法定义。

# 这些方法用于将从数据集获取的样本整理成 Tensor。
# 因为 Python 2 不支持序列化静态方法，所以这些方法需要在全局范围内定义。

# default_collate 和 default_convert 通过 'dataloader.py' 对外暴露给用户。

import collections  # 导入 collections 模块
import contextlib  # 导入 contextlib 模块
import copy  # 导入 copy 模块
import re  # 导入 re 模块
from typing import Callable, Dict, Optional, Tuple, Type, Union  # 导入类型提示

import torch  # 导入 torch 模块


np_str_obj_array_pattern = re.compile(r"[SaUO]")  # 编译正则表达式，用于匹配 NumPy 中的字符串和对象数组


def default_convert(data):
    r"""
    Convert each NumPy array element into a :class:`torch.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
    If the input is not an NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~torch.utils.data.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~torch.utils.data.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with `int`
        >>> default_convert(0)
        0
        >>> # Example with NumPy array
        >>> default_convert(np.array([0, 1]))
        tensor([0, 1])
        >>> # Example with NamedTuple
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert(Point(np.array(0), np.array(0)))
        Point(x=tensor(0), y=tensor(0))
        >>> # Example with List
        >>> default_convert([np.array([0, 1]), np.array([2, 3])])
        [tensor([0, 1]), tensor([2, 3])]
    """
    elem_type = type(data)  # 获取数据的类型
    if isinstance(data, torch.Tensor):  # 如果数据已经是 torch.Tensor 类型
        return data  # 直接返回数据
    elif (
        elem_type.__module__ == "numpy"  # 如果数据类型属于 numpy 模块
        and elem_type.__name__ != "str_"  # 并且不是字符串类型
        and elem_type.__name__ != "string_"  # 也不是 string_ 类型
    ):
        # 如果是 ndarray 类型且其中包含字符串或对象数组
        if (
            elem_type.__name__ == "ndarray"
            and np_str_obj_array_pattern.search(data.dtype.str) is not None
        ):
            return data  # 返回原始数据
        return torch.as_tensor(data)  # 否则转换成 torch.Tensor 返回
    elif isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                # 如果数据是可变映射类型，由于映射类型可能有额外的属性，因此不能简单地使用 `type(data)(...)` 来创建新的映射。
                # 创建一个克隆对象并更新，如果映射类型是可变的。
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            else:
                # 如果数据是不可变映射类型，创建一个新的映射对象并递归转换每个元素。
                return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # 如果映射类型不支持 `copy()` / `update(mapping)` 或 `__init__(iterable)` 操作。
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        # 如果数据是命名元组类型，则递归转换每个字段的值并返回转换后的命名元组。
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        # 如果数据是普通元组类型，则递归转换每个元素并返回转换后的列表，用于向后兼容。
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        try:
            if isinstance(data, collections.abc.MutableSequence):
                # 如果数据是可变序列类型，由于序列类型可能有额外的属性，因此不能简单地使用 `type(data)(...)` 来创建新的序列。
                # 创建一个克隆对象并更新，如果序列类型是可变的。
                clone = copy.copy(data)  # type: ignore[arg-type]
                for i, d in enumerate(data):
                    clone[i] = default_convert(d)
                return clone
            else:
                # 如果数据是不可变序列类型，创建一个新的序列对象并递归转换每个元素。
                return elem_type([default_convert(d) for d in data])
        except TypeError:
            # 如果序列类型不支持 `copy()` / `__setitem__(index, item)` 或 `__init__(iterable)` 操作（如 `range` 类型）。
            return [default_convert(d) for d in data]
    else:
        # 如果数据不属于以上任何类型，则直接返回数据本身，即已经是基本类型或不需要转换的对象。
        return data
# 默认的集合错误消息格式，用于在出错时生成错误消息
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

# 定义了一个名为 collate 的函数
def collate(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    r"""
    通用的集合函数，用于处理每个批次中的元素集合类型。

    该函数还打开函数注册表以处理特定的元素类型。`default_collate_fn_map`
    提供了张量、numpy 数组、数字和字符串的默认集合函数。

    Args:
        batch: 要集合的单个批次
        collate_fn_map: 可选的字典，将元素类型映射到相应的集合函数。
            如果元素类型不在此字典中，
            该函数将按照插入顺序遍历字典中的每个键，
            如果元素类型是键的子类，则调用相应的集合函数。

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        ...     # 扩展此函数以处理张量批次
        ...     return torch.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {torch.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # 通过原地修改 `default_collate_fn_map` 扩展 `default_collate`
        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

    Note:
        每个集合函数需要一个用于批次的位置参数和一个用于集合函数字典的关键字参数 `collate_fn_map`。
    """
    # 获取批次中第一个元素
    elem = batch[0]
    # 获取第一个元素的类型
    elem_type = type(elem)

    # 如果 collate_fn_map 不是 None
    if collate_fn_map is not None:
        # 如果元素类型在 collate_fn_map 中
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        # 否则遍历 collate_fn_map 中的每个类型
        for collate_type in collate_fn_map:
            # 如果元素是 collate_type 的实例
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )
    # 检查元素是否是映射类型（如字典）
    if isinstance(elem, collections.abc.Mapping):
        try:
            # 如果映射类型是可变的
            if isinstance(elem, collections.abc.MutableMapping):
                # 映射类型可能有额外的属性，因此不能简单地用 `type(data)(...)` 来创建新的映射。
                # 创建一个克隆对象，并在映射类型可变时更新它。
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                # 创建一个新的不可变映射对象，其中的每个键值对都使用 collate 函数进行聚合
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            # 映射类型可能不支持 `copy()` / `update(mapping)` 或 `__init__(iterable)` 操作，
            # 返回一个新的字典，其中每个键值对都使用 collate 函数进行聚合
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    # 如果元素是命名元组
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        # 返回一个新的命名元组对象，其中每个字段都使用 collate 函数对对应的数据样本进行聚合
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch)
            )
        )
    elif isinstance(elem, collections.abc.Sequence):
        # 检查批次中的元素是否具有一致的大小
        it = iter(batch)
        elem_size = len(next(it))  # 获取第一个元素的长度作为标准大小
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # 转置批次数据，将每个元素的相同索引位置数据组合成元组列表

        if isinstance(elem, tuple):
            # 如果元素类型是元组，则递归调用 collate 函数对每个元组中的样本进行处理
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # 保持向后兼容性。
        else:
            try:
                if isinstance(elem, collections.abc.MutableSequence):
                    # 如果元素类型是可变序列，则创建一个克隆对象，并更新其中的数据
                    clone = copy.copy(elem)  # 创建元素的浅复制
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone  # 返回更新后的克隆对象
                else:
                    # 对于其他类型的序列，创建一个新的相同类型的序列，其中每个元素都经过 collate 处理
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                # 如果序列类型不支持复制或更新操作，返回对每个元素都经过 collate 处理后的列表
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
# 默认的数据整理函数，用于将给定批次的数据整理成可用于训练的形式
def default_collate(batch):
    r"""
    # 使用 torch.Tensor 对象的 `collate_tensor_fn` 来整理包含 Tensor 的批次
    """

    # 确定批次中第一个元素
    elem = batch[0]

    # 定义一个空的输出变量
    out = None

    # 如果批次中的元素是嵌套的，抛出运行时错误
    if elem.is_nested:
        raise RuntimeError(
            "Batches of nested tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )

    # 如果批次中的元素是稀疏张量之一，抛出运行时错误
    if elem.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        raise RuntimeError(
            "Batches of sparse tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )

    # 如果当前在一个后台进程中，为了避免额外的复制，将数据直接连接到共享内存张量中
    if torch.utils.data.get_worker_info() is not None:
        numel = sum(x.numel() for x in batch)  # 计算批次中所有张量元素的总数
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)  # 创建共享存储
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))  # 将共享存储连接到输出张量

    # 使用 torch.stack 将批次中的所有张量堆叠起来，形成一个新的张量
    return torch.stack(batch, 0, out=out)


# 将 numpy.ndarray 和 memmap 类型的批次整理成可以使用的形式
def collate_numpy_array_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # 确定批次中第一个元素
    elem = batch[0]

    # 如果批次中存在字符串类或对象数组，抛出类型错误
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    # 使用 `collate` 函数将批次中的每个 numpy 数组转换为 torch.Tensor，然后整理
    return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


# 将批次中的标量 numpy 数组整理成 torch.Tensor 形式
def collate_numpy_scalar_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # 直接将批次中的所有元素转换为 torch.Tensor
    return torch.as_tensor(batch)


# 将批次中的浮点数列表整理成 torch.Tensor 形式
def collate_float_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # 将批次中的浮点数列表转换为 torch.Tensor，数据类型为 torch.float64
    return torch.tensor(batch, dtype=torch.float64)


# 将批次中的整数列表整理成 torch.Tensor 形式
def collate_int_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # 将批次中的整数列表转换为 torch.Tensor
    return torch.tensor(batch)


# 将批次中的字符串列表整理成原始列表形式
def collate_str_fn(
    batch,
    *,
    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None,
):
    # 直接返回批次中的字符串列表
    return batch


# 默认的数据整理函数映射，用于根据数据类型选择合适的整理函数
default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {
    torch.Tensor: collate_tensor_fn,  # 对于 torch.Tensor 使用 collate_tensor_fn
}

# 尝试导入 numpy，如果成功则添加相应的整理函数到映射中
with contextlib.suppress(ImportError):
    import numpy as np

    # 对于 numpy.ndarray 和 memmap 类型，使用 collate_numpy_array_fn
    default_collate_fn_map[np.ndarray] = collate_numpy_array_fn

    # 对于 numpy 中的标量类型，使用 collate_numpy_scalar_fn
    # 参考：https://numpy.org/doc/stable/reference/arrays.scalars.html
    # 跳过字符串类型的标量
    default_collate_fn_map[(np.bool_, np.number, np.object_)] = collate_numpy_scalar_fn

# 对于 float 类型的数据，使用 collate_float_fn
default_collate_fn_map[float] = collate_float_fn

# 对于 int 类型的数据，使用 collate_int_fn
default_collate_fn_map[int] = collate_int_fn

# 对于 str 和 bytes 类型的数据，使用 collate_str_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn
    # 将输入的数据批次进行整理，将批次中的每个元素转换为一个带有额外外部维度（批次大小）的张量。
    
    # 输出类型可以是:class:`torch.Tensor`，一个`Sequence`类型的:class:`torch.Tensor`集合，或保持不变，具体取决于输入类型。
    # 当:class:`~torch.utils.data.DataLoader`中定义了`batch_size`或`batch_sampler`时，这将作为默认的整理函数使用。
    
    # 这里是根据输入元素类型到输出类型的映射关系：
    
    # * :class:`torch.Tensor` -> :class:`torch.Tensor`（添加外部维度批次大小）
    # * NumPy 数组 -> :class:`torch.Tensor`
    # * `float` -> :class:`torch.Tensor`
    # * `int` -> :class:`torch.Tensor`
    # * `str` -> `str`（保持不变）
    # * `bytes` -> `bytes`（保持不变）
    # * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
    # * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
    #   default_collate([V2_1, V2_2, ...]), ...]`
    # * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
    #   default_collate([V2_1, V2_2, ...]), ...]`
    
    # Args:
    #     batch: 待整理的单个批次数据
    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Two options to extend `default_collate` to handle specific type
        >>> # Option 1: Write custom collate function and invoke `default_collate`
        >>> def custom_collate(batch):
        ...     elem = batch[0]
        ...     if isinstance(elem, CustomType):  # Some custom condition
        ...         return ...
        ...     else:  # Fall back to `default_collate`
        ...         return default_collate(batch)
        >>> # Option 2: In-place modify `default_collate_fn_map`
        >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
        ...     return ...
        >>> default_collate_fn_map.update(CustomType, collate_customtype_fn)
        >>> # Invoke the main collate function with specified collate function map
        >>> return collate(batch, collate_fn_map=default_collate_fn_map)


注释：

# 使用默认的批量整理函数 `default_collate` 处理给定的批量数据 `batch`
# 返回整理后的结果，并指定使用 `default_collate_fn_map` 中的映射表进行处理
```