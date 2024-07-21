# `.\pytorch\functorch\einops\rearrange.py`

```
# 从未来导入注解功能，用于支持函数签名中的类型注解
from __future__ import annotations

# 导入 functools 库，用于创建带有缓存的函数
import functools
# 导入 typing 库中的类型定义
from typing import Callable, Dict, List, Sequence, Tuple, Union

# 导入 PyTorch 库
import torch

# 从 functorch._C 中导入 dim 函数作为 _C 别名
from functorch._C import dim as _C
# 导入 _parsing 模块中的特定函数和类
from ._parsing import (
    _ellipsis,
    AnonymousAxis,
    comma_separate,
    parse_pattern,
    validate_rearrange_expressions,
)

# 将 rearrange 函数添加到 __all__ 中，表示它是模块的公共接口之一
__all__ = ["rearrange"]

# 从 _C 模块中获取 dims 函数，并赋值给 dims 变量
dims = _C.dims

# 使用 functools 提供的 lru_cache 装饰器，缓存函数调用结果，最多缓存 256 个不同的调用
@functools.lru_cache(256)
# 定义一个函数 _create_rearrange_callable，接受 tensor_ndim、pattern 和 axes_lengths 作为参数，返回一个函数
def _create_rearrange_callable(
    tensor_ndim: int, pattern: str, **axes_lengths: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Translate an `einops`-style pattern into a callable that performs the rearrange using first-class dimensions.

    Since the an equivalent result is computed for tensors with the same number of dimensions, with the same pattern and
    specified axes lengths, this function can be memoized.

    Args:
        tensor_ndim (int): the number of dimensions in the tensor to rearrange
        pattern (str): the `einops`-style rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: a callable that performs the rearrangement
    """
    # 解析指定的重排列模式，生成左侧和右侧表达式
    left, right = parse_pattern(pattern, axes_lengths)
    # 验证重排列表达式是否有效
    validate_rearrange_expressions(left, right, axes_lengths)

    # 计算匿名维度的数量
    n_anon_dims = sum(not dim for dim in left.composition)

    # 如果左侧表达式包含省略号，则计算省略号维度的数量
    if left.has_ellipsis:
        n_ellipsis_dims = tensor_ndim - (len(left.composition) - 1)
        n_named_dims = len(left.identifiers) - 1

        # 检查模式中指定的维度数量是否不超过张量的维度数量
        if (pattern_ndim := n_anon_dims + n_named_dims) > tensor_ndim:
            raise ValueError(
                f"Number of dimensions in pattern ({pattern_ndim}) must be less than or equal to the number of "
                f"dimensions in the tensor ({tensor_ndim})"
            )
    else:
        n_ellipsis_dims = 0
        n_named_dims = len(left.identifiers)

        # 检查模式中指定的维度数量是否与张量的维度数量相等
        if (pattern_ndim := len(left.composition)) != tensor_ndim:
            raise ValueError(
                f"Number of dimensions in pattern ({pattern_ndim}) must be equal to the number of dimensions in "
                f"the tensor ({tensor_ndim})"
            )

    # 计算总维度数量
    n_dims = n_named_dims + n_ellipsis_dims + n_anon_dims

    # 如果维度数量为 0，返回一个恒等函数，用于处理零维张量的恒等重排列
    if n_dims == 0:
        return lambda tensor: tensor

    # 生成第一类维度的名称，形如 "d0", "d1", ...
    first_class_dims: Tuple[str, ...] = tuple(f"d{i}" for i in range(n_dims))
    # 创建一个映射，将左侧标识符映射到第一类维度名称
    identifier_dim_map: Dict[Union[str, AnonymousAxis], Tuple[str, ...]] = {}
    # 匿名轴的列表
    anon_axes: List[AnonymousAxis] = []

    # map the left-hand side identifiers to strings representing first class dims
    dims_i = 0
    for dimension in left.composition:
        # 遍历 left 对象的 composition 中的每一个 dimension
        if isinstance(dimension, list):
            # 如果 dimension 是列表类型
            for identifier in dimension:
                # 遍历 dimension 中的每一个 identifier
                # 匿名轴必须是字符串类型，非单元匿名轴不允许在重新排列中存在，而单元匿名轴表示为空列表
                assert isinstance(identifier, str)
                # 将 identifier 映射到对应的维度元组
                identifier_dim_map[identifier] = (first_class_dims[dims_i],)
                dims_i += 1
            if not dimension:
                # 如果 dimension 为空列表，表示单元匿名轴
                # 创建一个匿名轴对象，并将其映射到对应的维度元组中
                anon_axis = AnonymousAxis("1")
                identifier_dim_map[anon_axis] = (first_class_dims[dims_i],)
                anon_axes.append(anon_axis)
                dimension.append(anon_axis)
                dims_i += 1
        elif dimension == _ellipsis:
            # 如果 dimension 是省略号 _ellipsis
            identifier = _ellipsis
            # 将省略号映射到对应的维度元组中
            identifier_dim_map[identifier] = tuple(
                first_class_dims[dims_i + j] for j in range(n_ellipsis_dims)
            )
            dims_i += n_ellipsis_dims
        else:
            # 如果 dimension 类型不符合预期，抛出异常
            raise ValueError(f"Unexpected dimension: {dimension}")

    def composition_to_dims(
        composition: Sequence[Union[List[Union[str, AnonymousAxis]], str]]
    ) -> List[Union[str, Tuple[str, ...]]]:
        """Convert a `ParsedExpression.composition` into a `Tensor.__getitem__` index of strings representing first
        class dims."""
        # 将 ParsedExpression.composition 转换为表示第一类维度的字符串索引列表
        dim_composition: List[Union[str, Tuple[str, ...]]] = []
        for dimension in composition:
            if isinstance(dimension, list):
                # 如果 dimension 是列表类型
                dim_composition.append(
                    tuple(
                        dim
                        for identifier in dimension
                        for dim in identifier_dim_map[identifier]
                    )
                )
            elif dimension == _ellipsis:
                # 如果 dimension 是省略号 _ellipsis
                dim_composition.extend(identifier_dim_map[_ellipsis])
            else:
                # 如果 dimension 类型不符合预期，抛出异常
                raise ValueError(f"Unexpected dimension: {dimension}")
        return dim_composition

    left_dims = composition_to_dims(left.composition)
    right_dims = composition_to_dims(right.composition)
    anon_dims = tuple(identifier_dim_map[axis][0] for axis in anon_axes)
    specified_lengths = tuple(
        (identifier_dim_map[axis][0], length) for axis, length in axes_lengths.items()
    )

    custom_rearrange_callable_name = "do_rearrange"
    custom_rearrange_callable_code = (
        (
            f"def {custom_rearrange_callable_name}(tensor):\n"
            f"    {comma_separate(first_class_dims)} = dims({n_dims})\n"
        )
        + (
            "".join(
                f"    {dim}.size = {length}\n" for (dim, length) in specified_lengths
            )
            if specified_lengths
            else ""
        )
        + f"    tensor = tensor[{comma_separate(left_dims)}].order({comma_separate(right_dims)})\n"
        + (
            f"    return tensor.sum({comma_separate([anon_dims])}, keepdim=False)\n"
            if anon_dims
            else "    return tensor\n"
        )
    )
    # 执行自定义重排列可调用代码，这里假设 custom_rearrange_callable_code 是一个包含可调用代码的字符串变量，
    # 它将被执行并修改当前的局部命名空间
    exec(custom_rearrange_callable_code)
    # 返回局部命名空间中指定名称的对象，这里假设 custom_rearrange_callable_name 是一个字符串变量，
    # 表示在前面执行的代码中定义的可调用对象的名称
    return locals()[custom_rearrange_callable_name]
def rearrange(
    tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    pattern: str,
    **axes_lengths: int,
) -> torch.Tensor:
    r"""A native implementation of `einops.rearrange`, a reader-friendly smart element reordering for multidimensional
    tensors. This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    See: https://einops.rocks/api/rearrange/

    Args:
        tensor (Tensor or sequence of Tensor): the tensor(s) to rearrange
        pattern (str): the rearrangement pattern
        axes_lengths (int): any additional length specifications for dimensions

    Returns:
        Tensor: the rearranged tensor

    Examples:
        >>> # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        >>> images = torch.randn((32, 30, 40, 3))

        >>> # stack along first (batch) axis, output is a single array
        >>> rearrange(images, 'b h w c -> b h w c').shape
        torch.Size([32, 30, 40, 3])

        >>> # concatenate images along height (vertical axis), 960 = 32 * 30
        >>> rearrange(images, 'b h w c -> (b h) w c').shape
        torch.Size([960, 40, 3])

        >>> # concatenated images along horizontal axis, 1280 = 32 * 40
        >>> rearrange(images, 'b h w c -> h (b w) c').shape
        torch.Size([30, 1280, 3])

        >>> # reordered axes to "b c h w" format for deep learning
        >>> rearrange(images, 'b h w c -> b c h w').shape
        torch.Size([32, 3, 30, 40])

        >>> # flattened each image into a vector, 3600 = 30 * 40 * 3
        >>> rearrange(images, 'b h w c -> b (c h w)').shape
        torch.Size([32, 3600])

        >>> # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
        torch.Size([128, 15, 20, 3])

        >>> # space-to-depth operation
        >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
        torch.Size([32, 15, 20, 12])
    """
    # 如果输入不是 torch.Tensor 类型，则将其堆叠成一个 Tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.stack(tensor)

    # 创建一个重排操作的可调用函数，根据输入的维度和重排模式以及额外的长度参数
    rearrange_callable = _create_rearrange_callable(
        tensor.ndim, pattern, **axes_lengths
    )

    # 调用重排函数并返回结果
    return rearrange_callable(tensor)
```