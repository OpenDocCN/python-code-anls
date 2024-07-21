# `.\pytorch\torch\testing\_creation.py`

```
"""
This module contains tensor creation utilities.
"""

import collections.abc  # 导入 collections.abc 模块，用于处理集合类型的抽象基类
import math  # 导入 math 模块，提供数学函数
import warnings  # 导入 warnings 模块，用于发出警告
from typing import cast, List, Optional, Tuple, Union  # 导入类型提示相关的功能

import torch  # 导入 PyTorch 库

_INTEGRAL_TYPES = [  # 定义整数类型列表
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint16,
    torch.uint32,
    torch.uint64,
]
_FLOATING_TYPES = [  # 定义浮点数类型列表
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]
_FLOATING_8BIT_TYPES = [  # 定义8位浮点数类型列表
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]
_COMPLEX_TYPES = [  # 定义复数类型列表
    torch.complex32,
    torch.complex64,
    torch.complex128,
]
_BOOLEAN_OR_INTEGRAL_TYPES = [torch.bool, *_INTEGRAL_TYPES]  # 包含布尔和整数类型的列表
_FLOATING_OR_COMPLEX_TYPES = [*_FLOATING_TYPES, *_COMPLEX_TYPES]  # 包含浮点数和复数类型的列表


def _uniform_random_(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """
    Fills the input tensor `t` with values uniformly drawn from the interval [low, high).

    Args:
    - t (torch.Tensor): The input tensor to be filled.
    - low (float): Lower bound of the uniform distribution.
    - high (float): Upper bound of the uniform distribution.

    Returns:
    - torch.Tensor: The filled tensor `t`.

    Notes:
    - If the difference `high - low` is larger than the maximum representable value of `t`'s dtype,
      the range is scaled before and after the PRNG to ensure it fits within bounds.
    """
    if high - low >= torch.finfo(t.dtype).max:
        return t.uniform_(low / 2, high / 2).mul_(2)
    else:
        return t.uniform_(low, high)


def make_tensor(
    *shape: Union[int, torch.Size, List[int], Tuple[int, ...]],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    noncontiguous: bool = False,
    exclude_zero: bool = False,
    memory_format: Optional[torch.memory_format] = None,
) -> torch.Tensor:
    """
    Creates a tensor with the specified shape, dtype, and device, filled with values uniformly drawn
    from [low, high).

    Args:
    - *shape (Union[int, torch.Size, List[int], Tuple[int, ...]]): Size of each dimension of the tensor.
    - dtype (torch.dtype): Data type of the tensor.
    - device (Union[str, torch.device]): Device to place the tensor on.
    - low (Optional[float]): Lower bound of the uniform distribution.
    - high (Optional[float]): Upper bound of the uniform distribution.
    - requires_grad (bool): Whether to track operations on the tensor for gradient computation.
    - noncontiguous (bool): Whether the tensor should be non-contiguous (not densely packed in memory).
    - exclude_zero (bool): Whether to exclude zero from the uniform distribution.
    - memory_format (Optional[torch.memory_format]): Memory format for the tensor.

    Returns:
    - torch.Tensor: A tensor filled with values uniformly drawn from [low, high).

    Notes:
    - Default values for `low` and `high` depend on the dtype and are clamped to dtype's representable range
      if outside.
    """
    r"""Creates a tensor with the given :attr:`shape`, :attr:`device`, and :attr:`dtype`, and filled with
    values uniformly drawn from ``[low, high)``.

    If :attr:`low` or :attr:`high` are specified and are outside the range of the :attr:`dtype`'s representable
    finite values then they are clamped to the lowest or highest representable finite value, respectively.
    If ``None``, then the following table describes the default values for :attr:`low` and :attr:`high`,
    which depend on :attr:`dtype`.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``low``    | ``high`` |
    +===========================+============+==========+
    | boolean type              | ``0``      | ``2``    |
    +---------------------------+------------+----------+
    | unsigned integral type    | ``0``      | ``10``   |
    +---------------------------+------------+----------+
    | signed integral types     | ``-9``     | ``10``   |
    +---------------------------+------------+----------+
    | floating types            | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    | complex types             | ``-9``     | ``9``    |
    +---------------------------+------------+----------+
    """
    # 创建一个张量的工具函数，返回一个新的张量对象
    def make_tensor(
        # 定义张量的形状，可以是单个整数或整数序列
        shape: Tuple[int, ...],
        # 定义返回张量的数据类型
        dtype: torch.dtype,
        # 定义返回张量的设备，可以是字符串或torch.device对象
        device: Union[str, torch.device],
        # 设置给定范围的下限（包含），如果提供的数字超出dtype的最小值则会被截断，为None时根据dtype确定
        low: Optional[Number] = None,
        # 设置给定范围的上限（不包含），如果提供的数字超出dtype的最大值则会被截断，为None时根据dtype确定
        high: Optional[Number] = None,
        # 是否记录返回张量上的autograd操作，默认为False
        requires_grad: Optional[bool] = False,
        # 返回张量是否是非连续的（noncontiguous），仅在张量元素数量大于等于2时有效，与memory_format互斥
        noncontiguous: Optional[bool] = False,
        # 如果为True，则将张量中的零替换为dtype的小正数值，依据dtype类型的定义
        exclude_zero: Optional[bool] = False,
        # 返回张量的内存格式，与noncontiguous互斥
        memory_format: Optional[torch.memory_format] = None,
    ):
        # 警告：使用low==high参数值创建浮点或复数类型张量已经废弃，请使用torch.full代替
        """
        Args:
            shape (Tuple[int, ...]): Single integer or a sequence of integers defining the shape of the output tensor.
            dtype (:class:`torch.dtype`): The data type of the returned tensor.
            device (Union[str, torch.device]): The device of the returned tensor.
            low (Optional[Number]): Sets the lower limit (inclusive) of the given range. If a number is provided it is
                clamped to the least representable finite value of the given dtype. When ``None`` (default),
                this value is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
            high (Optional[Number]): Sets the upper limit (exclusive) of the given range. If a number is provided it is
                clamped to the greatest representable finite value of the given dtype. When ``None`` (default) this value
                is determined based on the :attr:`dtype` (see the table above). Default: ``None``.
    
                .. deprecated:: 2.1
    
                    Passing ``low==high`` to :func:`~torch.testing.make_tensor` for floating or complex types is deprecated
                    since 2.1 and will be removed in 2.3. Use :func:`torch.full` instead.
    
            requires_grad (Optional[bool]): If autograd should record operations on the returned tensor. Default: ``False``.
            noncontiguous (Optional[bool]): If `True`, the returned tensor will be noncontiguous. This argument is
                ignored if the constructed tensor has fewer than two elements. Mutually exclusive with ``memory_format``.
            exclude_zero (Optional[bool]): If ``True`` then zeros are replaced with the dtype's small positive value
                depending on the :attr:`dtype`. For bool and integer types zero is replaced with one. For floating
                point types it is replaced with the dtype's smallest positive normal number (the "tiny" value of the
                :attr:`dtype`'s :func:`~torch.finfo` object), and for complex types it is replaced with a complex number
                whose real and imaginary parts are both the smallest positive normal number representable by the complex
                type. Default ``False``.
            memory_format (Optional[torch.memory_format]): The memory format of the returned tensor. Mutually exclusive
                with ``noncontiguous``.
    
        Raises:
            ValueError: If ``requires_grad=True`` is passed for integral `dtype`
            ValueError: If ``low >= high``.
            ValueError: If either :attr:`low` or :attr:`high` is ``nan``.
            ValueError: If both :attr:`noncontiguous` and :attr:`memory_format` are passed.
            TypeError: If :attr:`dtype` isn't supported by this function.
        """
    """
    Examples:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> from torch.testing import make_tensor
        >>> # Creates a float tensor with values in [-1, 1)
        >>> make_tensor((3,), device='cpu', dtype=torch.float32, low=-1, high=1)
        >>> # xdoctest: +SKIP
        tensor([ 0.1205, 0.2282, -0.6380])
        >>> # Creates a bool tensor on CUDA
        >>> make_tensor((2, 2), device='cuda', dtype=torch.bool)
        tensor([[False, False],
                [False, True]], device='cuda:0')
    """

    def modify_low_high(
        low: Optional[float],
        high: Optional[float],
        *,
        lowest_inclusive: float,
        highest_exclusive: float,
        default_low: float,
        default_high: float,
    ) -> Tuple[float, float]:
        """
        Modifies (and raises ValueError when appropriate) low and high values given by the user (input_low, input_high)
        if required.

        Args:
            low: Optional[float] - User-provided lower bound (inclusive) for value generation.
            high: Optional[float] - User-provided upper bound (exclusive) for value generation.
            lowest_inclusive: float - Lowest inclusive boundary supported by the data type.
            highest_exclusive: float - Highest exclusive boundary supported by the data type.
            default_low: float - Default lower bound if not provided by the user.
            default_high: float - Default upper bound if not provided by the user.

        Returns:
            Tuple[float, float]: Adjusted lower and upper bounds after validation and adjustment.

        Raises:
            ValueError: If `low` or `high` are NaN, or if they violate specific conditions.
        """

        def clamp(a: float, l: float, h: float) -> float:
            """Clamps value `a` between `l` (inclusive) and `h` (inclusive)."""
            return min(max(a, l), h)

        # Assign default values if not provided by the user
        low = low if low is not None else default_low
        high = high if high is not None else default_high

        # Check for NaN values in low or high
        if any(isinstance(value, float) and math.isnan(value) for value in [low, high]):
            raise ValueError(
                f"`low` and `high` cannot be NaN, but got {low=} and {high=}"
            )
        # Deprecated warning for equal low and high values for floating or complex types
        elif low == high and dtype in _FLOATING_OR_COMPLEX_TYPES:
            warnings.warn(
                "Passing `low==high` to `torch.testing.make_tensor` for floating or complex types "
                "is deprecated since 2.1 and will be removed in 2.3. "
                "Use `torch.full(...)` instead.",
                FutureWarning,
                stacklevel=3,
            )
        # Validate low < high condition
        elif low >= high:
            raise ValueError(f"`low` must be less than `high`, but got {low} >= {high}")
        # Check if high is within expected range
        elif high < lowest_inclusive or low >= highest_exclusive:
            raise ValueError(
                f"The value interval specified by `low` and `high` is [{low}, {high}), "
                f"but {dtype} only supports [{lowest_inclusive}, {highest_exclusive})"
            )

        # Clamp low and high values within supported range
        low = clamp(low, lowest_inclusive, highest_exclusive)
        high = clamp(high, lowest_inclusive, highest_exclusive)

        if dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
            # Adjust low and high for boolean or integral types
            # 1. `low` is ceiled to avoid creating values smaller than `low` and thus outside the specified interval
            # 2. Following the same reasoning as for 1., `high` should be floored. However, the higher bound of
            #    `torch.randint` is exclusive, and thus we need to ceil here as well.
            return math.ceil(low), math.ceil(high)

        return low, high

    if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
        shape = shape[0]  # type: ignore[assignment]
    shape = cast(Tuple[int, ...], tuple(shape))
    # 如果 `noncontiguous` 和 `memory_format` 都为真，则抛出 ValueError 异常
    if noncontiguous and memory_format is not None:
        raise ValueError(
            f"The parameters `noncontiguous` and `memory_format` are mutually exclusive, "
            f"but got {noncontiguous=} and {memory_format=}"
        )

    # 如果 `requires_grad` 为真且 `dtype` 是布尔值或整数类型，则抛出 ValueError 异常
    if requires_grad and dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        raise ValueError(
            f"`requires_grad=True` is not supported for boolean and integral dtypes, but got {dtype=}"
        )

    # 如果 `dtype` 是 torch.bool 类型
    if dtype is torch.bool:
        # 调用 modify_low_high 函数，获取修正后的 low 和 high 值
        low, high = cast(
            Tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest_inclusive=0,
                highest_exclusive=2,
                default_low=0,
                default_high=2,
            ),
        )
        # 使用 torch.randint 生成一个随机整数张量
        result = torch.randint(low, high, shape, device=device, dtype=dtype)

    # 如果 `dtype` 是布尔类型或整数类型
    elif dtype in _BOOLEAN_OR_INTEGRAL_TYPES:
        # 调用 modify_low_high 函数，获取修正后的 low 和 high 值
        low, high = cast(
            Tuple[int, int],
            modify_low_high(
                low,
                high,
                lowest_inclusive=torch.iinfo(dtype).min,
                highest_exclusive=torch.iinfo(dtype).max
                # 理论上，`highest_exclusive` 应该是最大值加1。然而，`torch.randint` 内部将边界转换为 int64，可能会溢出。
                # 换句话说：`torch.randint` 无法采样到 2**63 - 1，即 `torch.int64` 的最大值，这里需要进行调整。
                + (1 if dtype is not torch.int64 else 0),
                # 对于 `torch.uint8` 这里是不正确的，但是因为我们在使用默认值后，会将最小值夹到 `lowest` 即0，所以这里不需要特别处理
                default_low=-9,
                default_high=10,
            ),
        )
        # 使用 torch.randint 生成一个随机整数张量
        result = torch.randint(low, high, shape, device=device, dtype=dtype)

    # 如果 `dtype` 是浮点数或复数类型
    elif dtype in _FLOATING_OR_COMPLEX_TYPES:
        # 调用 modify_low_high 函数，获取修正后的 low 和 high 值
        low, high = modify_low_high(
            low,
            high,
            lowest_inclusive=torch.finfo(dtype).min,
            highest_exclusive=torch.finfo(dtype).max,
            default_low=-9,
            default_high=9,
        )
        # 使用 torch.empty 创建一个空的张量
        result = torch.empty(shape, device=device, dtype=dtype)
        # 使用 _uniform_random_ 函数填充张量的随机值
        _uniform_random_(
            torch.view_as_real(result) if dtype in _COMPLEX_TYPES else result, low, high
        )

    # 如果 `dtype` 是8位浮点数类型
    elif dtype in _FLOATING_8BIT_TYPES:
        # 调用 modify_low_high 函数，获取修正后的 low 和 high 值
        low, high = modify_low_high(
            low,
            high,
            lowest_inclusive=torch.finfo(dtype).min,
            highest_exclusive=torch.finfo(dtype).max,
            default_low=-9,
            default_high=9,
        )
        # 使用 torch.empty 创建一个空的张量，dtype 为 torch.float32
        result = torch.empty(shape, device=device, dtype=torch.float32)
        # 使用 _uniform_random_ 函数填充张量的随机值
        _uniform_random_(result, low, high)
        # 将结果张量转换为指定的 dtype
        result = result.to(dtype)
    else:
        # 如果请求的数据类型不被 torch.testing.make_tensor() 支持，抛出类型错误异常
        raise TypeError(
            f"The requested dtype '{dtype}' is not supported by torch.testing.make_tensor()."
            " To request support, file an issue at: https://github.com/pytorch/pytorch/issues"
        )

    if noncontiguous and result.numel() > 1:
        # 如果要求非连续存储且结果张量元素数量大于1，则进行重复插值和切片操作
        result = torch.repeat_interleave(result, 2, dim=-1)
        result = result[..., ::2]
    elif memory_format is not None:
        # 如果指定了内存格式，则克隆结果张量并应用指定的内存格式
        result = result.clone(memory_format=memory_format)

    if exclude_zero:
        # 如果要排除零值，则将结果张量中的零值替换为相应类型的最小非零值
        result[result == 0] = (
            1 if dtype in _BOOLEAN_OR_INTEGRAL_TYPES else torch.finfo(dtype).tiny
        )

    if dtype in _FLOATING_OR_COMPLEX_TYPES:
        # 如果数据类型是浮点数或复数类型，则根据需要设置梯度计算
        result.requires_grad = requires_grad

    # 返回处理后的结果张量
    return result
```