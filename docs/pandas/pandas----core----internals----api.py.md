# `D:\src\scipysrc\pandas\pandas\core\internals\api.py`

```
"""
This is a pseudo-public API for downstream libraries.  We ask that downstream
authors

1) Try to avoid using internals directly altogether, and failing that,
2) Use only functions exposed here (or in core.internals)

"""

# 引入未来版本特性模块
from __future__ import annotations

# 类型检查标记
from typing import TYPE_CHECKING
import warnings

# 引入 NumPy 库
import numpy as np

# 引入 pandas 内部的 BlockPlacement 类
from pandas._libs.internals import BlockPlacement

# 引入 pandas 的公共数据类型函数
from pandas.core.dtypes.common import pandas_dtype
# 引入 pandas 的具体数据类型模块
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    PeriodDtype,
)

# 引入 pandas 的数组模块
from pandas.core.arrays import (
    DatetimeArray,
    TimedeltaArray,
)
# 引入 pandas 的构造模块中的 extract_array 函数
from pandas.core.construction import extract_array
# 引入 pandas 内部的块模块
from pandas.core.internals.blocks import (
    check_ndim,
    ensure_block_shape,
    extract_pandas_array,
    get_block_type,
    maybe_coerce_values,
)

# 如果支持类型检查，则引入 ArrayLike 和 Dtype 类型
if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Dtype,
    )

    # 引入 pandas 内部的 Block 类
    from pandas.core.internals.blocks import Block


def _make_block(values: ArrayLike, placement: np.ndarray) -> Block:
    """
    This is an analogue to blocks.new_block(_2d) that ensures:
    1) correct dimension for EAs that support 2D (`ensure_block_shape`), and
    2) correct EA class for datetime64/timedelta64 (`maybe_coerce_values`).

    The input `values` is assumed to be either numpy array or ExtensionArray:
    - In case of a numpy array, it is assumed to already be in the expected
      shape for Blocks (2D, (cols, rows)).
    - In case of an ExtensionArray the input can be 1D, also for EAs that are
      internally stored as 2D.

    For the rest no preprocessing or validation is done, except for those dtypes
    that are internally stored as EAs but have an exact numpy equivalent (and at
    the moment use that numpy dtype), i.e. datetime64/timedelta64.
    """
    # 获取值的数据类型
    dtype = values.dtype
    # 根据数据类型获取块的类型
    klass = get_block_type(dtype)
    # 创建块的放置对象
    placement_obj = BlockPlacement(placement)

    # 如果数据类型是 ExtensionDtype 且支持二维，则确保块的形状为二维
    if (isinstance(dtype, ExtensionDtype) and dtype._supports_2d) or isinstance(
        values, (DatetimeArray, TimedeltaArray)
    ):
        values = ensure_block_shape(values, ndim=2)

    # 可能强制转换值
    values = maybe_coerce_values(values)
    # 返回创建的块对象
    return klass(values, ndim=2, placement=placement_obj)


def make_block(
    values, placement, klass=None, ndim=None, dtype: Dtype | None = None
) -> Block:
    """
    This is a pseudo-public analogue to blocks.new_block.

    We ask that downstream libraries use this rather than any fully-internal
    APIs, including but not limited to:

    - core.internals.blocks.make_block
    - Block.make_block
    - Block.make_block_same_class
    - Block.__init__
    """
    # 发出警告，表明此函数即将被弃用
    warnings.warn(
        # GH#56815
        "make_block is deprecated and will be removed in a future version. "
        "Use pd.api.internals.create_dataframe_from_blocks or "
        "(recommended) higher-level public APIs instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # 如果提供了 dtype 参数，则将其转换为 pandas 数据类型
    if dtype is not None:
        dtype = pandas_dtype(dtype)
    # 调用函数 extract_pandas_array 处理给定的值和数据类型，返回处理后的结果和数据类型
    values, dtype = extract_pandas_array(values, dtype, ndim)

    # 从 pandas 库中导入 ExtensionBlock 类
    from pandas.core.internals.blocks import ExtensionBlock

    # 如果 klass 是 ExtensionBlock 类型，并且 values 的数据类型是 PeriodDtype 类型
    if klass is ExtensionBlock and isinstance(values.dtype, PeriodDtype):
        # GH-44681 将 PeriodArray 存储在 NDArrayBackedExtensionBlock 中，而不是 ExtensionBlock 中
        # -> 在这种情况下仍然允许使用 ExtensionBlock 以保持向后兼容性
        klass = None

    # 如果 klass 为 None
    if klass is None:
        # 如果 dtype 为 None，则将其设为 values 的数据类型；否则保持 dtype 不变
        dtype = dtype or values.dtype
        # 根据数据类型获取相应的数据块类型
        klass = get_block_type(dtype)

    # 如果 placement 不是 BlockPlacement 类型，则将其转换为 BlockPlacement 对象
    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)

    # 推断 values 的维度，并更新 ndim
    ndim = maybe_infer_ndim(values, placement, ndim)

    # 如果 values 的数据类型是 PeriodDtype 或 DatetimeTZDtype 类型
    if isinstance(values.dtype, (PeriodDtype, DatetimeTZDtype)):
        # GH#41168 确保我们可以传递 1D 的 dt64tz 值
        # 更一般地说，任何不是 is_1d_only_ea_dtype 的 EA 数据类型
        # 提取数组的实际值（使用 numpy 数组形式）
        values = extract_array(values, extract_numpy=True)
        # 确保数组的形状符合块的形状要求
        values = ensure_block_shape(values, ndim)

    # 检查 values 的维度是否符合预期
    check_ndim(values, placement, ndim)

    # 可能强制转换 values 的值
    values = maybe_coerce_values(values)

    # 返回根据 klass 类型创建的块对象，传入 values 和 ndim 参数，使用给定的 placement
    return klass(values, ndim=ndim, placement=placement)
# 如果未提供 `ndim` 参数，根据 `placement` 和 `values` 推断其值。
def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    """
    If `ndim` is not provided, infer it from placement and values.
    如果没有提供 `ndim` 参数，则根据 `placement` 和 `values` 推断其值。
    """
    # GH#38134 Block constructor now assumes ndim is not None
    # 如果 values 的数据类型不是 numpy 的数据类型
    if ndim is None:
        # 如果 values 的数据类型不是 numpy 的数据类型
        if not isinstance(values.dtype, np.dtype):
            # 如果 placement 的长度不为 1，则设置 ndim 为 1
            if len(placement) != 1:
                ndim = 1
            else:
                # 否则设置 ndim 为 2
                ndim = 2
        else:
            # 否则使用 values 的维度作为 ndim 的值
            ndim = values.ndim
    # 返回推断或提供的 ndim 值
    return ndim
```