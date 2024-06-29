# `D:\src\scipysrc\pandas\pandas\core\internals\blocks.py`

```
# 引入未来的注解功能，使得类型提示更加强大
from __future__ import annotations

# 导入用于检查类型的工具
import inspect
# 导入正则表达式模块
import re
# 导入类型提示相关的工具
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    final,
)
# 导入警告模块
import warnings
# 弱引用模块，用于创建对象的弱引用
import weakref

# 导入 NumPy 库
import numpy as np

# 从 pandas 库中导入各种子模块和类
# pandas._libs 下的模块
from pandas._libs import (
    NaT,
    internals as libinternals,
    lib,
)
# pandas._libs.internals 下的模块
from pandas._libs.internals import (
    BlockPlacement,
    BlockValuesRefs,
)
# pandas._libs.missing 下的模块
from pandas._libs.missing import NA
# pandas._typing 下的类型定义
from pandas._typing import (
    ArrayLike,
    AxisInt,
    DtypeBackend,
    DtypeObj,
    FillnaOptions,
    IgnoreRaise,
    InterpolateOptions,
    QuantileInterpolation,
    Self,
    Shape,
    npt,
)
# pandas.errors 下的错误类型
from pandas.errors import (
    AbstractMethodError,
    OutOfBoundsDatetime,
)
# pandas.util._decorators 下的装饰器
from pandas.util._decorators import cache_readonly
# pandas.util._exceptions 下的异常处理工具
from pandas.util._exceptions import find_stack_level
# pandas.util._validators 下的验证器
from pandas.util._validators import validate_bool_kwarg

# pandas.core.dtypes.astype 下的类型转换函数
from pandas.core.dtypes.astype import (
    astype_array_safe,
    astype_is_view,
)
# pandas.core.dtypes.cast 下的类型转换函数和异常
from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    convert_dtypes,
    find_result_type,
    np_can_hold_element,
)
# pandas.core.dtypes.common 下的通用类型检查函数
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
)
# pandas.core.dtypes.dtypes 下的特定数据类型定义
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    NumpyEADtype,
    PeriodDtype,
)
# pandas.core.dtypes.generic 下的泛型数据类型
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCNumpyExtensionArray,
    ABCSeries,
)
# pandas.core.dtypes.missing 下的缺失数据处理函数
from pandas.core.dtypes.missing import (
    is_valid_na_for_dtype,
    isna,
    na_value_for_dtype,
)
# pandas.core 下的核心模块
from pandas.core import missing
# pandas.core.algorithms 下的算法模块
import pandas.core.algorithms as algos
# pandas.core.array_algos.putmask 下的掩码操作函数
from pandas.core.array_algos.putmask import (
    extract_bool_array,
    putmask_inplace,
    putmask_without_repeat,
    setitem_datetimelike_compat,
    validate_putmask,
)
# pandas.core.array_algos.quantile 下的分位数计算函数
from pandas.core.array_algos.quantile import quantile_compat
# pandas.core.array_algos.replace 下的替换操作函数
from pandas.core.array_algos.replace import (
    compare_or_regex_search,
    replace_regex,
    should_use_regex,
)
# pandas.core.array_algos.transforms 下的转换函数
from pandas.core.array_algos.transforms import shift
# pandas.core.arrays 下的数组类型
from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
# pandas.core.base 下的基础对象
from pandas.core.base import PandasObject
# pandas.core.common 下的通用功能
import pandas.core.common as com
# pandas.core.computation 下的表达式计算相关
from pandas.core.computation import expressions
# pandas.core.construction 下的构造函数
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
# pandas.core.indexers 下的索引器
from pandas.core.indexers import check_setitem_lengths
# pandas.core.indexes.base 下的索引基类
from pandas.core.indexes.base import get_values_for_csv

# 如果是类型检查环境，则导入额外的类型
if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Sequence,
    )
    # pandas.core.api 下的 Index 类型
    from pandas.core.api import Index
    # pandas.core.arrays._mixins 下的扩展数组混合类型
    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

# 使用 NumPy 创建一个对象类型的数据类型，用于快速比较
_dtype_obj = np.dtype("object")

# 定义 Block 类，继承自 PandasObject 和 libinternals.Block
class Block(PandasObject, libinternals.Block):
    """
    pandas 数据块类的扩展定义，继承自 PandasObject 和 libinternals.Block
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """

    # 定义一个用于存储同一数据类型的多维数据单元的类，存放在 pandas 数据结构中
    values: np.ndarray | ExtensionArray
    # 数据单元的维度
    ndim: int
    # BlockValuesRefs 类型的引用
    refs: BlockValuesRefs
    # __init__ 方法的类型是 Callable

    # 禁用 __slots__，允许动态添加属性
    __slots__ = ()
    # 默认情况下不是数值类型
    is_numeric = False

    @final
    @cache_readonly
    def _validate_ndim(self) -> bool:
        """
        We validate dimension for blocks that can hold 2D values, which for now
        means numpy dtypes or DatetimeTZDtype.
        """
        # 获取当前数据单元的数据类型
        dtype = self.dtype
        # 如果数据类型不是 ExtensionDtype，或者是 DatetimeTZDtype，则返回 True
        return not isinstance(dtype, ExtensionDtype) or isinstance(
            dtype, DatetimeTZDtype
        )

    @final
    @cache_readonly
    def is_object(self) -> bool:
        # 判断当前数据单元是否是对象类型
        return self.values.dtype == _dtype_obj

    @final
    @cache_readonly
    def is_extension(self) -> bool:
        # 判断当前数据单元是否是扩展类型（不是 numpy 数据类型）
        return not lib.is_np_dtype(self.values.dtype)

    @final
    @cache_readonly
    def _can_consolidate(self) -> bool:
        # 判断当前数据单元是否可以进行合并
        # 对于 DatetimeTZDtype 类型，可以进行合并，但目前不实现此功能
        return not self.is_extension

    @final
    @cache_readonly
    def _consolidate_key(self):
        # 返回用于合并的键
        return self._can_consolidate, self.dtype.name

    @final
    @cache_readonly
    def _can_hold_na(self) -> bool:
        """
        Can we store NA values in this Block?
        """
        # 判断当前数据单元是否能够存储 NA 值
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in "iub"
        return dtype._can_hold_na

    @final
    @property
    def is_bool(self) -> bool:
        """
        We can be bool if a) we are bool dtype or b) object dtype with bool objects.
        """
        # 判断当前数据单元是否是布尔类型
        return self.values.dtype == np.dtype(bool)

    @final
    def external_values(self):
        # 返回外部值的引用
        return external_values(self.values)

    @final
    @cache_readonly
    def fill_value(self):
        # 在 reindex_indexer 中使用的填充值
        return na_value_for_dtype(self.dtype, compat=False)

    @final
    def _standardize_fill_value(self, value):
        # 标准化填充值，如果传入标量 None，则在此处转换它
        if self.dtype != _dtype_obj and is_valid_na_for_dtype(value, self.dtype):
            value = self.fill_value
        return value

    @property
    def mgr_locs(self) -> BlockPlacement:
        # 获取管理位置（BlockPlacement 对象）
        return self._mgr_locs

    @mgr_locs.setter
    def mgr_locs(self, new_mgr_locs: BlockPlacement) -> None:
        # 设置管理位置（BlockPlacement 对象）
        self._mgr_locs = new_mgr_locs

    @final
    def make_block(
        self,
        values,
        placement: BlockPlacement | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> Block:
        """
        Create a new block, with type inference propagate any values that are
        not specified
        """
        # 创建一个新的数据块，如果未指定，则推断类型并传播任何值
        if placement is None:
            placement = self._mgr_locs
        # 如果是扩展类型，确保数据块的形状与当前维度匹配
        if self.is_extension:
            values = ensure_block_shape(values, ndim=self.ndim)

        return new_block(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def make_block_same_class(
        self,
        values,
        placement: BlockPlacement | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> Self:
        """Wrap given values in a block of same type as self."""
        # 在2.0之前，我们调用ensure_wrapped_if_datetimelike，因为fastparquet依赖它，从2.0开始，调用者负责这个操作。
        # 如果未提供placement参数，则使用self._mgr_locs作为默认值
        if placement is None:
            placement = self._mgr_locs

        # 我们假设maybe_coerce_values已经被调用过
        # 返回一个与当前对象类型相同的新对象，传入values作为数据，placement作为位置信息，ndim为当前对象的维度，refs为引用信息
        return type(self)(values, placement=placement, ndim=self.ndim, refs=refs)

    @final
    def __repr__(self) -> str:
        # 这里不打印所有的项
        name = type(self).__name__
        if self.ndim == 1:
            result = f"{name}: {len(self)} dtype: {self.dtype}"
        else:
            shape = " x ".join([str(s) for s in self.shape])
            result = f"{name}: {self.mgr_locs.indexer}, {shape}, dtype: {self.dtype}"

        return result

    @final
    def __len__(self) -> int:
        # 返回self.values的长度
        return len(self.values)

    @final
    def slice_block_columns(self, slc: slice) -> Self:
        """
        Perform __getitem__-like, return result as block.
        """
        # 根据给定的slice对象slc，生成新的_mgr_locs作为新的位置信息
        new_mgr_locs = self._mgr_locs[slc]

        # 根据给定的slice对象slc，对self的数据进行切片操作，生成新的数据new_values
        new_values = self._slice(slc)
        refs = self.refs
        # 返回一个与当前对象类型相同的新对象，传入new_values作为数据，new_mgr_locs作为位置信息，ndim为当前对象的维度，refs为引用信息
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def take_block_columns(self, indices: npt.NDArray[np.intp]) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        # 注意：这个方法只能从internals.concat中调用，我们可以验证不会对1列块（即ExtensionBlock）进行操作。

        # 根据给定的indices数组，生成新的_mgr_locs作为新的位置信息
        new_mgr_locs = self._mgr_locs[indices]

        # 根据给定的indices数组，对self的数据进行切片操作，生成新的数据new_values
        new_values = self._slice(indices)
        # 返回一个与当前对象类型相同的新对象，传入new_values作为数据，new_mgr_locs作为位置信息，ndim为当前对象的维度，refs为None
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=None)

    @final
    def getitem_block_columns(
        self, slicer: slice, new_mgr_locs: BlockPlacement, ref_inplace_op: bool = False
    ) -> Self:
        """
        Perform __getitem__-like, return result as block.

        Only supports slices that preserve dimensionality.
        """
        # 根据给定的slice对象slicer，对self的数据进行切片操作，生成新的数据new_values
        new_values = self._slice(slicer)
        # 如果ref_inplace_op为False或self.refs具有引用，则使用self.refs作为refs；否则，refs为None
        refs = self.refs if not ref_inplace_op or self.refs.has_reference() else None
        # 返回一个与当前对象类型相同的新对象，传入new_values作为数据，new_mgr_locs作为位置信息，ndim为当前对象的维度，refs为refs
        return type(self)(new_values, new_mgr_locs, self.ndim, refs=refs)

    @final
    def _can_hold_element(self, element: Any) -> bool:
        """require the same dtype as ourselves"""
        # 提取element的数组表示，使用numpy数组提取方式
        element = extract_array(element, extract_numpy=True)
        # 检查self.values是否能够容纳element的数据类型
        return can_hold_element(self.values, element)
    # 确定是否应该原地设置 self.values[indexer] = value 或者需要进行类型转换
    def should_store(self, value: ArrayLike) -> bool:
        """
        Should we set self.values[indexer] = value inplace or do we need to cast?

        Parameters
        ----------
        value : np.ndarray or ExtensionArray

        Returns
        -------
        bool
        """
        # 返回值的数据类型是否与 self 的数据类型相同
        return value.dtype == self.dtype

    # ---------------------------------------------------------------------
    # Apply/Reduce and Helpers

    @final
    def apply(self, func, **kwargs) -> list[Block]:
        """
        apply the function to my values; return a block if we are not
        one
        """
        # 将函数 func 应用于 self.values，返回结果
        result = func(self.values, **kwargs)

        # 可能需要对结果进行类型转换处理
        result = maybe_coerce_values(result)
        return self._split_op_result(result)

    @final
    def reduce(self, func) -> list[Block]:
        # 对二维数据进行应用函数操作，并将结果重塑为具有相同 mgr_locs 的单行 Block
        # 压缩操作将在更高级别完成
        assert self.ndim == 2

        result = func(self.values)

        # 如果 self.values 是一维的，则直接使用结果值
        if self.values.ndim == 1:
            res_values = result
        else:
            # 否则，将结果重塑为二维的，每行一个值
            res_values = result.reshape(-1, 1)

        # 创建一个新的 Block，并返回其作为列表元素
        nb = self.make_block(res_values)
        return [nb]

    @final
    def _split_op_result(self, result: ArrayLike) -> list[Block]:
        # 参考：split_and_operate
        # 如果结果的维度大于1并且其数据类型是 ExtensionDtype 的实例
        if result.ndim > 1 and isinstance(result.dtype, ExtensionDtype):
            # TODO(EA2D): 对于二维的 ExtensionArray，这一步可能是不必要的
            # 如果得到一个二维的 ExtensionArray，需要将其拆分成一维的片段
            nbs = []
            for i, loc in enumerate(self._mgr_locs):
                if not is_1d_only_ea_dtype(result.dtype):
                    vals = result[i : i + 1]
                else:
                    vals = result[i]

                bp = BlockPlacement(loc)
                block = self.make_block(values=vals, placement=bp)
                nbs.append(block)
            return nbs

        # 否则，创建一个新的 Block，并返回其作为列表元素
        nb = self.make_block(result)
        return [nb]

    @final
    def _split(self) -> Generator[Block, None, None]:
        """
        Split a block into a list of single-column blocks.
        """
        # 确保 self 是二维的
        assert self.ndim == 2

        # 遍历 self._mgr_locs，并根据索引切分 self.values，每次生成一个单列 Block
        for i, ref_loc in enumerate(self._mgr_locs):
            vals = self.values[slice(i, i + 1)]

            bp = BlockPlacement(ref_loc)
            nb = type(self)(vals, placement=bp, ndim=2, refs=self.refs)
            yield nb

    @final
    def split_and_operate(self, func, *args, **kwargs) -> list[Block]:
        """
        Split the block and apply func column-by-column.

        Parameters
        ----------
        func : Block method
        *args
        **kwargs

        Returns
        -------
        List[Block]
        """
        # 确保 self 是二维的且行数不为1
        assert self.ndim == 2 and self.shape[0] != 1

        # 分割 Block，并逐列应用 func 函数
        res_blocks = []
        for nb in self._split():
            rbs = func(nb, *args, **kwargs)
            res_blocks.extend(rbs)
        return res_blocks
    # ---------------------------------------------------------------------
    # Up/Down-casting

    @final
    def coerce_to_target_dtype(self, other, warn_on_upcast: bool = False) -> Block:
        """
        coerce the current block to a dtype compat for other
        we will return a block, possibly object, and not raise

        we can also safely try to coerce to the same dtype
        and will receive the same block
        """
        # 查找适合当前块和其他块的结果数据类型
        new_dtype = find_result_type(self.values.dtype, other)

        # 如果新数据类型与当前块的数据类型相同，抛出断言错误，避免递归错误
        if new_dtype == self.dtype:
            raise AssertionError(
                "Something has gone wrong, please report a bug at "
                "https://github.com/pandas-dev/pandas/issues"
            )

        # 在将来的 pandas 版本中，默认情况下，将 `nan` 设置到整数系列不会引发警告
        if (
            is_scalar(other)
            and is_integer_dtype(self.values.dtype)
            and isna(other)
            and other is not NaT
            and not (
                isinstance(other, (np.datetime64, np.timedelta64)) and np.isnat(other)
            )
        ):
            warn_on_upcast = False
        elif (
            isinstance(other, np.ndarray)
            and other.ndim == 1
            and is_integer_dtype(self.values.dtype)
            and is_float_dtype(other.dtype)
            and lib.has_only_ints_or_nan(other)
        ):
            warn_on_upcast = False

        # 如果 warn_on_upcast 为真，发出警告，指示不兼容的数据类型设置将在将来的 pandas 版本中引发错误
        if warn_on_upcast:
            warnings.warn(
                f"Setting an item of incompatible dtype is deprecated "
                "and will raise an error in a future version of pandas. "
                f"Value '{other}' has dtype incompatible with {self.values.dtype}, "
                "please explicitly cast to a compatible dtype first.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        # 如果当前块的数据类型与新数据类型相同，抛出断言错误
        if self.values.dtype == new_dtype:
            raise AssertionError(
                f"Did not expect new dtype {new_dtype} to equal self.dtype "
                f"{self.values.dtype}. Please report a bug at "
                "https://github.com/pandas-dev/pandas/issues."
            )

        try:
            # 尝试将当前块强制转换为新的数据类型并返回
            return self.astype(new_dtype)
        except OutOfBoundsDatetime as err:
            # 如果尝试将低分辨率的 dt64 上升到高分辨率时，可能存在超出边界的条目
            # 抛出带有更多信息的 OutOfBoundsDatetime 错误
            raise OutOfBoundsDatetime(
                f"Incompatible (high-resolution) value for dtype='{self.dtype}'. "
                "Explicitly cast before operating."
            ) from err

    @final
    def convert(self) -> list[Block]:
        """
        Attempt to coerce any object types to better types. Return a copy
        of the block (if copy = True).
        """
        # 如果不是对象类型，则返回当前块的浅拷贝的列表
        if not self.is_object:
            return [self.copy(deep=False)]

        # 如果数据维度不是1或者第一个维度不是1，则对块进行分割并应用 convert 方法
        blocks = self.split_and_operate(Block.convert)
        # 检查所有块的 dtype 是否都是对象类型，如果是，则返回当前块的浅拷贝的列表
        if all(blk.dtype.kind == "O" for blk in blocks):
            return [self.copy(deep=False)]
        return blocks

        # 如果值的维度是2，则检查是否只有一行，如果是，则将 values 变量设置为其第一行的值
        values = self.values
        if values.ndim == 2:
            # 上面的检查确保只有 values.shape[0] == 1，避免使用 .ravel 方法可能会产生拷贝
            values = values[0]

        # 使用 lib.maybe_convert_objects 函数尝试将对象类型转换为更好的类型
        res_values = lib.maybe_convert_objects(
            values,  # type: ignore[arg-type]
            convert_non_numeric=True,
        )
        refs = None
        # 如果 res_values 和 values 是同一个对象，则将 refs 设置为 self.refs
        if res_values is values:
            refs = self.refs

        # 确保 res_values 的块形状与 self 的维度一致
        res_values = ensure_block_shape(res_values, self.ndim)
        # 可能强制转换 res_values 的值
        res_values = maybe_coerce_values(res_values)
        # 使用 make_block 方法创建新的块对象，使用 res_values 作为值，refs 作为引用
        return [self.make_block(res_values, refs=refs)]

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: DtypeBackend = "numpy_nullable",
    ) -> list[Block]:
        # 如果 infer_objects 为 True 并且当前块是对象类型，则使用 convert 方法进行转换
        if infer_objects and self.is_object:
            blks = self.convert()
        else:
            blks = [self]

        # 如果没有任何类型转换标志为 True，则返回 blks 的浅拷贝列表
        if not any(
            [convert_floating, convert_integer, convert_boolean, convert_string]
        ):
            return [b.copy(deep=False) for b in blks]

        # 存储结果的空列表
        rbs = []
        for blk in blks:
            # 逐列确定块的数据类型
            # 如果块的维度是1或者 self 的第一个维度是1，则将其作为单块处理，否则作为列表处理
            sub_blks = (
                [blk] if blk.ndim == 1 or self.shape[0] == 1 else list(blk._split())
            )
            # 对每个子块使用 convert_dtypes 函数进行数据类型转换
            dtypes = [
                convert_dtypes(
                    b.values,
                    convert_string,
                    convert_integer,
                    convert_boolean,
                    convert_floating,
                    infer_objects,
                    dtype_backend,
                )
                for b in sub_blks
            ]
            # 如果所有子块的数据类型都与当前块的数据类型相同，则将当前块的浅拷贝添加到结果列表
            if all(dtype == self.dtype for dtype in dtypes):
                rbs.append(blk.copy(deep=False))
                continue

            # 否则，对每个子块进行数据类型转换，并根据情况进行拷贝
            for dtype, b in zip(dtypes, sub_blks):
                rbs.append(b.astype(dtype=dtype, squeeze=b.ndim != 1))
        return rbs

    # ---------------------------------------------------------------------
    # Array-Like Methods

    @final
    @cache_readonly
    def dtype(self) -> DtypeObj:
        # 返回当前块的值的数据类型
        return self.values.dtype

    @final
    # 对数据类型进行转换操作，返回转换后的块对象
    def astype(
        self,
        dtype: DtypeObj,
        errors: IgnoreRaise = "raise",
        squeeze: bool = False,
    ) -> Block:
        """
        Coerce to the new dtype.

        Parameters
        ----------
        dtype : np.dtype or ExtensionDtype
            新的数据类型
        errors : str, {'raise', 'ignore'}, default 'raise'
            - ``raise`` : 允许引发异常
            - ``ignore`` : 忽略异常，出错时返回原始对象
            异常处理方式
        squeeze : bool, default False
            如果仅给定一个列，将值挤压成 ndim=1

        Returns
        -------
        Block
            返回转换后的块对象
        """
        values = self.values
        # 如果 squeeze=True 且当前数据为二维且是 1D 的扩展类型，则进行挤压操作
        if squeeze and values.ndim == 2 and is_1d_only_ea_dtype(dtype):
            if values.shape[0] != 1:
                raise ValueError("Can not squeeze with more than one column.")
            values = values[0, :]  # type: ignore[call-overload]

        # 安全地对值数组进行类型转换
        new_values = astype_array_safe(values, dtype, errors=errors)

        # 可能需要强制转换新值
        new_values = maybe_coerce_values(new_values)

        refs = None
        # 如果旧值和新值的数据类型是视图，则保留引用关系
        if astype_is_view(values.dtype, new_values.dtype):
            refs = self.refs

        # 创建新的块对象
        newb = self.make_block(new_values, refs=refs)
        # 检查新块对象的形状是否与当前对象形状相同
        if newb.shape != self.shape:
            raise TypeError(
                f"cannot set astype for dtype "
                f"({self.dtype.name} [{self.shape}]) to different shape "
                f"({newb.dtype.name} [{newb.shape}])"
            )
        return newb

    @final
    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Block:
        """convert to our native types format"""
        # 获取用于 CSV 输出的值数组
        result = get_values_for_csv(
            self.values,
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )
        return self.make_block(result)

    @final
    def copy(self, deep: bool = True) -> Self:
        """copy constructor"""
        values = self.values
        refs: BlockValuesRefs | None
        if deep:
            # 如果进行深拷贝，则复制值数组
            values = values.copy()
            refs = None
        else:
            # 否则保留引用关系
            refs = self.refs
        return type(self)(values, placement=self._mgr_locs, ndim=self.ndim, refs=refs)

    # ---------------------------------------------------------------------
    # Copy-on-Write Helpers

    def _maybe_copy(self, inplace: bool) -> Self:
        # 如果 inplace=True，则根据是否存在引用来决定进行深拷贝或浅拷贝
        if inplace:
            deep = self.refs.has_reference()
            return self.copy(deep=deep)
        return self.copy()

    @final
    def _get_refs_and_copy(self, inplace: bool):
        refs = None
        copy = not inplace
        # 如果 inplace=True，则根据是否存在引用来决定是否需要进行拷贝操作
        if inplace:
            if self.refs.has_reference():
                copy = True
            else:
                refs = self.refs
        return copy, refs

    # ---------------------------------------------------------------------
    # Replace

    @final
    def replace(
        self,
        to_replace,
        value,
        inplace: bool = False,
        # 如果我们从 replace_list 调用，可能预先计算了 mask
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> list[Block]:
        """
        用 value 替换 to_replace 的值，可能会在此处创建新的块，这只是一个调用 putmask 的过程。
        """

        # 注意：在 NDFrame.replace 中的检查确保我们永远不会用类似列表的 to_replace 或 value 进入这里，因为那些情况会经过 replace_list 处理
        values = self.values

        if not self._can_hold_element(to_replace):
            # 我们无法容纳 `to_replace`，所以可以立即知道替换操作是空操作。
            # 注意：如果 to_replace 是一个列表，NDFrame.replace 将调用 replace_list 而不是 replace。
            return [self.copy(deep=False)]

        if mask is None:
            mask = missing.mask_missing(values, to_replace)
        if not mask.any():
            # 注意：我们在这里以 test_replace_extension_other 错误进入，因为 _can_hold_element 是错误的。
            return [self.copy(deep=False)]

        elif self._can_hold_element(value):
            # TODO(CoW): 也许在这里分割，根据 mask 为 True 的列和其余列？
            blk = self._maybe_copy(inplace)
            putmask_inplace(blk.values, mask, value)
            return [blk]

        elif self.ndim == 1 or self.shape[0] == 1:
            if value is None or value is NA:
                blk = self.astype(np.dtype(object))
            else:
                blk = self.coerce_to_target_dtype(value)
            return blk.replace(
                to_replace=to_replace,
                value=value,
                inplace=True,
                mask=mask,
            )

        else:
            # 拆分以便仅在必要时进行类型转换
            blocks = []
            for i, nb in enumerate(self._split()):
                blocks.extend(
                    type(self).replace(
                        nb,
                        to_replace=to_replace,
                        value=value,
                        inplace=True,
                        mask=mask[i : i + 1],
                    )
                )
            return blocks

    @final
    def _replace_regex(
        self,
        to_replace,
        value,
        inplace: bool = False,
        mask=None,
    ):
        """
        用正则表达式替换 to_replace 的值，返回替换后的副本。
        """
    ) -> list[Block]:
        """
        Replace elements by the given value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        inplace : bool, default False
            Perform inplace modification.
        mask : array-like of bool, optional
            True indicate corresponding element is ignored.

        Returns
        -------
        List[Block]
            A list containing the modified Block object.

        """
        # 检查是否可以替换元素，若不满足条件则返回原对象的浅拷贝列表
        if not self._can_hold_element(to_replace):
            return [self.copy(deep=False)]

        # 编译正则表达式对象
        rx = re.compile(to_replace)

        # 复制当前对象，如果需要则进行原地修改
        block = self._maybe_copy(inplace)

        # 调用替换函数，根据正则表达式替换元素
        replace_regex(block.values, rx, value, mask)
        
        # 返回包含修改后 Block 对象的列表
        return [block]

    @final
    def replace_list(
        self,
        src_list: Iterable[Any],
        dest_list: Sequence[Any],
        inplace: bool = False,
        regex: bool = False,
    ) -> list[Block]:
        """
        See BlockManager.replace_list docstring.
        """
        # 获取 self.values 的引用
        values = self.values

        # 排除我们知道不会包含的元素
        # 生成源列表和目标列表的元组对，仅包含可以存放在当前 BlockManager 中的元素
        pairs = [
            (x, y) for x, y in zip(src_list, dest_list) if self._can_hold_element(x)
        ]
        # 如果没有符合条件的元组对，则返回当前 BlockManager 的一个浅拷贝列表
        if not len(pairs):
            return [self.copy(deep=False)]

        # 计算源列表长度减1，用于后续判断
        src_len = len(pairs) - 1

        # 如果 values 的数据类型是字符串类型
        if is_string_dtype(values.dtype):
            # 计算掩码一次，以避免在每次调用 comp 前重复相同的计算
            na_mask = ~isna(values)
            # 生成多个掩码数组的迭代器，每个数组与 pairs 中的源元素相关联
            masks: Iterable[npt.NDArray[np.bool_]] = (
                extract_bool_array(
                    cast(
                        ArrayLike,
                        compare_or_regex_search(
                            values, s[0], regex=regex, mask=na_mask
                        ),
                    )
                )
                for s in pairs
            )
        else:
            # 如果不需要检查正则表达式，则使用 missing.mask_missing 生成掩码数组的迭代器
            masks = (missing.mask_missing(values, s[0]) for s in pairs)

        # 如果 inplace=True，则将迭代器转换为列表以材料化掩码数组，因为在替换过程中掩码可能会改变
        if inplace:
            masks = list(masks)

        # 在此处不要设置引用，否则稍后再次检查时可能会错误地认为存在引用
        rb = [self]

        # 遍历 pairs 中的元组对和对应的掩码数组
        for i, ((src, dest), mask) in enumerate(zip(pairs, masks)):
            new_rb: list[Block] = []

            # _replace_coerce 可能会将一个块拆分为单列块，因此跟踪索引以知道在掩码中的位置
            for blk_num, blk in enumerate(rb):
                if len(rb) == 1:
                    m = mask
                else:
                    mib = mask
                    assert not isinstance(mib, bool)
                    m = mib[blk_num : blk_num + 1]

                # 调用块的 _replace_coerce 方法进行替换操作
                result = blk._replace_coerce(
                    to_replace=src,
                    value=dest,
                    mask=m,
                    inplace=inplace,
                    regex=regex,
                )

                # 如果不是最后一个替换操作，清理中间引用，以避免不必要的复制
                if i != src_len:
                    for b in result:
                        ref = weakref.ref(b)
                        b.refs.referenced_blocks.pop(
                            b.refs.referenced_blocks.index(ref)
                        )

                # 将结果扩展到新的结果块列表中
                new_rb.extend(result)
            
            # 更新 rb 到新的结果块列表
            rb = new_rb
        
        # 返回最终的结果块列表
        return rb

    @final
    # 替换指定的值，根据给定的布尔数组（掩码）进行替换操作，返回替换后的块列表
    def _replace_coerce(
        self,
        to_replace,
        value,
        mask: npt.NDArray[np.bool_],
        inplace: bool = True,
        regex: bool = False,
    ) -> list[Block]:
        """
        Replace value corresponding to the given boolean array with another
        value.

        Parameters
        ----------
        to_replace : object or pattern
            Scalar to replace or regular expression to match.
        value : object
            Replacement object.
        mask : np.ndarray[bool]
            True indicate corresponding element is ignored.
        inplace : bool, default True
            Perform inplace modification.
        regex : bool, default False
            If true, perform regular expression substitution.

        Returns
        -------
        List[Block]
        """
        # 如果需要使用正则表达式替换
        if should_use_regex(regex, to_replace):
            return self._replace_regex(
                to_replace,
                value,
                inplace=inplace,
                mask=mask,
            )
        else:
            # 如果 value 为 None
            if value is None:
                # 处理特定情况 gh-45601, gh-45836, gh-46634
                if mask.any():
                    # 检查是否存在引用
                    has_ref = self.refs.has_reference()
                    # 将块转换为对象类型，处理 inplace 情况
                    nb = self.astype(np.dtype(object))
                    # 如果不是 inplace，则进行复制操作
                    if not inplace:
                        nb = nb.copy()
                    # 如果 inplace 为真且存在引用，则进行复制操作
                    elif inplace and has_ref and nb.refs.has_reference():
                        # 在 astype 操作中不复制，并且之前存在引用
                        nb = nb.copy()
                    # 使用掩码进行原位赋值操作
                    putmask_inplace(nb.values, mask, value)
                    return [nb]
                # 如果不满足替换条件，则返回当前块
                return [self]
            # 如果不使用正则表达式替换，则调用 replace 方法
            return self.replace(
                to_replace=to_replace,
                value=value,
                inplace=inplace,
                mask=mask,
            )

    # ---------------------------------------------------------------------
    # 2D Methods - Shared by NumpyBlock and NDArrayBackedExtensionBlock
    #  but not ExtensionBlock

    # 兼容性函数，对于仅支持 1D 扩展数组的情况
    def _maybe_squeeze_arg(self, arg: np.ndarray) -> np.ndarray:
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return arg

    # 兼容性函数，对于仅支持 1D 扩展数组的情况
    def _unwrap_setitem_indexer(self, indexer):
        """
        For compatibility with 1D-only ExtensionArrays.
        """
        return indexer

    # 注意：此属性不能设置为 cache_readonly，因为在 mgr.set_values 中我们会钉住新的 .values，可能会有不同的形状 GH#42631
    @property
    def shape(self) -> Shape:
        return self.values.shape
    def iget(self, i: int | tuple[int, int] | tuple[slice, int]) -> np.ndarray:
        # 当索引参数为 tuple[slice, int] 时，slice 始终是 slice(None)
        # 注意：仅当 self.ndim == 2 时才会到达此处
        # Invalid index type "Union[int, Tuple[int, int], Tuple[slice, int]]"
        # for "Union[ndarray[Any, Any], ExtensionArray]"; expected type
        # "Union[int, integer[Any]]"
        return self.values[i]  # type: ignore[index]

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
    ) -> ArrayLike:
        """返回我值的一个切片"""
        
        return self.values[slicer]

    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
        """
        就地修改块值为新的项目值。

        如果 copy=True，先在修改之前复制底层值（用于写时复制）。

        Notes
        -----
        `set_inplace` 永远不会创建新数组或新块，而 `setitem`
        可能会创建新数组并始终创建新块。

        调用者负责检查 values.dtype == self.dtype。
        """
        if copy:
            self.values = self.values.copy()
        self.values[locs] = values

    @final
    def take_nd(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt,
        new_mgr_locs: BlockPlacement | None = None,
        fill_value=lib.no_default,
    ) -> Block:
        """
        根据索引器取值并返回作为块。

        """
        values = self.values

        if fill_value is lib.no_default:
            fill_value = self.fill_value
            allow_fill = False
        else:
            allow_fill = True

        # 注意：algos.take_nd 具有类似 coerce_to_target_dtype 的提升逻辑
        new_values = algos.take_nd(
            values, indexer, axis=axis, allow_fill=allow_fill, fill_value=fill_value
        )

        # 被三个管理器的不同调用点调用，都满足这些断言
        if isinstance(self, ExtensionBlock):
            # 注意：在此情况下，algos.take_nd 调用中会忽略 'axis' 关键字参数
            assert not (self.ndim == 1 and new_mgr_locs is None)
        assert not (axis == 0 and new_mgr_locs is None)

        if new_mgr_locs is None:
            new_mgr_locs = self._mgr_locs

        if new_values.dtype != self.dtype:
            return self.make_block(new_values, new_mgr_locs)
        else:
            return self.make_block_same_class(new_values, new_mgr_locs)

    def _unstack(
        self,
        unstacker,
        fill_value,
        new_placement: npt.NDArray[np.intp],
        needs_masking: npt.NDArray[np.bool_],
    ):
        """
        Return a list of unstacked blocks of self
        
        Parameters
        ----------
        unstacker : reshape._Unstacker
            An instance of `_Unstacker` used for reshaping operations.
        fill_value : int
            Only used in ExtensionBlock._unstack
            Integer value used for filling missing elements, specific to ExtensionBlock.
        new_placement : np.ndarray[np.intp]
            Array indicating new placement of elements after unstacking.
        allow_fill : bool
            Boolean flag indicating whether to allow filling missing values.
        needs_masking : np.ndarray[bool]
            Boolean array specifying elements needing masking.

        Returns
        -------
        blocks : list of Block
            New blocks of unstacked values.
        mask : array-like of bool
            The mask of columns of `blocks` we should keep.
        """
        new_values, mask = unstacker.get_new_values(
            self.values.T, fill_value=fill_value
        )
        
        mask = mask.any(0)
        # TODO: in all tests we have mask.all(); can we rely on that?
        
        # Note: these next two lines ensure that
        #  mask.sum() == sum(len(nb.mgr_locs) for nb in blocks)
        #  which the calling function needs in order to pass verify_integrity=False
        #  to the BlockManager constructor
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]
        
        bp = BlockPlacement(new_placement)
        blocks = [new_block_2d(new_values, placement=bp)]
        return blocks, mask

    # ---------------------------------------------------------------------
    ```python`
        def setitem(self, indexer, value) -> Block:
            """
            Attempt self.values[indexer] = value, possibly creating a new array.
    
            Parameters
            ----------
            indexer : tuple, list-like, array-like, slice, int
                The subset of self.values to set
            value : object
                The value being set
    
            Returns
            -------
            Block
    
            Notes
            -----
            `indexer` is a direct slice/positional indexer. `value` must
            be a compatible shape.
            """
    
            # 标准化填充值，确保与数据结构兼容
            value = self._standardize_fill_value(value)
    
            # 将self.values转换为numpy数组
            values = cast(np.ndarray, self.values)
    
            # 如果数据维度为2，将其转置
            if self.ndim == 2:
                values = values.T
    
            # 检查长度是否匹配
            check_setitem_lengths(indexer, value, values)
    
            # 如果数据类型不是对象类型
            if self.dtype != _dtype_obj:
                # GH48933: extract_array会将pd.Series转换为np.ndarray
                value = extract_array(value, extract_numpy=True)
    
            try:
                # 检查是否能够用当前的dtype存储value
                casted = np_can_hold_element(values.dtype, value)
            except LossySetitemError:
                # 如果当前dtype无法存储value，将value强制转换为目标dtype
                nb = self.coerce_to_target_dtype(value, warn_on_upcast=True)
                return nb.setitem(indexer, value)
            else:
                # 如果数据类型是对象类型
                if self.dtype == _dtype_obj:
                    # TODO: 避免必须构造values[indexer]
                    vi = values[indexer]
                    # 检查vi是否类列表对象，这里在test_iloc_setitem_custom_object中验证is_scalar失败
                    if lib.is_list_like(vi):
                        casted = setitem_datetimelike_compat(values, len(vi), casted)
    
                # 检查是否需要复制self
                self = self._maybe_copy(inplace=True)
    
                # 再次将values转为numpy数组，这里是为了处理可能的转置情况
                values = cast(np.ndarray, self.values.T)
    
                # 如果casted是一维数组且长度为1
                if isinstance(casted, np.ndarray) and casted.ndim == 1 and len(casted) == 1:
                    # NumPy 1.25弃用警告：https://github.com/numpy/numpy/pull/10615
                    casted = casted[0, ...]
    
                try:
                    # 尝试设置values[indexer]的值为casted
                    values[indexer] = casted
                except (TypeError, ValueError) as err:
                    # 如果casted是类列表对象，则抛出值错误异常
                    if is_list_like(casted):
                        raise ValueError(
                            "setting an array element with a sequence."
                        ) from err
                    raise
    
            # 返回更新后的self
            return self
    def putmask(self, mask, new) -> list[Block]:
        """
        将数据根据掩码进行屏蔽；可能会创建新的块数据类型。

        返回结果块列表。

        Parameters
        ----------
        mask : np.ndarray[bool], SparseArray[bool], or BooleanArray
            控制屏蔽操作的掩码数组，可以是布尔型的numpy数组、稀疏数组或者布尔数组。
        new : a ndarray/object
            新数据，可以是任意ndarray或对象。

        Returns
        -------
        List[Block]
            返回处理后的块列表。
        """
        orig_mask = mask  # 保存原始掩码数据
        values = cast(np.ndarray, self.values)  # 获取self对象的值，并进行类型转换为ndarray
        mask, noop = validate_putmask(values.T, mask)  # 校验掩码，确保格式正确，并返回处理后的掩码和操作标记
        assert not isinstance(new, (ABCIndex, ABCSeries, ABCDataFrame))  # 确保new不是索引对象或数据系列

        if new is lib.no_default:
            new = self.fill_value  # 如果new为lib.no_default，则使用默认填充值self.fill_value

        new = self._standardize_fill_value(new)  # 标准化填充值new
        new = extract_array(new, extract_numpy=True)  # 提取数组形式的new，并确保是numpy数组类型

        if noop:
            return [self.copy(deep=False)]  # 如果无需操作，则返回当前对象的浅复制列表

        try:
            casted = np_can_hold_element(values.dtype, new)  # 检查是否能够将new的元素转换为values的dtype

            self = self._maybe_copy(inplace=True)  # 可能进行对象的复制操作，根据inplace参数决定
            values = cast(np.ndarray, self.values)  # 再次获取self对象的值，并进行类型转换为ndarray

            putmask_without_repeat(values.T, mask, casted)  # 使用屏蔽操作将casted应用到values上
            return [self]  # 返回包含当前对象的列表
        except LossySetitemError:
            if self.ndim == 1 or self.shape[0] == 1:
                # 不需要分割列

                if not is_list_like(new):
                    # 如果new不是列表型数据，则调用coerce_to_target_dtype方法处理，并递归调用putmask
                    return self.coerce_to_target_dtype(
                        new, warn_on_upcast=True
                    ).putmask(mask, new)
                else:
                    indexer = mask.nonzero()[0]  # 获取非零掩码的索引
                    nb = self.setitem(indexer, new[indexer])  # 使用索引器对对象进行设置
                    return [nb]  # 返回包含新块的列表

            else:
                is_array = isinstance(new, np.ndarray)  # 检查new是否为ndarray类型

                res_blocks = []  # 初始化结果块列表
                for i, nb in enumerate(self._split()):  # 遍历对象的分割块
                    n = new
                    if is_array:
                        # 对于每列使用不同的值
                        n = new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]  # 获取子掩码
                    rbs = nb.putmask(submask, n)  # 对分割块应用子掩码和新数据n进行屏蔽操作
                    res_blocks.extend(rbs)  # 将处理后的块添加到结果块列表中
                return res_blocks  # 返回结果块列表
    ) -> list[Block]:
        """
        fillna on the block with the value. If we fail, then convert to
        block to hold objects instead and try again
        """
        # 确保 inplace 参数为布尔值，用于指示是否原地修改数据
        inplace = validate_bool_kwarg(inplace, "inplace")

        if not self._can_hold_na:
            # 如果块不支持 NA 值，则可以直接返回当前块的浅复制
            noop = True
        else:
            # 获取当前块中 NA 值的掩码
            mask = isna(self.values)
            # 根据掩码验证并处理填充操作
            mask, noop = validate_putmask(self.values, mask)

        if noop:
            # 如果无法处理值，则返回当前块的浅复制
            return [self.copy(deep=False)]

        if limit is not None:
            # 如果设置了限制条件，则根据限制条件调整掩码
            mask[mask.cumsum(self.ndim - 1) > limit] = False

        if inplace:
            # 如果 inplace 为 True，则在原地修改块的值并返回修改后的块列表
            nbs = self.putmask(mask.T, value)
        else:
            # 如果 inplace 为 False，则返回根据条件生成的新块列表
            nbs = self.where(value, ~mask.T)
        return extend_blocks(nbs)

    def pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> list[Block]:
        if not self._can_hold_na:
            # 如果当前块不包含 NA 值，则填充操作无效，返回当前块的浅复制
            return [self.copy(deep=False)]

        copy, refs = self._get_refs_and_copy(inplace)

        # 调用 NumpyExtensionArray 的方法进行填充或反向填充
        vals = cast(NumpyExtensionArray, self.array_values)
        new_values = vals.T._pad_or_backfill(
            method=method,
            limit=limit,
            limit_area=limit_area,
            copy=copy,
        ).T

        # 提取新值并生成与当前块相同类型的块对象，并返回块列表
        data = extract_array(new_values, extract_numpy=True)
        return [self.make_block_same_class(data, refs=refs)]

    @final
    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        index: Index,
        inplace: bool = False,
        limit: int | None = None,
        limit_direction: Literal["forward", "backward", "both"] = "forward",
        limit_area: Literal["inside", "outside"] | None = None,
        **kwargs,
        ) -> list[Block]:
        # 如果当前块不包含 NA 值，则插值操作无效，返回当前块的浅复制
        if not self._can_hold_na:
            return [self.copy(deep=False)]

        # 获取副本和引用
        copy, refs = self._get_refs_and_copy(inplace)

        # 调度到 NumpyExtensionArray 方法进行插值
        vals = cast(NumpyExtensionArray, self.array_values)
        new_values = vals.T._interpolate(
            method=method,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            copy=copy,
            **kwargs,
        ).T

        # 提取新值并生成与当前块相同类型的块对象，并返回块列表
        data = extract_array(new_values, extract_numpy=True)
        return [self.make_block_same_class(data, refs=refs)]
    ) -> list[Block]:
        # 验证 inplace 参数是否为布尔类型，并进行处理
        inplace = validate_bool_kwarg(inplace, "inplace")
        # 错误: 非重叠的相等性检查 [...]
        if method == "asfreq":  # 错误类型: 忽略[比较重叠]
            # clean_fill_method 曾允许这种操作
            missing.clean_fill_method(method)

        if not self._can_hold_na:
            # 如果没有缺失值，则插值操作无效
            return [self.copy(deep=False)]

        if self.dtype == _dtype_obj:
            # GH#53631
            # 如果数据类型为 object，抛出类型错误
            name = {1: "Series", 2: "DataFrame"}[self.ndim]
            raise TypeError(f"{name} cannot interpolate with object dtype.")

        copy, refs = self._get_refs_and_copy(inplace)

        # 分派至 EA 方法（Estimated Arithmetic）。
        new_values = self.array_values.interpolate(
            method=method,
            axis=self.ndim - 1,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            copy=copy,
            **kwargs,
        )
        # 提取数组并确保为 NumPy 数组
        data = extract_array(new_values, extract_numpy=True)
        # 返回与当前对象相同类别的数据块列表
        return [self.make_block_same_class(data, refs=refs)]

    @final
    def diff(self, n: int) -> list[Block]:
        """返回值差异的数据块"""
        # 只有在 ndim == 2 时才会执行到这里
        # TODO(EA2D): 对于 2D EA，转置将是不必要的
        new_values = algos.diff(self.values.T, n, axis=0).T
        return [self.make_block(values=new_values)]

    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """将数据块按 periods 进行移动，可能会升级数据类型"""
        # 如果需要，将整数转换为浮点数。还需要处理布尔值等等
        axis = self.ndim - 1

        # 注意: 这里 periods 永远不会为 0，因为在 NDFrame.shift 的顶部已经处理了
        # 如果将来有所改变，我们可以检查 periods=0 并可能避免强制转换。

        if not lib.is_scalar(fill_value) and self.dtype != _dtype_obj:
            # 对于 object 数据类型，无需升级，用户可以传递几乎任何奇怪的 fill_value
            # 参见 test_shift_object_non_scalar_fill
            raise ValueError("fill_value must be a scalar")

        fill_value = self._standardize_fill_value(fill_value)

        try:
            # 错误: "np_can_hold_element" 的第一个参数类型与预期不符
            # "Union[dtype[Any], ExtensionDtype]"; 期望 "dtype[Any]"
            casted = np_can_hold_element(
                self.dtype,  # 错误类型: 忽略[arg-type]
                fill_value,
            )
        except LossySetitemError:
            nb = self.coerce_to_target_dtype(fill_value)
            return nb.shift(periods, fill_value=fill_value)

        else:
            values = cast(np.ndarray, self.values)
            new_values = shift(values, periods, axis, casted)
            # 返回与当前数据块相同类别的新数据块列表
            return [self.make_block_same_class(new_values)]

    @final
    @final
    # 定义一个方法，用于计算给定分位数的值
    def quantile(
        self,
        qs: Index,  # 传入的参数是一个索引对象，其中包含dtype为float64的数据
        interpolation: QuantileInterpolation = "linear",
    ) -> Block:
        """
        compute the quantiles of the

        Parameters
        ----------
        qs : Index
            The quantiles to be computed in float64.
        interpolation : str, default 'linear'
            Type of interpolation.

        Returns
        -------
        Block
        """
        # 我们应该总是具有 ndim == 2，因为 Series 会分派到 DataFrame
        assert self.ndim == 2
        assert is_list_like(qs)  # 调用者负责确保 qs 是类列表对象

        # 调用 quantile_compat 函数计算分位数
        result = quantile_compat(self.values, np.asarray(qs._values), interpolation)
        # 确保返回的结果形状符合二维要求，例如 IntegerArray、SparseArray 等情况
        result = ensure_block_shape(result, ndim=2)
        # 创建一个新的二维数据块并返回，使用 self._mgr_locs 控制放置
        return new_block_2d(result, placement=self._mgr_locs)

    @final
    # 定义一个方法，用于对数值进行指定精度的四舍五入
    def round(self, decimals: int) -> Self:
        """
        Rounds the values.
        If the block is not of an integer or float dtype, nothing happens.
        This is consistent with DataFrame.round behavivor.
        (Note: Series.round would raise)

        Parameters
        ----------
        decimals: int,
            Number of decimal places to round to.
            Caller is responsible for validating this
        """
        # 如果数据块不是数值类型或布尔类型，则返回副本
        if not self.is_numeric or self.is_bool:
            return self.copy(deep=False)
        # TODO: round only defined on BaseMaskedArray
        # Series also does this, so would need to fix both places
        # error: Item "ExtensionArray" of "Union[ndarray[Any, Any], ExtensionArray]"
        # has no attribute "round"
        # 调用数据块的 values 属性执行四舍五入操作，忽略 Union 类型的属性错误
        values = self.values.round(decimals)  # type: ignore[union-attr]

        refs = None
        # 如果四舍五入后的值与原始值相同，则保留引用信息
        if values is self.values:
            refs = self.refs

        # 创建一个与当前类相同的数据块，并返回
        return self.make_block_same_class(values, refs=refs)

    # ---------------------------------------------------------------------
    # Abstract Methods Overridden By EABackedBlock and NumpyBlock
    def delete(self, loc) -> list[Block]:
        """Deletes the locs from the block.

        We split the block to avoid copying the underlying data. We create new
        blocks for every connected segment of the initial block that is not deleted.
        The new blocks point to the initial array.
        """
        # 检查 loc 是否为可迭代对象，如果不是则转为列表
        if not is_list_like(loc):
            loc = [loc]

        # 如果块是一维的
        if self.ndim == 1:
            # 获取块的值数组
            values = cast(np.ndarray, self.values)
            # 删除指定位置的元素
            values = np.delete(values, loc)
            # 更新管理位置
            mgr_locs = self._mgr_locs.delete(loc)
            # 返回一个新块对象的列表，指向原始数组
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]

        # 如果删除位置中的最大索引超出了块的值数组的长度
        if np.max(loc) >= self.values.shape[0]:
            raise IndexError

        # 将最大的删除位置作为索引添加到 loc 中，以便收集最后一个索引器后的所有列（如果有的话）
        loc = np.concatenate([loc, [self.values.shape[0]]])
        # 获取管理位置数组的 numpy 数组表示
        mgr_locs_arr = self._mgr_locs.as_array
        # 用于存储新块的列表
        new_blocks: list[Block] = []

        # 上一个位置的初始化为 -1
        previous_loc = -1
        # TODO(CoW): This is tricky, if parent block goes out of scope
        # all split blocks are referencing each other even though they
        # don't share data
        # 如果存在引用，则将其分配给 refs，否则为 None
        refs = self.refs if self.refs.has_reference() else None
        # 遍历删除位置 loc
        for idx in loc:
            # 如果当前位置与上一个位置相差 1，则表示两者之间没有列
            if idx == previous_loc + 1:
                # 没有列需要处理
                pass
            else:
                # 使用切片获取值数组中上一个位置到当前位置之间的数据
                values = self.values[previous_loc + 1 : idx, :]  # type: ignore[call-overload]
                # 获取管理位置数组中上一个位置到当前位置之间的数据
                locs = mgr_locs_arr[previous_loc + 1 : idx]
                # 创建一个新的块对象 nb
                nb = type(self)(
                    values, placement=BlockPlacement(locs), ndim=self.ndim, refs=refs
                )
                # 将新块对象 nb 添加到 new_blocks 列表中
                new_blocks.append(nb)

            # 更新上一个位置为当前位置 idx
            previous_loc = idx

        # 返回新块对象的列表
        return new_blocks

    @property
    def is_view(self) -> bool:
        """return a boolean if I am possibly a view"""
        # 抛出抽象方法错误，需要在子类中实现
        raise AbstractMethodError(self)

    @property
    def array_values(self) -> ExtensionArray:
        """
        The array that Series.array returns. Always an ExtensionArray.
        """
        # 抛出抽象方法错误，需要在子类中实现
        raise AbstractMethodError(self)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return an internal format, currently just the ndarray
        this is often overridden to handle to_dense like operations
        """
        # 抛出抽象方法错误，需要在子类中实现
        raise AbstractMethodError(self)
    """
    Mixin for Block subclasses backed by ExtensionArray.
    """

    values: ExtensionArray

    @final
    def shift(self, periods: int, fill_value: Any = None) -> list[Block]:
        """
        Shift the block by `periods`.

        Dispatches to underlying ExtensionArray and re-boxes in an
        ExtensionBlock.
        """
        # Transpose since EA.shift is always along axis=0, while we want to shift
        # along rows.
        new_values = self.values.T.shift(periods=periods, fill_value=fill_value).T
        return [self.make_block_same_class(new_values)]

    @final
    def setitem(self, indexer, value):
        """
        Attempt self.values[indexer] = value, possibly creating a new array.

        This differs from Block.setitem by not allowing setitem to change
        the dtype of the Block.

        Parameters
        ----------
        indexer : tuple, list-like, array-like, slice, int
            The subset of self.values to set
        value : object
            The value being set

        Returns
        -------
        Block

        Notes
        -----
        `indexer` is a direct slice/positional indexer. `value` must
        be a compatible shape.
        """
        orig_indexer = indexer
        orig_value = value

        # Unwrap indexer to handle advanced indexing
        indexer = self._unwrap_setitem_indexer(indexer)
        # Ensure value has correct shape
        value = self._maybe_squeeze_arg(value)

        # Handle 2D values by transposing conditionally
        values = self.values
        if values.ndim == 2:
            # TODO(GH#45419): string[pyarrow] tests break if we transpose
            # unconditionally
            values = values.T

        # Check lengths of indexer and value to ensure they match
        check_setitem_lengths(indexer, value, values)

        try:
            # Attempt to set the value using indexer
            values[indexer] = value
        except (ValueError, TypeError):
            # Handle specific cases for IntervalDtype and NDArrayBackedExtensionBlock
            if isinstance(self.dtype, IntervalDtype):
                # Handle dtype coercion for IntervalDtype
                nb = self.coerce_to_target_dtype(orig_value, warn_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)
            elif isinstance(self, NDArrayBackedExtensionBlock):
                # Handle dtype coercion for NDArrayBackedExtensionBlock
                nb = self.coerce_to_target_dtype(orig_value, warn_on_upcast=True)
                return nb.setitem(orig_indexer, orig_value)
            else:
                # Raise exception for other cases
                raise
        else:
            # Return self if setitem succeeds
            return self
    # 将 self 对象的转置值赋给 arr 变量
    arr = self.values.T

    # 调用 extract_bool_array 函数处理 cond 变量，转换为布尔数组
    cond = extract_bool_array(cond)

    # 保存原始的 other 和 cond 变量
    orig_other = other
    orig_cond = cond

    # 调用 _maybe_squeeze_arg 方法，可能对 other 进行降维处理
    other = self._maybe_squeeze_arg(other)

    # 调用 _maybe_squeeze_arg 方法，可能对 cond 进行降维处理
    cond = self._maybe_squeeze_arg(cond)

    # 如果 other 为 lib.no_default，则将其设置为 self.fill_value
    if other is lib.no_default:
        other = self.fill_value

    # 调用 validate_putmask 函数，验证条件是否符合放置掩码操作的要求
    icond, noop = validate_putmask(arr, ~cond)

    # 如果 noop 为 True，避免对 Interval/PeriodDtype 类型对象进行操作，直接返回一个仅拷贝不深拷贝的列表
    if noop:
        # GH#44181, GH#45135
        # 避免 a) 对 Interval/PeriodDtype 类型对象抛出异常，b) 不必要的对象类型提升
        return [self.copy(deep=False)]

    # 尝试在 arr 上应用 where 操作，根据条件 cond 和替换值 other 进行筛选
    try:
        res_values = arr._where(cond, other).T
    except (ValueError, TypeError):
        # 处理可能发生的异常情况
        if self.ndim == 1 or self.shape[0] == 1:
            if isinstance(self.dtype, IntervalDtype):
                # TestSetitemFloatIntervalWithIntIntervalValues
                # 转换为目标数据类型后再次调用 where 方法
                blk = self.coerce_to_target_dtype(orig_other)
                return blk.where(orig_other, orig_cond)

            elif isinstance(self, NDArrayBackedExtensionBlock):
                # NB: not (yet) the same as
                #  isinstance(values, NDArrayBackedExtensionArray)
                # 转换为目标数据类型后再次调用 where 方法
                blk = self.coerce_to_target_dtype(orig_other)
                return blk.where(orig_other, orig_cond)

            else:
                # 抛出异常
                raise

        else:
            # 在 Block.putmask 中使用的相同模式
            # 判断 orig_other 是否为数组或扩展数组类型
            is_array = isinstance(orig_other, (np.ndarray, ExtensionArray))

            # 初始化结果块列表
            res_blocks = []

            # 遍历 self._split() 的结果，对每个子块进行处理
            for i, nb in enumerate(self._split()):
                n = orig_other
                if is_array:
                    # 如果是数组，每列可能有不同的值
                    n = orig_other[:, i : i + 1]

                # 获取子条件掩码
                submask = orig_cond[:, i : i + 1]

                # 调用 nb 的 where 方法
                rbs = nb.where(n, submask)

                # 将结果扩展到结果块列表中
                res_blocks.extend(rbs)

            # 返回处理后的结果块列表
            return res_blocks

    # 根据 res_values 创建与当前对象同类的块
    nb = self.make_block_same_class(res_values)

    # 返回包含 nb 的列表
    return [nb]
    def putmask(self, mask, new) -> list[Block]:
        """
        See Block.putmask.__doc__
        """
        # 转换掩码为布尔数组
        mask = extract_bool_array(mask)
        # 如果 new 为 lib.no_default，则使用 fill_value 替代
        if new is lib.no_default:
            new = self.fill_value

        orig_new = new
        orig_mask = mask
        # 可能对 new 进行压缩处理
        new = self._maybe_squeeze_arg(new)
        # 可能对 mask 进行压缩处理
        mask = self._maybe_squeeze_arg(mask)

        # 如果 mask 全为 False，返回当前对象的浅拷贝组成的列表
        if not mask.any():
            return [self.copy(deep=False)]

        # 尝试在 inplace 模式下复制当前对象
        self = self._maybe_copy(inplace=True)
        values = self.values
        # 如果 values 是二维的，转置处理
        if values.ndim == 2:
            values = values.T

        try:
            # 调用者需确保长度匹配，将值根据 mask 进行替换
            values._putmask(mask, new)
        except (TypeError, ValueError):
            # 处理异常情况
            if self.ndim == 1 or self.shape[0] == 1:
                if isinstance(self.dtype, IntervalDtype):
                    # 讨论对通用情况的支持
                    blk = self.coerce_to_target_dtype(orig_new, warn_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)

                elif isinstance(self, NDArrayBackedExtensionBlock):
                    # 注意：目前与 isinstance(values, NDArrayBackedExtensionArray) 不同
                    blk = self.coerce_to_target_dtype(orig_new, warn_on_upcast=True)
                    return blk.putmask(orig_mask, orig_new)

                else:
                    raise

            else:
                # 在 Block.putmask 中使用相同的模式
                is_array = isinstance(orig_new, (np.ndarray, ExtensionArray))

                res_blocks = []
                # 拆分当前对象并根据子掩码和值进行替换
                for i, nb in enumerate(self._split()):
                    n = orig_new
                    if is_array:
                        # 每列有不同的值
                        n = orig_new[:, i : i + 1]

                    submask = orig_mask[:, i : i + 1]
                    rbs = nb.putmask(submask, n)
                    res_blocks.extend(rbs)
                return res_blocks

        # 返回包含当前对象的列表
        return [self]

    @final
    def delete(self, loc) -> list[Block]:
        # 如果是一维数组，则删除指定位置的值
        if self.ndim == 1:
            values = self.values.delete(loc)
            mgr_locs = self._mgr_locs.delete(loc)
            return [type(self)(values, placement=mgr_locs, ndim=self.ndim)]
        elif self.values.ndim == 1:
            # 通过 to_stata 方法执行
            return []
        # 调用父类的 delete 方法
        return super().delete(loc)

    @final
    @cache_readonly
    def array_values(self) -> ExtensionArray:
        # 返回当前对象的值
        return self.values

    @final
    # 返回以对象数据类型为盒式值，例如时间戳/时间差的数组
    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """
        return object dtype as boxed values, such as Timestamps/Timedelta
        """
        # 将对象的值存储到 values 变量中
        values: ArrayLike = self.values
        # 如果 dtype 与 _dtype_obj 相等，则将 values 转换为对象类型
        if dtype == _dtype_obj:
            values = values.astype(object)
        # TODO(EA2D): 对于二维的 EAs，不需要重新调整形状
        # 将 values 转换为 NumPy 数组，并重新调整为与 self.shape 相同的形状
        return np.asarray(values).reshape(self.shape)

    @final
    # 在指定的地方进行填充或者向后填充
    def pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        inplace: bool = False,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
    ) -> list[Block]:
        # 将对象的值存储到 values 变量中
        values = self.values

        # 准备填充或向后填充的参数
        kwargs: dict[str, Any] = {"method": method, "limit": limit}
        # 如果 values._pad_or_backfill 方法的参数中包含 limit_area
        if "limit_area" in inspect.signature(values._pad_or_backfill).parameters:
            kwargs["limit_area"] = limit_area
        # 如果 limit_area 不为 None 但 values._pad_or_backfill 方法中不包含 limit_area 参数，则抛出 NotImplementedError
        elif limit_area is not None:
            raise NotImplementedError(
                f"{type(values).__name__} does not implement limit_area "
                "(added in pandas 2.2). 3rd-party ExtensionArray authors "
                "need to add this argument to _pad_or_backfill."
            )

        # 如果 values 是二维的数组
        if values.ndim == 2:
            # NDArrayBackedExtensionArray.fillna 假定 axis=0
            # 对 values 的转置进行填充或向后填充，然后再进行转置以恢复原始形状
            new_values = values.T._pad_or_backfill(**kwargs).T
        else:
            # 对 values 进行填充或向后填充
            new_values = values._pad_or_backfill(**kwargs)
        # 使用新的值创建与原始对象相同类别的 Block 列表
        return [self.make_block_same_class(new_values)]
class ExtensionBlock(EABackedBlock):
    """
    Block for holding extension types.

    Notes
    -----
    This holds all 3rd-party extension array types. It's also the immediate
    parent class for our internal extension types' blocks.

    ExtensionArrays are limited to 1-D.
    """

    values: ExtensionArray  # 定义一个名为 `values` 的属性，类型为 ExtensionArray

    def fillna(
        self,
        value,  # 填充缺失值所用的值
        limit: int | None = None,  # 最多填充的数量限制，默认为 None
        inplace: bool = False,  # 是否原地修改，默认为 False
    ) -> list[Block]:  # 返回一个 Block 类型的列表
        if isinstance(self.dtype, IntervalDtype):
            # 如果数据类型是 IntervalDtype，调用 Block.fillna 处理填充（test_fillna_interval）
            if limit is not None:
                raise ValueError("limit must be None")
            return super().fillna(
                value=value,
                limit=limit,
                inplace=inplace,
            )
        if self._can_hold_na and not self.values._hasna:
            # 如果可以容纳缺失值且当前值没有缺失值
            refs = self.refs  # 获取引用
            new_values = self.values  # 新的值为当前值
        else:
            copy, refs = self._get_refs_and_copy(inplace)  # 获取复制标志和引用
            try:
                new_values = self.values.fillna(value=value, limit=limit, copy=copy)  # 填充缺失值
            except TypeError:
                # 处理未实现复制关键字的第三方扩展数组类型
                refs = None
                new_values = self.values.fillna(value=value, limit=limit)
                # 在重试后发出警告，以防 TypeError 是由于无效的填充值引起的
                warnings.warn(
                    # GH#53278
                    "ExtensionArray.fillna added a 'copy' keyword in pandas "
                    "2.1.0. In a future version, ExtensionArray subclasses will "
                    "need to implement this keyword or an exception will be "
                    "raised. In the interim, the keyword is ignored by "
                    f"{type(self.values).__name__}.",
                    DeprecationWarning,
                    stacklevel=find_stack_level(),
                )

        return [self.make_block_same_class(new_values, refs=refs)]  # 返回相同类型的 Block 对象列表

    @cache_readonly
    def shape(self) -> Shape:
        # TODO(EA2D): override unnecessary with 2D EAs
        if self.ndim == 1:
            return (len(self.values),)  # 如果是一维数据，返回长度为唯一元组
        return len(self._mgr_locs), len(self.values)  # 否则返回管理位置和值的长度
    def iget(self, i: int | tuple[int, int] | tuple[slice, int]):
        # 当传入的索引是 tuple[slice, int] 时，slice 将总是 slice(None)
        # 我们本可以让注解更具体，但是 mypy 会抱怨覆盖不匹配：
        # Literal[0] | tuple[Literal[0], int] | tuple[slice, int]

        # 注意：仅当 self.ndim == 2 时才会执行到这里

        if isinstance(i, tuple):
            # TODO(EA2D): 对于二维的 EA 来说，这里是不必要的
            col, loc = i
            if not com.is_null_slice(col) and col != 0:
                raise IndexError(f"{self} 只包含一个项目")
            if isinstance(col, slice):
                # 上面的 is_null_slice 检查保证 col 是 slice(None)
                # 我们需要的是一个包含所有列和行 loc 的视图
                if loc < 0:
                    loc += len(self.values)
                # 注意：loc:loc+1 和 [[loc]] 的区别在于当从 fast_xs 调用时
                # 我们希望得到一个视图
                return self.values[loc : loc + 1]
            return self.values[loc]
        else:
            if i != 0:
                raise IndexError(f"{self} 只包含一个项目")
            return self.values

    def set_inplace(self, locs, values: ArrayLike, copy: bool = False) -> None:
        # 当 locs 是 ndarray 时，应该满足 locs.tolist() == [0]
        # 当 locs 是 BlockPlacement 时，应该满足 list(locs) == [0]
        if copy:
            self.values = self.values.copy()
        self.values[:] = values

    def _maybe_squeeze_arg(self, arg):
        """
        如果需要，将 (N, 1) 的 ndarray 压缩为 (N,)
        """
        # 例如，如果我们传入 putmask 的二维 mask
        if (
            isinstance(arg, (np.ndarray, ExtensionArray))
            and arg.ndim == self.values.ndim + 1
        ):
            # TODO(EA2D): 对于二维的 EA 来说，这里是不必要的
            assert arg.shape[1] == 1
            # error: No overload variant of "__getitem__" of "ExtensionArray"
            # matches argument type "Tuple[slice, int]"
            arg = arg[:, 0]  # type: ignore[call-overload]
        elif isinstance(arg, ABCDataFrame):
            # 2022-01-06 只在 setitem 时执行到这里
            # TODO: 我们应该避免 DataFrame 进入这里吗？
            assert arg.shape[1] == 1
            arg = arg._ixs(0, axis=1)._values

        return arg
    def _unwrap_setitem_indexer(self, indexer):
        """
        Adapt a 2D-indexer to our 1D values.

        This is intended for 'setitem', not 'iget' or '_slice'.
        """
        # TODO: ATM this doesn't work for iget/_slice, can we change that?
        # 检查传入的索引器是否是二维元组
        if isinstance(indexer, tuple) and len(indexer) == 2:
            # TODO(EA2D): not needed with 2D EAs
            #  Should never have length > 2.  Caller is responsible for checking.
            #  Length 1 is reached vis setitem_single_block and setitem_single_column
            #  each of which pass indexer=(pi,)
            # 检查索引器中的每个元素是否为二维 ndarray
            if all(isinstance(x, np.ndarray) and x.ndim == 2 for x in indexer):
                # GH#44703 went through indexing.maybe_convert_ix
                # 如果第二个元素的大小为1且全部为0，并且第一个元素的第二维大小为1
                if not (
                    indexer[1].size == 1 and (indexer[1] == 0).all() and indexer[0].shape[1] == 1
                ):
                    raise NotImplementedError(
                        "This should not be reached. Please report a bug at "
                        "github.com/pandas-dev/pandas/"
                    )
                # 将索引器更新为第一个元素的第一列
                indexer = indexer[0][:, 0]

            elif lib.is_integer(indexer[1]) and indexer[1] == 0:
                # reached via setitem_single_block passing the whole indexer
                # 如果第二个元素是整数且为0，则更新索引器为第一个元素
                indexer = indexer[0]

            elif com.is_null_slice(indexer[1]):
                # 如果第二个元素表示空切片，则更新索引器为第一个元素
                indexer = indexer[0]

            elif is_list_like(indexer[1]) and indexer[1][0] == 0:
                # 如果第二个元素是列表且第一个元素为0，则更新索引器为第一个元素
                indexer = indexer[0]

            else:
                raise NotImplementedError(
                    "This should not be reached. Please report a bug at "
                    "github.com/pandas-dev/pandas/"
                )
        return indexer

    @property
    def is_view(self) -> bool:
        """Extension arrays are never treated as views."""
        # 扩展数组从不被视为视图，因此始终返回 False
        return False

    # error: Cannot override writeable attribute with read-only property
    @cache_readonly
    def is_numeric(self) -> bool:  # type: ignore[override]
        # 返回该属性对应的值是否是数值类型
        return self.values.dtype._is_numeric

    def _slice(
        self, slicer: slice | npt.NDArray[np.bool_] | npt.NDArray[np.intp]
        # 用于处理切片操作的方法，接受的参数类型为 slice 或者布尔型或整型的 NumPy 数组
    ) -> ExtensionArray:
        """
        Return a slice of my values.

        Parameters
        ----------
        slicer : slice, ndarray[int], or ndarray[bool]
            Valid (non-reducing) indexer for self.values.

        Returns
        -------
        ExtensionArray
        """
        # Notes: ndarray[bool] is only reachable when via get_rows_with_mask, which
        #  is only for Series, i.e. self.ndim == 1.

        # 如果当前对象是二维的，说明是通过 getitem_block 通过 _slice_take_blocks_ax0 进入的
        # TODO(EA2D): 在二维 ExtensionArray 中不需要这部分代码了
        if self.ndim == 2:
            if not isinstance(slicer, slice):
                # 如果 slicer 不是 slice 类型，抛出断言错误
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            # GH#32959 只有在虚拟维度0上的全片段才是有效的
            # TODO(EA2D): 在二维 ExtensionArray 中不需要这部分代码了
            # 使用 range(1) 而不是 self._mgr_locs，避免在 [::-1] 上抛出异常
            #  参见 test_iloc_getitem_slice_negative_step_ea_block
            new_locs = range(1)[slicer]
            if not len(new_locs):
                # 如果 new_locs 的长度为0，抛出断言错误
                raise AssertionError(
                    "invalid slicing for a 1-ndim ExtensionArray", slicer
                )
            slicer = slice(None)

        # 返回根据 slicer 切片后的 self.values
        return self.values[slicer]

    @final
    def slice_block_rows(self, slicer: slice) -> Self:
        """
        Perform __getitem__-like specialized to slicing along index.
        """
        # GH#42787 原则上这等同于 values[..., slicer]，但我们暂时不要求 ExtensionArray 的子类支持这种形式
        new_values = self.values[slicer]
        # 返回一个新的类型为 Self 的对象，使用 new_values、self._mgr_locs、当前维度信息和引用信息构造
        return type(self)(new_values, self._mgr_locs, ndim=self.ndim, refs=self.refs)

    def _unstack(
        self,
        unstacker,
        fill_value,
        new_placement: npt.NDArray[np.intp],
        needs_masking: npt.NDArray[np.bool_],
        # ExtensionArray-safe unstack.
        # We override Block._unstack, which unstacks directly on the
        # values of the array. For EA-backed blocks, this would require
        # converting to a 2-D ndarray of objects.
        # Instead, we unstack an ndarray of integer positions, followed by
        # a `take` on the actual values.

        # Caller is responsible for ensuring self.shape[-1] == len(unstacker.index)
        # 获取 unstacker 的 arange_result 属性，包含新值和掩码信息
        new_values, mask = unstacker.arange_result

        # Note: these next two lines ensure that
        #  mask.sum() == sum(len(nb.mgr_locs) for nb in blocks)
        #  which the calling function needs in order to pass verify_integrity=False
        #  to the BlockManager constructor
        # 重新调整 new_values 和 new_placement，确保满足调用函数的特定要求
        new_values = new_values.T[mask]
        new_placement = new_placement[mask]

        # needs_masking[i] calculated once in BlockManager.unstack tells
        #  us if there are any -1s in the relevant indices.  When False,
        #  that allows us to go through a faster path in 'take', among
        #  other things avoiding e.g. Categorical._validate_scalar.
        # 使用列表推导式创建 blocks 列表，每个元素都是一个新的 Block 对象
        blocks = [
            # TODO: could cast to object depending on fill_value?
            # 根据 indices 获取对应的值，根据需要进行填充，并创建一个新的 Block 对象
            type(self)(
                self.values.take(
                    indices, allow_fill=needs_masking[i], fill_value=fill_value
                ),
                BlockPlacement(place),
                ndim=2,
            )
            for i, (indices, place) in enumerate(zip(new_values, new_placement))
        ]
        # 返回生成的 blocks 列表和掩码 mask
        return blocks, mask
class NumpyBlock(Block):
    values: np.ndarray
    __slots__ = ()

    @property
    def is_view(self) -> bool:
        """Return a boolean indicating if the values are possibly a view."""
        return self.values.base is not None

    @property
    def array_values(self) -> ExtensionArray:
        """Return the values wrapped in a NumpyExtensionArray."""
        return NumpyExtensionArray(self.values)

    def get_values(self, dtype: DtypeObj | None = None) -> np.ndarray:
        """Return the values optionally casted to a specified dtype."""
        if dtype == _dtype_obj:
            return self.values.astype(_dtype_obj)
        return self.values

    @cache_readonly
    def is_numeric(self) -> bool:  # type: ignore[override]
        """Check if the values are of a numeric kind."""
        dtype = self.values.dtype
        kind = dtype.kind
        return kind in "fciub"


class NDArrayBackedExtensionBlock(EABackedBlock):
    """
    Block backed by an NDArrayBackedExtensionArray.
    """

    values: NDArrayBackedExtensionArray

    @property
    def is_view(self) -> bool:
        """Return a boolean indicating if the values are possibly a view."""
        # Check if the underlying ndarray of values has a base attribute.
        return self.values._ndarray.base is not None


class DatetimeLikeBlock(NDArrayBackedExtensionBlock):
    """Block for datetime64[ns], timedelta64[ns]."""

    __slots__ = ()
    is_numeric = False
    values: DatetimeArray | TimedeltaArray


# -----------------------------------------------------------------
# Constructor Helpers


def maybe_coerce_values(values: ArrayLike) -> ArrayLike:
    """
    Input validation for values passed to __init__. Ensure that
    any datetime64/timedelta64 dtypes are in nanoseconds.  Ensure
    that we do not have string dtypes.

    Parameters
    ----------
    values : np.ndarray or ExtensionArray

    Returns
    -------
    values : np.ndarray or ExtensionArray
    """
    # Caller is responsible for ensuring NumpyExtensionArray is already extracted.

    if isinstance(values, np.ndarray):
        values = ensure_wrapped_if_datetimelike(values)

        if issubclass(values.dtype.type, str):
            values = np.array(values, dtype=object)

    if isinstance(values, (DatetimeArray, TimedeltaArray)) and values.freq is not None:
        # freq is only stored in DatetimeIndex/TimedeltaIndex, not in Series/DataFrame
        values = values._with_freq(None)

    return values


def get_block_type(dtype: DtypeObj) -> type[Block]:
    """
    Find the appropriate Block subclass to use for the given values and dtype.

    Parameters
    ----------
    dtype : numpy or pandas dtype

    Returns
    -------
    cls : class, subclass of Block
    """
    if isinstance(dtype, DatetimeTZDtype):
        return DatetimeLikeBlock
    elif isinstance(dtype, PeriodDtype):
        return NDArrayBackedExtensionBlock
    elif isinstance(dtype, ExtensionDtype):
        # Note: need to be sure NumpyExtensionArray is unwrapped before we get here
        return ExtensionBlock

    # We use kind checks because it is much more performant
    # than is_foo_dtype
    kind = dtype.kind
    if kind in "Mm":
        return DatetimeLikeBlock
    return NumpyBlock


注释：


# 返回变量 NumpyBlock 的值作为函数的返回结果
# 定义一个函数 `new_block_2d`，创建一个二维块对象的实例。
# 参数 `values`：数组样式的输入数据。
# 参数 `placement`：块的放置位置，必须是 `BlockPlacement` 类的实例。
# 参数 `refs`（可选）：块值引用或者为 `None`。
# 返回：一个 `Block` 类的实例。
def new_block_2d(
    values: ArrayLike, placement: BlockPlacement, refs: BlockValuesRefs | None = None
) -> Block:
    # 从数据类型获取块的类型
    klass = get_block_type(values.dtype)

    # 可能对输入数据进行类型转换
    values = maybe_coerce_values(values)
    
    # 返回一个 `Block` 类的实例，二维情况下使用 `ndim=2`，给定的 `placement` 和 `refs`
    return klass(values, ndim=2, placement=placement, refs=refs)


# 定义一个函数 `new_block`，创建一个新的块对象的实例。
# 参数 `values`：输入的数据。
# 参数 `placement`：块的放置位置，必须是 `BlockPlacement` 类的实例。
# 参数 `ndim`：块对象的维度。
# 参数 `refs`（可选）：块值引用或者为 `None`。
# 返回：一个 `Block` 类的实例。
def new_block(
    values,
    placement: BlockPlacement,
    *,
    ndim: int,
    refs: BlockValuesRefs | None = None,
) -> Block:
    # 从数据类型获取块的类型
    klass = get_block_type(values.dtype)

    # 返回一个 `Block` 类的实例，给定的 `ndim`，`placement` 和 `refs`
    return klass(values, ndim=ndim, placement=placement, refs=refs)


# 定义一个函数 `check_ndim`，验证并推断数据的维度。
# 参数 `values`：数组样式的数据。
# 参数 `placement`：块的放置位置，必须是 `BlockPlacement` 类的实例。
# 参数 `ndim`：期望的维度。
# 返回：无。
def check_ndim(values, placement: BlockPlacement, ndim: int) -> None:
    """
    ndim inference and validation.

    Validates that values.ndim and ndim are consistent.
    Validates that len(values) and len(placement) are consistent.

    Parameters
    ----------
    values : array-like
    placement : BlockPlacement
    ndim : int

    Raises
    ------
    ValueError : the number of dimensions do not match
    """

    # 检查数据的维度是否大于期望的维度
    if values.ndim > ndim:
        raise ValueError(
            "Wrong number of dimensions. "
            f"values.ndim > ndim [{values.ndim} > {ndim}]"
        )

    # 如果数据类型不是仅限于一维的扩展数组，则进行进一步检查
    if not is_1d_only_ea_dtype(values.dtype):
        # TODO(EA2D): 对于二维扩展数组，不需要特殊处理
        if values.ndim != ndim:
            raise ValueError(
                "Wrong number of dimensions. "
                f"values.ndim != ndim [{values.ndim} != {ndim}]"
            )
        # 检查数据长度和放置位置长度是否一致
        if len(placement) != len(values):
            raise ValueError(
                f"Wrong number of items passed {len(values)}, "
                f"placement implies {len(placement)}"
            )
    # 对于二维扩展数组，如果放置位置长度不为1，则引发错误
    elif ndim == 2 and len(placement) != 1:
        raise ValueError("need to split")


# 定义一个函数 `extract_pandas_array`，确保在内部不允许使用 `NumpyExtensionArray` 或 `NumpyEADtype`。
# 参数 `values`：数组样式的数据。
# 参数 `dtype`：数据类型对象或 `None`。
# 参数 `ndim`：期望的数据维度。
# 返回：一个包含处理后的 `values` 和 `dtype` 的元组。
def extract_pandas_array(
    values: ArrayLike, dtype: DtypeObj | None, ndim: int
) -> tuple[ArrayLike, DtypeObj | None]:
    """
    Ensure that we don't allow NumpyExtensionArray / NumpyEADtype in internals.
    """
    # 目前，块应该尽可能由 ndarray 支持
    if isinstance(values, ABCNumpyExtensionArray):
        values = values.to_numpy()
        # 如果指定了二维维度，并且数据不是二维，则至少转换为二维
        if ndim and ndim > 1:
            values = np.atleast_2d(values)

    # 如果数据类型为 `NumpyEADtype`，则转换为其对应的 `numpy` 数据类型
    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype

    # 返回处理后的 `values` 和 `dtype`
    return values, dtype


# 定义一个函数 `extend_blocks`，返回给定结果的新扩展块。
# 参数 `result`：给定的结果。
# 参数 `blocks`（可选）：块的列表，默认为 `None`。
# 返回：一个 `Block` 类的实例列表。
def extend_blocks(result, blocks=None) -> list[Block]:
    """return a new extended blocks, given the result"""
    # 如果 `blocks` 为 `None`，则初始化为空列表
    if blocks is None:
        blocks = []
    # 如果结果是列表类型，则进行如下操作
    if isinstance(result, list):
        # 遍历列表中的每个元素
        for r in result:
            # 如果元素 r 也是列表类型，则将其扩展到 blocks 列表中
            if isinstance(r, list):
                blocks.extend(r)
            # 否则直接将元素 r 添加到 blocks 列表中
            else:
                blocks.append(r)
    # 如果结果不是列表类型，则断言结果是 Block 类型，并将其添加到 blocks 列表中
    else:
        assert isinstance(result, Block), type(result)
        blocks.append(result)
    # 返回处理后的 blocks 列表
    return blocks
def ensure_block_shape(values: ArrayLike, ndim: int = 1) -> ArrayLike:
    """
    Reshape if possible to have values.ndim == ndim.
    """

    # 如果 values 的维度小于 ndim，则尝试进行 reshape
    if values.ndim < ndim:
        # 如果 values 的 dtype 不是仅支持一维的类型
        if not is_1d_only_ea_dtype(values.dtype):
            # TODO(EA2D): https://github.com/pandas-dev/pandas/issues/23023
            # block.shape is incorrect for "2D" ExtensionArrays
            # We can't, and don't need to, reshape.
            # 将 values 强制转换为 "np.ndarray | DatetimeArray | TimedeltaArray" 类型
            values = cast("np.ndarray | DatetimeArray | TimedeltaArray", values)
            # 对 values 进行 reshape 成 (1, -1) 形状
            values = values.reshape(1, -1)

    return values


def external_values(values: ArrayLike) -> ArrayLike:
    """
    The array that Series.values returns (public attribute).

    This has some historical constraints, and is overridden in block
    subclasses to return the correct array (e.g. period returns
    object ndarray and datetimetz a datetime64[ns] ndarray instead of
    proper extension array).
    """
    # 如果 values 是 PeriodArray 或者 IntervalArray 类型的实例
    if isinstance(values, (PeriodArray, IntervalArray)):
        # 返回转换为 object 类型的 values
        return values.astype(object)
    # 如果 values 是 DatetimeArray 或者 TimedeltaArray 类型的实例
    elif isinstance(values, (DatetimeArray, TimedeltaArray)):
        # NB: for datetime64tz this is different from np.asarray(values), since
        #  that returns an object-dtype ndarray of Timestamps.
        # Avoid raising in .astype in casting from dt64tz to dt64
        # 将 values 的内部 ndarray 提取出来
        values = values._ndarray

    # 如果 values 是 np.ndarray 类型的实例
    if isinstance(values, np.ndarray):
        # 创建 values 的一个视图
        values = values.view()
        # 设置 values 不可写
        values.flags.writeable = False

    # TODO(CoW) we should also mark our ExtensionArrays as read-only

    return values
```