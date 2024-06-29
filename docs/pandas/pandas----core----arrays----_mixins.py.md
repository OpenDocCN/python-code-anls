# `D:\src\scipysrc\pandas\pandas\core\arrays\_mixins.py`

```
# 从 __future__ 导入 annotations，用于支持在函数参数类型提示中使用字符串形式的类型
from __future__ import annotations

# 导入装饰器相关的模块
from functools import wraps
# 导入类型相关的模块
from typing import (
    TYPE_CHECKING,  # 用于类型检查时的特殊标记
    Any,             # 表示任意类型
    Literal,         # 用于指定字符串字面值类型
    cast,            # 用于强制类型转换
    overload,        # 用于定义重载的特殊装饰器
)

# 导入第三方库 numpy，并重命名为 np
import numpy as np

# 导入 pandas 内部的 C 库
from pandas._libs import lib
# 导入 Pandas 数组相关的模块
from pandas._libs.arrays import NDArrayBacked
# 导入 Pandas 时间序列相关的模块
from pandas._libs.tslibs import is_supported_dtype
# 导入 Pandas 中的类型定义
from pandas._typing import (
    ArrayLike,              # 类数组对象的类型
    AxisInt,                # 轴的整数索引类型
    Dtype,                  # 数据类型的类型
    F,                      # 泛型函数类型
    FillnaOptions,          # 填充选项的类型
    PositionalIndexer2D,    # 二维位置索引器类型
    PositionalIndexerTuple, # 元组位置索引器类型
    ScalarIndexer,          # 标量索引器类型
    Self,                   # 自引用类型
    SequenceIndexer,        # 序列索引器类型
    Shape,                  # 形状类型
    TakeIndexer,            # 获取索引器类型
    npt,                    # NumPy 类型
)

# 导入 Pandas 错误相关的模块
from pandas.errors import AbstractMethodError
# 导入 Pandas 内部工具中的装饰器相关模块
from pandas.util._decorators import doc
# 导入 Pandas 内部工具中的验证器相关模块
from pandas.util._validators import (
    validate_bool_kwarg,    # 用于验证布尔类型参数的函数
    validate_insert_loc,    # 用于验证插入位置的函数
)

# 导入 Pandas 核心数据类型通用模块
from pandas.core.dtypes.common import pandas_dtype
# 导入 Pandas 核心数据类型模块
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,        # 带时区的日期时间类型
    ExtensionDtype,         # 扩展数据类型
    PeriodDtype,            # 时期数据类型
)
# 导入 Pandas 核心数据类型缺失值相关模块
from pandas.core.dtypes.missing import array_equivalent

# 导入 Pandas 核心模块
from pandas.core import missing
# 导入 Pandas 核心算法模块
from pandas.core.algorithms import (
    take,                   # 用于数组取值的函数
    unique,                 # 用于获取唯一值的函数
    value_counts_internal as value_counts,  # 用于内部值计数的函数，重命名为 value_counts
)
# 导入 Pandas 核心数组算法模块中的分位数计算函数
from pandas.core.array_algos.quantile import quantile_with_mask
# 导入 Pandas 核心数组算法模块中的数据转换函数
from pandas.core.array_algos.transforms import shift
# 导入 Pandas 核心数组基础模块
from pandas.core.arrays.base import ExtensionArray
# 导入 Pandas 核心构造模块
from pandas.core.construction import extract_array
# 导入 Pandas 核心索引器模块
from pandas.core.indexers import check_array_indexer
# 导入 Pandas 核心排序模块
from pandas.core.sorting import nargminmax

# 如果是类型检查模式，导入 Sequence 抽象基类和特定类型定义
if TYPE_CHECKING:
    from collections.abc import Sequence
    from pandas._typing import (
        NumpySorter,        # NumPy 排序器类型
        NumpyValueArrayLike, # NumPy 值类数组对象类型
    )
    from pandas import Series

# 定义装饰器函数 ravel_compat，用于将二维数组展平后再进行 Cython 操作，并按需重新整形
def ravel_compat(meth: F) -> F:
    """
    Decorator to ravel a 2D array before passing it to a cython operation,
    then reshape the result to our own shape.
    """

    @wraps(meth)
    def method(self, *args, **kwargs):
        # 如果数组是一维的，则直接调用原方法
        if self.ndim == 1:
            return meth(self, *args, **kwargs)

        # 否则，获取数组的标志位
        flags = self._ndarray.flags
        # 将数组展平为一维数组
        flat = self.ravel("K")
        # 调用原方法进行计算
        result = meth(flat, *args, **kwargs)
        # 根据数组的存储顺序重新整形结果
        order = "F" if flags.f_contiguous else "C"
        return result.reshape(self.shape, order=order)

    return cast(F, method)


# 错误：在“NDArrayBacked”基类中的“delete/ravel/T/repeat/copy”定义与“ExtensionArray”基类中的定义不兼容
# 定义一个继承自 NDArrayBacked 和 ExtensionArray 的扩展数组类 NDArrayBackedExtensionArray
class NDArrayBackedExtensionArray(NDArrayBacked, ExtensionArray):  # type: ignore[misc]
    """
    ExtensionArray that is backed by a single NumPy ndarray.
    """

    _ndarray: np.ndarray  # 用于存储数据的 NumPy ndarray 对象

    # 用于表示 self._ndarray 中的 NA 值的标量，例如对于分类数据是 -1，对于时期数据是 iNaT
    # 在非对象 dtype 中，self.isna() 函数应当精确标记出 self._ndarray 中的 _internal_fill_value 位置
    _internal_fill_value: Any

    # 定义一个内部方法 _box_func，用于将 numpy 类型包装成我们的 dtype.type 类型（如果有必要）
    def _box_func(self, x):
        """
        Wrap numpy type in our dtype.type if necessary.
        """
        return x

    # 定义一个抽象方法 _validate_scalar，用于验证标量值
    def _validate_scalar(self, value):
        # 由 NDArrayBackedExtensionIndex.insert 方法使用
        raise AbstractMethodError(self)
    # ------------------------------------------------------------------------

    def view(self, dtype: Dtype | None = None) -> ArrayLike:
        # 处理 datetime64、datetime64tz、timedelta64 和 period 这些特定的 dtype。
        # 其他类型的 dtype 将直接透传到底层的 ndarray。
        if dtype is None or dtype is self.dtype:
            # 如果 dtype 为 None 或者与当前对象的 dtype 相同，则返回当前对象的数据。
            return self._from_backing_data(self._ndarray)

        if isinstance(dtype, type):
            # 有时会传入非 dtype 对象，例如 np.ndarray；这些对象会透传到底层的 ndarray。
            return self._ndarray.view(dtype)

        dtype = pandas_dtype(dtype)
        arr = self._ndarray

        if isinstance(dtype, PeriodDtype):
            # 如果 dtype 是 PeriodDtype 类型，则构造相应的数组类型并返回。
            cls = dtype.construct_array_type()
            return cls(arr.view("i8"), dtype=dtype)
        elif isinstance(dtype, DatetimeTZDtype):
            # 如果 dtype 是 DatetimeTZDtype 类型，则构造相应的数组类型并返回。
            dt_cls = dtype.construct_array_type()
            dt64_values = arr.view(f"M8[{dtype.unit}]")
            return dt_cls._simple_new(dt64_values, dtype=dtype)
        elif lib.is_np_dtype(dtype, "M") and is_supported_dtype(dtype):
            # 如果 dtype 是日期时间相关的 NumPy dtype，则构造 DatetimeArray 并返回。
            from pandas.core.arrays import DatetimeArray

            dt64_values = arr.view(dtype)
            return DatetimeArray._simple_new(dt64_values, dtype=dtype)

        elif lib.is_np_dtype(dtype, "m") and is_supported_dtype(dtype):
            # 如果 dtype 是时间差相关的 NumPy dtype，则构造 TimedeltaArray 并返回。
            from pandas.core.arrays import TimedeltaArray

            td64_values = arr.view(dtype)
            return TimedeltaArray._simple_new(td64_values, dtype=dtype)

        # 如果以上条件都不符合，则尝试将数组视图转换为指定的 dtype。
        # 错误信息: "view" 方法的参数 "dtype" 类型为 "Union[ExtensionDtype, dtype[Any]]"，
        # 但期望的类型为 "Union[dtype[Any], None, type, _SupportsDType, str, ...]"
        return arr.view(dtype=dtype)  # type: ignore[arg-type]

    def take(
        self,
        indices: TakeIndexer,
        *,
        allow_fill: bool = False,
        fill_value: Any = None,
        axis: AxisInt = 0,
    ) -> Self:
        if allow_fill:
            # 如果允许填充，则验证并设置填充值。
            fill_value = self._validate_scalar(fill_value)

        # 使用 take 函数从 ndarray 中取出指定 indices 的数据。
        new_data = take(
            self._ndarray,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value,
            axis=axis,
        )
        # 返回一个新的对象，其数据为取出的新数据。
        return self._from_backing_data(new_data)

    # ------------------------------------------------------------------------

    def equals(self, other) -> bool:
        # 检查对象类型和 dtype 是否相同，若不同则返回 False。
        if type(self) is not type(other):
            return False
        # 检查对象的 dtype 是否相同，若不同则返回 False。
        if self.dtype != other.dtype:
            return False
        # 比较两个对象的数据是否相等，使用 dtype_equal=True 表示比较时考虑 dtype。
        return bool(array_equivalent(self._ndarray, other._ndarray, dtype_equal=True))

    @classmethod
    def _from_factorized(cls, values, original):
        # 断言 values 的 dtype 与原始对象的数据的 dtype 相同。
        assert values.dtype == original._ndarray.dtype
        # 使用原始对象的 _from_backing_data 方法构造新对象并返回。
        return original._from_backing_data(values)

    def _values_for_argsort(self) -> np.ndarray:
        # 返回对象的 ndarray 数据。
        return self._ndarray
    # 返回一个包含当前对象的 ndarray 和内部填充值的元组
    def _values_for_factorize(self):
        return self._ndarray, self._internal_fill_value

    # 对 Pandas 对象进行哈希化处理，返回一个 npt.NDArray[np.uint64] 类型的哈希数组
    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        from pandas.core.util.hashing import hash_array

        # 获取当前对象的 ndarray
        values = self._ndarray
        # 调用 hash_array 函数对 ndarray 进行哈希化处理
        return hash_array(
            values, encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    # 覆盖基类的 argmin 方法，并添加了 axis 关键字参数
    def argmin(self, axis: AxisInt = 0, skipna: bool = True):  # type: ignore[override]
        # 验证 skipna 参数是否为布尔类型
        validate_bool_kwarg(skipna, "skipna")
        # 如果 skipna=False 且存在 NA 值，则抛出 ValueError 异常
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        # 调用 nargminmax 函数进行 argmin 操作
        return nargminmax(self, "argmin", axis=axis)

    # 覆盖基类的 argmax 方法，并添加了 axis 关键字参数
    def argmax(self, axis: AxisInt = 0, skipna: bool = True):  # type: ignore[override]
        # 验证 skipna 参数是否为布尔类型
        validate_bool_kwarg(skipna, "skipna")
        # 如果 skipna=False 且存在 NA 值，则抛出 ValueError 异常
        if not skipna and self._hasna:
            raise ValueError("Encountered an NA value with skipna=False")
        # 调用 nargminmax 函数进行 argmax 操作
        return nargminmax(self, "argmax", axis=axis)

    # 返回当前对象的唯一值，类型为 Self
    def unique(self) -> Self:
        # 使用 unique 函数获取当前对象 ndarray 的唯一值
        new_data = unique(self._ndarray)
        # 调用 _from_backing_data 方法，根据 new_data 创建一个新的 Self 对象
        return self._from_backing_data(new_data)

    # 类方法，用于按指定轴合并同一类型的 ExtensionArray 对象序列
    @classmethod
    @doc(ExtensionArray._concat_same_type)
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        # 检查待合并对象的 dtype 是否相同，若不同则抛出 ValueError 异常
        if not lib.dtypes_all_equal([x.dtype for x in to_concat]):
            dtypes = {str(x.dtype) for x in to_concat}
            raise ValueError("to_concat must have the same dtype", dtypes)

        # 调用基类的 _concat_same_type 方法，进行同类型对象的合并操作
        return super()._concat_same_type(to_concat, axis=axis)

    # 返回使用 searchsorted 方法在当前对象中搜索指定值的结果
    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        # 验证并获取 value 参数的 ndarray 表示
        npvalue = self._validate_setitem_value(value)
        # 调用当前对象 ndarray 的 searchsorted 方法进行搜索操作
        return self._ndarray.searchsorted(npvalue, side=side, sorter=sorter)

    # 返回移动后的当前对象，移动周期由 periods 指定，填充值由 fill_value 指定
    @doc(ExtensionArray.shift)
    def shift(self, periods: int = 1, fill_value=None) -> Self:
        # 注意：shift 操作始终沿 axis=0 进行
        axis = 0
        # 验证并获取填充值的标量表示
        fill_value = self._validate_scalar(fill_value)
        # 调用 shift 函数对当前对象 ndarray 进行移动操作
        new_values = shift(self._ndarray, periods, axis, fill_value)

        # 调用 _from_backing_data 方法，根据 new_values 创建一个新的 Self 对象
        return self._from_backing_data(new_values)

    # 设置当前对象的指定索引 key 处的值为 value
    def __setitem__(self, key, value) -> None:
        # 检查并规范化 key 参数作为数组索引
        key = check_array_indexer(self, key)
        # 验证并获取 value 参数的设置值
        value = self._validate_setitem_value(value)
        # 设置当前对象 ndarray 的指定索引处的值为 value
        self._ndarray[key] = value

    # 验证并返回用于设置当前对象的 value 参数的有效值
    def _validate_setitem_value(self, value):
        return value

    # 重载 __getitem__ 方法的标量索引版本
    @overload
    def __getitem__(self, key: ScalarIndexer) -> Any: ...

    # 重载 __getitem__ 方法的序列索引版本
    @overload
    def __getitem__(
        self,
        key: SequenceIndexer | PositionalIndexerTuple,
        ...
    ) -> Any: ...
    # 返回类型注释表明该方法返回调用者自身类型
    ) -> Self: ...

    # 从对象中获取指定键的元素或者任意类型的返回值
    def __getitem__(
        self,
        key: PositionalIndexer2D,
    ) -> Self | Any:
        # 如果键是整数类型，使用快速路径获取对应的元素
        if lib.is_integer(key):
            result = self._ndarray[key]
            # 如果对象是一维的，则对结果应用包装函数并返回
            if self.ndim == 1:
                return self._box_func(result)
            # 否则，返回从底层数据创建的新对象
            return self._from_backing_data(result)

        # 如果键不是整数类型，将其转换为数组形式，支持 NumPy 数组的扩展操作
        key = extract_array(key, extract_numpy=True)  # type: ignore[assignment]
        # 检查并调整数组索引，确保其符合对象的要求
        key = check_array_indexer(self, key)
        # 使用调整后的索引从底层数据中获取结果
        result = self._ndarray[key]
        # 如果结果是标量，则应用包装函数并返回
        if lib.is_scalar(result):
            return self._box_func(result)

        # 否则，返回从底层数据创建的新对象
        result = self._from_backing_data(result)
        return result

    # 根据指定的填充方法进行填充或后向填充操作，并返回新对象
    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,  # 指定填充方法的枚举类型
        limit: int | None = None,  # 填充操作的限制次数
        limit_area: Literal["inside", "outside"] | None = None,  # 填充的区域限制
        copy: bool = True,  # 是否复制数据进行填充操作
    ) -> Self:
        # 创建一个表示缺失值的掩码
        mask = self.isna()
        # 如果存在缺失值，则根据对象维度选择填充轴，默认为轴 0
        if mask.any():
            func = missing.get_fill_func(method, ndim=self.ndim)

            # 获取数据的转置副本用于填充操作
            npvalues = self._ndarray.T
            if copy:
                npvalues = npvalues.copy()
            # 执行填充操作，并根据需要限制填充的次数和区域
            func(npvalues, limit=limit, limit_area=limit_area, mask=mask.T)
            npvalues = npvalues.T

            # 根据复制标志创建新的填充后的对象或者直接更新当前对象
            if copy:
                new_values = self._from_backing_data(npvalues)
            else:
                new_values = self

        else:
            # 如果没有缺失值，则根据复制标志决定是返回当前对象的副本还是自身
            if copy:
                new_values = self.copy()
            else:
                new_values = self
        return new_values

    # 使用指定的值填充缺失值，并返回新对象
    @doc(ExtensionArray.fillna)
    def fillna(self, value, limit: int | None = None, copy: bool = True) -> Self:
        # 创建一个表示缺失值的掩码
        mask = self.isna()
        # 如果指定了填充限制且小于对象长度，则根据条件修改掩码
        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit  # type: ignore[union-attr]
            if modify.any():
                # 只有在需要修改时才复制掩码
                mask = mask.copy()
                mask[modify] = False
        # 根据掩码和对象长度验证填充值的大小，并可能进行调整
        value = missing.check_value_size(
            value,
            mask,  # type: ignore[arg-type]
            len(self),
        )

        # 如果存在缺失值，则根据填充值填充对象的副本或者当前对象的副本
        if mask.any():
            if copy:
                new_values = self.copy()
            else:
                new_values = self[:]
            new_values[mask] = value
        else:
            # 即使没有要填充的内容，也要验证填充值的合法性
            self._validate_setitem_value(value)

            # 根据复制标志返回对象的副本或者当前对象的副本
            if not copy:
                new_values = self[:]
            else:
                new_values = self.copy()
        return new_values
    # ------------------------------------------------------------------------
    # Reductions

    # 将归约结果进行封装处理
    def _wrap_reduction_result(self, axis: AxisInt | None, result) -> Any:
        # 如果轴为 None 或者数组为一维，则对结果应用 _box_func 封装
        if axis is None or self.ndim == 1:
            return self._box_func(result)
        # 否则，使用 _from_backing_data 对结果进行处理
        return self._from_backing_data(result)

    # ------------------------------------------------------------------------
    # __array_function__ methods

    # 实现类似于 np.putmask(self, mask, value) 的功能
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        """
        Analogue to np.putmask(self, mask, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        # 验证并处理设置项的值
        value = self._validate_setitem_value(value)

        # 使用 np.putmask 将 value 应用于 self._ndarray
        np.putmask(self._ndarray, mask, value)

    # 实现类似于 np.where(mask, self, value) 的功能
    def _where(self: Self, mask: npt.NDArray[np.bool_], value) -> Self:
        """
        Analogue to np.where(mask, self, value)

        Parameters
        ----------
        mask : np.ndarray[bool]
        value : scalar or listlike

        Raises
        ------
        TypeError
            If value cannot be cast to self.dtype.
        """
        # 验证并处理设置项的值
        value = self._validate_setitem_value(value)

        # 使用 np.where 处理 mask，并将结果保存到 res_values 中
        res_values = np.where(mask, self._ndarray, value)
        # 如果结果的数据类型与 self._ndarray 的数据类型不同，抛出断言错误
        if res_values.dtype != self._ndarray.dtype:
            raise AssertionError(
                # GH#56410
                "Something has gone wrong, please report a bug at "
                "github.com/pandas-dev/pandas/"
            )
        # 使用 _from_backing_data 将结果数据封装并返回
        return self._from_backing_data(res_values)

    # ------------------------------------------------------------------------
    # Index compat methods

    # 在特定位置插入新元素并返回新的 ExtensionArray 对象
    def insert(self, loc: int, item) -> Self:
        """
        Make new ExtensionArray inserting new item at location. Follows
        Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        type(self)
        """
        # 验证插入位置 loc 的有效性
        loc = validate_insert_loc(loc, len(self))

        # 验证并处理插入的单个元素 item
        code = self._validate_scalar(item)

        # 构建新的值数组 new_vals，将 code 插入到指定位置 loc 处
        new_vals = np.concatenate(
            (
                self._ndarray[:loc],
                np.asarray([code], dtype=self._ndarray.dtype),
                self._ndarray[loc:],
            )
        )
        # 使用 _from_backing_data 封装新的值数组并返回新的 ExtensionArray 对象
        return self._from_backing_data(new_vals)

    # ------------------------------------------------------------------------
    # Additional array methods
    # These are not part of the EA API, but we implement them because
    # pandas assumes they're there.
    def value_counts(self, dropna: bool = True) -> Series:
        """
        Return a Series containing counts of unique values.

        Parameters
        ----------
        dropna : bool, default True
            Don't include counts of NA values.

        Returns
        -------
        Series
        """
        # 如果数组的维度不是1，则抛出未实现的错误
        if self.ndim != 1:
            raise NotImplementedError

        from pandas import (
            Index,
            Series,
        )

        if dropna:
            # 从当前对象中筛选出非缺失值的数组，并获取其内部的 numpy 数组
            values = self[~self.isna()]._ndarray  # type: ignore[operator]
        else:
            # 直接获取当前对象内部的 numpy 数组
            values = self._ndarray

        # 调用 pandas 的 value_counts 函数统计值的频数
        result = value_counts(values, sort=False, dropna=dropna)

        # 将结果的索引部分转换成合适的格式
        index_arr = self._from_backing_data(np.asarray(result.index._data))
        index = Index(index_arr, name=result.index.name)
        # 返回一个新的 Series 对象，包含统计结果，不进行复制操作
        return Series(result._values, index=index, name=result.name, copy=False)

    def _quantile(
        self,
        qs: npt.NDArray[np.float64],
        interpolation: str,
    ) -> Self:
        # TODO: 如果分类变量未排序，可能需要禁用此方法？

        # 创建一个布尔掩码，标识当前对象中的缺失值
        mask = np.asarray(self.isna())
        # 获取当前对象内部的 numpy 数组
        arr = self._ndarray
        # 获取填充缺失值时使用的填充值
        fill_value = self._internal_fill_value

        # 调用 quantile_with_mask 函数计算带掩码的分位数
        res_values = quantile_with_mask(arr, mask, fill_value, qs, interpolation)

        # 将结果数组转换为适当的类型，以便在 _from_backing_data 中使用
        res_values = self._cast_quantile_result(res_values)
        # 返回一个新的对象，使用计算得到的结果数据
        return self._from_backing_data(res_values)

    # TODO: 看看是否可以与其他调度封装方法共享此函数
    def _cast_quantile_result(self, res_values: np.ndarray) -> np.ndarray:
        """
        Cast the result of quantile_with_mask to an appropriate dtype
        to pass to _from_backing_data in _quantile.
        """
        # 直接返回量化计算结果的 numpy 数组
        return res_values

    # ------------------------------------------------------------------------
    # 类似于 numpy 的方法

    @classmethod
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        """
        Analogous to np.empty(shape, dtype=dtype)

        Parameters
        ----------
        shape : tuple[int]
        dtype : ExtensionDtype
        """
        # 基本实现使用了一种简单的方法来确定底层数组的 dtype
        # 从序列中创建一个空的数组对象，使用指定的 dtype
        arr = cls._from_sequence([], dtype=dtype)
        # 使用 numpy 创建一个空的数组作为底层数据
        backing = np.empty(shape, dtype=arr._ndarray.dtype)
        # 返回一个新的数组对象，使用 numpy 创建的底层数据
        return arr._from_backing_data(backing)
```