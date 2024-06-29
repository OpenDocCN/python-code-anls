# `D:\src\scipysrc\pandas\pandas\core\arrays\masked.py`

```
    # 引入了未来的注解功能，允许在类型检查中使用 annotations
from __future__ import annotations

    # 引入了多个类型提示相关的模块和函数
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

    # 引入警告模块，用于生成警告信息
import warnings

    # 引入 NumPy 库
import numpy as np

    # 引入 Pandas 内部的一些 C 库和函数
from pandas._libs import (
    lib,
    missing as libmissing,
)

    # 引入 Pandas 的时间序列库的支持数据类型检查函数
from pandas._libs.tslibs import is_supported_dtype

    # 引入 Pandas 兼容性相关的函数和常量
from pandas.compat import (
    IS64,
    is_platform_windows,
)

    # 引入 Pandas 异常处理模块中的抽象方法错误类
from pandas.errors import AbstractMethodError

    # 引入 Pandas 内部的文档装饰器
from pandas.util._decorators import doc

    # 引入 Pandas 核心数据类型基类和常见的数据类型判断函数
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_bool,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)

    # 引入 Pandas 核心数据类型模块中的掩码数据类型基类
from pandas.core.dtypes.dtypes import BaseMaskedDtype

    # 引入 Pandas 核心数据类型模块中的缺失数据处理函数
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
    notna,
)

    # 引入 Pandas 核心模块和函数：算法、数组处理、缺失值处理、运算等
from pandas.core import (
    algorithms as algos,
    arraylike,
    missing,
    nanops,
    ops,
)

    # 引入 Pandas 核心算法模块中的数组处理函数
from pandas.core.algorithms import (
    factorize_array,
    isin,
    map_array,
    mode,
    take,
)

    # 引入 Pandas 核心数组算法模块中的掩码数据处理函数
from pandas.core.array_algos import (
    masked_accumulations,
    masked_reductions,
)

    # 引入 Pandas 核心数组算法模块中的分位数计算函数
from pandas.core.array_algos.quantile import quantile_with_mask

    # 引入 Pandas 核心数组混合操作类
from pandas.core.arraylike import OpsMixin

    # 引入 Pandas 核心数组工具模块中的数据类型推断函数
from pandas.core.arrays._utils import to_numpy_dtype_inference

    # 引入 Pandas 核心数组基类和扩展数组基类
from pandas.core.arrays.base import ExtensionArray

    # 引入 Pandas 核心构建模块中的数组构建、日期时间处理、数据提取函数
from pandas.core.construction import (
    array as pd_array,
    ensure_wrapped_if_datetimelike,
    extract_array,
)

    # 引入 Pandas 核心索引器模块中的数组索引检查函数
from pandas.core.indexers import check_array_indexer

    # 引入 Pandas 核心运算模块中的无效比较函数
from pandas.core.ops import invalid_comparison

    # 引入 Pandas 核心工具模块中的数组哈希计算函数
from pandas.core.util.hashing import hash_array

    # 如果开启类型检查，引入额外的类型注解
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import (
        Iterator,
        Sequence,
    )
    from pandas import Series
    from pandas.core.arrays import BooleanArray
    from pandas._typing import (
        NumpySorter,
        NumpyValueArrayLike,
        ArrayLike,
        AstypeArg,
        AxisInt,
        DtypeObj,
        FillnaOptions,
        InterpolateOptions,
        NpDtype,
        PositionalIndexer,
        Scalar,
        ScalarIndexer,
        Self,
        SequenceIndexer,
        Shape,
        npt,
    )
    from pandas._libs.missing import NAType
    from pandas.core.arrays import FloatingArray

    # 引入 Pandas 兼容性 Numpy 函数别名
from pandas.compat.numpy import function as nv


    # 定义掩码数组基类，继承自 OpsMixin 和 ExtensionArray
class BaseMaskedArray(OpsMixin, ExtensionArray):
    """
    Base class for masked arrays (which use _data and _mask to store the data).

    numpy based
    """

    # 我们的底层数据和掩码都是 ndarray 类型
    _data: np.ndarray
    _mask: npt.NDArray[np.bool_]

    # 类方法：创建新的掩码数组实例，接受数据数组和掩码数组作为参数，并返回新的实例
    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        result = BaseMaskedArray.__new__(cls)
        result._data = values
        result._mask = mask
        return result

    # 构造函数：初始化掩码数组对象，接受数据数组、掩码数组和复制标志作为参数
    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        # 定义一个初始化方法，接受两个参数：values 和 mask
        # 在子类中应该保证 values 已经被验证过
        if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_):
            # 如果 mask 不是布尔型的 NumPy 数组，抛出类型错误异常
            raise TypeError(
                "mask should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        if values.shape != mask.shape:
            # 如果 values 和 mask 的形状不匹配，抛出值错误异常
            raise ValueError("values.shape must match mask.shape")

        if copy:
            # 如果 copy 参数为 True，复制 values 和 mask
            values = values.copy()
            mask = mask.copy()

        # 将 values 赋值给对象的 _data 属性，将 mask 赋值给 _mask 属性
        self._data = values
        self._mask = mask

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy: bool = False) -> Self:
        # 从序列创建对象的类方法
        # 调用 _coerce_to_array 方法将 scalars 转换为 values 和 mask
        values, mask = cls._coerce_to_array(scalars, dtype=dtype, copy=copy)
        # 返回当前类的实例，使用 values 和 mask 初始化
        return cls(values, mask)

    @classmethod
    @doc(ExtensionArray._empty)
    def _empty(cls, shape: Shape, dtype: ExtensionDtype) -> Self:
        # 创建一个空的对象的类方法
        dtype = cast(BaseMaskedDtype, dtype)
        # 使用指定的形状和数据类型创建一个空的 NumPy 数组 values
        values: np.ndarray = np.empty(shape, dtype=dtype.type)
        # 使用指定的形状创建一个全部为 True 的布尔型 NumPy 数组 mask
        mask = np.ones(shape, dtype=bool)
        # 使用 values 和 mask 初始化当前类的实例 result
        result = cls(values, mask)
        # 如果 result 不是当前类的实例，或者数据类型不匹配，则抛出未实现错误异常
        if not isinstance(result, cls) or dtype != result.dtype:
            raise NotImplementedError(
                f"Default 'empty' implementation is invalid for dtype='{dtype}'"
            )
        # 返回创建的实例 result
        return result

    def _formatter(self, boxed: bool = False) -> Callable[[Any], str | None]:
        # 返回一个格式化函数，根据 boxed 参数决定是否返回格式化后的字符串
        # 参考 NEP 51: https://github.com/numpy/numpy/pull/22449
        return str

    @property
    def dtype(self) -> BaseMaskedDtype:
        # 返回当前对象的数据类型，应该由子类实现
        raise AbstractMethodError(self)

    @overload
    def __getitem__(self, item: ScalarIndexer) -> Any: ...
    
    @overload
    def __getitem__(self, item: SequenceIndexer) -> Self: ...
    
    def __getitem__(self, item: PositionalIndexer) -> Self | Any:
        # 获取对象的某个元素或子集的方法
        # 调用 check_array_indexer 方法检查 item 是否是有效的索引器
        item = check_array_indexer(self, item)

        # 根据索引 item 更新当前对象的掩码 newmask
        newmask = self._mask[item]
        if is_bool(newmask):
            # 如果 newmask 是布尔值
            # 这是标量索引
            if newmask:
                # 如果 newmask 是 True，返回数据类型的缺失值
                return self.dtype.na_value
            # 否则返回对应位置的数据值
            return self._data[item]

        # 否则返回一个新的对象，使用 _data[item] 和 newmask 初始化
        return self._simple_new(self._data[item], newmask)

    def _pad_or_backfill(
        self,
        *,
        method: FillnaOptions,
        limit: int | None = None,
        limit_area: Literal["inside", "outside"] | None = None,
        copy: bool = True,
    @doc(ExtensionArray.fillna)
    # 使用 @doc 装饰器将下面的方法与 ExtensionArray.fillna 文档相关联
    def fillna(self, value, limit: int | None = None, copy: bool = True) -> Self:
        # 获取当前对象的遮罩（掩码）
        mask = self._mask
        # 如果 limit 不为 None 且小于对象的长度，则创建一个布尔数组标记需要修改的位置
        if limit is not None and limit < len(self):
            modify = mask.cumsum() > limit
            # 如果有需要修改的位置，则将 mask 复制一份并修改相应位置为 False
            if modify.any():
                # 只在必要时复制 mask
                mask = mask.copy()
                mask[modify] = False

        # 检查并调整 value 的大小以匹配 mask 和当前对象的长度
        value = missing.check_value_size(value, mask, len(self))

        # 如果 mask 中有需要填充的位置
        if mask.any():
            # 如果需要复制对象
            if copy:
                # 创建当前对象的副本
                new_values = self.copy()
            else:
                # 否则将当前对象切片复制给 new_values
                new_values = self[:]
            # 将 mask 中的位置用 value 填充
            new_values[mask] = value
        else:
            # 如果 mask 中没有需要填充的位置
            if copy:
                # 如果需要复制对象，则创建当前对象的副本
                new_values = self.copy()
            else:
                # 否则将当前对象切片复制给 new_values
                new_values = self[:]
        # 返回填充后的新对象 new_values
        return new_values

    @classmethod
    def _coerce_to_array(
        cls, values, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # 抛出抽象方法错误，提示需要在子类中实现该方法
        raise AbstractMethodError(cls)
    def _validate_setitem_value(self, value):
        """
        Check if we have a scalar that we can cast losslessly.

        Raises
        ------
        TypeError: If the value cannot be cast to the dtype of the object.
        """
        kind = self.dtype.kind
        # TODO: get this all from np_can_hold_element?
        # 如果数据类型是布尔型并且值是布尔型，直接返回该值
        if kind == "b":
            if lib.is_bool(value):
                return value

        # 如果数据类型是浮点型，值可以是整数或浮点数
        elif kind == "f":
            if lib.is_integer(value) or lib.is_float(value):
                return value

        # 对于其他数据类型，值可以是整数或浮点数（且为整数类型的浮点数）
        else:
            if lib.is_integer(value) or (lib.is_float(value) and value.is_integer()):
                return value
            # TODO: unsigned checks

        # 如果以上条件都不满足，抛出类型错误异常
        raise TypeError(f"Invalid value '{value!s}' for dtype {self.dtype}")

    def __setitem__(self, key, value) -> None:
        """
        Set an item in the object with validation.

        Parameters
        ----------
        key : object
            Key/index to set.
        value : object
            Value to set.

        Raises
        ------
        TypeError: If the value cannot be validated for the dtype of the object.
        """
        key = check_array_indexer(self, key)

        # 如果值是标量并且是有效的 NA 值，则将对应的掩码位置设为 True
        if is_scalar(value):
            if is_valid_na_for_dtype(value, self.dtype):
                self._mask[key] = True
            else:
                # 否则，验证值的有效性，并将数据和掩码设置为相应的值
                value = self._validate_setitem_value(value)
                self._data[key] = value
                self._mask[key] = False
            return

        # 如果值不是标量，则尝试强制转换成数组，并设置数据和掩码
        value, mask = self._coerce_to_array(value, dtype=self.dtype)

        self._data[key] = value
        self._mask[key] = mask

    def __contains__(self, key) -> bool:
        """
        Check if the object contains a given key.

        Parameters
        ----------
        key : object
            Key to check for containment.

        Returns
        -------
        bool
            True if the key is contained in the object, False otherwise.
        """
        if isna(key) and key is not self.dtype.na_value:
            # GH#52840
            # 如果数据类型是浮点型并且 key 是浮点数，则检查是否有 NaN 且不在掩码内
            if self._data.dtype.kind == "f" and lib.is_float(key):
                return bool((np.isnan(self._data) & ~self._mask).any())

        # 否则，调用父类方法检查是否包含 key
        return bool(super().__contains__(key))

    def __iter__(self) -> Iterator:
        """
        Return an iterator over the object's elements.

        Returns
        -------
        Iterator
            Iterator over the elements of the object.
        """
        if self.ndim == 1:
            if not self._hasna:
                # 如果没有缺失值，直接迭代数据数组中的值
                for val in self._data:
                    yield val
            else:
                # 否则，根据掩码决定返回实际值或缺失值
                na_value = self.dtype.na_value
                for isna_, val in zip(self._mask, self._data):
                    if isna_:
                        yield na_value
                    else:
                        yield val
        else:
            # 对于多维数组，按顺序迭代索引
            for i in range(len(self)):
                yield self[i]

    def __len__(self) -> int:
        """
        Return the number of elements in the object.

        Returns
        -------
        int
            Number of elements in the object.
        """
        return len(self._data)

    @property
    def shape(self) -> Shape:
        """
        Return the shape of the object's data array.

        Returns
        -------
        Shape
            Shape tuple of the object's data array.
        """
        return self._data.shape

    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions of the object's data array.

        Returns
        -------
        int
            Number of dimensions of the object's data array.
        """
        return self._data.ndim

    def swapaxes(self, axis1, axis2) -> Self:
        """
        Swap the axes of the object's data and mask arrays.

        Parameters
        ----------
        axis1 : int
            First axis to swap.
        axis2 : int
            Second axis to swap.

        Returns
        -------
        Self
            New instance of the object with swapped axes.
        """
        data = self._data.swapaxes(axis1, axis2)
        mask = self._mask.swapaxes(axis1, axis2)
        return self._simple_new(data, mask)

    def delete(self, loc, axis: AxisInt = 0) -> Self:
        """
        Delete elements from the object along the specified axis.

        Parameters
        ----------
        loc : slice, int, or array of int
            Indices to delete.
        axis : int, optional
            Axis along which to delete (default is 0).

        Returns
        -------
        Self
            New instance of the object with elements deleted.
        """
        data = np.delete(self._data, loc, axis=axis)
        mask = np.delete(self._mask, loc, axis=axis)
        return self._simple_new(data, mask)

    def reshape(self, *args, **kwargs) -> Self:
        """
        Reshape the object's data and mask arrays.

        Returns
        -------
        Self
            New instance of the object with reshaped data and mask.
        """
        data = self._data.reshape(*args, **kwargs)
        mask = self._mask.reshape(*args, **kwargs)
        return self._simple_new(data, mask)
    def ravel(self, *args, **kwargs) -> Self:
        """
        Flatten the data and mask arrays.

        Parameters
        ----------
        *args, **kwargs
            Additional arguments to pass to the `ravel` method of the data and mask arrays.

        Returns
        -------
        Self
            A new instance of the class with flattened data and mask arrays.
        """
        # TODO: need to make sure we have the same order for data/mask
        # Flatten the data array
        data = self._data.ravel(*args, **kwargs)
        # Flatten the mask array
        mask = self._mask.ravel(*args, **kwargs)
        # Return a new instance of the class with flattened data and mask arrays
        return type(self)(data, mask)

    @property
    def T(self) -> Self:
        """
        Transpose the data and mask arrays.

        Returns
        -------
        Self
            A new instance of the class with transposed data and mask arrays.
        """
        # Transpose the data and mask arrays and return a new instance of the class
        return self._simple_new(self._data.T, self._mask.T)

    def round(self, decimals: int = 0, *args, **kwargs):
        """
        Round each value in the array to the given number of decimals.

        Parameters
        ----------
        decimals : int, default 0
            Number of decimal places to round to. If negative, rounds to the left of the decimal point.
        *args, **kwargs
            Additional arguments and keywords passed to numpy's round function.

        Returns
        -------
        NumericArray
            Rounded values of the NumericArray.

        See Also
        --------
        numpy.around : Round values of an np.array.
        DataFrame.round : Round values of a DataFrame.
        Series.round : Round values of a Series.
        """
        if self.dtype.kind == "b":
            return self  # If the dtype is boolean, return self without rounding
        nv.validate_round(args, kwargs)  # Validate additional arguments
        # Round the data array
        values = np.round(self._data, decimals=decimals, **kwargs)
        # Return rounded values with potentially masked results
        return self._maybe_mask_result(values, self._mask.copy())

    # ------------------------------------------------------------------
    # Unary Methods

    def __invert__(self) -> Self:
        """
        Invert (bitwise NOT) the data array.

        Returns
        -------
        Self
            A new instance of the class with inverted data array.
        """
        return self._simple_new(~self._data, self._mask.copy())

    def __neg__(self) -> Self:
        """
        Negate (arithmetic negative) the data array.

        Returns
        -------
        Self
            A new instance of the class with negated data array.
        """
        return self._simple_new(-self._data, self._mask.copy())

    def __pos__(self) -> Self:
        """
        Return a copy of the instance.

        Returns
        -------
        Self
            A copy of the instance.
        """
        return self.copy()

    def __abs__(self) -> Self:
        """
        Compute the absolute values of the data array.

        Returns
        -------
        Self
            A new instance of the class with absolute values of the data array.
        """
        return self._simple_new(abs(self._data), self._mask.copy())

    # ------------------------------------------------------------------

    def _values_for_json(self) -> np.ndarray:
        """
        Return the instance's data as a NumPy array suitable for JSON serialization.

        Returns
        -------
        np.ndarray
            NumPy array representation of the instance's data.
        """
        return np.asarray(self, dtype=object)

    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the instance to a NumPy array.

        Parameters
        ----------
        dtype : numpy.dtype or None, optional
            Desired data type for the array.
        copy : bool, default False
            Whether to copy the data.
        na_value : object, default lib.no_default
            Value to interpret as NA (Not Available).

        Returns
        -------
        np.ndarray
            NumPy array representation of the instance.

        Notes
        -----
        This method is equivalent to pandas' ExtensionArray.tolist.
        """
        if self.ndim > 1:
            return [x.tolist() for x in self]  # Convert multidimensional array to list
        dtype = None if self._hasna else self._data.dtype
        # Convert to NumPy array and then to list
        return self.to_numpy(dtype=dtype, na_value=libmissing.NA).tolist()

    @overload
    def astype(self, dtype: npt.DTypeLike, copy: bool = ...) -> np.ndarray: ...
    
    @overload
    def astype(self, dtype: ExtensionDtype, copy: bool = ...) -> ExtensionArray: ...

    @overload
    def astype(self, dtype: AstypeArg, copy: bool = ...) -> ArrayLike: ...
    # 将数组转换为指定的数据类型
    def astype(self, dtype: AstypeArg, copy: bool = True) -> ArrayLike:
        # 将 dtype 转换为 Pandas 的数据类型对象
        dtype = pandas_dtype(dtype)

        # 如果目标 dtype 与当前数组的 dtype 相同
        if dtype == self.dtype:
            # 如果需要复制，则返回当前数组的副本
            if copy:
                return self.copy()
            # 否则返回当前数组本身
            return self

        # 如果目标 dtype 是另一种可空掩码 dtype，则进行优化处理
        if isinstance(dtype, BaseMaskedDtype):
            # TODO 处理 FloatingArray 情况下的 NaNs
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                # 将数据转换为目标 dtype 对应的 numpy 数据类型
                data = self._data.astype(dtype.numpy_dtype, copy=copy)
            # 根据数据是否复制来决定是否复制掩码，而不是直接依赖于 `copy` 关键字
            mask = self._mask if data is self._data else self._mask.copy()
            # 使用目标 dtype 的构造函数创建新的数组对象
            cls = dtype.construct_array_type()
            return cls(data, mask, copy=False)

        # 如果目标 dtype 是扩展 dtype
        if isinstance(dtype, ExtensionDtype):
            # 使用目标 dtype 的构造函数创建数组对象
            eacls = dtype.construct_array_type()
            # 从当前对象中创建新的数组对象
            return eacls._from_sequence(self, dtype=dtype, copy=copy)

        # 定义用于 NA 值的变量，可以是 float、datetime 或没有默认值
        na_value: float | np.datetime64 | lib.NoDefault

        # 强制转换
        if dtype.kind == "f":
            # 在 astype 中，dtype=float 也意味着 na_value=np.nan
            na_value = np.nan
        elif dtype.kind == "M":
            na_value = np.datetime64("NaT")
        else:
            na_value = lib.no_default

        # 如果 dtype 是整数或无符号整数类型，并且存在 NA 值
        if dtype.kind in "iu" and self._hasna:
            raise ValueError("cannot convert NA to integer")
        # 如果 dtype 是布尔类型，并且存在 NA 值
        if dtype.kind == "b" and self._hasna:
            # 注意：astype_nansafe 会将 np.nan 转换为 True
            raise ValueError("cannot convert float NaN to bool")

        # 将数据转换为 numpy 数组
        data = self.to_numpy(dtype=dtype, na_value=na_value, copy=copy)
        return data

    # 数组对象的优先级，比 ndarray 高，使得操作会优先调用本对象的方法
    __array_priority__ = 1000

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        """
        数组接口，返回对象的值
        这里返回一个对象数组以保留标量值
        """
        return self.to_numpy(dtype=dtype)

    # 可处理的类型元组，用于判断对象可以处理的类型范围
    _HANDLED_TYPES: tuple[type, ...]
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        """
        处理数组对象与 NumPy ufunc（通用函数）的交互。

        对于 MaskedArray 输入，将 ufunc 应用于 ._data，并对结果进行遮罩。

        Parameters:
        ufunc (np.ufunc): 要应用的 NumPy ufunc 对象。
        method (str): 调用的方法名称。
        *inputs: 输入参数列表。
        **kwargs: 关键字参数，包括 "out" 参数。

        Returns:
        根据不同的调用情况返回处理后的结果。
        """
        out = kwargs.get("out", ())

        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES + (BaseMaskedArray,)):
                return NotImplemented

        # 对于二元操作，使用自定义的双下划线方法
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # 例如，test_ufunc_with_out
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        mask = np.zeros(len(self), dtype=bool)
        inputs2 = []
        for x in inputs:
            if isinstance(x, BaseMaskedArray):
                mask |= x._mask
                inputs2.append(x._data)
            else:
                inputs2.append(x)

        def reconstruct(x: np.ndarray):
            """
            根据输入的 NumPy 数组重构返回适当的数组对象。

            Parameters:
            x (np.ndarray): 要重构的 NumPy 数组。

            Returns:
            重构后的适当的数组对象，例如 BooleanArray、IntegerArray 或 FloatingArray。
            """
            from pandas.core.arrays import (
                BooleanArray,
                FloatingArray,
                IntegerArray,
            )

            if x.dtype.kind == "b":
                m = mask.copy()
                return BooleanArray(x, m)
            elif x.dtype.kind in "iu":
                m = mask.copy()
                return IntegerArray(x, m)
            elif x.dtype.kind == "f":
                m = mask.copy()
                if x.dtype == np.float16:
                    # 在例如 np.sqrt 中使用 BooleanArray 时达到此处
                    # 我们不支持 float16，需要转换为 np.float32
                    x = x.astype(np.float32)
                return FloatingArray(x, m)
            else:
                x[mask] = np.nan
            return x

        result = getattr(ufunc, method)(*inputs2, **kwargs)
        if ufunc.nout > 1:
            # 例如，np.divmod
            return tuple(reconstruct(x) for x in result)
        elif method == "reduce":
            # 例如，np.add.reduce；test_ufunc_reduce_raises
            if self._mask.any():
                return self._na_value
            return result
        else:
            return reconstruct(result)

    def __arrow_array__(self, type=None):
        """
        将自身转换为 pyarrow Array 对象。

        Parameters:
        type: 要转换为的 pyarrow 类型。

        Returns:
        转换后的 pyarrow Array 对象。
        """
        import pyarrow as pa

        return pa.array(self._data, mask=self._mask, type=type)

    @property
    def _hasna(self) -> bool:
        """
        # Note: this is expensive right now! The hope is that we can
        # make this faster by having an optional mask, but not have to change
        # source code using it..
        """
        return self._mask.any()  # type: ignore[return-value]

    def _propagate_mask(
        self, mask: npt.NDArray[np.bool_] | None, other
    ) -> npt.NDArray[np.bool_]:
        """
        Propagates the mask to handle missing values in comparison operations.

        Args:
            mask: Optional mask array indicating missing values.
            other: Object to compare with.

        Returns:
            npt.NDArray[np.bool_]: Mask array after propagation.

        Notes:
            - If mask is None, it initializes with a copy of self._mask.
            - Handles special cases for missing values and comparison operations.
        """
        if mask is None:
            mask = self._mask.copy()  # TODO: need test for BooleanArray needing a copy
            if other is libmissing.NA:
                # GH#45421 don't alter inplace
                mask = mask | True
            elif is_list_like(other) and len(other) == len(mask):
                mask = mask | isna(other)
        else:
            mask = self._mask | mask
        return mask  # type: ignore[return-value]

    _logical_method = _arith_method

    def _cmp_method(self, other, op) -> BooleanArray:
        """
        Compares the current object with another using a specified operation.

        Args:
            other: Object to compare with.
            op: Operation to perform the comparison.

        Returns:
            BooleanArray: Boolean array result of the comparison.

        Raises:
            NotImplementedError: If the comparison involves multi-dimensional structures.
            ValueError: If lengths of self and other do not match for comparison.

        Notes:
            - Handles comparison involving pd.NA and warns about future numpy behavior.
            - Uses _propagate_mask to handle missing values in comparison.
        """
        from pandas.core.arrays import BooleanArray

        mask = None

        if isinstance(other, BaseMaskedArray):
            other, mask = other._data, other._mask

        elif is_list_like(other):
            other = np.asarray(other)
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            if len(self) != len(other):
                raise ValueError("Lengths must match to compare")

        if other is libmissing.NA:
            # numpy does not handle pd.NA well as "other" scalar (it returns
            # a scalar False instead of an array)
            # This may be fixed by NA.__array_ufunc__. Revisit this check
            # once that's implemented.
            result = np.zeros(self._data.shape, dtype="bool")
            mask = np.ones(self._data.shape, dtype="bool")
        else:
            with warnings.catch_warnings():
                # numpy may show a FutureWarning or DeprecationWarning:
                #     elementwise comparison failed; returning scalar instead,
                #     but in the future will perform elementwise comparison
                # before returning NotImplemented. We fall back to the correct
                # behavior today, so that should be fine to ignore.
                warnings.filterwarnings("ignore", "elementwise", FutureWarning)
                warnings.filterwarnings("ignore", "elementwise", DeprecationWarning)
                method = getattr(self._data, f"__{op.__name__}__")
                result = method(other)

                if result is NotImplemented:
                    result = invalid_comparison(self._data, other, op)

        mask = self._propagate_mask(mask, other)
        return BooleanArray(result, mask, copy=False)
    def _maybe_mask_result(
        self, result: np.ndarray | tuple[np.ndarray, np.ndarray], mask: np.ndarray
    ):
        """
        对结果进行可能的遮罩处理，根据结果的数据类型选择适当的处理方式并返回处理后的对象。

        Parameters
        ----------
        result : array-like or tuple[array-like]
            结果数据，可以是数组或数组元组。
        mask : array-like bool
            遮罩数组，用于标记结果中的无效值。

        Returns
        -------
        array-like
            处理后的结果对象，可能是浮点数组、布尔数组、时间增量数组或整数数组。
        """
        if isinstance(result, tuple):
            # 如果结果是元组，例如 divmod 返回的两个数组
            div, mod = result
            return (
                self._maybe_mask_result(div, mask),  # 递归处理 div 部分
                self._maybe_mask_result(mod, mask),  # 递归处理 mod 部分
            )

        if result.dtype.kind == "f":
            from pandas.core.arrays import FloatingArray

            return FloatingArray(result, mask, copy=False)  # 返回浮点数组对象

        elif result.dtype.kind == "b":
            from pandas.core.arrays import BooleanArray

            return BooleanArray(result, mask, copy=False)  # 返回布尔数组对象

        elif lib.is_np_dtype(result.dtype, "m") and is_supported_dtype(result.dtype):
            # 如果数据类型是时间增量，并且是支持的数据类型
            from pandas.core.arrays import TimedeltaArray

            result[mask] = result.dtype.type("NaT")  # 标记 mask 中的时间增量为 NaT

            if not isinstance(result, TimedeltaArray):
                return TimedeltaArray._simple_new(result, dtype=result.dtype)  # 返回时间增量数组对象

            return result

        elif result.dtype.kind in "iu":
            from pandas.core.arrays import IntegerArray

            return IntegerArray(result, mask, copy=False)  # 返回整数数组对象

        else:
            result[mask] = np.nan  # 标记 mask 中的无效值为 NaN
            return result

    def isna(self) -> np.ndarray:
        """
        返回对象的遮罩数组副本。

        Returns
        -------
        np.ndarray
            遮罩数组副本，表示对象中的缺失值位置。
        """
        return self._mask.copy()

    @property
    def _na_value(self):
        """
        返回数据类型的缺失值。

        Returns
        -------
        scalar
            数据类型的缺失值。
        """
        return self.dtype.na_value

    @property
    def nbytes(self) -> int:
        """
        返回对象数据占用的字节数。

        Returns
        -------
        int
            对象数据占用的总字节数。
        """
        return self._data.nbytes + self._mask.nbytes

    @classmethod
    def _concat_same_type(
        cls,
        to_concat: Sequence[Self],
        axis: AxisInt = 0,
    ) -> Self:
        """
        合并同类型的对象数组。

        Parameters
        ----------
        to_concat : Sequence[Self]
            要合并的对象数组序列。
        axis : AxisInt, optional
            合并的轴向，默认为0。

        Returns
        -------
        Self
            合并后的对象。
        """
        data = np.concatenate([x._data for x in to_concat], axis=axis)
        mask = np.concatenate([x._mask for x in to_concat], axis=axis)
        return cls(data, mask)

    def _hash_pandas_object(
        self, *, encoding: str, hash_key: str, categorize: bool
    ) -> npt.NDArray[np.uint64]:
        """
        对 Pandas 对象进行哈希处理。

        Parameters
        ----------
        encoding : str
            编码方式。
        hash_key : str
            哈希键。
        categorize : bool
            是否进行分类。

        Returns
        -------
        npt.NDArray[np.uint64]
            哈希处理后的无符号 64 位整数数组。
        """
        hashed_array = hash_array(
            self._data, encoding=encoding, hash_key=hash_key, categorize=categorize
        )
        hashed_array[self.isna()] = hash(self.dtype.na_value)  # 将缺失值位置的哈希值设为数据类型的缺失值的哈希值
        return hashed_array

    def take(
        self,
        indexer,
        *,
        allow_fill: bool = False,
        fill_value: Scalar | None = None,
        axis: AxisInt = 0,
    ) -> Self:
        # 我们总是使用内部的 1 来填充
        # 以避免上溢
        data_fill_value = (
            self.dtype._internal_fill_value if isna(fill_value) else fill_value
        )
        # 使用 take 函数获取根据索引取得的数据
        result = take(
            self._data,
            indexer,
            fill_value=data_fill_value,
            allow_fill=allow_fill,
            axis=axis,
        )

        # 使用 take 函数获取根据索引取得的掩码
        mask = take(
            self._mask, indexer, fill_value=True, allow_fill=allow_fill, axis=axis
        )

        # 如果需要填充
        # 只在索引处为 null 的地方填充
        # 而不是现有的缺失值
        # TODO(jreback) 如果填充值是非 NA 浮点数会怎样？
        if allow_fill and notna(fill_value):
            fill_mask = np.asarray(indexer) == -1
            result[fill_mask] = fill_value
            mask = mask ^ fill_mask

        # 返回一个新的对象，使用 _simple_new 方法创建
        return self._simple_new(result, mask)

    # error: Return type "BooleanArray" of "isin" incompatible with return type
    # "ndarray" in supertype "ExtensionArray"
    def isin(self, values: ArrayLike) -> BooleanArray:  # type: ignore[override]
        from pandas.core.arrays import BooleanArray

        # 算法 isin 最终会将值转换为 ndarray，所以在这里先做转换没有额外成本
        values_arr = np.asarray(values)
        # 使用 isin 函数检查 self._data 中的值是否在 values_arr 中
        result = isin(self._data, values_arr)

        if self._hasna:
            # 检查 values_arr 是否包含 NA 值，并且与 self.dtype.na_value 相同
            values_have_NA = values_arr.dtype == object and any(
                val is self.dtype.na_value for val in values_arr
            )

            # 目前 NA 值不会传播，因此根据 NA 的存在设置 result 的值
            # 参见 https://github.com/pandas-dev/pandas/pull/38379 进行了一些讨论
            result[self._mask] = values_have_NA

        # 创建一个布尔型数组对象，返回结果
        mask = np.zeros(self._data.shape, dtype=bool)
        return BooleanArray(result, mask, copy=False)

    def copy(self) -> Self:
        # 复制数据和掩码，返回一个新对象
        data = self._data.copy()
        mask = self._mask.copy()
        return self._simple_new(data, mask)

    @doc(ExtensionArray.duplicated)
    def duplicated(
        self, keep: Literal["first", "last", False] = "first"
    ) -> npt.NDArray[np.bool_]:
        # 获取数据和掩码，使用 algos.duplicated 计算重复值
        values = self._data
        mask = self._mask
        return algos.duplicated(values, keep=keep, mask=mask)

    def unique(self) -> Self:
        """
        计算唯一值的 BaseMaskedArray。

        返回
        -------
        uniques : BaseMaskedArray
        """
        # 使用 algos.unique_with_mask 计算唯一值和掩码
        uniques, mask = algos.unique_with_mask(self._data, self._mask)
        # 返回一个新对象，使用 _simple_new 方法创建
        return self._simple_new(uniques, mask)

    @doc(ExtensionArray.searchsorted)
    def searchsorted(
        self,
        value: NumpyValueArrayLike | ExtensionArray,
        side: Literal["left", "right"] = "left",
        sorter: NumpySorter | None = None,
    # 返回一个 numpy 数组或者一个整数索引数组
        ) -> npt.NDArray[np.intp] | np.intp:
            # 如果数组中有缺失值，抛出 ValueError 异常，因为 searchsorted 要求数组是有序的，而有缺失值时无法排序。
            if self._hasna:
                raise ValueError(
                    "searchsorted requires array to be sorted, which is impossible "
                    "with NAs present."
                )
            # 如果 value 是 ExtensionArray 类型，将其转换为 object 类型，避免在基类的 searchsorted 中发生不必要的类型转换，提升性能。
            if isinstance(value, ExtensionArray):
                value = value.astype(object)
            # 调用底层的 _data 的 searchsorted 方法进行搜索，并返回结果。
            # side 和 sorter 是传递给 searchsorted 的参数。
            return self._data.searchsorted(value, side=side, sorter=sorter)
    
        # 根据 ExtensionArray.factorize 的文档注释
        @doc(ExtensionArray.factorize)
        def factorize(
            self,
            use_na_sentinel: bool = True,
        ) -> tuple[np.ndarray, ExtensionArray]:
            # 获取数据数组和掩码数组
            arr = self._data
            mask = self._mask
    
            # 使用一个标志值（sentinel）来表示缺失值；根据需要重编码并在 uniques 中添加 NA 值。
            codes, uniques = factorize_array(arr, use_na_sentinel=True, mask=mask)
    
            # 检查 factorize_array 是否正确保持了数据类型。
            assert uniques.dtype == self.dtype.numpy_dtype, (uniques.dtype, self.dtype)
    
            # 检查是否有缺失值。
            has_na = mask.any()
            
            # 根据 use_na_sentinel 或者数据中是否有缺失值来确定返回的 uniques 数组的大小。
            if use_na_sentinel or not has_na:
                size = len(uniques)
            else:
                # 为 NA 值腾出位置。
                size = len(uniques) + 1
            
            # 创建一个与 uniques 大小相同的布尔类型数组 uniques_mask。
            uniques_mask = np.zeros(size, dtype=bool)
            
            # 如果不使用 na sentinel 且数据中存在缺失值，则需要处理 NA 值的索引和代码。
            if not use_na_sentinel and has_na:
                # 找到第一个 NA 值的索引。
                na_index = mask.argmax()
                # 根据 NA 值的索引确定 NA 的代码。
                if na_index == 0:
                    na_code = np.intp(0)
                else:
                    na_code = codes[:na_index].max() + 1
                # 调整大于等于 na_code 的代码值。
                codes[codes >= na_code] += 1
                codes[codes == -1] = na_code
                # 在 uniques 中插入 NA 值的占位符，实际上不会用到该值，因为 uniques_mask 中的相应位置已设置为 True。
                uniques = np.insert(uniques, na_code, 0)
                uniques_mask[na_code] = True
            
            # 使用 _simple_new 方法创建一个新的 ExtensionArray，包含更新后的 uniques 和 uniques_mask。
            uniques_ea = self._simple_new(uniques, uniques_mask)
    
            # 返回编码数组 codes 和新的 ExtensionArray uniques_ea。
            return codes, uniques_ea
    
        # 根据 ExtensionArray._values_for_argsort 的文档注释
        @doc(ExtensionArray._values_for_argsort)
        def _values_for_argsort(self) -> np.ndarray:
            # 直接返回底层数据数组 _data。
            return self._data
    # 返回一个包含每个唯一值计数的 Series

    # 导入需要的模块和类
    from pandas import (
        Index,
        Series,
    )
    from pandas.arrays import IntegerArray

    # 调用算法计算数组中每个值的计数
    keys, value_counts, na_counter = algos.value_counts_arraylike(
        self._data, dropna=dropna, mask=self._mask
    )
    # 创建一个布尔数组作为索引的掩码
    mask_index = np.zeros((len(value_counts),), dtype=np.bool_)
    mask = mask_index.copy()

    # 如果存在缺失值，则将最后一个索引设置为 True
    if na_counter > 0:
        mask_index[-1] = True

    # 创建一个整数数组对象
    arr = IntegerArray(value_counts, mask)
    # 创建索引对象
    index = Index(
        self.dtype.construct_array_type()(
            keys,  # type: ignore[arg-type]
            mask_index,
        )
    )
    # 返回一个 Series 对象
    return Series(arr, index=index, name="count", copy=False)

    # 返回众数

    # 如果 dropna 为 True，则计算众数并创建相应的掩码
    if dropna:
        result = mode(self._data, dropna=dropna, mask=self._mask)
        res_mask = np.zeros(result.shape, dtype=np.bool_)
    # 否则计算众数和掩码
    else:
        result, res_mask = mode(self._data, dropna=dropna, mask=self._mask)
    # 创建一个新对象并返回排序后的结果
    result = type(self)(result, res_mask)  # type: ignore[arg-type]
    return result[result.argsort()]

    # 检查两个 ExtensionArray 是否相等

    # 如果两个对象的类型不同，则返回 False
    if type(self) != type(other):
        return False
    # 如果两个对象的数据类型不同，则返回 False
    if other.dtype != self.dtype:
        return False

    # 如果两个对象的掩码不相等，则返回 False
    if not np.array_equal(self._mask, other._mask):
        return False

    # 获取两个对象的数据并比较是否等价
    left = self._data[~self._mask]
    right = other._data[~other._mask]
    return array_equivalent(left, right, strict_nan=True, dtype_equal=True)

    # 计算分位数

    def _quantile(
        self, qs: npt.NDArray[np.float64], interpolation: str
    ) -> BaseMaskedArray:
        """
        Dispatch to quantile_with_mask, needed because we do not have
        _from_factorized.

        Notes
        -----
        We assume that all impacted cases are 1D-only.
        """
        # 调用 quantile_with_mask 函数进行分发，因为我们没有 _from_factorized 方法
        res = quantile_with_mask(
            self._data,
            mask=self._mask,
            # TODO(GH#40932): na_value_for_dtype(self.dtype.numpy_dtype)
            #  instead of np.nan
            # 使用 na_value_for_dtype(self.dtype.numpy_dtype) 替代 np.nan
            fill_value=np.nan,
            qs=qs,
            interpolation=interpolation,
        )

        if self._hasna:
            # Our result mask is all-False unless we are all-NA, in which
            #  case it is all-True.
            if self.ndim == 2:
                # I think this should be out_mask=self.isna().all(axis=1)
                #  but am holding off until we have tests
                raise NotImplementedError
            if self.isna().all():
                # 如果所有值都是 NA，则结果的掩码是全真
                out_mask = np.ones(res.shape, dtype=bool)

                if is_integer_dtype(self.dtype):
                    # We try to maintain int dtype if possible for not all-na case
                    # as well
                    # 如果数据类型是整数，则尝试保持整数类型以便处理非全 NA 的情况
                    res = np.zeros(res.shape, dtype=self.dtype.numpy_dtype)
            else:
                # 如果不是所有值都是 NA，则结果的掩码是全假
                out_mask = np.zeros(res.shape, dtype=bool)
        else:
            # 如果没有 NA 值，则结果的掩码是全假
            out_mask = np.zeros(res.shape, dtype=bool)
        return self._maybe_mask_result(res, mask=out_mask)

    # ------------------------------------------------------------------
    # Reductions

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        if name in {"any", "all", "min", "max", "sum", "prod", "mean", "var", "std"}:
            # 如果 name 是预定义的聚合函数名称，则调用相应的函数
            result = getattr(self, name)(skipna=skipna, **kwargs)
        else:
            # 否则，调用 nanops 模块中相应名称的函数进行操作
            data = self._data
            mask = self._mask
            op = getattr(nanops, f"nan{name}")
            axis = kwargs.pop("axis", None)
            result = op(data, axis=axis, skipna=skipna, mask=mask, **kwargs)

        if keepdims:
            if isna(result):
                # 如果结果为 NA，则返回包装后的 NA 结果
                return self._wrap_na_result(name=name, axis=0, mask_size=(1,))
            else:
                # 否则，将结果重新形状为长度为 1 的数组，并生成全假掩码
                result = result.reshape(1)
                mask = np.zeros(1, dtype=bool)
                return self._maybe_mask_result(result, mask)

        if isna(result):
            # 如果结果为 NA，则返回 libmissing.NA
            return libmissing.NA
        else:
            return result

    def _wrap_reduction_result(self, name: str, result, *, skipna, axis):
        if isinstance(result, np.ndarray):
            if skipna:
                # 如果 skipna=True，则保留所有-NA的行/列的掩码
                mask = self._mask.all(axis=axis)
            else:
                # 如果 skipna=False，则保留任意-NA的行/列的掩码
                mask = self._mask.any(axis=axis)

            return self._maybe_mask_result(result, mask)
        return result
    # 封装不可用结果的掩码，默认为全是 True 的布尔数组
    mask = np.ones(mask_size, dtype=bool)

    # 根据数据类型确定浮点数类型，根据 self.dtype 的值选择 "float32" 或 "float64"
    float_dtyp = "float32" if self.dtype == "Float32" else "float64"
    
    # 根据聚合函数名确定结果的 NumPy 数据类型
    if name in ["mean", "median", "var", "std", "skew", "kurt"]:
        np_dtype = float_dtyp
    elif name in ["min", "max"] or self.dtype.itemsize == 8:
        np_dtype = self.dtype.numpy_dtype.name
    else:
        # 检查是否运行在 Windows 平台或者是 32 位系统
        is_windows_or_32bit = is_platform_windows() or not IS64
        # 根据平台选择整数类型和无符号整数类型
        int_dtyp = "int32" if is_windows_or_32bit else "int64"
        uint_dtyp = "uint32" if is_windows_or_32bit else "uint64"
        # 根据数据类型的种类选择对应的 NumPy 数据类型
        np_dtype = {"b": int_dtyp, "i": int_dtyp, "u": uint_dtyp, "f": float_dtyp}[self.dtype.kind]

    # 创建一个包含单个元素的 NumPy 数组，类型为 np_dtype
    value = np.array([1], dtype=np_dtype)
    
    # 调用 _maybe_mask_result 方法，处理可能存在的结果掩码
    return self._maybe_mask_result(value, mask=mask)

def _wrap_min_count_reduction_result(
    self, name: str, result, *, skipna, min_count, axis
):
    # 如果 min_count 为 0 并且结果是 NumPy 数组，则返回结果并使用全零掩码
    if min_count == 0 and isinstance(result, np.ndarray):
        return self._maybe_mask_result(result, np.zeros(result.shape, dtype=bool))
    
    # 否则，调用 _wrap_reduction_result 方法，封装聚合结果
    return self._wrap_reduction_result(name, result, skipna=skipna, axis=axis)

def sum(
    self,
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = 0,
    **kwargs,
):
    # 使用 nv.validate_sum 方法验证参数
    nv.validate_sum((), kwargs)

    # 调用 masked_reductions.sum 函数，对数据进行求和操作
    result = masked_reductions.sum(
        self._data,
        self._mask,
        skipna=skipna,
        min_count=min_count,
        axis=axis,
    )
    # 调用 _wrap_min_count_reduction_result 方法，封装带有最小计数的聚合结果
    return self._wrap_min_count_reduction_result(
        "sum", result, skipna=skipna, min_count=min_count, axis=axis
    )

def prod(
    self,
    *,
    skipna: bool = True,
    min_count: int = 0,
    axis: AxisInt | None = 0,
    **kwargs,
):
    # 使用 nv.validate_prod 方法验证参数
    nv.validate_prod((), kwargs)

    # 调用 masked_reductions.prod 函数，对数据进行乘积操作
    result = masked_reductions.prod(
        self._data,
        self._mask,
        skipna=skipna,
        min_count=min_count,
        axis=axis,
    )
    # 调用 _wrap_min_count_reduction_result 方法，封装带有最小计数的聚合结果
    return self._wrap_min_count_reduction_result(
        "prod", result, skipna=skipna, min_count=min_count, axis=axis
    )

def mean(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
    # 使用 nv.validate_mean 方法验证参数
    nv.validate_mean((), kwargs)
    
    # 调用 masked_reductions.mean 函数，对数据进行均值计算
    result = masked_reductions.mean(
        self._data,
        self._mask,
        skipna=skipna,
        axis=axis,
    )
    # 调用 _wrap_reduction_result 方法，封装均值计算的结果
    return self._wrap_reduction_result("mean", result, skipna=skipna, axis=axis)

def var(
    self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs
):
    # 使用 nv.validate_stat_ddof_func 方法验证参数，指定函数名为 "var"
    nv.validate_stat_ddof_func((), kwargs, fname="var")
    
    # 调用 masked_reductions.var 函数，对数据进行方差计算
    result = masked_reductions.var(
        self._data,
        self._mask,
        skipna=skipna,
        axis=axis,
        ddof=ddof,
    )
    # 调用 _wrap_reduction_result 方法，封装方差计算的结果
    return self._wrap_reduction_result("var", result, skipna=skipna, axis=axis)
    # 定义一个方法 `std`，用于计算标准差
    def std(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, ddof: int = 1, **kwargs
    ):
        # 使用 nv 模块验证标准差函数的自由度设置
        nv.validate_stat_ddof_func((), kwargs, fname="std")
        # 调用 masked_reductions 模块的 std 函数计算标准差
        result = masked_reductions.std(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
            ddof=ddof,
        )
        # 将计算结果进行封装并返回，包括跳过 NA 值和指定的轴
        return self._wrap_reduction_result("std", result, skipna=skipna, axis=axis)

    # 定义一个方法 `min`，用于计算最小值
    def min(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        # 使用 nv 模块验证最小值函数的参数
        nv.validate_min((), kwargs)
        # 调用 masked_reductions 模块的 min 函数计算最小值
        result = masked_reductions.min(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        # 将计算结果进行封装并返回，包括跳过 NA 值和指定的轴
        return self._wrap_reduction_result("min", result, skipna=skipna, axis=axis)

    # 定义一个方法 `max`，用于计算最大值
    def max(self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs):
        # 使用 nv 模块验证最大值函数的参数
        nv.validate_max((), kwargs)
        # 调用 masked_reductions 模块的 max 函数计算最大值
        result = masked_reductions.max(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        # 将计算结果进行封装并返回，包括跳过 NA 值和指定的轴
        return self._wrap_reduction_result("max", result, skipna=skipna, axis=axis)

    # 定义一个方法 `map`，用于应用映射函数到数组中的每个元素
    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        # 将数据转换为 NumPy 数组，并调用 map_array 函数进行映射操作
        return map_array(self.to_numpy(), mapper, na_action=na_action)

    # 重载方法 `any`，用于计算是否存在任意真值
    @overload
    def any(
        self, *, skipna: Literal[True] = ..., axis: AxisInt | None = ..., **kwargs
    ) -> np.bool_: ...
    
    # 重载方法 `any`，用于计算是否存在任意真值，指定了跳过 NA 值的参数
    @overload
    def any(
        self, *, skipna: bool, axis: AxisInt | None = ..., **kwargs
    ) -> np.bool_ | NAType: ...

    # 定义方法 `any`，用于计算是否存在任意真值
    def any(
        self, *, skipna: bool = True, axis: AxisInt | None = 0, **kwargs
    ) -> np.bool_ | NAType:
        # 调用 masked_reductions 模块的 any 函数计算是否存在任意真值
        result = masked_reductions.any(
            self._data,
            self._mask,
            skipna=skipna,
            axis=axis,
        )
        # 将计算结果进行封装并返回，包括跳过 NA 值和指定的轴
        return self._wrap_reduction_result("any", result, skipna=skipna, axis=axis)
    ) -> np.bool_ | NAType:
        """
        Return whether any element is truthy.

        Returns False unless there is at least one element that is truthy.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be False, as for an empty array.
            If `skipna` is False, the result will still be True if there is
            at least one element that is truthy, otherwise NA will be returned
            if there are NA's present.
        axis : int, optional, default 0
            This parameter specifies the axis along which the logical operation
            is applied. By default, it operates along the first dimension (axis=0).
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`
            Returns a boolean indicating if any element in the array-like object
            is true. If `skipna=False` and NA values are present, it may return
            NA instead.

        See Also
        --------
        numpy.any : Numpy version of this method.
        BaseMaskedArray.all : Return whether all elements are truthy.

        Examples
        --------
        The result indicates whether any element is truthy (and by default
        skips NAs):

        >>> pd.array([True, False, True]).any()
        True
        >>> pd.array([True, False, pd.NA]).any()
        True
        >>> pd.array([False, False, pd.NA]).any()
        False
        >>> pd.array([], dtype="boolean").any()
        False
        >>> pd.array([pd.NA], dtype="boolean").any()
        False
        >>> pd.array([pd.NA], dtype="Float64").any()
        False

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, False, pd.NA]).any(skipna=False)
        True
        >>> pd.array([1, 0, pd.NA]).any(skipna=False)
        True
        >>> pd.array([False, False, pd.NA]).any(skipna=False)
        <NA>
        >>> pd.array([0, 0, pd.NA]).any(skipna=False)
        <NA>
        """
        # Validate that no positional arguments are passed using nv.validate_any
        nv.validate_any((), kwargs)

        # Create a copy of the data, including masked values
        values = self._data.copy()

        # Replace masked values with the corresponding falsey value of the dtype
        np.putmask(values, self._mask, self.dtype._falsey_value)

        # Check if any element in the modified array is truthy
        result = values.any()

        # If skipna is True, return the result directly
        if skipna:
            return result
        else:
            # If skipna is False, additional checks are performed
            # If result is True or the length of the array is zero or no mask elements are true,
            # return the result; otherwise, return the NA value of the dtype
            if result or len(self) == 0 or not self._mask.any():
                return result
            else:
                return self.dtype.na_value
    ) -> np.bool_ | NAType:
        """
        Return whether all elements are truthy.

        Returns True unless there is at least one element that is falsey.
        By default, NAs are skipped. If ``skipna=False`` is specified and
        missing values are present, similar :ref:`Kleene logic <boolean.kleene>`
        is used as for logical operations.

        .. versionchanged:: 1.4.0

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values. If the entire array is NA and `skipna` is
            True, then the result will be True, as for an empty array.
            If `skipna` is False, the result will still be False if there is
            at least one element that is falsey, otherwise NA will be returned
            if there are NA's present.
        axis : int, optional, default 0
            The axis along which the operation is performed.
        **kwargs : any, default None
            Additional keywords have no effect but might be accepted for
            compatibility with NumPy.

        Returns
        -------
        bool or :attr:`pandas.NA`
            Returns True if all elements are truthy, or :attr:`pandas.NA`
            if there are NA values present and `skipna=False`.

        See Also
        --------
        numpy.all : Numpy version of this method.
        BooleanArray.any : Return whether any element is truthy.

        Examples
        --------
        The result indicates whether all elements are truthy (and by default
        skips NAs):

        >>> pd.array([True, True, pd.NA]).all()
        True
        >>> pd.array([1, 1, pd.NA]).all()
        True
        >>> pd.array([True, False, pd.NA]).all()
        False
        >>> pd.array([], dtype="boolean").all()
        True
        >>> pd.array([pd.NA], dtype="boolean").all()
        True
        >>> pd.array([pd.NA], dtype="Float64").all()
        True

        With ``skipna=False``, the result can be NA if this is logically
        required (whether ``pd.NA`` is True or False influences the result):

        >>> pd.array([True, True, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([1, 1, pd.NA]).all(skipna=False)
        <NA>
        >>> pd.array([True, False, pd.NA]).all(skipna=False)
        False
        >>> pd.array([1, 0, pd.NA]).all(skipna=False)
        False
        """
        nv.validate_all((), kwargs)  # 调用nv模块的validate_all函数，验证空元组和关键字参数kwargs

        values = self._data.copy()  # 复制self对象的数据到values变量
        np.putmask(values, self._mask, self.dtype._truthy_value)  # 使用self对象的掩码属性将数据中的值用dtype的_truthy_value替换
        result = values.all(axis=axis)  # 对values进行逻辑与操作，沿指定轴axis进行

        if skipna:
            return result  # 如果skipna为True，直接返回逻辑与操作的结果
        else:
            if not result or len(self) == 0 or not self._mask.any():
                return result  # 如果逻辑与操作的结果为假，self对象长度为0，或者掩码属性没有任何真值，则返回结果
            else:
                return self.dtype.na_value  # 否则返回dtype的na_value值
    ) -> FloatingArray:
        """
        See NDFrame.interpolate.__doc__.
        """
        # NB: we return type(self) even if copy=False
        if self.dtype.kind == "f":
            # 如果数据类型是浮点数
            if copy:
                # 如果需要复制数据
                data = self._data.copy()
                mask = self._mask.copy()
            else:
                # 如果不需要复制数据
                data = self._data
                mask = self._mask
        elif self.dtype.kind in "iu":
            # 如果数据类型是无符号整数或有符号整数
            copy = True
            data = self._data.astype("f8")  # 转换数据类型为 float64
            mask = self._mask.copy()
        else:
            # 如果数据类型不在已知的类型范围内，则抛出未实现错误
            raise NotImplementedError(
                f"interpolate is not implemented for dtype={self.dtype}"
            )

        # 调用 missing.interpolate_2d_inplace 方法对数据进行插值处理
        missing.interpolate_2d_inplace(
            data,
            method=method,
            axis=0,
            index=index,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            mask=mask,
            **kwargs,
        )
        # 如果不需要复制数据，则返回当前对象
        if not copy:
            return self  # type: ignore[return-value]
        # 如果数据类型是浮点数，返回一个新的 FloatingArray 对象
        if self.dtype.kind == "f":
            return type(self)._simple_new(data, mask)  # type: ignore[return-value]
        else:
            # 如果数据类型不是浮点数，则返回一个新的 FloatingArray 对象
            from pandas.core.arrays import FloatingArray

            return FloatingArray._simple_new(data, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
    ) -> BaseMaskedArray:
        # 获取数据和掩码
        data = self._data
        mask = self._mask

        # 使用 getattr 调用 masked_accumulations 模块中的指定操作
        op = getattr(masked_accumulations, name)
        # 执行指定操作，更新数据和掩码
        data, mask = op(data, mask, skipna=skipna, **kwargs)

        # 返回一个新的对象，包含更新后的数据和掩码
        return self._simple_new(data, mask)

    # ------------------------------------------------------------------
    # GroupBy Methods

    def _groupby_op(
        self,
        *,
        how: str,
        has_dropped_na: bool,
        min_count: int,
        ngroups: int,
        ids: npt.NDArray[np.intp],
        **kwargs,
        from pandas.core.groupby.ops import WrappedCythonOp
        # 从 pandas 库导入 WrappedCythonOp 类

        kind = WrappedCythonOp.get_kind_from_how(how)
        # 使用 WrappedCythonOp 类的方法获取操作类型 kind

        op = WrappedCythonOp(how=how, kind=kind, has_dropped_na=has_dropped_na)
        # 使用 WrappedCythonOp 类创建操作对象 op，传入操作类型、kind 和是否已经删除了 NA 值的标志

        # libgroupby functions are responsible for NOT altering mask
        # libgroupby 函数负责不修改 mask

        mask = self._mask
        # 获取当前对象的 _mask 属性作为 mask

        if op.kind != "aggregate":
            # 如果操作的类型不是 "aggregate"
            result_mask = mask.copy()
            # 则复制 mask 到 result_mask
        else:
            # 否则
            result_mask = np.zeros(ngroups, dtype=bool)
            # 创建一个与 ngroups 大小相同的布尔类型的全零数组作为 result_mask

        if how == "rank" and kwargs.get("na_option") in ["top", "bottom"]:
            # 如果操作类型是 "rank" 并且 na_option 参数是 "top" 或 "bottom"
            result_mask[:] = False
            # 将 result_mask 全部设为 False

        res_values = op._cython_op_ndim_compat(
            self._data,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )
        # 使用 op 对象的 _cython_op_ndim_compat 方法进行操作，返回操作结果 res_values

        if op.how == "ohlc":
            # 如果操作类型是 "ohlc"
            arity = op._cython_arity.get(op.how, 1)
            # 获取操作的 arity（通常是返回值的数量）
            result_mask = np.tile(result_mask, (arity, 1)).T
            # 对 result_mask 进行扩展以适应返回值的数量

        if op.how in ["idxmin", "idxmax"]:
            # 如果操作类型是 "idxmin" 或 "idxmax"
            # 结果值是索引，直接返回 res_values
            return res_values
        else:
            # 否则
            # res_values 应该已经具有正确的数据类型，我们只需要将其包装成 MaskedArray
            return self._maybe_mask_result(res_values, result_mask)
def transpose_homogeneous_masked_arrays(
    masked_arrays: Sequence[BaseMaskedArray],
) -> list[BaseMaskedArray]:
    """Transpose masked arrays in a list, but faster.

    Input should be a list of 1-dim masked arrays of equal length and all have the
    same dtype. The caller is responsible for ensuring validity of input data.
    """
    # 将输入的 masked_arrays 转换为列表，确保可以对其进行迭代和处理
    masked_arrays = list(masked_arrays)
    # 获取第一个 masked array 的数据类型
    dtype = masked_arrays[0].dtype

    # 收集每个 masked array 的数据部分并按照列进行堆叠，以便进行快速转置
    values = [arr._data.reshape(1, -1) for arr in masked_arrays]
    # 使用 np.concatenate 函数将数据堆叠在一起，构成转置后的值矩阵
    transposed_values = np.concatenate(
        values,
        axis=0,
        out=np.empty(
            (len(masked_arrays), len(masked_arrays[0])),
            order="F",  # 使用 Fortran 风格的内存布局，有助于某些操作的性能
            dtype=dtype.numpy_dtype,  # 转置后的数据类型与原始数据类型相同
        ),
    )

    # 收集每个 masked array 的掩码部分并按照列进行堆叠
    masks = [arr._mask.reshape(1, -1) for arr in masked_arrays]
    # 使用 np.concatenate 函数将掩码堆叠在一起，构成转置后的掩码矩阵
    transposed_masks = np.concatenate(
        masks, axis=0, out=np.empty_like(transposed_values, dtype=bool)
    )

    # 根据数据类型创建转置后的 masked array 列表
    arr_type = dtype.construct_array_type()
    transposed_arrays: list[BaseMaskedArray] = []
    # 遍历转置后的值矩阵的列，并创建相应的 masked array 对象
    for i in range(transposed_values.shape[1]):
        transposed_arr = arr_type(transposed_values[:, i], mask=transposed_masks[:, i])
        transposed_arrays.append(transposed_arr)

    # 返回转置后的 masked array 列表
    return transposed_arrays
```