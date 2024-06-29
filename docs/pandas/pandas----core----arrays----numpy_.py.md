# `D:\src\scipysrc\pandas\pandas\core\arrays\numpy_.py`

```
# 引入将来的注解特性，允许类型在定义之前使用
from __future__ import annotations

# 引入类型检查相关的模块和类型
from typing import (
    TYPE_CHECKING,  # 类型检查开关
    Literal,        # 字面类型
)

# 引入第三方库 numpy，并使用别名 np
import numpy as np

# 引入 pandas 内部的 C 库
from pandas._libs import lib

# 引入 pandas 时间序列相关的工具
from pandas._libs.tslibs import is_supported_dtype

# 引入 pandas 兼容的 numpy 函数
from pandas.compat.numpy import function as nv

# 引入 pandas 数据类型相关的模块
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import isna

# 引入 pandas 核心模块及其子模块
from pandas.core import (
    arraylike,  # 数组样式相关
    missing,    # 缺失值处理
    nanops,     # NaN 值操作
    ops,        # 操作相关
)

# 引入 pandas 核心数组样式相关的模块
from pandas.core.arraylike import OpsMixin

# 引入 pandas 核心数组扩展数组的混合类
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray

# 引入 pandas 核心构造相关的模块
from pandas.core.construction import ensure_wrapped_if_datetimelike

# 引入 pandas 核心字符串对象数组相关的混合类
from pandas.core.strings.object_array import ObjectStringArrayMixin

# 如果类型检查开关打开，则引入进一步的类型
if TYPE_CHECKING:
    from pandas._typing import (
        AxisInt,           # 轴的整数类型
        Dtype,             # 数据类型
        FillnaOptions,     # 填充选项
        InterpolateOptions,  # 插值选项
        NpDtype,           # NumPy 数据类型
        Scalar,            # 标量
        Self,              # 自身
        npt,               # NumPy 类型
    )

    from pandas import Index  # 索引类型

# 定义一个类 NumpyExtensionArray，继承自 OpsMixin、NDArrayBackedExtensionArray 和 ObjectStringArrayMixin
# 忽略类型检查相关的错误提示
class NumpyExtensionArray(  # type: ignore[misc]
    OpsMixin,
    NDArrayBackedExtensionArray,
    ObjectStringArrayMixin,
):
    """
    A pandas ExtensionArray for NumPy data.

    This is mostly for internal compatibility, and is not especially
    useful on its own.

    Parameters
    ----------
    values : ndarray
        The NumPy ndarray to wrap. Must be 1-dimensional.
    copy : bool, default False
        Whether to copy `values`.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Examples
    --------
    >>> pd.arrays.NumpyExtensionArray(np.array([0, 1, 2, 3]))
    <NumpyExtensionArray>
    [0, 1, 2, 3]
    Length: 4, dtype: int64
    """

    # 指定这个类的类型标识，表示是一个 NumPy 扩展数组
    _typ = "npy_extension"
    # 指定此类的数组优先级
    __array_priority__ = 1000
    # 定义类的成员变量 _ndarray 为 NumPy 数组
    _ndarray: np.ndarray
    # 定义类的成员变量 _dtype 为 NumpyEADtype 类型
    _dtype: NumpyEADtype
    # 定义类的内部填充值为 NaN
    _internal_fill_value = np.nan

    # ------------------------------------------------------------------------
    # Constructors

    # 定义构造函数，接受 values 参数作为 np.ndarray 或 NumpyExtensionArray，copy 参数默认为 False
    def __init__(
        self, values: np.ndarray | NumpyExtensionArray, copy: bool = False
    ) -> None:
        # 如果传入的 values 是当前类的实例，则获取其内部的 _ndarray 属性作为 values
        if isinstance(values, type(self)):
            values = values._ndarray
        # 检查 values 是否为 NumPy 数组，如果不是则抛出 ValueError 异常
        if not isinstance(values, np.ndarray):
            raise ValueError(
                f"'values' must be a NumPy array, not {type(values).__name__}"
            )

        # 检查 values 的维度是否为 0，NumPyExtensionArray 必须是一维数组
        if values.ndim == 0:
            # 抛出异常，说明 NumPyExtensionArray 必须是一维的
            raise ValueError("NumpyExtensionArray must be 1-dimensional.")

        # 如果 copy 标志为 True，则复制 values 数组
        if copy:
            values = values.copy()

        # 使用 values 的 dtype 创建一个 NumpyEADtype 对象，并调用父类的初始化方法
        dtype = NumpyEADtype(values.dtype)
        super().__init__(values, dtype)

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> NumpyExtensionArray:
        # 如果 dtype 是 NumpyEADtype 的实例，则获取其内部的 _dtype 属性
        if isinstance(dtype, NumpyEADtype):
            dtype = dtype._dtype

        # 将 scalars 转换为 NumPy 数组，使用指定的 dtype，忽略类型检查错误
        result = np.asarray(scalars, dtype=dtype)  # type: ignore[arg-type]
        # 如果 result 的维度大于 1，且 scalars 没有 dtype 属性，并且 dtype 是 None 或者 object 类型
        if (
            result.ndim > 1
            and not hasattr(scalars, "dtype")
            and (dtype is None or dtype == object)
        ):
            # 例如，处理元组列表等情况，将其构造为一维对象数组
            result = construct_1d_object_array_from_listlike(scalars)

        # 如果 copy 标志为 True，并且 result 和 scalars 是同一个对象，则复制 result
        if copy and result is scalars:
            result = result.copy()
        # 返回一个新的 NumpyExtensionArray 对象，使用 result 初始化
        return cls(result)

    def _from_backing_data(self, arr: np.ndarray) -> NumpyExtensionArray:
        # 使用当前对象的类型，以 arr 作为 backing data，创建一个新的 NumpyExtensionArray 对象
        return type(self)(arr)

    # ------------------------------------------------------------------------
    # Data

    @property
    def dtype(self) -> NumpyEADtype:
        # 返回当前对象的数据类型 _dtype
        return self._dtype

    # ------------------------------------------------------------------------
    # NumPy Array Interface

    def __array__(
        self, dtype: NpDtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        # 将当前对象的 _ndarray 转换为 NumPy 数组，并使用指定的 dtype
        return np.asarray(self._ndarray, dtype=dtype)
    # 实现了__array_ufunc__方法，用于处理NumPy的通用函数（ufunc）与自定义类型的交互
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        # 从arraylike模块中调用可能分派给dunder操作的ufunc处理
        out = kwargs.get("out", ())

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        # 如果分派成功，则返回结果
        if result is not NotImplemented:
            return result

        # 如果kwargs中包含"out"参数，则调用dispatch_ufunc_with_out处理
        if "out" in kwargs:
            # 例如：test_ufunc_unary
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        # 如果method为"reduce"，则调用dispatch_reduction_ufunc处理
        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            # 如果成功处理，则返回结果
            if result is not NotImplemented:
                # 例如：tests.series.test_ufunc.TestNumpyReductions
                return result

        # 将非NumpyExtensionArray的输入转换为其_ndarray属性
        inputs = tuple(
            x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in inputs
        )
        # 如果存在输出参数out，则将其转换为对应的_ndarray属性
        if out:
            kwargs["out"] = tuple(
                x._ndarray if isinstance(x, NumpyExtensionArray) else x for x in out
            )
        # 调用ufunc对象的method方法处理inputs和kwargs，获取结果
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # 如果ufunc有多个输出，则将结果重新封装成当前类型的实例
        if ufunc.nout > 1:
            # 多个返回值；重新包装数组类似的结果
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # 没有返回值
            return None
        elif method == "reduce":
            # 如果结果是np.ndarray类型，则返回当前类型的实例
            if isinstance(result, np.ndarray):
                # 例如：test_np_reduce_2d
                return type(self)(result)

            # 例如：test_np_max_nested_tuples
            return result
        else:
            # 单个返回值；重新包装数组类似的结果
            return type(self)(result)

    # ------------------------------------------------------------------------
    # Pandas ExtensionArray 接口

    # 将当前数组转换为指定dtype类型的数组
    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        # 如果dtype与当前数组的dtype相同，则返回复制或原数组
        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self

        # 使用astype_array函数将当前_ndarray转换为指定dtype类型的数组
        result = astype_array(self._ndarray, dtype=dtype, copy=copy)
        return result

    # 返回一个布尔数组，指示当前数组中的每个元素是否为缺失值
    def isna(self) -> np.ndarray:
        return isna(self._ndarray)

    # 验证并返回用于填充的标量值，如果fill_value为None，则使用dtype的缺失值
    def _validate_scalar(self, fill_value):
        if fill_value is None:
            # 主要用于子类
            fill_value = self.dtype.na_value
        return fill_value

    # 返回一个包含当前_ndarray和适合于因子化的填充值的元组
    def _values_for_factorize(self) -> tuple[np.ndarray, float | None]:
        if self.dtype.kind in "iub":
            fv = None
        else:
            fv = np.nan
        return self._ndarray, fv

    # Base EA类（以及所有其他EA类）不具有limit_area关键字
    # ------------------------------------------------------------------------
    # Reductions

    def any(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Self:
        """
        Compute if at least one element is True along a specified axis.
        """
        # Validate and prepare output parameters
        nv.validate_any((), {"out": out, "keepdims": keepdims})
        
        # Compute any() operation across the ndarray, handling NaN values
        result = nanops.nanany(self._ndarray, axis=axis, skipna=skipna)
        
        # Wrap and return the reduction result with appropriate dimensions
        return self._wrap_reduction_result(axis, result)

    def all(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ) -> Self:
        """
        Compute if all elements are True along a specified axis.
        """
        # Validate and prepare output parameters
        nv.validate_all((), {"out": out, "keepdims": keepdims})
        
        # Compute all() operation across the ndarray, handling NaN values
        result = nanops.nanall(self._ndarray, axis=axis, skipna=skipna)
        
        # Wrap and return the reduction result with appropriate dimensions
        return self._wrap_reduction_result(axis, result)

    def min(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        """
        Compute the minimum value along a specified axis.
        """
        # Validate and prepare parameters for minimum calculation
        nv.validate_min((), kwargs)
        
        # Compute minimum operation across the ndarray, ignoring NaNs
        result = nanops.nanmin(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        
        # Wrap and return the reduction result with appropriate dimensions
        return self._wrap_reduction_result(axis, result)

    def max(
        self, *, axis: AxisInt | None = None, skipna: bool = True, **kwargs
    ) -> Scalar:
        """
        Compute the maximum value along a specified axis.
        """
        # Validate and prepare parameters for maximum calculation
        nv.validate_max((), kwargs)
        
        # Compute maximum operation across the ndarray, ignoring NaNs
        result = nanops.nanmax(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        
        # Wrap and return the reduction result with appropriate dimensions
        return self._wrap_reduction_result(axis, result)
    ) -> Scalar:
        # 使用验证器验证参数
        nv.validate_max((), kwargs)
        # 计算沿指定轴的最大值，忽略 NaN 值，并返回结果
        result = nanops.nanmax(
            values=self._ndarray, axis=axis, mask=self.isna(), skipna=skipna
        )
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)

    def sum(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        # 使用验证器验证参数
        nv.validate_sum((), kwargs)
        # 计算沿指定轴的和，忽略 NaN 值，并返回结果
        result = nanops.nansum(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)

    def prod(
        self,
        *,
        axis: AxisInt | None = None,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs,
    ) -> Scalar:
        # 使用验证器验证参数
        nv.validate_prod((), kwargs)
        # 计算沿指定轴的乘积，忽略 NaN 值，并返回结果
        result = nanops.nanprod(
            self._ndarray, axis=axis, skipna=skipna, min_count=min_count
        )
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)

    def mean(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # 使用验证器验证参数
        nv.validate_mean((), {"dtype": dtype, "out": out, "keepdims": keepdims})
        # 计算沿指定轴的均值，忽略 NaN 值，并返回结果
        result = nanops.nanmean(self._ndarray, axis=axis, skipna=skipna)
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)

    def median(
        self,
        *,
        axis: AxisInt | None = None,
        out=None,
        overwrite_input: bool = False,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # 使用验证器验证参数
        nv.validate_median(
            (), {"out": out, "overwrite_input": overwrite_input, "keepdims": keepdims}
        )
        # 计算沿指定轴的中位数，忽略 NaN 值，并返回结果
        result = nanops.nanmedian(self._ndarray, axis=axis, skipna=skipna)
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)

    def std(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # 使用验证器验证参数
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="std"
        )
        # 计算沿指定轴的标准差，忽略 NaN 值，并返回结果
        result = nanops.nanstd(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)

    def var(
        self,
        *,
        axis: AxisInt | None = None,
        dtype: NpDtype | None = None,
        out=None,
        ddof: int = 1,
        keepdims: bool = False,
        skipna: bool = True,
    ):
        # 使用验证器验证参数
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="var"
        )
        # 计算沿指定轴的方差，忽略 NaN 值，并返回结果
        result = nanops.nanvar(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        # 封装并返回减少操作的结果
        return self._wrap_reduction_result(axis, result)
    # 定义一个方法 sem，计算标准误差的方法
    def sem(
        self,
        *,
        axis: AxisInt | None = None,  # 指定计算的轴向，默认为 None
        dtype: NpDtype | None = None,  # 指定返回结果的数据类型，默认为 None
        out=None,  # 输出结果的存储位置，默认为 None
        ddof: int = 1,  # 自由度的修正值，默认为 1
        keepdims: bool = False,  # 是否保持减少维度后的维度，默认为 False
        skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    ):
        # 使用验证函数验证参数有效性
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="sem"
        )
        # 调用 nansem 函数计算标准误差
        result = nanops.nansem(self._ndarray, axis=axis, skipna=skipna, ddof=ddof)
        # 封装计算结果并返回
        return self._wrap_reduction_result(axis, result)

    # 定义一个方法 kurt，计算峰度的方法
    def kurt(
        self,
        *,
        axis: AxisInt | None = None,  # 指定计算的轴向，默认为 None
        dtype: NpDtype | None = None,  # 指定返回结果的数据类型，默认为 None
        out=None,  # 输出结果的存储位置，默认为 None
        keepdims: bool = False,  # 是否保持减少维度后的维度，默认为 False
        skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    ):
        # 使用验证函数验证参数有效性
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="kurt"
        )
        # 调用 nankurt 函数计算峰度
        result = nanops.nankurt(self._ndarray, axis=axis, skipna=skipna)
        # 封装计算结果并返回
        return self._wrap_reduction_result(axis, result)

    # 定义一个方法 skew，计算偏度的方法
    def skew(
        self,
        *,
        axis: AxisInt | None = None,  # 指定计算的轴向，默认为 None
        dtype: NpDtype | None = None,  # 指定返回结果的数据类型，默认为 None
        out=None,  # 输出结果的存储位置，默认为 None
        keepdims: bool = False,  # 是否保持减少维度后的维度，默认为 False
        skipna: bool = True,  # 是否跳过 NaN 值，默认为 True
    ):
        # 使用验证函数验证参数有效性
        nv.validate_stat_ddof_func(
            (), {"dtype": dtype, "out": out, "keepdims": keepdims}, fname="skew"
        )
        # 调用 nanskew 函数计算偏度
        result = nanops.nanskew(self._ndarray, axis=axis, skipna=skipna)
        # 封装计算结果并返回
        return self._wrap_reduction_result(axis, result)

    # ------------------------------------------------------------------------
    # Additional Methods

    # 定义一个方法 to_numpy，将扩展数组转换为 NumPy 数组的方法
    def to_numpy(
        self,
        dtype: npt.DTypeLike | None = None,  # 指定返回结果的数据类型，默认为 None
        copy: bool = False,  # 是否复制数组，默认为 False
        na_value: object = lib.no_default,  # 指定 NaN 值的替换值，默认为 lib.no_default
    ) -> np.ndarray:  # 返回一个 NumPy 数组
        # 检查是否存在 NaN 值，并根据需要替换为 na_value
        mask = self.isna()
        if na_value is not lib.no_default and mask.any():
            result = self._ndarray.copy()
            result[mask] = na_value
        else:
            result = self._ndarray

        # 将结果转换为指定的数据类型
        result = np.asarray(result, dtype=dtype)

        # 如果需要复制数组，则进行复制操作
        if copy and result is self._ndarray:
            result = result.copy()

        # 返回转换后的 NumPy 数组
        return result

    # ------------------------------------------------------------------------
    # Ops

    # 定义一个方法 __invert__，执行按位取反的操作
    def __invert__(self) -> NumpyExtensionArray:  # 返回一个扩展数组
        return type(self)(~self._ndarray)

    # 定义一个方法 __neg__，执行取负数的操作
    def __neg__(self) -> NumpyExtensionArray:  # 返回一个扩展数组
        return type(self)(-self._ndarray)

    # 定义一个方法 __pos__，执行取正数的操作
    def __pos__(self) -> NumpyExtensionArray:  # 返回一个扩展数组
        return type(self)(+self._ndarray)

    # 定义一个方法 __abs__，执行取绝对值的操作
    def __abs__(self) -> NumpyExtensionArray:  # 返回一个扩展数组
        return type(self)(abs(self._ndarray))
    # 定义一个比较方法，用于处理与另一个对象的比较操作
    def _cmp_method(self, other, op):
        # 如果 other 是 NumpyExtensionArray 类型，则使用其底层的 ndarray 进行比较
        if isinstance(other, NumpyExtensionArray):
            other = other._ndarray

        # 将 other 转换为适合进行操作的标量形式
        other = ops.maybe_prepare_scalar_for_op(other, (len(self),))
        
        # 获取特定操作符 op 对应的 Pandas 的数组操作函数
        pd_op = ops.get_array_op(op)
        
        # 确保 other 如果是日期时间类型，进行适当的封装处理
        other = ensure_wrapped_if_datetimelike(other)
        
        # 对 self 的 ndarray 与 other 执行相应的操作
        result = pd_op(self._ndarray, other)

        # 对于 divmod 或 ops.rdivmod 操作，需要特殊处理结果
        if op is divmod or op is ops.rdivmod:
            a, b = result
            if isinstance(a, np.ndarray):
                # 如果 a 是 ndarray 类型，则可能是 ExtensionArray，不需要再次封装
                return self._wrap_ndarray_result(a), self._wrap_ndarray_result(b)
            return a, b

        # 如果结果是 ndarray 类型，则可能是 ExtensionArray，不需要再次封装
        if isinstance(result, np.ndarray):
            return self._wrap_ndarray_result(result)
        
        # 返回处理结果
        return result

    # 将 _cmp_method 方法赋值给 _arith_method，这两个方法功能上是等效的
    _arith_method = _cmp_method

    # 定义一个方法，用于封装 ndarray 类型的结果
    def _wrap_ndarray_result(self, result: np.ndarray):
        # 如果结果的 dtype 是 timedelta64[ns]，则返回 TimedeltaArray 类型的结果
        if result.dtype.kind == "m" and is_supported_dtype(result.dtype):
            from pandas.core.arrays import TimedeltaArray
            return TimedeltaArray._simple_new(result, dtype=result.dtype)
        
        # 否则返回与 self 类型相同的结果
        return type(self)(result)

    # ------------------------------------------------------------------------
    # 字符串方法接口

    # _str_na_value 表示字符串方法中的缺失值，设定为 np.nan
    _str_na_value = np.nan
```