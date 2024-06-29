# `D:\src\scipysrc\pandas\pandas\core\arrays\string_.py`

```
# 从未来模块导入类型标注的支持
from __future__ import annotations
# 导入类型相关的声明，用于静态类型检查
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    cast,
)

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 从 pandas._config 模块中导入 get_option 函数
from pandas._config import get_option

# 从 pandas._libs 模块中导入 lib 和 libmissing 别名
from pandas._libs import (
    lib,
    missing as libmissing,
)
# 从 pandas._libs.arrays 模块中导入 NDArrayBacked 类
from pandas._libs.arrays import NDArrayBacked
# 从 pandas._libs.lib 模块中导入 ensure_string_array 函数
from pandas._libs.lib import ensure_string_array
# 从 pandas.compat 模块中导入 pa_version_under10p1 函数
from pandas.compat import pa_version_under10p1
# 从 pandas.compat.numpy 模块中导入 function 别名
from pandas.compat.numpy import function as nv
# 从 pandas.util._decorators 模块中导入 doc 装饰器
from pandas.util._decorators import doc

# 从 pandas.core.dtypes.base 模块中导入多个类和函数
from pandas.core.dtypes.base import (
    ExtensionDtype,
    StorageExtensionDtype,
    register_extension_dtype,
)
# 从 pandas.core.dtypes.common 模块中导入多个函数
from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)

# 从 pandas.core 模块中导入 ops 对象
from pandas.core import ops
# 从 pandas.core.array_algos 模块中导入 masked_reductions 函数
from pandas.core.array_algos import masked_reductions
# 从 pandas.core.arrays.base 模块中导入 ExtensionArray 类
from pandas.core.arrays.base import ExtensionArray
# 从 pandas.core.arrays.floating 模块中导入 FloatingArray 和 FloatingDtype 类
from pandas.core.arrays.floating import (
    FloatingArray,
    FloatingDtype,
)
# 从 pandas.core.arrays.integer 模块中导入 IntegerArray 和 IntegerDtype 类
from pandas.core.arrays.integer import (
    IntegerArray,
    IntegerDtype,
)
# 从 pandas.core.arrays.numpy_ 模块中导入 NumpyExtensionArray 类
from pandas.core.arrays.numpy_ import NumpyExtensionArray
# 从 pandas.core.construction 模块中导入 extract_array 函数
from pandas.core.construction import extract_array
# 从 pandas.core.indexers 模块中导入 check_array_indexer 函数
from pandas.core.indexers import check_array_indexer
# 从 pandas.core.missing 模块中导入 isna 函数

from pandas.core.missing import isna

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 pyarrow 库
    import pyarrow

    # 从 pandas._typing 模块中导入多个类型
    from pandas._typing import (
        AxisInt,
        Dtype,
        DtypeObj,
        NumpySorter,
        NumpyValueArrayLike,
        Scalar,
        Self,
        npt,
        type_t,
    )

    # 从 pandas 模块中导入 Series 类

    from pandas import Series


# 注册 StringDtype 类作为扩展的数据类型
@register_extension_dtype
class StringDtype(StorageExtensionDtype):
    """
    Extension dtype for string data.

    .. warning::

       StringDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    storage : {"python", "pyarrow", "pyarrow_numpy"}, optional
        If not given, the value of ``pd.options.mode.string_storage``.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    BooleanDtype : Extension dtype for boolean data.

    Examples
    --------
    >>> pd.StringDtype()
    string[python]

    >>> pd.StringDtype(storage="pyarrow")
    string[pyarrow]
    """

    # 类变量，指定数据类型的名称为 "string"
    name: ClassVar[str] = "string"  # type: ignore[misc]

    #: StringDtype().na_value uses pandas.NA except the implementation that
    # follows NumPy semantics, which uses nan.
    @property
    # 属性方法，根据存储类型返回缺失值
    def na_value(self) -> libmissing.NAType | float:  # type: ignore[override]
        # 如果存储类型是 "pyarrow_numpy"，返回 np.nan
        if self.storage == "pyarrow_numpy":
            return np.nan
        # 否则返回 pandas 中定义的 NA 值
        else:
            return libmissing.NA

    # 元数据，指定存储类型作为元组的一部分
    _metadata = ("storage",)
    def __init__(self, storage=None) -> None:
        # 如果未提供存储类型，则根据选项决定默认的存储方式
        if storage is None:
            infer_string = get_option("future.infer_string")
            if infer_string:
                storage = "pyarrow_numpy"
            else:
                storage = get_option("mode.string_storage")
        
        # 检查存储类型是否在允许的集合中，如果不在则引发 ValueError 异常
        if storage not in {"python", "pyarrow", "pyarrow_numpy"}:
            raise ValueError(
                f"Storage must be 'python', 'pyarrow' or 'pyarrow_numpy'. "
                f"Got {storage} instead."
            )
        
        # 如果存储类型是 pyarrow 或 pyarrow_numpy，且 pyarrow 版本小于 10.0.1，则引发 ImportError 异常
        if storage in ("pyarrow", "pyarrow_numpy") and pa_version_under10p1:
            raise ImportError(
                "pyarrow>=10.0.1 is required for PyArrow backed StringArray."
            )
        
        # 将存储类型存储在对象实例中
        self.storage = storage

    @property
    def type(self) -> type[str]:
        # 返回对象的类型为 str
        return str

    @classmethod
    def construct_from_string(cls, string) -> Self:
        """
        Construct a StringDtype from a string.

        Parameters
        ----------
        string : str
            The type of the name. The storage type will be taking from `string`.
            Valid options and their storage types are

            ========================== ==============================================
            string                     result storage
            ========================== ==============================================
            ``'string'``               pd.options.mode.string_storage, default python
            ``'string[python]'``       python
            ``'string[pyarrow]'``      pyarrow
            ========================== ==============================================

        Returns
        -------
        StringDtype

        Raise
        -----
        TypeError
            If the string is not a valid option.
        """
        # 检查传入的参数是否为字符串类型，如果不是则引发 TypeError 异常
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        
        # 根据输入的字符串选择合适的存储类型构造 StringDtype 对象，并返回
        if string == "string":
            return cls()
        elif string == "string[python]":
            return cls(storage="python")
        elif string == "string[pyarrow]":
            return cls(storage="pyarrow")
        elif string == "string[pyarrow_numpy]":
            return cls(storage="pyarrow_numpy")
        else:
            # 如果输入的字符串不在预期的选项中，则引发 TypeError 异常
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")

    # https://github.com/pandas-dev/pandas/issues/36126
    # error: Signature of "construct_array_type" incompatible with supertype
    # "ExtensionDtype"
    def construct_array_type(  # type: ignore[override]
        self,
    ) -> type_t[BaseStringArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        from pandas.core.arrays.string_arrow import (
            ArrowStringArray,
            ArrowStringArrayNumpySemantics,
        )

        # 根据存储类型返回相应的数组类型
        if self.storage == "python":
            return StringArray
        elif self.storage == "pyarrow":
            return ArrowStringArray
        else:
            return ArrowStringArrayNumpySemantics

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseStringArray:
        """
        Construct StringArray from pyarrow Array/ChunkedArray.
        """
        # 如果存储类型是 pyarrow，则使用 ArrowStringArray 构造 StringArray
        if self.storage == "pyarrow":
            from pandas.core.arrays.string_arrow import ArrowStringArray

            return ArrowStringArray(array)
        elif self.storage == "pyarrow_numpy":
            # 如果存储类型是 pyarrow_numpy，则使用 ArrowStringArrayNumpySemantics 构造 StringArray
            from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics

            return ArrowStringArrayNumpySemantics(array)
        else:
            import pyarrow

            if isinstance(array, pyarrow.Array):
                chunks = [array]
            else:
                # 如果是 pyarrow.ChunkedArray，则获取其所有的 chunk
                chunks = array.chunks

            results = []
            for arr in chunks:
                # 将每个 chunk 转换为 numpy 数组，并拼接结果，避免大型字符串数据在拼接 pyarrow 数组时的溢出问题
                arr = arr.to_numpy(zero_copy_only=False)
                arr = ensure_string_array(arr, na_value=libmissing.NA)
                results.append(arr)

        if len(chunks) == 0:
            # 如果 chunks 为空，则返回一个空的 numpy 数组对象
            arr = np.array([], dtype=object)
        else:
            # 否则将所有结果拼接成一个 numpy 数组
            arr = np.concatenate(results)

        # 构造一个新的 StringArray，跳过内部验证以提高效率，参见 GH#47781
        new_string_array = StringArray.__new__(StringArray)
        NDArrayBacked.__init__(
            new_string_array,
            arr,
            StringDtype(storage="python"),
        )
        return new_string_array
class BaseStringArray(ExtensionArray):
    """
    Mixin class for StringArray, ArrowStringArray.
    """

    @doc(ExtensionArray.tolist)
    def tolist(self) -> list:
        # 如果数组的维度大于1，则递归调用每个元素的 tolist 方法并返回列表
        if self.ndim > 1:
            return [x.tolist() for x in self]
        # 否则将当前数组转换为 NumPy 数组后返回其列表表示
        return list(self.to_numpy())

    @classmethod
    def _from_scalars(cls, scalars, dtype: DtypeObj) -> Self:
        # 如果推断出标量的类型不是字符串或者空类型，则抛出 ValueError 异常
        if lib.infer_dtype(scalars, skipna=True) not in ["string", "empty"]:
            # TODO: 需要确保任何缺失值对于字符串是有效的
            raise ValueError
        # 否则根据标量创建相应类型的实例并返回
        return cls._from_sequence(scalars, dtype=dtype)


# error: Definition of "_concat_same_type" in base class "NDArrayBacked" is
# incompatible with definition in base class "ExtensionArray"
class StringArray(BaseStringArray, NumpyExtensionArray):  # type: ignore[misc]
    """
    Extension array for string data.

    .. warning::

       StringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : array-like
        The array of data.

        .. warning::

           Currently, this expects an object-dtype ndarray
           where the elements are Python strings
           or nan-likes (``None``, ``np.nan``, ``NA``).
           This may change without warning in the future. Use
           :meth:`pandas.array` with ``dtype="string"`` for a stable way of
           creating a `StringArray` from any sequence.

        .. versionchanged:: 1.5.0

           StringArray now accepts array-likes containing
           nan-likes(``None``, ``np.nan``) for the ``values`` parameter
           in addition to strings and :attr:`pandas.NA`

    copy : bool, default False
        Whether to copy the array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`array`
        The recommended function for creating a StringArray.
    Series.str
        The string methods are available on Series backed by
        a StringArray.

    Notes
    -----
    StringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(["This is", "some text", None, "data."], dtype="string")
    <StringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string

    Unlike arrays instantiated with ``dtype="object"``, ``StringArray``
    will convert the values to strings.

    >>> pd.array(["1", 1], dtype="object")
    <NumpyExtensionArray>
    ['1', 1]
    Length: 2, dtype: object
    >>> pd.array(["1", 1], dtype="string")
    <StringArray>
    ['1', '1']
    Length: 2, dtype: string

    However, instantiating StringArrays directly with non-strings will raise an error.

    For comparison methods, `StringArray` returns a :class:`pandas.BooleanArray`:

    >>> pd.array(["a", None, "c"], dtype="string") == "a"
    <BooleanArray>
    [True, <NA>, False]
    Length: 3, dtype: boolean
    """

    # undo the NumpyExtensionArray hack
    # 定义类变量 `_typ`，表示这是一个扩展类型的对象
    _typ = "extension"

    # 初始化方法，接受 `values` 参数并可能进行复制操作
    def __init__(self, values, copy: bool = False) -> None:
        # 提取数组形式的 `values`
        values = extract_array(values)

        # 调用父类的初始化方法，传入提取后的 `values` 和 `copy` 参数
        super().__init__(values, copy=copy)

        # 如果 `values` 不是当前类的实例，则进行验证
        if not isinstance(values, type(self)):
            self._validate()

        # 使用 `NDArrayBacked` 类的初始化方法，传入内部的 ndarray 和 StringDtype
        NDArrayBacked.__init__(self, self._ndarray, StringDtype(storage="python"))

    # 内部方法，用于验证存储的数据是否只包含 NA 或字符串
    def _validate(self) -> None:
        """Validate that we only store NA or strings."""
        # 如果 `_ndarray` 的长度不为 0，并且不是字符串数组，则引发 ValueError
        if len(self._ndarray) and not lib.is_string_array(self._ndarray, skipna=True):
            raise ValueError("StringArray requires a sequence of strings or pandas.NA")

        # 如果 `_ndarray` 的 dtype 不是 "object"，则引发 ValueError
        if self._ndarray.dtype != "object":
            raise ValueError(
                "StringArray requires a sequence of strings or pandas.NA. Got "
                f"'{self._ndarray.dtype}' dtype instead."
            )

        # 检查是否需要将 NaN 值转换为 pd.NA
        if self._ndarray.ndim > 2:
            # 如果 `_ndarray` 的维度大于 2，则将其展平为一维再进行转换
            lib.convert_nans_to_NA(self._ndarray.ravel("K"))
        else:
            # 否则直接将 `_ndarray` 中的 NaN 值转换为 pd.NA
            lib.convert_nans_to_NA(self._ndarray)

    # 类方法，从标量序列创建新的 StringArray 对象
    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype: Dtype | None = None, copy: bool = False
    ) -> Self:
        # 如果指定了 dtype 并且不是字符串类型，则转换为 StringDtype
        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "python"

        # 导入 pandas 的 BaseMaskedArray 类
        from pandas.core.arrays.masked import BaseMaskedArray

        # 如果 scalars 是 BaseMaskedArray 类型，则避免转换为对象 dtype
        if isinstance(scalars, BaseMaskedArray):
            na_values = scalars._mask
            result = scalars._data
            # 确保结果是字符串数组，并保持原样不转换 NA 值
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            result[na_values] = libmissing.NA

        else:
            # 如果 scalars 是 pyarrow 数组，则先转换为 numpy 数组
            if lib.is_pyarrow_array(scalars):
                scalars = np.array(scalars)

            # 确保 scalars 转换为字符串数组，将 NaN 值转换为 `libmissing.NA`
            result = lib.ensure_string_array(scalars, na_value=libmissing.NA, copy=copy)

        # 手动创建新的 StringArray 对象，并初始化为内部结果，避免了 `__init__` 中的验证步骤，因此更快
        new_string_array = cls.__new__(cls)
        NDArrayBacked.__init__(new_string_array, result, StringDtype(storage="python"))

        return new_string_array

    # 类方法，从字符串序列创建新的 StringArray 对象
    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # 调用 `_from_sequence` 方法，传入字符串序列和指定的 dtype
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    # 类方法
    @classmethod
    # 创建一个空的 StringArray 对象，使用 np.empty 初始化，元素类型为 object
    # 并将所有元素设置为 libmissing.NA
    def _empty(cls, shape, dtype) -> StringArray:
        values = np.empty(shape, dtype=object)
        values[:] = libmissing.NA
        return cls(values).astype(dtype, copy=False)

    # 将当前对象转换为 pyarrow Array 对象
    def __arrow_array__(self, type=None):
        """
        Convert myself into a pyarrow Array.
        """
        import pyarrow as pa

        # 如果 type 参数为 None，则默认使用 pa.string() 类型
        if type is None:
            type = pa.string()

        # 复制当前对象的值到 values 变量，并将缺失值替换为 None
        values = self._ndarray.copy()
        values[self.isna()] = None
        # 使用 pyarrow 的 array 函数创建 Array 对象，from_pandas=True 表示从 pandas 对象创建
        return pa.array(values, type=type, from_pandas=True)

    # 为进行 factorize 操作准备值，返回 arr 和 None
    def _values_for_factorize(self) -> tuple[np.ndarray, None]:
        arr = self._ndarray.copy()
        # 获取缺失值的掩码
        mask = self.isna()
        # 将缺失值替换为 None
        arr[mask] = None
        return arr, None

    # 重写父类方法 __setitem__，用于设置数组元素
    def __setitem__(self, key, value) -> None:
        # 使用 extract_array 函数从 value 中提取数组，同时将其转换为 numpy 数组
        value = extract_array(value, extract_numpy=True)
        # 如果 value 是当前对象的子类，则使用其内部 _ndarray 属性
        if isinstance(value, type(self)):
            value = value._ndarray

        # 检查 key 是否为合法的数组索引
        key = check_array_indexer(self, key)
        # 检查 key 和 value 是否为标量
        scalar_key = lib.is_scalar(key)
        scalar_value = lib.is_scalar(value)
        if scalar_key and not scalar_value:
            raise ValueError("setting an array element with a sequence.")

        # 验证新值的类型和内容
        if scalar_value:
            # 如果 value 是缺失值，则设置为 libmissing.NA
            if isna(value):
                value = libmissing.NA
            # 如果 value 不是字符串，则抛出类型错误异常
            elif not isinstance(value, str):
                raise TypeError(
                    f"Cannot set non-string value '{value}' into a StringArray."
                )
        else:
            # 如果 value 不是类数组对象，则转换为 numpy 数组，并确保元素类型为 object
            if not is_array_like(value):
                value = np.asarray(value, dtype=object)
            # 如果 value 包含非字符串元素，则抛出类型错误异常
            if len(value) and not lib.is_string_array(value, skipna=True):
                raise TypeError("Must provide strings.")

            # 获取缺失值的掩码，并将其替换为 libmissing.NA
            mask = isna(value)
            if mask.any():
                value = value.copy()
                value[isna(value)] = libmissing.NA

        # 调用父类的 __setitem__ 方法，设置 key 对应的值为 value
        super().__setitem__(key, value)

    # 使用掩码 mask 将 value 中的值插入当前对象
    def _putmask(self, mask: npt.NDArray[np.bool_], value) -> None:
        # NDArrayBackedExtensionArray._putmask 方法使用 np.putmask 处理掩码，
        # 但不正确处理 None 或 pd.NA，因此使用基类实现，该实现使用 __setitem__
        ExtensionArray._putmask(self, mask, value)
    # 将 Series 对象转换为指定数据类型 dtype
    def astype(self, dtype, copy: bool = True):
        # 将 dtype 转换为 pandas 的数据类型
        dtype = pandas_dtype(dtype)

        # 如果当前数据类型与目标数据类型相同
        if dtype == self.dtype:
            # 如果需要复制数据，则返回副本
            if copy:
                return self.copy()
            # 否则直接返回当前对象
            return self

        # 如果目标数据类型是整数类型
        elif isinstance(dtype, IntegerDtype):
            # 复制数据数组，并处理缺失值
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            # 将处理后的数据数组转换为整数数组，并创建 IntegerArray 对象返回
            values = arr.astype(dtype.numpy_dtype)
            return IntegerArray(values, mask, copy=False)

        # 如果目标数据类型是浮点数类型
        elif isinstance(dtype, FloatingDtype):
            # 复制数据数组，并处理缺失值
            arr_ea = self.copy()
            mask = self.isna()
            arr_ea[mask] = "0"
            # 将处理后的数据数组转换为浮点数数组，并创建 FloatingArray 对象返回
            values = arr_ea.astype(dtype.numpy_dtype)
            return FloatingArray(values, mask, copy=False)

        # 如果目标数据类型是扩展类型
        elif isinstance(dtype, ExtensionDtype):
            # 调用 ExtensionArray 的 astype 方法，并跳过 NumpyExtensionArray.astype 方法
            return ExtensionArray.astype(self, dtype, copy)

        # 如果目标数据类型是浮点数的子类型
        elif np.issubdtype(dtype, np.floating):
            # 复制数据数组，并处理缺失值
            arr = self._ndarray.copy()
            mask = self.isna()
            arr[mask] = 0
            # 将处理后的数据数组转换为目标数据类型，同时将缺失值设为 NaN
            values = arr.astype(dtype)
            values[mask] = np.nan
            return values

        # 对于其他情况，调用父类的 astype 方法进行处理
        return super().astype(dtype, copy)
    ) -> npt.NDArray[np.intp] | np.intp:
        # 如果数组中存在缺失值，抛出数值错误，因为要求数组排序，而存在缺失值无法排序
        if self._hasna:
            raise ValueError(
                "searchsorted requires array to be sorted, which is impossible "
                "with NAs present."
            )
        # 调用父类的 searchsorted 方法进行查找并返回结果
        return super().searchsorted(value=value, side=side, sorter=sorter)

    def _cmp_method(self, other, op):
        from pandas.arrays import BooleanArray

        if isinstance(other, StringArray):
            other = other._ndarray

        # 创建一个布尔掩码，标记当前对象和其他对象是否有缺失值
        mask = isna(self) | isna(other)
        # 创建一个有效数据的掩码
        valid = ~mask

        # 如果 other 不是标量，并且长度与当前对象不匹配，则抛出 ValueError
        if not lib.is_scalar(other):
            if len(other) != len(self):
                # 避免在 other 是二维时进行不正确的广播
                raise ValueError(
                    f"Lengths of operands do not match: {len(self)} != {len(other)}"
                )

            # 将 other 转换为 NumPy 数组，并仅保留有效数据
            other = np.asarray(other)
            other = other[valid]

        # 如果操作属于算术二元操作符集合
        if op.__name__ in ops.ARITHMETIC_BINOPS:
            # 创建一个与当前对象同样大小的对象数组，数据类型为 object
            result = np.empty_like(self._ndarray, dtype="object")
            # 将缺失值位置填充为 NA
            result[mask] = libmissing.NA
            # 将有效数据位置的操作结果填充到结果数组中
            result[valid] = op(self._ndarray[valid], other)
            # 返回结果作为 StringArray 对象
            return StringArray(result)
        else:
            # 逻辑操作
            # 创建一个布尔数组，长度与当前对象的 ndarray 相同
            result = np.zeros(len(self._ndarray), dtype="bool")
            # 将有效数据位置的操作结果填充到结果数组中
            result[valid] = op(self._ndarray[valid], other)
            # 返回结果作为 BooleanArray 对象，带有掩码信息
            return BooleanArray(result, mask)

    _arith_method = _cmp_method

    # ------------------------------------------------------------------------
    # String methods interface
    # error: Incompatible types in assignment (expression has type "NAType",
    # base class "NumpyExtensionArray" defined the type as "float")
    # 定义字符串数组的 NA 值为 libmissing.NA，忽略类型检查错误
    _str_na_value = libmissing.NA  # type: ignore[assignment]

    def _str_map(
        self, f, na_value=None, dtype: Dtype | None = None, convert: bool = True
        ):
            # 从 pandas.arrays 中导入 BooleanArray
            from pandas.arrays import BooleanArray

            # 如果未指定 dtype，则使用 StringDtype，存储格式为 Python 对象
            if dtype is None:
                dtype = StringDtype(storage="python")
            # 如果未指定 na_value，则使用 self 的默认 na_value
            if na_value is None:
                na_value = self.dtype.na_value

            # 创建一个布尔掩码，标识 self 中的缺失值
            mask = isna(self)
            # 将 self 转换为 NumPy 数组
            arr = np.asarray(self)

            # 如果目标 dtype 是整数或布尔类型
            if is_integer_dtype(dtype) or is_bool_dtype(dtype):
                # 声明一个类型为 IntegerArray 或 BooleanArray 的构造函数变量
                constructor: type[IntegerArray | BooleanArray]
                if is_integer_dtype(dtype):
                    constructor = IntegerArray
                else:
                    constructor = BooleanArray

                # 检查 na_value 是否为缺失值
                na_value_is_na = isna(na_value)
                if na_value_is_na:
                    na_value = 1
                # 如果目标 dtype 是布尔类型，确保 na_value 转换为布尔值
                elif dtype == np.dtype("bool"):
                    na_value = bool(na_value)

                # 调用 lib.map_infer_mask 函数，推断映射结果和掩码
                result = lib.map_infer_mask(
                    arr,
                    f,
                    mask.view("uint8"),
                    convert=False,
                    na_value=na_value,
                    # 使用指定的 dtype，进行类型强制转换
                    dtype=np.dtype(cast(type, dtype)),
                )

                # 如果 na_value 不是缺失值，则将掩码置为 False
                if not na_value_is_na:
                    mask[:] = False

                # 使用 constructor 构造结果对象，返回构造后的对象和掩码
                return constructor(result, mask)

            # 如果目标 dtype 是字符串类型但不是对象类型
            elif is_string_dtype(dtype) and not is_object_dtype(dtype):
                # 使用 lib.map_infer_mask 函数，推断映射结果和掩码，返回 StringArray 对象
                result = lib.map_infer_mask(
                    arr, f, mask.view("uint8"), convert=False, na_value=na_value
                )
                return StringArray(result)
            else:
                # 当结果类型是对象类型时，使用 lib.map_infer_mask 函数推断映射结果和掩码
                return lib.map_infer_mask(arr, f, mask.view("uint8"))
```