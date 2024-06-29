# `D:\src\scipysrc\pandas\pandas\core\arrays\boolean.py`

```
from __future__ import annotations
# 导入未来的 annotations 特性，用于支持类型注解中的递归类型引用

import numbers
# 导入 numbers 模块，用于数值类型的操作

from typing import (
    TYPE_CHECKING,
    ClassVar,
    cast,
)
# 导入 typing 模块中的类型，包括 TYPE_CHECKING、ClassVar 和 cast

import numpy as np
# 导入 NumPy 库，用于数组操作

from pandas._libs import (
    lib,
    missing as libmissing,
)
# 从 pandas._libs 中导入 lib 和 libmissing 模块

from pandas.core.dtypes.common import is_list_like
# 从 pandas.core.dtypes.common 中导入 is_list_like 函数，用于检查对象是否类列表

from pandas.core.dtypes.dtypes import register_extension_dtype
# 从 pandas.core.dtypes.dtypes 中导入 register_extension_dtype 函数，用于注册扩展数据类型

from pandas.core.dtypes.missing import isna
# 从 pandas.core.dtypes.missing 中导入 isna 函数，用于检查缺失值

from pandas.core import ops
# 从 pandas.core 中导入 ops 模块，用于操作符重载

from pandas.core.array_algos import masked_accumulations
# 从 pandas.core.array_algos 中导入 masked_accumulations 函数，用于掩码累积运算

from pandas.core.arrays.masked import (
    BaseMaskedArray,
    BaseMaskedDtype,
)
# 从 pandas.core.arrays.masked 中导入 BaseMaskedArray 和 BaseMaskedDtype 类

if TYPE_CHECKING:
    import pyarrow
    # 如果 TYPE_CHECKING 为真，导入 pyarrow 模块

    from pandas._typing import (
        DtypeObj,
        Self,
        npt,
        type_t,
    )
    # 从 pandas._typing 中导入 DtypeObj、Self、npt 和 type_t 类型

    from pandas.core.dtypes.dtypes import ExtensionDtype
    # 从 pandas.core.dtypes.dtypes 中导入 ExtensionDtype 类型

@register_extension_dtype
class BooleanDtype(BaseMaskedDtype):
    """
    Extension dtype for boolean data.

    .. warning::

       BooleanDtype is considered experimental. The implementation and
       parts of the API may change without warning.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    StringDtype : Extension dtype for string data.

    Examples
    --------
    >>> pd.BooleanDtype()
    BooleanDtype
    """

    name: ClassVar[str] = "boolean"
    # 类变量 name 指定为 "boolean"

    # The value used to fill '_data' to avoid upcasting
    _internal_fill_value = False
    # _internal_fill_value 用于填充 '_data' 以避免向上转型

    # https://github.com/python/mypy/issues/4125
    # error: Signature of "type" incompatible with supertype "BaseMaskedDtype"
    @property
    def type(self) -> type:  # type: ignore[override]
        return np.bool_
    # type 属性返回 np.bool_，用于指定数据类型为布尔值

    @property
    def kind(self) -> str:
        return "b"
    # kind 属性返回 "b"，表示数据类型的种类为布尔型

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype("bool")
    # numpy_dtype 属性返回 NumPy 的 bool 数据类型对象

    @classmethod
    def construct_array_type(cls) -> type_t[BooleanArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return BooleanArray
    # construct_array_type 类方法返回与此数据类型关联的数组类型

    def __repr__(self) -> str:
        return "BooleanDtype"
    # __repr__ 方法返回描述该对象的字符串表示形式为 "BooleanDtype"

    @property
    def _is_boolean(self) -> bool:
        return True
    # _is_boolean 属性返回 True，指示该数据类型为布尔型

    @property
    def _is_numeric(self) -> bool:
        return True
    # _is_numeric 属性返回 True，指示该数据类型是数值型的

    def __from_arrow__(
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BooleanArray:
        """
        Construct BooleanArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow

        # 检查输入的 array 是否为布尔类型的 pyarrow.Array 或 pyarrow.ChunkedArray
        if array.type != pyarrow.bool_() and not pyarrow.types.is_null(array.type):
            raise TypeError(f"Expected array of boolean type, got {array.type} instead")

        if isinstance(array, pyarrow.Array):
            # 如果 array 是 pyarrow.Array 类型，则将其作为唯一的 chunk 处理
            chunks = [array]
            length = len(array)
        else:
            # 如果 array 是 pyarrow.ChunkedArray 类型，则获取其所有的 chunk
            chunks = array.chunks
            length = array.length()

        if pyarrow.types.is_null(array.type):
            # 如果 array 中的数据都是 null，则生成一个全为 True 的 mask 和空的 data 数组
            mask = np.ones(length, dtype=bool)
            # 由于数据全为 null，所以不需要初始化 data 数组
            data = np.empty(length, dtype=bool)
            return BooleanArray(data, mask)

        results = []
        for arr in chunks:
            # 对每个 chunk 进行处理，获取其 buffers
            buflist = arr.buffers()
            # 根据 buffers 创建 pyarrow.BooleanArray，并转换为 numpy 数组
            data = pyarrow.BooleanArray.from_buffers(
                arr.type, len(arr), [None, buflist[1]], offset=arr.offset
            ).to_numpy(zero_copy_only=False)
            if arr.null_count != 0:
                # 如果 chunk 中存在 null 值，则创建相应的 mask
                mask = pyarrow.BooleanArray.from_buffers(
                    arr.type, len(arr), [None, buflist[0]], offset=arr.offset
                ).to_numpy(zero_copy_only=False)
                mask = ~mask  # 取反，将 null 的位置设为 True
            else:
                mask = np.zeros(len(arr), dtype=bool)  # 否则创建全为 False 的 mask

            # 根据 data 和 mask 创建 BooleanArray 对象
            bool_arr = BooleanArray(data, mask)
            results.append(bool_arr)

        if not results:
            # 如果结果列表为空，则返回一个空的 BooleanArray 对象
            return BooleanArray(
                np.array([], dtype=np.bool_), np.array([], dtype=np.bool_)
            )
        else:
            # 否则将结果列表中的所有 BooleanArray 进行合并，并返回合并后的结果
            return BooleanArray._concat_same_type(results)
# 定义一个函数，将输入的值数组强制转换为带有掩码的 NumPy 数组的元组返回

def coerce_to_array(
    values, mask=None, copy: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce the input values array to numpy arrays with a mask.

    Parameters
    ----------
    values : 1D list-like
        输入的值数组，类似于列表，一维
    mask : bool 1D array, optional
        布尔类型的一维数组，掩码，可选参数
    copy : bool, default False
        如果为 True，复制输入的数组

    Returns
    -------
    tuple of (values, mask)
        返回值是一个元组，包含 values 和 mask
    """

    # 如果输入的 values 是 BooleanArray 类型的实例
    if isinstance(values, BooleanArray):
        if mask is not None:
            raise ValueError("cannot pass mask for BooleanArray input")
        # 获取 BooleanArray 对象的数据和掩码
        values, mask = values._data, values._mask
        # 如果需要复制，则复制数据和掩码
        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask

    mask_values = None
    
    # 如果 values 是 NumPy 数组且数据类型是布尔型
    if isinstance(values, np.ndarray) and values.dtype == np.bool_:
        if copy:
            # 如果需要复制，复制数据
            values = values.copy()
    # 如果 values 是 NumPy 数组且数据类型是整数、浮点数或者布尔型
    elif isinstance(values, np.ndarray) and values.dtype.kind in "iufcb":
        # 判断 values 中的缺失值
        mask_values = isna(values)

        # 创建一个布尔型的数组
        values_bool = np.zeros(len(values), dtype=bool)
        # 将非缺失值的元素转换为布尔型
        values_bool[~mask_values] = values[~mask_values].astype(bool)

        # 如果转换后的值不符合原始值的数据类型，则抛出类型错误
        if not np.all(
            values_bool[~mask_values].astype(values.dtype) == values[~mask_values]
        ):
            raise TypeError("Need to pass bool-like values")

        values = values_bool
    else:
        # 将 values 转换为 NumPy 数组，数据类型为对象
        values_object = np.asarray(values, dtype=object)

        # 推断 values_object 的数据类型
        inferred_dtype = lib.infer_dtype(values_object, skipna=True)
        integer_like = ("floating", "integer", "mixed-integer-float")
        
        # 如果推断的数据类型不是布尔型、空或者整数样式的数据类型，则抛出类型错误
        if inferred_dtype not in ("boolean", "empty") + integer_like:
            raise TypeError("Need to pass bool-like values")

        # 获取 values_object 的缺失值掩码
        mask_values = cast("npt.NDArray[np.bool_]", isna(values_object))
        # 创建一个布尔型的数组
        values = np.zeros(len(values), dtype=bool)
        # 将非缺失值的元素转换为布尔型
        values[~mask_values] = values_object[~mask_values].astype(bool)

        # 如果原始数据是整数样式的，则验证它是否为 0 或 1
        if (inferred_dtype in integer_like) and not (
            np.all(
                values[~mask_values].astype(float)
                == values_object[~mask_values].astype(float)
            )
        ):
            raise TypeError("Need to pass bool-like values")

    # 如果 mask 和 mask_values 都为 None，则创建一个形状与 values 相同的布尔型数组
    if mask is None and mask_values is None:
        mask = np.zeros(values.shape, dtype=bool)
    elif mask is None:
        mask = mask_values
    else:
        # 如果 mask 是 NumPy 数组且数据类型是布尔型
        if isinstance(mask, np.ndarray) and mask.dtype == np.bool_:
            # 如果 mask_values 不为 None，则合并 mask 和 mask_values
            if mask_values is not None:
                mask = mask | mask_values
            else:
                # 如果需要复制，则复制 mask
                if copy:
                    mask = mask.copy()
        else:
            # 将 mask 转换为布尔型数组
            mask = np.array(mask, dtype=bool)
            # 如果 mask_values 不为 None，则合并 mask 和 mask_values
            if mask_values is not None:
                mask = mask | mask_values

    # 检查 values 和 mask 的形状是否匹配
    if values.shape != mask.shape:
        raise ValueError("values.shape and mask.shape must match")

    # 返回 values 和 mask 的元组
    return values, mask
    """
    Array of boolean (True/False) data with missing values.

    This is a pandas Extension array for boolean data, under the hood
    represented by 2 numpy arrays: a boolean array with the data and
    a boolean array with the mask (True indicating missing).

    BooleanArray implements Kleene logic (sometimes called three-value
    logic) for logical operations. See :ref:`boolean.kleene` for more.

    To construct an BooleanArray from generic array-like input, use
    :func:`pandas.array` specifying ``dtype="boolean"`` (see examples
    below).

    .. warning::

       BooleanArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : numpy.ndarray
        A 1-d boolean-dtype array with the data.
    mask : numpy.ndarray
        A 1-d boolean-dtype array indicating missing values (True
        indicates missing).
    copy : bool, default False
        Whether to copy the `values` and `mask` arrays.

    Attributes
    ----------
    None

    Methods
    -------
    None

    Returns
    -------
    BooleanArray

    Examples
    --------
    Create an BooleanArray with :func:`pandas.array`:

    >>> pd.array([True, False, None], dtype="boolean")
    <BooleanArray>
    [True, False, <NA>]
    Length: 3, dtype: boolean
    """

    # 定义真值集合，用于解析字符串到布尔值
    _TRUE_VALUES = {"True", "TRUE", "true", "1", "1.0"}
    # 定义假值集合，用于解析字符串到布尔值
    _FALSE_VALUES = {"False", "FALSE", "false", "0", "0.0"}

    @classmethod
    def _simple_new(cls, values: np.ndarray, mask: npt.NDArray[np.bool_]) -> Self:
        # 创建一个新的 BooleanArray 实例，使用指定的值和掩码
        result = super()._simple_new(values, mask)
        # 设置实例的数据类型为 BooleanDtype
        result._dtype = BooleanDtype()
        return result

    def __init__(
        self, values: np.ndarray, mask: np.ndarray, copy: bool = False
    ) -> None:
        # 如果传入的值不是布尔类型的 numpy 数组，则抛出类型错误
        if not (isinstance(values, np.ndarray) and values.dtype == np.bool_):
            raise TypeError(
                "values should be boolean numpy array. Use "
                "the 'pd.array' function instead"
            )
        # 设置实例的数据类型为 BooleanDtype
        self._dtype = BooleanDtype()
        # 调用父类的构造函数，初始化实例
        super().__init__(values, mask, copy=copy)

    @property
    def dtype(self) -> BooleanDtype:
        # 返回实例的数据类型
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: list[str],
        *,
        dtype: ExtensionDtype,
        copy: bool = False,
        true_values: list[str] | None = None,
        false_values: list[str] | None = None,
        none_values: list[str] | None = None,
    ) -> BooleanArray:
        # 将类的_TRUE_VALUES与传入的true_values取并集，构成最终的true值集合
        true_values_union = cls._TRUE_VALUES.union(true_values or [])
        # 将类的_FALSE_VALUES与传入的false_values取并集，构成最终的false值集合
        false_values_union = cls._FALSE_VALUES.union(false_values or [])

        # 如果none_values为None，则将其设为空列表
        if none_values is None:
            none_values = []

        # 定义一个内部函数map_string，用于将字符串映射为布尔值或None
        def map_string(s) -> bool | None:
            if s in true_values_union:
                return True
            elif s in false_values_union:
                return False
            elif s in none_values:
                return None
            else:
                raise ValueError(f"{s} cannot be cast to bool")

        # 将输入的strings转换为numpy数组对象
        scalars = np.array(strings, dtype=object)
        # 判断数组中的每个元素是否为缺失值，生成对应的布尔掩码
        mask = isna(scalars)
        # 对于非缺失值的元素，应用map_string函数进行映射处理
        scalars[~mask] = list(map(map_string, scalars[~mask]))
        # 调用类方法_from_sequence，构建BooleanArray对象并返回
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    _HANDLED_TYPES = (np.ndarray, numbers.Number, bool, np.bool_)

    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # 如果指定了dtype，确保其为"boolean"
        if dtype:
            assert dtype == "boolean"
        # 调用coerce_to_array函数，将value转换为numpy数组对象，并返回
        return coerce_to_array(value, copy=copy)

    def _logical_method(self, other, op):
        # 断言操作符名称在预定义的集合中
        assert op.__name__ in {"or_", "ror_", "and_", "rand_", "xor", "rxor"}
        # 判断other是否为标量
        other_is_scalar = lib.is_scalar(other)
        # 初始化mask为None
        mask = None

        # 如果other是BooleanArray对象，则获取其_data和_mask属性
        if isinstance(other, BooleanArray):
            other, mask = other._data, other._mask
        # 如果other是类数组对象，则转换为numpy数组，并检查维度
        elif is_list_like(other):
            other = np.asarray(other, dtype="bool")
            if other.ndim > 1:
                raise NotImplementedError("can only perform ops with 1-d structures")
            other, mask = coerce_to_array(other, copy=False)
        # 如果other是numpy中的布尔值对象，则转换为其单一值表示
        elif isinstance(other, np.bool_):
            other = other.item()

        # 如果other是标量且不是缺失值NA且不是布尔值，则引发类型错误
        if other_is_scalar and other is not libmissing.NA and not lib.is_bool(other):
            raise TypeError(
                "'other' should be pandas.NA or a bool. "
                f"Got {type(other).__name__} instead."
            )

        # 如果other不是标量且长度与self不匹配，则引发数值错误
        if not other_is_scalar and len(self) != len(other):
            raise ValueError("Lengths must match")

        # 根据操作符名称选择对应的操作函数进行逻辑运算，得到结果和新的掩码
        if op.__name__ in {"or_", "ror_"}:
            result, mask = ops.kleene_or(self._data, other, self._mask, mask)
        elif op.__name__ in {"and_", "rand_"}:
            result, mask = ops.kleene_and(self._data, other, self._mask, mask)
        else:
            # 即xor, rxor操作
            result, mask = ops.kleene_xor(self._data, other, self._mask, mask)

        # 调用_maybe_mask_result方法，处理结果并返回BooleanArray对象
        return self._maybe_mask_result(result, mask)

    def _accumulate(
        self, name: str, *, skipna: bool = True, **kwargs
        # 获取对象的数据和掩码
        data = self._data
        mask = self._mask
        
        # 检查操作名称是否为累积最小值或累积最大值
        if name in ("cummin", "cummax"):
            # 获取对应的累积函数并应用于数据和掩码
            op = getattr(masked_accumulations, name)
            data, mask = op(data, mask, skipna=skipna, **kwargs)
            # 返回一个新的 BaseMaskedArray 对象，使用更新后的数据和掩码
            return self._simple_new(data, mask)
        else:
            # 导入 IntegerArray 类用于处理整数数组
            from pandas.core.arrays import IntegerArray
            
            # 将数据转换为整数类型的 IntegerArray，然后调用累积方法
            return IntegerArray(data.astype(int), mask)._accumulate(
                name, skipna=skipna, **kwargs
            )
```