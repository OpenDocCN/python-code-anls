# `D:\src\scipysrc\pandas\pandas\core\arrays\numeric.py`

```
from __future__ import annotations  # 允许类型注解中使用字符串形式的类型

import numbers  # 引入 numbers 模块，用于处理数字相关的类型和操作
from typing import (  # 引入 typing 模块，用于类型提示
    TYPE_CHECKING,  # 用于类型检查时的条件判断
    Any,  # 表示任意类型
)

import numpy as np  # 引入 NumPy 库，用于数值计算

from pandas._libs import (  # 从 pandas._libs 中导入 lib 和 libmissing 模块
    lib,  # pandas 库的 C 语言库函数
    missing as libmissing,  # pandas 库中的缺失值处理函数，别名为 libmissing
)
from pandas.errors import AbstractMethodError  # 从 pandas.errors 中导入 AbstractMethodError 异常
from pandas.util._decorators import cache_readonly  # 从 pandas.util._decorators 中导入 cache_readonly 装饰器

from pandas.core.dtypes.common import (  # 从 pandas.core.dtypes.common 中导入若干函数
    is_integer_dtype,  # 判断是否为整数类型的函数
    is_string_dtype,  # 判断是否为字符串类型的函数
    pandas_dtype,  # 返回 pandas 中的数据类型
)

from pandas.core.arrays.masked import (  # 从 pandas.core.arrays.masked 中导入 BaseMaskedArray 和 BaseMaskedDtype 类
    BaseMaskedArray,  # pandas 中的基础掩码数组类型
    BaseMaskedDtype,  # pandas 中的基础掩码数据类型
)

if TYPE_CHECKING:  # 如果是类型检查模式
    from collections.abc import (  # 从 collections.abc 中导入若干类
        Callable,  # 可调用对象类型
        Mapping,  # 映射类型
    )

    import pyarrow  # 导入 pyarrow 库，用于 Apache Arrow 和 pandas 之间的数据交互

    from pandas._typing import (  # 从 pandas._typing 中导入若干类型
        DtypeObj,  # 数据类型对象
        Self,  # 自身类型
        npt,  # NumPy 数组类型
    )

    from pandas.core.dtypes.dtypes import ExtensionDtype  # 从 pandas.core.dtypes.dtypes 中导入 ExtensionDtype 类


class NumericDtype(BaseMaskedDtype):  # 定义 NumericDtype 类，继承自 BaseMaskedDtype 类
    _default_np_dtype: np.dtype  # 默认的 NumPy 数据类型
    _checker: Callable[[Any], bool]  # 检查器，接受任意类型参数并返回布尔值

    def __repr__(self) -> str:  # 定义 __repr__ 方法，返回对象的字符串表示形式
        return f"{self.name}Dtype()"  # 返回带有数据类型名称的字符串

    @cache_readonly  # 使用 cache_readonly 装饰器，将方法转换为只读缓存属性
    def is_signed_integer(self) -> bool:  # 判断是否为有符号整数类型的方法
        return self.kind == "i"  # 返回数据类型的种类是否为 'i'，即有符号整数

    @cache_readonly  # 使用 cache_readonly 装饰器，将方法转换为只读缓存属性
    def is_unsigned_integer(self) -> bool:  # 判断是否为无符号整数类型的方法
        return self.kind == "u"  # 返回数据类型的种类是否为 'u'，即无符号整数

    @property  # 将 _is_numeric 方法定义为属性
    def _is_numeric(self) -> bool:  # 判断数据类型是否为数值类型的方法
        return True  # 始终返回 True，表示该数据类型为数值类型

    def __from_arrow__(  # 定义 __from_arrow__ 方法，从 pyarrow 数组构造 IntegerArray/FloatingArray
        self, array: pyarrow.Array | pyarrow.ChunkedArray
    ) -> BaseMaskedArray:  # 方法接受 pyarrow.Array 或 pyarrow.ChunkedArray，并返回 BaseMaskedArray 对象
        """
        Construct IntegerArray/FloatingArray from pyarrow Array/ChunkedArray.
        """
        import pyarrow  # 再次导入 pyarrow，确保在方法内部可以使用它

        from pandas.core.arrays.arrow._arrow_utils import (  # 从 pandas.core.arrays.arrow._arrow_utils 中导入 pyarrow_array_to_numpy_and_mask 函数
            pyarrow_array_to_numpy_and_mask,  # 将 pyarrow 数组转换为 NumPy 数组和掩码的工具函数
        )

        array_class = self.construct_array_type()  # 使用当前数据类型构造数组类型

        pyarrow_type = pyarrow.from_numpy_dtype(self.type)  # 根据当前数据类型生成对应的 pyarrow 类型
        if not array.type.equals(pyarrow_type) and not pyarrow.types.is_null(  # 如果数组类型不等于 pyarrow_type，并且不是空类型
            array.type
        ):
            # test_from_arrow_type_error raise for string, but allow
            #  through itemsize conversion GH#31896
            rt_dtype = pandas_dtype(array.type.to_pandas_dtype())  # 将 pyarrow 类型转换为 pandas 数据类型
            if rt_dtype.kind not in "iuf":  # 如果转换后的数据类型的种类不是 'i', 'u', 'f' 中的任何一种
                # Could allow "c" or potentially disallow float<->int conversion,
                #  but at the moment we specifically test that uint<->int works
                raise TypeError(  # 抛出类型错误异常
                    f"Expected array of {self} type, got {array.type} instead"  # 提示期望的数组类型与实际数组类型不匹配
                )

            array = array.cast(pyarrow_type)  # 将数组类型强制转换为 pyarrow_type

        if isinstance(array, pyarrow.ChunkedArray):  # 如果数组是 ChunkedArray 类型
            # TODO this "if" can be removed when requiring pyarrow >= 10.0, which fixed
            # combine_chunks for empty arrays https://github.com/apache/arrow/pull/13757
            if array.num_chunks == 0:  # 如果 ChunkedArray 没有块
                array = pyarrow.array([], type=array.type)  # 创建一个空数组，类型与 array 相同
            else:
                array = array.combine_chunks()  # 合并数组的块

        data, mask = pyarrow_array_to_numpy_and_mask(array, dtype=self.numpy_dtype)  # 将 pyarrow 数组转换为 NumPy 数组和掩码
        return array_class(data.copy(), ~mask, copy=False)  # 返回构造的数组对象，数据拷贝，掩码取反，不复制数据

    @classmethod  # 类方法装饰器，用于定义类方法
    def _get_dtype_mapping(cls) -> Mapping[np.dtype, NumericDtype]:  # 获取数据类型映射的抽象方法
        raise AbstractMethodError(cls)  # 抛出抽象方法错误异常，子类必须实现该方法
    # 定义一个类方法，用于将输入的 dtype 参数标准化为 NumericDtype 类型
    @classmethod
    def _standardize_dtype(cls, dtype: NumericDtype | str | np.dtype) -> NumericDtype:
        """
        Convert a string representation or a numpy dtype to NumericDtype.
        """
        # 如果 dtype 是字符串并且以 "Int", "UInt", "Float" 开头，则将其转换为小写
        if isinstance(dtype, str) and (dtype.startswith(("Int", "UInt", "Float"))):
            # 避免因为 np.dtype("Int64") 而引发的 DeprecationWarning
            # 参考：https://github.com/numpy/numpy/pull/7476
            dtype = dtype.lower()

        # 如果 dtype 不是 NumericDtype 类型，则尝试使用映射表将其转换为 NumericDtype
        if not isinstance(dtype, NumericDtype):
            mapping = cls._get_dtype_mapping()
            try:
                dtype = mapping[np.dtype(dtype)]
            except KeyError as err:
                # 如果转换失败，抛出 ValueError 异常并包含错误信息
                raise ValueError(f"invalid dtype specified {dtype}") from err
        # 返回标准化后的 dtype
        return dtype

    # 定义一个类方法，用于安全地将 values 数组按指定的 dtype 进行类型转换
    @classmethod
    def _safe_cast(cls, values: np.ndarray, dtype: np.dtype, copy: bool) -> np.ndarray:
        """
        Safely cast the values to the given dtype.

        "safe" in this context means the casting is lossless.
        """
        # 抽象方法错误：在子类中应该实现这个方法
        raise AbstractMethodError(cls)
def _coerce_to_data_and_mask(
    values, dtype, copy: bool, dtype_cls: type[NumericDtype], default_dtype: np.dtype
):
    # 获取类型检查器
    checker = dtype_cls._checker

    mask = None  # 初始化掩码为空
    inferred_type = None  # 推断类型为空

    # 如果未指定 dtype 且 values 有 dtype 属性
    if dtype is None and hasattr(values, "dtype"):
        # 如果 values 的 dtype 符合类型检查器要求，则将其作为 dtype
        if checker(values.dtype):
            dtype = values.dtype

    # 如果指定了 dtype，则标准化该 dtype
    if dtype is not None:
        dtype = dtype_cls._standardize_dtype(dtype)

    # 构建数据类型的数组类型
    cls = dtype_cls.construct_array_type()

    # 如果 values 是 cls 的实例
    if isinstance(values, cls):
        # 提取数据和掩码
        values, mask = values._data, values._mask
        # 如果指定了 dtype，则将数据类型转换为指定的 numpy 数据类型
        if dtype is not None:
            values = values.astype(dtype.numpy_dtype, copy=False)

        # 如果需要拷贝，则复制数据和掩码
        if copy:
            values = values.copy()
            mask = mask.copy()
        return values, mask, dtype, inferred_type

    original = values  # 备份原始数据

    # 如果不需要拷贝，则将 values 转换为 numpy 数组
    if not copy:
        values = np.asarray(values)
    else:
        values = np.array(values, copy=copy)

    inferred_type = None  # 重置推断类型为空

    # 如果 values 的 dtype 是对象类型或字符串类型
    if values.dtype == object or is_string_dtype(values.dtype):
        # 推断数据类型
        inferred_type = lib.infer_dtype(values, skipna=True)
        # 如果推断出的类型是布尔型且未指定 dtype
        if inferred_type == "boolean" and dtype is None:
            name = dtype_cls.__name__.strip("_")
            # 抛出类型错误异常
            raise TypeError(f"{values.dtype} cannot be converted to {name}")

    # 如果 values 的 dtype 是布尔型且符合类型检查器要求
    elif values.dtype.kind == "b" and checker(dtype):
        # 如果不需要拷贝，则将 values 转换为指定的默认数据类型
        if not copy:
            values = np.asarray(values, dtype=default_dtype)
        else:
            values = np.array(values, dtype=default_dtype, copy=copy)

    # 如果 values 的 dtype 不是整型、浮点型或无符号整型
    elif values.dtype.kind not in "iuf":
        name = dtype_cls.__name__.strip("_")
        # 抛出类型错误异常
        raise TypeError(f"{values.dtype} cannot be converted to {name}")

    # 如果 values 不是一维列表样式
    if values.ndim != 1:
        # 抛出类型错误异常
        raise TypeError("values must be a 1D list-like")

    # 如果 mask 为空
    if mask is None:
        # 如果 values 的 dtype 是整型或无符号整型
        if values.dtype.kind in "iu":
            # 快速路径，生成全零掩码
            mask = np.zeros(len(values), dtype=np.bool_)
        else:
            # 使用库函数判断数值型 NA 值
            mask = libmissing.is_numeric_na(values)
    else:
        # 断言 mask 的长度与 values 相同
        assert len(mask) == len(values)

    # 如果 mask 不是一维列表样式
    if mask.ndim != 1:
        # 抛出类型错误异常
        raise TypeError("mask must be a 1D list-like")

    # 如果未指定 dtype，则使用默认数据类型
    if dtype is None:
        dtype = default_dtype
    else:
        # 否则将 dtype 转换为 numpy 数据类型
        dtype = dtype.numpy_dtype

    # 如果 dtype 是整数类型且 values 的 dtype 是浮点型且长度大于 0
    if is_integer_dtype(dtype) and values.dtype.kind == "f" and len(values) > 0:
        # 如果 mask 全为真
        if mask.all():
            # 生成全为 1 的数据，数据类型为 dtype
            values = np.ones(values.shape, dtype=dtype)
        else:
            # 找到最大值的索引
            idx = np.nanargmax(values)
            # 如果将 values[idx] 转换为整数后不等于 original[idx]
            if int(values[idx]) != original[idx]:
                # 数据在转换过程中失去精度
                inferred_type = lib.infer_dtype(original, skipna=True)
                # 如果推断类型不是浮点型或混合整数浮点型且 mask 没有任何真值
                if (
                    inferred_type not in ["floating", "mixed-integer-float"]
                    and not mask.any()
                ):
                    # 将 original 转换为指
    # 如果推断类型为字符串或Unicode
    if inferred_type in ("string", "unicode"):
        # 将数值转换为指定的数据类型，如果无法解析为浮点数则引发 ValueError
        values = values.astype(dtype, copy=copy)
    else:
        # 否则，使用数据类型类的安全类型转换方法进行转换
        values = dtype_cls._safe_cast(values, dtype, copy=False)

    # 返回转换后的数值，掩码，数据类型和推断类型
    return values, mask, dtype, inferred_type
    """
    Base class for IntegerArray and FloatingArray.
    """

    # 类型提示：指定_dtype_cls为NumericDtype类型
    _dtype_cls: type[NumericDtype]

    # 初始化方法，接收values（numpy数组）、mask（布尔类型的numpy数组）、copy（是否复制，默认为False）
    def __init__(
        self, values: np.ndarray, mask: npt.NDArray[np.bool_], copy: bool = False
    ) -> None:
        # 获取dtype类的检查器
        checker = self._dtype_cls._checker
        # 检查values是否为numpy数组且其dtype是否符合检查器的要求
        if not (isinstance(values, np.ndarray) and checker(values.dtype)):
            # 根据dtype类的kind属性决定描述符为'floating'或'integer'
            descr = (
                "floating"
                if self._dtype_cls.kind == "f"  # type: ignore[comparison-overlap]
                else "integer"
            )
            # 抛出类型错误，指示应该使用pd.array函数而不是传入的values
            raise TypeError(
                f"values should be {descr} numpy array. Use "
                "the 'pd.array' function instead"
            )
        # 如果values的dtype为np.float16，则抛出类型错误
        if values.dtype == np.float16:
            raise TypeError("FloatingArray does not support np.float16 dtype.")

        # 调用父类BaseMaskedArray的初始化方法，传入values、mask和copy参数
        super().__init__(values, mask, copy=copy)

    # 缓存修饰器，返回当前实例的数据类型NumericDtype
    @cache_readonly
    def dtype(self) -> NumericDtype:
        # 获取dtype类的dtype映射
        mapping = self._dtype_cls._get_dtype_mapping()
        # 返回当前数据的对应dtype
        return mapping[self._data.dtype]

    # 类方法，将value强制转换为数组，返回值和掩码
    @classmethod
    def _coerce_to_array(
        cls, value, *, dtype: DtypeObj, copy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # 获取类的dtype_cls属性
        dtype_cls = cls._dtype_cls
        # 获取dtype类的默认numpy数据类型
        default_dtype = dtype_cls._default_np_dtype
        # 使用_coerce_to_data_and_mask函数将value、dtype、copy、dtype_cls和default_dtype转换为数据和掩码
        values, mask, _, _ = _coerce_to_data_and_mask(
            value, dtype, copy, dtype_cls, default_dtype
        )
        # 返回转换后的values和mask
        return values, mask

    # 类方法，从字符串序列创建实例，接收strings（字符串序列）、dtype（扩展dtype）和copy（是否复制，默认为False）
    @classmethod
    def _from_sequence_of_strings(
        cls, strings, *, dtype: ExtensionDtype, copy: bool = False
    ) -> Self:
        # 导入to_numeric函数，用于将字符串转换为数值
        from pandas.core.tools.numeric import to_numeric

        # 使用to_numeric函数将strings转换为标量，如果出错则抛出异常，使用numpy_nullable作为dtype的后端
        scalars = to_numeric(strings, errors="raise", dtype_backend="numpy_nullable")
        # 调用_from_sequence方法，将标量转换为实例，并传入dtype和copy参数
        return cls._from_sequence(scalars, dtype=dtype, copy=copy)

    # 类属性，定义可以处理的数据类型为numpy数组和数字类型
    _HANDLED_TYPES = (np.ndarray, numbers.Number)
```