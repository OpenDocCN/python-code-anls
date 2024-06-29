# `.\numpy\numpy\_typing\__init__.py`

```py
# 导入 `numpy.typing` 的私有模块。
from __future__ import annotations

# 导入 `ufunc` 模块。
from .. import ufunc

# 导入 `_utils` 模块中的 `set_module` 函数。
from .._utils import set_module

# 导入 `TYPE_CHECKING` 常量和 `final` 装饰器。
from typing import TYPE_CHECKING, final

# 使用 `@final` 装饰器，限制了不能创建任意的 `NBitBase` 子类。
@final
# 设置模块为 "numpy.typing"
@set_module("numpy.typing")
class NBitBase:
    """
    用于静态类型检查期间表示 `numpy.number` 精度的类型。

    仅用于静态类型检查，`NBitBase` 表示一个层级子类的基类。
    每个后续的子类用于表示更低精度的层级，例如 `64Bit > 32Bit > 16Bit`。

    .. versionadded:: 1.20

    Examples
    --------
    下面是一个典型的用法示例：`NBitBase` 用于注释一个函数，该函数接受任意精度的浮点数和整数作为参数，
    并返回最大精度的新浮点数（例如 `np.float16 + np.int64 -> np.float64`）。
    """

    # 检查是否子类的名称符合允许的命名规则
    def __init_subclass__(cls) -> None:
        # 允许的类名集合
        allowed_names = {
            "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
            "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        }
        # 如果子类的名称不在允许的名称集合中，则抛出错误
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        # 调用父类的初始化子类方法
        super().__init_subclass__()

# 禁止关于继承了 `@final` 装饰器类的子类化错误
class _256Bit(NBitBase):  # type: ignore[misc]
    pass

class _128Bit(_256Bit):  # type: ignore[misc]
    pass

class _96Bit(_128Bit):  # type: ignore[misc]
    pass

class _80Bit(_96Bit):  # type: ignore[misc]
    pass

class _64Bit(_80Bit):  # type: ignore[misc]
    pass

class _32Bit(_64Bit):  # type: ignore[misc]
    pass

class _16Bit(_32Bit):  # type: ignore[misc]
    pass

class _8Bit(_16Bit):  # type: ignore[misc]
    pass

# 导入 `_nested_sequence` 模块中的 `_NestedSequence` 类。
from ._nested_sequence import (
    _NestedSequence as _NestedSequence,
)

# 导入 `_nbit` 模块中的一系列 `_NBit*` 类。
from ._nbit import (
    _NBitByte as _NBitByte,
    _NBitShort as _NBitShort,
    _NBitIntC as _NBitIntC,
    _NBitIntP as _NBitIntP,
    _NBitInt as _NBitInt,
    _NBitLong as _NBitLong,
    _NBitLongLong as _NBitLongLong,
    _NBitHalf as _NBitHalf,
)
    # 导入模块中的 _NBitSingle 别名为 _NBitSingle
    _NBitSingle as _NBitSingle,
    # 导入模块中的 _NBitDouble 别名为 _NBitDouble
    _NBitDouble as _NBitDouble,
    # 导入模块中的 _NBitLongDouble 别名为 _NBitLongDouble
    _NBitLongDouble as _NBitLongDouble,
# 从._char_codes模块导入以下变量作为别名
from ._char_codes import (
    _BoolCodes as _BoolCodes,  # 布尔类型的字符编码
    _UInt8Codes as _UInt8Codes,  # 无符号8位整数的字符编码
    _UInt16Codes as _UInt16Codes,  # 无符号16位整数的字符编码
    _UInt32Codes as _UInt32Codes,  # 无符号32位整数的字符编码
    _UInt64Codes as _UInt64Codes,  # 无符号64位整数的字符编码
    _Int8Codes as _Int8Codes,  # 有符号8位整数的字符编码
    _Int16Codes as _Int16Codes,  # 有符号16位整数的字符编码
    _Int32Codes as _Int32Codes,  # 有符号32位整数的字符编码
    _Int64Codes as _Int64Codes,  # 有符号64位整数的字符编码
    _Float16Codes as _Float16Codes,  # 16位浮点数的字符编码
    _Float32Codes as _Float32Codes,  # 32位浮点数的字符编码
    _Float64Codes as _Float64Codes,  # 64位浮点数的字符编码
    _Complex64Codes as _Complex64Codes,  # 64位复数的字符编码
    _Complex128Codes as _Complex128Codes,  # 128位复数的字符编码
    _ByteCodes as _ByteCodes,  # 字节的字符编码
    _ShortCodes as _ShortCodes,  # 短整数的字符编码
    _IntCCodes as _IntCCodes,  # C语言整数的字符编码
    _IntPCodes as _IntPCodes,  # 平台相关整数的字符编码
    _IntCodes as _IntCodes,  # 整数的字符编码
    _LongCodes as _LongCodes,  # 长整数的字符编码
    _LongLongCodes as _LongLongCodes,  # 长长整数的字符编码
    _UByteCodes as _UByteCodes,  # 无符号字节的字符编码
    _UShortCodes as _UShortCodes,  # 无符号短整数的字符编码
    _UIntCCodes as _UIntCCodes,  # 无符号C语言整数的字符编码
    _UIntPCodes as _UIntPCodes,  # 无符号平台相关整数的字符编码
    _UIntCodes as _UIntCodes,  # 无符号整数的字符编码
    _ULongCodes as _ULongCodes,  # 无符号长整数的字符编码
    _ULongLongCodes as _ULongLongCodes,  # 无符号长长整数的字符编码
    _HalfCodes as _HalfCodes,  # 半精度浮点数的字符编码
    _SingleCodes as _SingleCodes,  # 单精度浮点数的字符编码
    _DoubleCodes as _DoubleCodes,  # 双精度浮点数的字符编码
    _LongDoubleCodes as _LongDoubleCodes,  # 长双精度浮点数的字符编码
    _CSingleCodes as _CSingleCodes,  # C语言单精度浮点数的字符编码
    _CDoubleCodes as _CDoubleCodes,  # C语言双精度浮点数的字符编码
    _CLongDoubleCodes as _CLongDoubleCodes,  # C语言长双精度浮点数的字符编码
    _DT64Codes as _DT64Codes,  # datetime64类型的字符编码
    _TD64Codes as _TD64Codes,  # timedelta64类型的字符编码
    _StrCodes as _StrCodes,  # 字符串的字符编码
    _BytesCodes as _BytesCodes,  # 字节串的字符编码
    _VoidCodes as _VoidCodes,  # 空类型的字符编码
    _ObjectCodes as _ObjectCodes,  # 对象类型的字符编码
)

# 从._scalars模块导入以下变量作为别名
from ._scalars import (
    _CharLike_co as _CharLike_co,  # 类似字符的协变类型的字符编码
    _BoolLike_co as _BoolLike_co,  # 类似布尔值的协变类型的字符编码
    _UIntLike_co as _UIntLike_co,  # 类似无符号整数的协变类型的字符编码
    _IntLike_co as _IntLike_co,  # 类似有符号整数的协变类型的字符编码
    _FloatLike_co as _FloatLike_co,  # 类似浮点数的协变类型的字符编码
    _ComplexLike_co as _ComplexLike_co,  # 类似复数的协变类型的字符编码
    _TD64Like_co as _TD64Like_co,  # 类似timedelta64的协变类型的字符编码
    _NumberLike_co as _NumberLike_co,  # 类似数字的协变类型的字符编码
    _ScalarLike_co as _ScalarLike_co,  # 类似标量的协变类型的字符编码
    _VoidLike_co as _VoidLike_co,  # 类似空类型的协变类型的字符编码
)

# 从._shape模块导入以下变量作为别名
from ._shape import (
    _Shape as _Shape,  # 形状的字符编码
    _ShapeLike as _ShapeLike,  # 类似形状的字符编码
)

# 从._dtype_like模块导入以下变量作为别名
from ._dtype_like import (
    DTypeLike as DTypeLike,  # 类似数据类型的字符编码
    _DTypeLike as _DTypeLike,  # 数据类型的字符编码
    _SupportsDType as _SupportsDType,  # 支持数据类型的字符编码
    _VoidDTypeLike as _VoidDTypeLike,  # 类似空数据类型的字符编码
    _DTypeLikeBool as _DTypeLikeBool,  # 类似布尔数据类型的字符编码
    _DTypeLikeUInt as _DTypeLikeUInt,  # 类似无符号整数数据类型的字符编码
    _DTypeLikeInt as _DTypeLikeInt,  # 类似有符号整数数据类型的字符编码
    _DTypeLikeFloat as _DTypeLikeFloat,  # 类似浮点数数据类型的字符编码
    _DTypeLikeComplex as _DTypeLikeComplex,  # 类似复数数据类型的字符编码
    _DTypeLikeTD64 as _DTypeLikeTD64,  # 类似timedelta64数据类型的字符编码
    _DTypeLikeDT64 as _DTypeLikeDT64,  # 类似datetime64数据类型的字符编码
    _DTypeLikeObject as _DTypeLikeObject,  # 类似对象数据类型的字符编码
    _DTypeLikeVoid as _DTypeLikeVoid,  # 类似空数据类型的字符编码
    _DTypeLikeStr as _DTypeLikeStr,  # 类似字符串数据类型的字符编码
    _DTypeLikeBytes as _DTypeLikeBytes,  # 类似字节串数据类型的字符编码
    _DTypeLikeComplex_co as _DTypeLikeComplex_co,  # 类似复数的协变类型的数据类型的字符编码
)

# 从._array_like模块导入以下变量作为别名
from ._array_like import (
    NDArray as NDArray,  # N维数组的字符编码
    ArrayLike as ArrayLike,  # 类似数组的字符编码
    _ArrayLike as _ArrayLike,  # 数组的字符编码
    _FiniteNestedSequence as _FiniteNestedSequence,  # 有限嵌套序列的字符编码
    _SupportsArray as _SupportsArray,  # 支持数组的字符编码
    _SupportsArrayFunc as _SupportsArrayFunc,  # 支持数组函数的字符编码
    _ArrayLikeInt as _ArrayLikeInt,  # 类似整数数组的字符编码
    _Array
    _ArrayLikeComplex_co as _ArrayLikeComplex_co,
    _ArrayLikeNumber_co as _ArrayLikeNumber_co,
    _ArrayLikeTD64_co as _ArrayLikeTD64_co,
    _ArrayLikeDT64_co as _ArrayLikeDT64_co,
    _ArrayLikeObject_co as _ArrayLikeObject_co,
    _ArrayLikeVoid_co as _ArrayLikeVoid_co,
    _ArrayLikeStr_co as _ArrayLikeStr_co,
    _ArrayLikeBytes_co as _ArrayLikeBytes_co,
    _ArrayLikeUnknown as _ArrayLikeUnknown,
    _UnknownType as _UnknownType,



# 定义类型别名，用于类型注解和提示
_ArrayLikeComplex_co as _ArrayLikeComplex_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeNumber_co as _ArrayLikeNumber_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeTD64_co as _ArrayLikeTD64_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeDT64_co as _ArrayLikeDT64_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeObject_co as _ArrayLikeObject_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeVoid_co as _ArrayLikeVoid_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeStr_co as _ArrayLikeStr_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeBytes_co as _ArrayLikeBytes_co,
# 定义类型别名，用于类型注解和提示
_ArrayLikeUnknown as _ArrayLikeUnknown,
# 定义类型别名，用于类型注解和提示
_UnknownType as _UnknownType,
if TYPE_CHECKING:
    # 如果是类型检查阶段，导入特定的 _ufunc 模块中的几个 ufunc 子类
    from ._ufunc import (
        _UFunc_Nin1_Nout1 as _UFunc_Nin1_Nout1,
        _UFunc_Nin2_Nout1 as _UFunc_Nin2_Nout1,
        _UFunc_Nin1_Nout2 as _UFunc_Nin1_Nout2,
        _UFunc_Nin2_Nout2 as _UFunc_Nin2_Nout2,
        _GUFunc_Nin2_Nout1 as _GUFunc_Nin2_Nout1,
    )
else:
    # 如果不是类型检查阶段，在运行时将 ufunc 子类声明为 ufunc 的别名
    # 这有助于像 Jedi 这样的自动补全工具 (numpy/numpy#19834)
    _UFunc_Nin1_Nout1 = ufunc
    _UFunc_Nin2_Nout1 = ufunc
    _UFunc_Nin1_Nout2 = ufunc
    _UFunc_Nin2_Nout2 = ufunc
    _GUFunc_Nin2_Nout1 = ufunc
```