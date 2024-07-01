# `.\numpy\numpy\typing\tests\data\reveal\type_check.pyi`

```py
import sys  # 导入sys模块，用于访问系统相关的信息
from typing import Any, Literal  # 导入typing模块中的Any和Literal类型

import numpy as np  # 导入NumPy库，并使用"np"作为别名
import numpy.typing as npt  # 导入NumPy的类型提示模块
from numpy._typing import _16Bit, _32Bit, _64Bit, _128Bit  # 导入NumPy的特定位数类型定义

if sys.version_info >= (3, 11):
    from typing import assert_type  # 根据Python版本导入不同版本的assert_type
else:
    from typing_extensions import assert_type  # 兼容Python旧版本导入assert_type

f8: np.float64  # 声明变量f8为NumPy中的64位浮点数类型
f: float  # 声明变量f为Python内置的浮点数类型

# 定义不同类型的NumPy数组
AR_i8: npt.NDArray[np.int64]
AR_i4: npt.NDArray[np.int32]
AR_f2: npt.NDArray[np.float16]
AR_f8: npt.NDArray[np.float64]
AR_f16: npt.NDArray[np.floating[_128Bit]]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]

AR_LIKE_f: list[float]  # 声明AR_LIKE_f为浮点数列表

class RealObj:
    real: slice  # 定义RealObj类包含一个slice类型的属性real

class ImagObj:
    imag: slice  # 定义ImagObj类包含一个slice类型的属性imag

assert_type(np.mintypecode(["f8"], typeset="qfQF"), str)  # 断言函数返回类型为字符串

assert_type(np.real(RealObj()), slice)  # 断言np.real函数应返回slice类型
assert_type(np.real(AR_f8), npt.NDArray[np.float64])  # 断言np.real函数应接受AR_f8并返回64位浮点数数组类型
assert_type(np.real(AR_c16), npt.NDArray[np.float64])  # 断言np.real函数应接受AR_c16并返回64位浮点数数组类型
assert_type(np.real(AR_LIKE_f), npt.NDArray[Any])  # 断言np.real函数应接受AR_LIKE_f并返回任意类型的NumPy数组

assert_type(np.imag(ImagObj()), slice)  # 断言np.imag函数应返回slice类型
assert_type(np.imag(AR_f8), npt.NDArray[np.float64])  # 断言np.imag函数应接受AR_f8并返回64位浮点数数组类型
assert_type(np.imag(AR_c16), npt.NDArray[np.float64])  # 断言np.imag函数应接受AR_c16并返回64位浮点数数组类型
assert_type(np.imag(AR_LIKE_f), npt.NDArray[Any])  # 断言np.imag函数应接受AR_LIKE_f并返回任意类型的NumPy数组

assert_type(np.iscomplex(f8), np.bool)  # 断言np.iscomplex函数应接受f8并返回布尔类型
assert_type(np.iscomplex(AR_f8), npt.NDArray[np.bool])  # 断言np.iscomplex函数应接受AR_f8并返回布尔数组类型
assert_type(np.iscomplex(AR_LIKE_f), npt.NDArray[np.bool])  # 断言np.iscomplex函数应接受AR_LIKE_f并返回布尔数组类型

assert_type(np.isreal(f8), np.bool)  # 断言np.isreal函数应接受f8并返回布尔类型
assert_type(np.isreal(AR_f8), npt.NDArray[np.bool])  # 断言np.isreal函数应接受AR_f8并返回布尔数组类型
assert_type(np.isreal(AR_LIKE_f), npt.NDArray[np.bool])  # 断言np.isreal函数应接受AR_LIKE_f并返回布尔数组类型

assert_type(np.iscomplexobj(f8), bool)  # 断言np.iscomplexobj函数应接受f8并返回布尔类型
assert_type(np.isrealobj(f8), bool)  # 断言np.isrealobj函数应接受f8并返回布尔类型

assert_type(np.nan_to_num(f8), np.float64)  # 断言np.nan_to_num函数应接受f8并返回64位浮点数类型
assert_type(np.nan_to_num(f, copy=True), Any)  # 断言np.nan_to_num函数应接受f并返回任意类型，参数copy设置为True
assert_type(np.nan_to_num(AR_f8, nan=1.5), npt.NDArray[np.float64])  # 断言np.nan_to_num函数应接受AR_f8并返回64位浮点数数组类型，参数nan设置为1.5
assert_type(np.nan_to_num(AR_LIKE_f, posinf=9999), npt.NDArray[Any])  # 断言np.nan_to_num函数应接受AR_LIKE_f并返回任意类型的NumPy数组，参数posinf设置为9999

assert_type(np.real_if_close(AR_f8), npt.NDArray[np.float64])  # 断言np.real_if_close函数应接受AR_f8并返回64位浮点数数组类型
assert_type(np.real_if_close(AR_c16), npt.NDArray[np.float64] | npt.NDArray[np.complex128])  # 断言np.real_if_close函数应接受AR_c16并返回64位浮点数数组类型或128位复数数组类型
assert_type(np.real_if_close(AR_c8), npt.NDArray[np.float32] | npt.NDArray[np.complex64])  # 断言np.real_if_close函数应接受AR_c8并返回32位浮点数数组类型或64位复数数组类型
assert_type(np.real_if_close(AR_LIKE_f), npt.NDArray[Any])  # 断言np.real_if_close函数应接受AR_LIKE_f并返回任意类型的NumPy数组

assert_type(np.typename("h"), Literal["short"])  # 断言np.typename函数应接受"h"并返回"short"字面值类型
assert_type(np.typename("B"), Literal["unsigned char"])  # 断言np.typename函数应接受"B"并返回"unsigned char"字面值类型
assert_type(np.typename("V"), Literal["void"])  # 断言np.typename函数应接受"V"并返回"void"字面值类型
assert_type(np.typename("S1"), Literal["character"])  # 断言np.typename函数应接受"S1"并返回"character"字面值类型

assert_type(np.common_type(AR_i4), type[np.float64])  # 断言np.common_type函数应接受AR_i4并返回np.float64类型
assert_type(np.common_type(AR_f2), type[np.float16])  # 断言np.common_type函数应接受AR_f2并返回np.float16类型
assert_type(np.common_type(AR_f2, AR_i4), type[np.floating[_16Bit | _64Bit]])  # 断言np.common_type函数应接受AR_f2和AR_i4并返回np.floating[16位或64位]类型
assert_type(np.common_type(AR_f16, AR_i4), type[np.floating[_64Bit | _128Bit]])  # 断言np.common_type函数应接受AR_f16和AR_i4并返回np.floating[64位或128位]类型
assert_type(
    np.common_type(AR_c8, AR_f2),
    type[np.complexfloating[_16Bit | _32Bit, _16Bit | _32Bit]],
)  # 断言np.common_type函数应接受AR_c8和AR_f2并返回np.complexfloating[16位或32位, 16位或32位]类型
assert_type(
    np.common_type(AR_f2, AR_c8, AR_i4),
    type[np.complexfloating[_16Bit | _32Bit | _64Bit, _16Bit | _32Bit | _64Bit]],
)  # 断言np.common_type函数应接受AR_f2、AR_c8和AR_i4并返回np.complexfloating[16位或32位或64位, 16位或32位或64位]类型
```