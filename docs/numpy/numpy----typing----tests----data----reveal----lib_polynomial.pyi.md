# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\lib_polynomial.pyi`

```py
import sys
from typing import Any, NoReturn
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

# 如果 Python 版本 >= 3.11，导入 typing 中的 assert_type 函数
if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    # 否则，从 typing_extensions 中导入 assert_type 函数
    from typing_extensions import assert_type

# 定义一些特定类型的 NumPy 数组变量
AR_b: npt.NDArray[np.bool]  # 布尔类型数组
AR_u4: npt.NDArray[np.uint32]  # 32位无符号整数类型数组
AR_i8: npt.NDArray[np.int64]  # 64位整数类型数组
AR_f8: npt.NDArray[np.float64]  # 双精度浮点数类型数组
AR_c16: npt.NDArray[np.complex128]  # 复数类型数组
AR_O: npt.NDArray[np.object_]  # 对象类型数组

poly_obj: np.poly1d  # 多项式对象 poly_obj

# 使用 assert_type 函数检查 poly_obj 的成员变量类型
assert_type(poly_obj.variable, str)  # 多项式变量名应为字符串
assert_type(poly_obj.order, int)  # 多项式阶数应为整数
assert_type(poly_obj.o, int)  # 未知变量 'o' 的数据类型
assert_type(poly_obj.roots, npt.NDArray[Any])  # 多项式的根应为任意类型的 NumPy 数组
assert_type(poly_obj.r, npt.NDArray[Any])  # 未知变量 'r' 的数据类型
assert_type(poly_obj.coeffs, npt.NDArray[Any])  # 多项式系数应为任意类型的 NumPy 数组
assert_type(poly_obj.c, npt.NDArray[Any])  # 未知变量 'c' 的数据类型
assert_type(poly_obj.coef, npt.NDArray[Any])  # 多项式系数应为任意类型的 NumPy 数组
assert_type(poly_obj.coefficients, npt.NDArray[Any])  # 多项式系数应为任意类型的 NumPy 数组
assert_type(poly_obj.__hash__, None)  # 多项式对象的哈希值为 None

# 使用 assert_type 函数检查多项式对象的方法返回类型
assert_type(poly_obj(1), Any)  # 对多项式求值应返回任意类型
assert_type(poly_obj([1]), npt.NDArray[Any])  # 对多项式数组求值应返回任意类型的 NumPy 数组
assert_type(poly_obj(poly_obj), np.poly1d)  # 对多项式对象求值应返回另一个多项式对象

assert_type(len(poly_obj), int)  # 获取多项式长度应返回整数
assert_type(-poly_obj, np.poly1d)  # 对多项式取负应返回另一个多项式对象
assert_type(+poly_obj, np.poly1d)  # 对多项式取正应返回另一个多项式对象

# 使用 assert_type 函数检查多项式对象与标量的运算结果类型
assert_type(poly_obj * 5, np.poly1d)  # 多项式与标量相乘应返回另一个多项式对象
assert_type(5 * poly_obj, np.poly1d)  # 标量与多项式相乘应返回另一个多项式对象
assert_type(poly_obj + 5, np.poly1d)  # 多项式与标量相加应返回另一个多项式对象
assert_type(5 + poly_obj, np.poly1d)  # 标量与多项式相加应返回另一个多项式对象
assert_type(poly_obj - 5, np.poly1d)  # 多项式与标量相减应返回另一个多项式对象
assert_type(5 - poly_obj, np.poly1d)  # 标量与多项式相减应返回另一个多项式对象
assert_type(poly_obj**1, np.poly1d)  # 多项式的整数次幂应返回另一个多项式对象
assert_type(poly_obj**1.0, np.poly1d)  # 多项式的浮点数次幂应返回另一个多项式对象
assert_type(poly_obj / 5, np.poly1d)  # 多项式除以标量应返回另一个多项式对象
assert_type(5 / poly_obj, np.poly1d)  # 标量除以多项式应返回另一个多项式对象

assert_type(poly_obj[0], Any)  # 获取多项式的第一个元素类型应为任意类型
poly_obj[0] = 5  # 将多项式的第一个元素赋值为 5
assert_type(iter(poly_obj), Iterator[Any])  # 对多项式对象进行迭代应返回任意类型的迭代器
assert_type(poly_obj.deriv(), np.poly1d)  # 对多项式进行求导应返回另一个多项式对象
assert_type(poly_obj.integ(), np.poly1d)  # 对多项式进行积分应返回另一个多项式对象

# 使用 assert_type 函数检查 np.poly 和 np.polyint 函数的返回类型
assert_type(np.poly(poly_obj), npt.NDArray[np.floating[Any]])  # 从多项式创建系数数组应返回浮点数类型的 NumPy 数组
assert_type(np.poly(AR_f8), npt.NDArray[np.floating[Any]])  # 从数组创建多项式系数应返回浮点数类型的 NumPy 数组
assert_type(np.poly(AR_c16), npt.NDArray[np.floating[Any]])  # 从复数数组创建多项式系数应返回浮点数类型的 NumPy 数组

assert_type(np.polyint(poly_obj), np.poly1d)  # 对多项式进行不定积分应返回另一个多项式对象
assert_type(np.polyint(AR_f8), npt.NDArray[np.floating[Any]])  # 对浮点数类型数组进行不定积分应返回浮点数类型的 NumPy 数组
assert_type(np.polyint(AR_f8, k=AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 对复数数组进行不定积分应返回复数类型的 NumPy 数组
assert_type(np.polyint(AR_O, m=2), npt.NDArray[np.object_])  # 对对象类型数组进行不定积分应返回对象类型的 NumPy 数组

assert_type(np.polyder(poly_obj), np.poly1d)  # 对多项式进行求导数应返回另一个多项式对象
assert_type(np.polyder(AR_f8), npt.NDArray[np.floating[Any]])  # 对浮点数类型数组进行求导数应返回浮点数类型的 NumPy 数组
assert_type(np.polyder(AR_c16), npt.NDArray[np.complexfloating[Any, Any]])  # 对复数数组进行求导数应返回复数类型的 NumPy 数组
assert_type(np.polyder(AR_O, m=2), npt.NDArray[np.object_])  # 对对象类型数组进行求导数应返回对象类型的 NumPy 数组

# 使用 assert_type 函数检查 np.polyfit 函数的返回类型
assert_type(np.polyfit(AR_f8, AR_f8, 2), npt.NDArray[np.float64])  # 多项式拟合应返回浮点数类型的 NumPy 数组
assert_type(
    np.polyfit(AR_f8, AR_i8, 1, full=True),
    tuple[
        npt.NDArray[np.float64],  # 拟合系数数组应为浮点数类型的 NumPy 数组
        npt.NDArray[np.float64],  # 拟合残差数组应为浮点数类型的 NumPy 数组
        npt.NDArray[np.int32],    # 拟合秩数组应为 32 位整数类型的 NumPy 数组
        npt.NDArray[np.float64],  # 拟合奇异值数组应为浮点数类型的 NumPy 数组
        npt.NDArray[np.float64],  # 拟合条件数数组应为浮点数类型的 NumPy 数组
    ],
)
assert_type(
    np.polyfit(AR_u4, AR_f8, 1.0, cov="unscaled"),
    tuple[
        n
    # 定义一个包含五个元素的元组，每个元素都是 NumPy 数组，具有不同的数据类型
    tuple[
        npt.NDArray[np.complex128],  # 第一个元素，复数类型的 NumPy 数组
        npt.NDArray[np.float64],     # 第二个元素，64 位浮点数类型的 NumPy 数组
        npt.NDArray[np.int32],       # 第三个元素，32 位整数类型的 NumPy 数组
        npt.NDArray[np.float64],     # 第四个元素，64 位浮点数类型的 NumPy 数组
        npt.NDArray[np.float64],     # 第五个元素，64 位浮点数类型的 NumPy 数组
    ],
assert_type(
    # 对 np.polyfit 函数的调用，用于多项式拟合
    np.polyfit(AR_u4, AR_c16, 1.0, cov=True),
    # 期望返回的类型，一个元组，包含两个 np.complex128 类型的数组
    tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.complex128],
    ],
)

assert_type(
    # 对 np.polyval 函数的调用，用于计算多项式在给定点的值
    np.polyval(AR_b, AR_b),
    # 期望返回的类型，一个 np.int64 类型的数组
    npt.NDArray[np.int64]
)

assert_type(
    np.polyval(AR_u4, AR_b),
    npt.NDArray[np.unsignedinteger[Any]]
)

assert_type(
    np.polyval(AR_i8, AR_i8),
    npt.NDArray[np.signedinteger[Any]]
)

assert_type(
    np.polyval(AR_f8, AR_i8),
    npt.NDArray[np.floating[Any]]
)

assert_type(
    np.polyval(AR_i8, AR_c16),
    npt.NDArray[np.complexfloating[Any, Any]]
)

assert_type(
    np.polyval(AR_O, AR_O),
    npt.NDArray[np.object_]
)

assert_type(
    np.polyadd(poly_obj, AR_i8),
    np.poly1d
)

assert_type(
    np.polyadd(AR_f8, poly_obj),
    np.poly1d
)

assert_type(
    np.polyadd(AR_b, AR_b),
    npt.NDArray[np.bool]
)

assert_type(
    np.polyadd(AR_u4, AR_b),
    npt.NDArray[np.unsignedinteger[Any]]
)

assert_type(
    np.polyadd(AR_i8, AR_i8),
    npt.NDArray[np.signedinteger[Any]]
)

assert_type(
    np.polyadd(AR_f8, AR_i8),
    npt.NDArray[np.floating[Any]]
)

assert_type(
    np.polyadd(AR_i8, AR_c16),
    npt.NDArray[np.complexfloating[Any, Any]]
)

assert_type(
    np.polyadd(AR_O, AR_O),
    npt.NDArray[np.object_]
)

assert_type(
    np.polysub(poly_obj, AR_i8),
    np.poly1d
)

assert_type(
    np.polysub(AR_f8, poly_obj),
    np.poly1d
)

assert_type(
    np.polysub(AR_b, AR_b),
    NoReturn
)

assert_type(
    np.polysub(AR_u4, AR_b),
    npt.NDArray[np.unsignedinteger[Any]]
)

assert_type(
    np.polysub(AR_i8, AR_i8),
    npt.NDArray[np.signedinteger[Any]]
)

assert_type(
    np.polysub(AR_f8, AR_i8),
    npt.NDArray[np.floating[Any]]
)

assert_type(
    np.polysub(AR_i8, AR_c16),
    npt.NDArray[np.complexfloating[Any, Any]]
)

assert_type(
    np.polysub(AR_O, AR_O),
    npt.NDArray[np.object_]
)

assert_type(
    np.polymul(poly_obj, AR_i8),
    np.poly1d
)

assert_type(
    np.polymul(AR_f8, poly_obj),
    np.poly1d
)

assert_type(
    np.polymul(AR_b, AR_b),
    npt.NDArray[np.bool]
)

assert_type(
    np.polymul(AR_u4, AR_b),
    npt.NDArray[np.unsignedinteger[Any]]
)

assert_type(
    np.polymul(AR_i8, AR_i8),
    npt.NDArray[np.signedinteger[Any]]
)

assert_type(
    np.polymul(AR_f8, AR_i8),
    npt.NDArray[np.floating[Any]]
)

assert_type(
    np.polymul(AR_i8, AR_c16),
    npt.NDArray[np.complexfloating[Any, Any]]
)

assert_type(
    np.polymul(AR_O, AR_O),
    npt.NDArray[np.object_]
)

assert_type(
    np.polydiv(poly_obj, AR_i8),
    tuple[np.poly1d, np.poly1d]
)

assert_type(
    np.polydiv(AR_f8, poly_obj),
    tuple[np.poly1d, np.poly1d]
)

assert_type(
    np.polydiv(AR_b, AR_b),
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
)

assert_type(
    np.polydiv(AR_u4, AR_b),
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
)

assert_type(
    np.polydiv(AR_i8, AR_i8),
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
)

assert_type(
    np.polydiv(AR_f8, AR_i8),
    tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]
)

assert_type(
    np.polydiv(AR_i8, AR_c16),
    tuple[npt.NDArray[np.complexfloating[Any, Any]], npt.NDArray[np.complexfloating[Any, Any]]]
)

assert_type(
    np.polydiv(AR_O, AR_O),
    tuple[npt.NDArray[Any], npt.NDArray[Any]]
)
```