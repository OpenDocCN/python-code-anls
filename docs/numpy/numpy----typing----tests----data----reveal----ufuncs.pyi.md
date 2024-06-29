# `D:\src\scipysrc\numpy\numpy\typing\tests\data\reveal\ufuncs.pyi`

```py
import sys
from typing import Literal, Any  # 导入 Literal 和 Any 类型

import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 类型注解模块

if sys.version_info >= (3, 11):
    from typing import assert_type  # 如果 Python 版本 >= 3.11，导入 assert_type 函数
else:
    from typing_extensions import assert_type  # 否则，从 typing_extensions 导入 assert_type 函数

i8: np.int64  # 声明 i8 为 np.int64 类型
f8: np.float64  # 声明 f8 为 np.float64 类型
AR_f8: npt.NDArray[np.float64]  # 声明 AR_f8 为 np.float64 类型的 NumPy 数组
AR_i8: npt.NDArray[np.int64]  # 声明 AR_i8 为 np.int64 类型的 NumPy 数组

assert_type(np.absolute.__doc__, str)  # 断言 np.absolute.__doc__ 是字符串类型
assert_type(np.absolute.types, list[str])  # 断言 np.absolute.types 是字符串列表类型

assert_type(np.absolute.__name__, Literal["absolute"])  # 断言 np.absolute.__name__ 是字面量类型为 "absolute"
assert_type(np.absolute.ntypes, Literal[20])  # 断言 np.absolute.ntypes 是字面量类型为 20
assert_type(np.absolute.identity, None)  # 断言 np.absolute.identity 是 None 类型
assert_type(np.absolute.nin, Literal[1])  # 断言 np.absolute.nin 是字面量类型为 1
assert_type(np.absolute.nout, Literal[1])  # 断言 np.absolute.nout 是字面量类型为 1
assert_type(np.absolute.nargs, Literal[2])  # 断言 np.absolute.nargs 是字面量类型为 2
assert_type(np.absolute.signature, None)  # 断言 np.absolute.signature 是 None 类型
assert_type(np.absolute(f8), Any)  # 断言 np.absolute(f8) 是任意类型
assert_type(np.absolute(AR_f8), npt.NDArray[Any])  # 断言 np.absolute(AR_f8) 是 NumPy 数组的任意类型

assert_type(np.absolute.at(AR_f8, AR_i8), None)  # 断言 np.absolute.at(AR_f8, AR_i8) 是 None 类型

assert_type(np.add.__name__, Literal["add"])  # 断言 np.add.__name__ 是字面量类型为 "add"
assert_type(np.add.ntypes, Literal[22])  # 断言 np.add.ntypes 是字面量类型为 22
assert_type(np.add.identity, Literal[0])  # 断言 np.add.identity 是字面量类型为 0
assert_type(np.add.nin, Literal[2])  # 断言 np.add.nin 是字面量类型为 2
assert_type(np.add.nout, Literal[1])  # 断言 np.add.nout 是字面量类型为 1
assert_type(np.add.nargs, Literal[3])  # 断言 np.add.nargs 是字面量类型为 3
assert_type(np.add.signature, None)  # 断言 np.add.signature 是 None 类型
assert_type(np.add(f8, f8), Any)  # 断言 np.add(f8, f8) 是任意类型
assert_type(np.add(AR_f8, f8), npt.NDArray[Any])  # 断言 np.add(AR_f8, f8) 是 NumPy 数组的任意类型
assert_type(np.add.at(AR_f8, AR_i8, f8), None)  # 断言 np.add.at(AR_f8, AR_i8, f8) 是 None 类型
assert_type(np.add.reduce(AR_f8, axis=0), Any)  # 断言 np.add.reduce(AR_f8, axis=0) 是任意类型
assert_type(np.add.accumulate(AR_f8), npt.NDArray[Any])  # 断言 np.add.accumulate(AR_f8) 是 NumPy 数组的任意类型
assert_type(np.add.reduceat(AR_f8, AR_i8), npt.NDArray[Any])  # 断言 np.add.reduceat(AR_f8, AR_i8) 是 NumPy 数组的任意类型
assert_type(np.add.outer(f8, f8), Any)  # 断言 np.add.outer(f8, f8) 是任意类型
assert_type(np.add.outer(AR_f8, f8), npt.NDArray[Any])  # 断言 np.add.outer(AR_f8, f8) 是 NumPy 数组的任意类型

assert_type(np.frexp.__name__, Literal["frexp"])  # 断言 np.frexp.__name__ 是字面量类型为 "frexp"
assert_type(np.frexp.ntypes, Literal[4])  # 断言 np.frexp.ntypes 是字面量类型为 4
assert_type(np.frexp.identity, None)  # 断言 np.frexp.identity 是 None 类型
assert_type(np.frexp.nin, Literal[1])  # 断言 np.frexp.nin 是字面量类型为 1
assert_type(np.frexp.nout, Literal[2])  # 断言 np.frexp.nout 是字面量类型为 2
assert_type(np.frexp.nargs, Literal[3])  # 断言 np.frexp.nargs 是字面量类型为 3
assert_type(np.frexp.signature, None)  # 断言 np.frexp.signature 是 None 类型
assert_type(np.frexp(f8), tuple[Any, Any])  # 断言 np.frexp(f8) 是包含任意类型的元组
assert_type(np.frexp(AR_f8), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # 断言 np.frexp(AR_f8) 是包含 NumPy 数组的任意类型的元组

assert_type(np.divmod.__name__, Literal["divmod"])  # 断言 np.divmod.__name__ 是字面量类型为 "divmod"
assert_type(np.divmod.ntypes, Literal[15])  # 断言 np.divmod.ntypes 是字面量类型为 15
assert_type(np.divmod.identity, None)  # 断言 np.divmod.identity 是 None 类型
assert_type(np.divmod.nin, Literal[2])  # 断言 np.divmod.nin 是字面量类型为 2
assert_type(np.divmod.nout, Literal[2])  # 断言 np.divmod.nout 是字面量类型为 2
assert_type(np.divmod.nargs, Literal[4])  # 断言 np.divmod.nargs 是字面量类型为 4
assert_type(np.divmod.signature, None)  # 断言 np.divmod.signature 是 None 类型
assert_type(np.divmod(f8, f8), tuple[Any, Any])  # 断言 np.divmod(f8, f8) 是包含任意类型的元组
assert_type(np.divmod(AR_f8, f8), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # 断言 np.divmod(AR_f8, f8) 是包含 NumPy 数组的任意类型的元组

assert_type(np.matmul.__name__, Literal["matmul"])  # 断言 np.matmul.__name__ 是字面量类型为 "matmul"
assert_type(np.matmul.ntypes, Literal[19])  # 断言 np.matmul.ntypes 是字面量类型为 19
assert_type(np.matmul.identity, None)  # 断言 np.matmul.identity 是 None 类型
assert_type(np.matmul.nin, Literal[2])  # 断言 np.matmul.nin 是字面量类型为 2
assert_type(np.matmul.nout, Literal[1])  # 断言 np.matmul.nout 是字面量类型为 1
assert_type(np.matmul.nargs, Literal[3])  # 断言 np.matmul.nargs 是字面量类型为 3
assert_type(np.matmul.signature, Literal["(n?,k),(k,m?)->(n?,m?)"])  # 断言 np.matmul.signature 是字面量类型为 "(n?,k),(k,m?)->(n?,m?)"
assert_type(np.matmul.identity, None)  # 断言 np.matmul.identity 是 None 类型
assert_type(np.matmul(AR_f8, AR_f8), Any)  # 断言 np.matmul(AR_f8, AR_f8) 是任意类型
assert_type(np.matmul(AR_f8, AR_f8, axes=[(0, 1), (0, 1), (0, 1)]), Any)  # 断言 np.matmul(AR_f8, AR_f8, axes=[(0, 1), (0, 1), (0, 1)]) 是任意类型

assert_type(np.vecdot.__name__, Literal
# 确定 np.vecdot.nout 的类型为单一类型输出
assert_type(np.vecdot.nout, Literal[1])

# 确定 np.vecdot.nargs 的类型为接受三个参数
assert_type(np.vecdot.nargs, Literal[3])

# 确定 np.vecdot.signature 的签名为 "(n),(n)->()"
assert_type(np.vecdot.signature, Literal["(n),(n)->()"])

# 确定 np.vecdot.identity 的值为 None
assert_type(np.vecdot.identity, None)

# 确定 np.vecdot(AR_f8, AR_f8) 的返回类型为任意类型
assert_type(np.vecdot(AR_f8, AR_f8), Any)

# 确定 np.bitwise_count.__name__ 的值为 'bitwise_count'
assert_type(np.bitwise_count.__name__, Literal['bitwise_count'])

# 确定 np.bitwise_count.ntypes 的类型为 11 种
assert_type(np.bitwise_count.ntypes, Literal[11])

# 确定 np.bitwise_count.identity 的值为 None
assert_type(np.bitwise_count.identity, None)

# 确定 np.bitwise_count.nin 的类型为单一输入
assert_type(np.bitwise_count.nin, Literal[1])

# 确定 np.bitwise_count.nout 的类型为单一输出
assert_type(np.bitwise_count.nout, Literal[1])

# 确定 np.bitwise_count.nargs 的类型为接受两个参数
assert_type(np.bitwise_count.nargs, Literal[2])

# 确定 np.bitwise_count.signature 的值为 None
assert_type(np.bitwise_count.signature, None)

# 确定 np.bitwise_count.identity 的值为 None
assert_type(np.bitwise_count.identity, None)

# 确定 np.bitwise_count(i8) 的返回类型为任意类型
assert_type(np.bitwise_count(i8), Any)

# 确定 np.bitwise_count(AR_i8) 的返回类型为 NumPy 数组，元素类型为任意类型
assert_type(np.bitwise_count(AR_i8), npt.NDArray[Any])
```