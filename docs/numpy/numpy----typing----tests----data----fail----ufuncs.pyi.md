# `.\numpy\numpy\typing\tests\data\fail\ufuncs.pyi`

```py
import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型注解

AR_f8: npt.NDArray[np.float64]  # 定义一个浮点64位类型的 NumPy 数组

np.sin.nin + "foo"  # E: Unsupported operand types  # 错误：不支持的操作类型，np.sin.nin 不是有效的属性

np.sin(1, foo="bar")  # E: No overload variant  # 错误：没有匹配的重载变体，np.sin 函数不接受名为 'foo' 的参数

np.abs(None)  # E: No overload variant  # 错误：没有匹配的重载变体，np.abs 函数不接受 None 作为参数

np.add(1, 1, 1)  # E: No overload variant  # 错误：没有匹配的重载变体，np.add 函数不接受三个参数的调用方式

np.add(1, 1, axis=0)  # E: No overload variant  # 错误：没有匹配的重载变体，np.add 函数不接受名为 'axis' 的关键字参数

np.matmul(AR_f8, AR_f8, where=True)  # E: No overload variant  # 错误：没有匹配的重载变体，np.matmul 函数不接受名为 'where' 的关键字参数

np.frexp(AR_f8, out=None)  # E: No overload variant  # 错误：没有匹配的重载变体，np.frexp 函数不接受名为 'out' 的关键字参数

np.frexp(AR_f8, out=AR_f8)  # E: No overload variant  # 错误：没有匹配的重载变体，np.frexp 函数不接受两个参数的调用方式

np.absolute.outer()  # E: "None" not callable  # 错误："None" 不可调用，np.absolute.outer 不是可调用对象

np.frexp.outer()  # E: "None" not callable  # 错误："None" 不可调用，np.frexp.outer 不是可调用对象

np.divmod.outer()  # E: "None" not callable  # 错误："None" 不可调用，np.divmod.outer 不是可调用对象

np.matmul.outer()  # E: "None" not callable  # 错误："None" 不可调用，np.matmul.outer 不是可调用对象

np.absolute.reduceat()  # E: "None" not callable  # 错误："None" 不可调用，np.absolute.reduceat 不是可调用对象

np.frexp.reduceat()  # E: "None" not callable  # 错误："None" 不可调用，np.frexp.reduceat 不是可调用对象

np.divmod.reduceat()  # E: "None" not callable  # 错误："None" 不可调用，np.divmod.reduceat 不是可调用对象

np.matmul.reduceat()  # E: "None" not callable  # 错误："None" 不可调用，np.matmul.reduceat 不是可调用对象

np.absolute.reduce()  # E: "None" not callable  # 错误："None" 不可调用，np.absolute.reduce 不是可调用对象

np.frexp.reduce()  # E: "None" not callable  # 错误："None" 不可调用，np.frexp.reduce 不是可调用对象

np.divmod.reduce()  # E: "None" not callable  # 错误："None" 不可调用，np.divmod.reduce 不是可调用对象

np.matmul.reduce()  # E: "None" not callable  # 错误："None" 不可调用，np.matmul.reduce 不是可调用对象

np.absolute.accumulate()  # E: "None" not callable  # 错误："None" 不可调用，np.absolute.accumulate 不是可调用对象

np.frexp.accumulate()  # E: "None" not callable  # 错误："None" 不可调用，np.frexp.accumulate 不是可调用对象

np.divmod.accumulate()  # E: "None" not callable  # 错误："None" 不可调用，np.divmod.accumulate 不是可调用对象

np.matmul.accumulate()  # E: "None" not callable  # 错误："None" 不可调用，np.matmul.accumulate 不是可调用对象

np.frexp.at()  # E: "None" not callable  # 错误："None" 不可调用，np.frexp.at 不是可调用对象

np.divmod.at()  # E: "None" not callable  # 错误："None" 不可调用，np.divmod.at 不是可调用对象

np.matmul.at()  # E: "None" not callable  # 错误："None" 不可调用，np.matmul.at 不是可调用对象
```