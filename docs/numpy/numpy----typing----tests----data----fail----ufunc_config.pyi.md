# `.\numpy\numpy\typing\tests\data\fail\ufunc_config.pyi`

```py
"""
Typing tests for `numpy._core._ufunc_config`.
"""

# 导入 NumPy 库
import numpy as np

# 定义一个函数 func1，接受一个字符串 a，一个整数 b，一个浮点数 c，返回空值
def func1(a: str, b: int, c: float) -> None: ...

# 定义一个函数 func2，接受一个字符串 a 和一个命名参数 b（必须通过关键字传递），返回空值
def func2(a: str, *, b: int) -> None: ...

# 定义一个类 Write1
class Write1:
    # 类 Write1 中的方法 write1，接受一个字符串 a，返回空值
    def write1(self, a: str) -> None: ...

# 定义一个类 Write2
class Write2:
    # 类 Write2 中的方法 write，接受两个字符串参数 a 和 b，返回空值
    def write(self, a: str, b: str) -> None: ...

# 定义一个类 Write3
class Write3:
    # 类 Write3 中的方法 write，接受一个命名参数 a（必须通过关键字传递），返回空值
    def write(self, *, a: str) -> None: ...

# 调用 NumPy 库中的 seterrcall 函数，传入 func1 函数作为参数，用于处理错误回调
np.seterrcall(func1)  # E: Argument 1 to "seterrcall" has incompatible type

# 调用 NumPy 库中的 seterrcall 函数，传入 func2 函数作为参数，用于处理错误回调
np.seterrcall(func2)  # E: Argument 1 to "seterrcall" has incompatible type

# 调用 NumPy 库中的 seterrcall 函数，传入 Write1 类的实例作为参数，用于处理错误回调
np.seterrcall(Write1())  # E: Argument 1 to "seterrcall" has incompatible type

# 调用 NumPy 库中的 seterrcall 函数，传入 Write2 类的实例作为参数，用于处理错误回调
np.seterrcall(Write2())  # E: Argument 1 to "seterrcall" has incompatible type

# 调用 NumPy 库中的 seterrcall 函数，传入 Write3 类的实例作为参数，用于处理错误回调
np.seterrcall(Write3())  # E: Argument 1 to "seterrcall" has incompatible type
```