# `.\numpy\numpy\typing\tests\data\pass\ufunc_config.py`

```py
"""Typing tests for `numpy._core._ufunc_config`."""

import numpy as np  # 导入 numpy 库


def func1(a: str, b: int) -> None:
    return None  # 函数 func1 接受一个字符串和一个整数参数，无返回值


def func2(a: str, b: int, c: float = 1.0) -> None:
    return None  # 函数 func2 接受一个字符串和两个整数参数（其中一个有默认值），无返回值


def func3(a: str, b: int) -> int:
    return 0  # 函数 func3 接受一个字符串和一个整数参数，返回整数


class Write1:
    def write(self, a: str) -> None:
        return None  # 类 Write1 的方法 write 接受一个字符串参数，无返回值


class Write2:
    def write(self, a: str, b: int = 1) -> None:
        return None  # 类 Write2 的方法 write 接受一个字符串和一个整数参数（其中一个有默认值），无返回值


class Write3:
    def write(self, a: str) -> int:
        return 0  # 类 Write3 的方法 write 接受一个字符串参数，返回整数


_err_default = np.geterr()  # 获取当前 numpy 的错误处理设置
_bufsize_default = np.getbufsize()  # 获取当前 numpy 的缓冲区大小设置
_errcall_default = np.geterrcall()  # 获取当前 numpy 的错误回调设置

try:
    np.seterr(all=None)  # 设置所有错误为默认值
    np.seterr(divide="ignore")  # 设置除法相关错误为忽略
    np.seterr(over="warn")  # 设置溢出错误为警告
    np.seterr(under="call")  # 设置下溢错误为调用回调函数
    np.seterr(invalid="raise")  # 设置无效操作错误为抛出异常
    np.geterr()  # 获取当前 numpy 的错误处理设置

    np.setbufsize(4096)  # 设置 numpy 的缓冲区大小为 4096
    np.getbufsize()  # 获取当前 numpy 的缓冲区大小设置

    np.seterrcall(func1)  # 设置 numpy 的错误回调函数为 func1
    np.seterrcall(func2)  # 设置 numpy 的错误回调函数为 func2
    np.seterrcall(func3)  # 设置 numpy 的错误回调函数为 func3
    np.seterrcall(Write1())  # 设置 numpy 的错误回调函数为 Write1 类的实例
    np.seterrcall(Write2())  # 设置 numpy 的错误回调函数为 Write2 类的实例
    np.seterrcall(Write3())  # 设置 numpy 的错误回调函数为 Write3 类的实例
    np.geterrcall()  # 获取当前 numpy 的错误回调设置

    with np.errstate(call=func1, all="call"):  # 在上下文中设置错误状态，call 错误使用 func1 处理，所有错误都调用错误回调
        pass

    with np.errstate(call=Write1(), divide="log", over="log"):  # 在上下文中设置错误状态，call 错误使用 Write1 类的实例处理，divide 和 over 错误使用对数处理
        pass

finally:
    np.seterr(**_err_default)  # 恢复 numpy 的错误处理设置为默认值
    np.setbufsize(_bufsize_default)  # 恢复 numpy 的缓冲区大小设置为默认值
    np.seterrcall(_errcall_default)  # 恢复 numpy 的错误回调设置为默认值
```