# `D:\src\scipysrc\scipy\scipy\fft\_debug_backends.py`

```
# 导入 numpy 库，命名为 np
import numpy as np

# 定义一个名为 NumPyBackend 的类，用于作为 numpy.fft 的后端
class NumPyBackend:
    """Backend that uses numpy.fft"""
    # 定义类属性 __ua_domain__，指定为 "numpy.scipy.fft"
    __ua_domain__ = "numpy.scipy.fft"

    # 定义静态方法 __ua_function__，用于接收并处理函数调用
    @staticmethod
    def __ua_function__(method, args, kwargs):
        # 从 kwargs 中移除键为 "overwrite_x" 的项
        kwargs.pop("overwrite_x", None)

        # 获取 np.fft 中与 method 同名的函数对象
        fn = getattr(np.fft, method.__name__, None)
        # 如果找不到对应的函数，则返回 NotImplemented
        return (NotImplemented if fn is None
                else fn(*args, **kwargs))


# 定义一个名为 EchoBackend 的类，用于输出 __ua_function__ 的参数信息
class EchoBackend:
    """Backend that just prints the __ua_function__ arguments"""
    # 定义类属性 __ua_domain__，指定为 "numpy.scipy.fft"
    __ua_domain__ = "numpy.scipy.fft"

    # 定义静态方法 __ua_function__，用于输出方法调用的参数信息
    @staticmethod
    def __ua_function__(method, args, kwargs):
        # 输出方法对象、位置参数和关键字参数，并用换行符分隔
        print(method, args, kwargs, sep='\n')
```