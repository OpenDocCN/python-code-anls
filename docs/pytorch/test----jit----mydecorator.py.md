# `.\pytorch\test\jit\mydecorator.py`

```py
"""
Decorator used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

# 导入 functools 模块，用于处理函数装饰器
import functools

# 定义名为 my_decorator 的装饰器函数，接受一个函数作为参数
def my_decorator(func):
    """Dummy decorator that removes itself when torchscripting"""

    # 使用 functools.wraps 装饰内部函数，保留原函数的元数据信息
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        # 调用被装饰的原函数，并返回其结果
        return func(*args, **kwargs)

    # torch.jit.script() 使用 __prepare_scriptable__ 方法来移除装饰器
    # 设置 wrapped_func 对象的 __prepare_scriptable__ 属性为一个 lambda 函数，其返回原函数 func
    wrapped_func.__prepare_scriptable__ = lambda: func

    # 返回经过装饰的函数对象 wrapped_func
    return wrapped_func
```