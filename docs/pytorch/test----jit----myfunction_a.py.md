# `.\pytorch\test\jit\myfunction_a.py`

```
"""
Helper function used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

# 从 jit.mydecorator 模块导入 my_decorator 装饰器
from jit.mydecorator import my_decorator
# 从 jit.myfunction_b 模块导入 my_function_b 函数
from jit.myfunction_b import my_function_b

# 使用 my_decorator 装饰器修饰 my_function_a 函数
@my_decorator
# 定义函数 my_function_a，接受一个 float 类型参数 x，返回一个 float 类型结果
def my_function_a(x: float) -> float:
    # 调用 my_function_b 函数，将 x 作为参数传入，然后将结果加上 1 返回
    return my_function_b(x) + 1
```