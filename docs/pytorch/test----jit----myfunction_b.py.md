# `.\pytorch\test\jit\myfunction_b.py`

```py
"""
Helper function used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

# 从 jit.mydecorator 模块导入 my_decorator 装饰器
from jit.mydecorator import my_decorator

# 使用 my_decorator 装饰器装饰以下函数，用于测试装饰器在不同模块中的解析能力
@my_decorator
# 定义一个名为 my_function_b 的函数，接受一个浮点数参数 x，返回一个浮点数
def my_function_b(x: float) -> float:
    # 调用 my_function_c 函数，将其结果与 2 相加后返回
    return my_function_c(x) + 2

# 定义一个名为 my_function_c 的函数，接受一个浮点数参数 x，返回 x + 3 的结果
def my_function_c(x: float) -> float:
    return x + 3
```