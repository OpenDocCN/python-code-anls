# `D:\src\scipysrc\scipy\scipy\special\_sf_error.py`

```
"""Warnings and Exceptions that can be raised by special functions."""
# 导入警告模块，用于处理警告信息
import warnings

# 定义一个特殊函数警告类，继承自内置的警告类Warning
class SpecialFunctionWarning(Warning):
    """Warning that can be emitted by special functions."""
    # 这是一个用于特殊函数可能发出的警告的警告类

# 设置警告过滤器，始终显示特定类别（SpecialFunctionWarning）的警告信息
warnings.simplefilter("always", category=SpecialFunctionWarning)

# 定义一个特殊函数错误类，继承自内置的异常类Exception
class SpecialFunctionError(Exception):
    """Exception that can be raised by special functions."""
    # 这是一个用于特殊函数可能抛出的异常的异常类
```