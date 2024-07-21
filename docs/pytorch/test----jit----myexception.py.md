# `.\pytorch\test\jit\myexception.py`

```
"""
Define exceptions used in test_exception.py. We define them in a
separate file on purpose to make sure the fully qualified exception class name
is captured correctly in such cases.
"""

# 定义一个自定义异常类 MyKeyError，继承自内置的 KeyError 类
class MyKeyError(KeyError):
    # 空的 pass 语句，表示该异常类不需要额外的定制行为
    pass
```