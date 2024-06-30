# `D:\src\scipysrc\seaborn\seaborn\_core\exceptions.py`

```
"""
Custom exceptions for the seaborn.objects interface.

This is very lightweight, but it's a separate module to avoid circular imports.

"""
# 导入用于未来注释的特性
from __future__ import annotations

# 定义一个自定义异常类 PlotSpecError，继承自 RuntimeError
class PlotSpecError(RuntimeError):
    """
    Error class raised from seaborn.objects.Plot for compile-time failures.

    In the declarative Plot interface, exceptions may not be triggered immediately
    by bad user input (and validation at input time may not be possible). This class
    is used to signal that indirect dependency. It should be raised in an exception
    chain when compile-time operations fail with an error message providing useful
    context (e.g., scaling errors could specify the variable that failed.)

    """
    
    # 类方法 _during，用于报告特定操作的失败
    @classmethod
    def _during(cls, step: str, var: str = "") -> PlotSpecError:
        """
        Initialize the class to report the failure of a specific operation.
        """
        # 初始化消息列表
        message = []
        # 如果有变量 var，添加包含变量名的失败信息
        if var:
            message.append(f"{step} failed for the `{var}` variable.")
        else:
            message.append(f"{step} failed.")
        # 添加信息，指示查看上面的回溯以获取更多信息
        message.append("See the traceback above for more information.")
        # 返回初始化的 PlotSpecError 实例，包含所有消息的字符串
        return cls(" ".join(message))
```