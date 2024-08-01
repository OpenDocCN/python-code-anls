# `.\DB-GPT-src\dbgpt\agent\resource\tool\exceptions.py`

```py
"""Exceptions for the tool."""

# 定义工具异常的基类，继承自内置的 Exception 类
class ToolException(Exception):
    """Common tool error exception."""

    # 初始化方法，接受消息和错误类型，默认为通用错误
    def __init__(self, message: str, error_type: str = "Common Error"):
        """Create a new ToolException instance."""
        # 调用父类 Exception 的初始化方法，传入消息
        super().__init__(message)
        # 设置实例的消息和错误类型属性
        self.message = message
        self.error_type = error_type

# 继承自 ToolException，表示创建工具时的异常
class CreateToolException(ToolException):
    """Create tool error exception."""

    # 初始化方法，接受消息和错误类型，默认为创建命令错误
    def __init__(self, message: str, error_type="Create Command Error"):
        """Create a new CreateToolException instance."""
        # 调用父类 ToolException 的初始化方法，传入消息和错误类型
        super().__init__(message, error_type)

# 继承自 ToolException，表示工具未找到的异常
class ToolNotFoundException(ToolException):
    """Tool not found exception."""

    # 初始化方法，接受消息和错误类型，默认为未找到命令错误
    def __init__(self, message: str, error_type="Not Command Error"):
        """Create a new ToolNotFoundException instance."""
        # 调用父类 ToolException 的初始化方法，传入消息和错误类型
        super().__init__(message, error_type)

# 继承自 ToolException，表示工具执行时的异常
class ToolExecutionException(ToolException):
    """Tool execution error exception."""

    # 初始化方法，接受消息和错误类型，默认为执行命令错误
    def __init__(self, message: str, error_type="Execution Command Error"):
        """Create a new ToolExecutionException instance."""
        # 调用父类 ToolException 的初始化方法，传入消息和错误类型
        super().__init__(message, error_type)
```