# `.\pytorch\torch\_export\error.py`

```py
# 导入枚举类 Enum
from enum import Enum

# 定义 ExportErrorType 枚举类，用于表示导出错误的类型
class ExportErrorType(Enum):
    # 用户提供给追踪器或其他公共 API 的无效输入
    INVALID_INPUT_TYPE = 1

    # 用户从其模型返回我们不支持的值
    INVALID_OUTPUT_TYPE = 2

    # 生成的 IR 不符合导出 IR 规范
    VIOLATION_OF_SPEC = 3

    # 用户的代码包含我们不支持的类型和功能
    NOT_SUPPORTED = 4

    # 用户的代码没有提供必要的细节，以便我们成功追踪和导出
    # 例如，我们使用许多装饰器，并要求用户为其模型进行注释
    MISSING_PROPERTY = 5

    # 用户在没有正确初始化的情况下使用 API
    UNINITIALIZED = 6


def internal_assert(pred: bool, assert_msg: str) -> None:
    """
    这是 exir 的自定义断言方法。在内部只抛出 InternalError。
    注意，其唯一目的是在保持类似 Python assert 语法的同时抛出我们自己的错误。
    """
    # 如果断言条件不满足，则抛出 InternalError 异常
    if not pred:
        raise InternalError(assert_msg)


class InternalError(Exception):
    """
    当在 EXIR 栈中违反内部不变性时引发。
    应提示用户向开发人员报告 Bug，并公开原始错误消息。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ExportError(Exception):
    """
    这种类型的异常是由用户代码直接引起的错误。
    通常在模型编写、追踪、使用我们的公共 API 和编写图形传递时发生用户错误。
    """

    def __init__(self, error_code: ExportErrorType, message: str) -> None:
        # 构造异常消息，包含错误码和用户提供的消息
        prefix = f"[{error_code}]: "
        super().__init__(prefix + message)
```