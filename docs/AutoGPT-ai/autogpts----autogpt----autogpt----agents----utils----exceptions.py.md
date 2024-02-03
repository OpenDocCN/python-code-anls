# `.\AutoGPT\autogpts\autogpt\autogpt\agents\utils\exceptions.py`

```py
# 导入 Optional 类型用于定义可选参数
from typing import Optional

# 定义 AgentException 类，作为执行代理程序中特定异常的基类
class AgentException(Exception):
    """Base class for specific exceptions relevant in the execution of Agents"""

    # 异常消息
    message: str

    # 可选的提示信息，可传递给 LLM 以减少此错误的再次发生
    hint: Optional[str] = None

    # 初始化方法，接受消息和其他参数
    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)

# AgentTerminated 类，代理程序终止或被终止
class AgentTerminated(AgentException):
    """The agent terminated or was terminated"""

# ConfigurationError 类，由于配置无效、不兼容或其他不正确导致的错误
class ConfigurationError(AgentException):
    """Error caused by invalid, incompatible or otherwise incorrect configuration"""

# InvalidAgentResponseError 类，LLM 偏离了规定的响应格式
class InvalidAgentResponseError(AgentException):
    """The LLM deviated from the prescribed response format"""

# UnknownCommandError 类，AI 尝试使用未知命令
class UnknownCommandError(AgentException):
    """The AI tried to use an unknown command"""

    # 提示信息，不要再尝试使用此命令
    hint = "Do not try to use this command again."

# DuplicateOperationError 类，提议的操作已经执行过
class DuplicateOperationError(AgentException):
    """The proposed operation has already been executed"""

# CommandExecutionError 类，尝试执行命令时发生错误
class CommandExecutionError(AgentException):
    """An error occurred when trying to execute the command"""

# InvalidArgumentError 类，命令接收到无效参数
class InvalidArgumentError(CommandExecutionError):
    """The command received an invalid argument"""

# OperationNotAllowedError 类，代理程序不允许执行提议的操作
class OperationNotAllowedError(CommandExecutionError):
    """The agent is not allowed to execute the proposed operation"""

# AccessDeniedError 类，由于拒绝访问所需资源，操作失败
class AccessDeniedError(CommandExecutionError):
    """The operation failed because access to a required resource was denied"""

# CodeExecutionError 类，操作（尝试运行任意代码）返回错误
class CodeExecutionError(CommandExecutionError):
    """The operation (an attempt to run arbitrary code) returned an error"""

# TooMuchOutputError 类，操作生成的输出超过代理程序可以处理的量
class TooMuchOutputError(CommandExecutionError):
    """The operation generated more output than what the Agent can process"""
```