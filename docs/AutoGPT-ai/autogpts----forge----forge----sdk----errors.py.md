# `.\AutoGPT\autogpts\forge\forge\sdk\errors.py`

```py
# 导入 Optional 类型
from typing import Optional

# 定义 NotFoundError 异常类
class NotFoundError(Exception):
    pass

# 定义 AgentException 异常类，作为执行代理程序中特定异常的基类
class AgentException(Exception):
    """Base class for specific exceptions relevant in the execution of Agents"""

    message: str

    hint: Optional[str] = None
    """A hint which can be passed to the LLM to reduce reoccurrence of this error"""

    # 初始化 AgentException 异常类
    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)

# 定义 ConfigurationError 异常类，表示由于配置无效、不兼容或其他不正确导致的错误
class ConfigurationError(AgentException):
    """Error caused by invalid, incompatible or otherwise incorrect configuration"""

# 定义 InvalidAgentResponseError 异常类，表示 LLM 偏离了规定的响应格式
class InvalidAgentResponseError(AgentException):
    """The LLM deviated from the prescribed response format"""

# 定义 UnknownCommandError 异常类，表示 AI 尝试使用未知命令
class UnknownCommandError(AgentException):
    """The AI tried to use an unknown command"""
    
    hint = "Do not try to use this command again."

# 定义 DuplicateOperationError 异常类，表示已经执行过的操作再次被提出
class DuplicateOperationError(AgentException):
    """The proposed operation has already been executed"""

# 定义 CommandExecutionError 异常类，表示尝试执行命令时发生错误
class CommandExecutionError(AgentException):
    """An error occured when trying to execute the command"""

# 定义 InvalidArgumentError 异常类，表示命令接收到无效参数
class InvalidArgumentError(CommandExecutionError):
    """The command received an invalid argument"""

# 定义 OperationNotAllowedError 异常类，表示代理程序不允许执行提出的操作
class OperationNotAllowedError(CommandExecutionError):
    """The agent is not allowed to execute the proposed operation"""

# 定义 AccessDeniedError 异常类，表示由于被拒绝访问所需资源而导致的操作失败
class AccessDeniedError(CommandExecutionError):
    """The operation failed because access to a required resource was denied"""

# 定义 CodeExecutionError 异常类，表示操作（尝试运行任意代码）返回错误
class CodeExecutionError(CommandExecutionError):
    """The operation (an attempt to run arbitrary code) returned an error"""

# 定义 TooMuchOutputError 异常类，表示操作生成的输出超过代理程序可以处理的量
class TooMuchOutputError(CommandExecutionError):
    """The operation generated more output than what the Agent can process"""
```