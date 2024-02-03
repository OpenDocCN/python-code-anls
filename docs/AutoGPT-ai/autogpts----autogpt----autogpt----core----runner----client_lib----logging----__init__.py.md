# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\client_lib\logging\__init__.py`

```py
# 导入 logging 模块
import logging

# 从当前目录下的 config 模块中导入 BelowLevelFilter, FancyConsoleFormatter, configure_root_logger 函数
from .config import BelowLevelFilter, FancyConsoleFormatter, configure_root_logger
# 从当前目录下的 helpers 模块中导入 dump_prompt 函数
from .helpers import dump_prompt

# 定义一个函数用于获取客户端日志记录器
def get_client_logger():
    # 在执行任何其他操作之前配置日志记录
    # 应用程序日志需要一个存放的地方
    client_logger = logging.getLogger("autogpt_client_application")
    # 设置客户端日志记录器的日志级别为 DEBUG
    client_logger.setLevel(logging.DEBUG)

    return client_logger

# 定义一个列表，包含了需要导出的函数和类
__all__ = [
    "configure_root_logger",
    "get_client_logger",
    "FancyConsoleFormatter",
    "BelowLevelFilter",
    "dump_prompt",
]
```