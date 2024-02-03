# `.\AutoGPT\benchmark\agbenchmark\utils\logging.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import logging
from colorama import Fore, Style

# 定义简单日志格式和调试日志格式
SIMPLE_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
DEBUG_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)03d  %(message)s"

# 配置日志记录器
def configure_logging(
    level: int = logging.INFO,
) -> None:
    """Configure the native logging module."""

    # 根据日志级别自动调整默认日志格式
    log_format = DEBUG_LOG_FORMAT if level == logging.DEBUG else SIMPLE_LOG_FORMAT

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FancyConsoleFormatter(log_format))

    # 配置根记录器
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[console_handler],
    )

# 自定义控制台日志格式
class FancyConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter designed for console output.

    This formatter enhances the standard logging output with color coding. The color
    coding is based on the level of the log message, making it easier to distinguish
    between different types of messages in the console output.

    The color for each level is defined in the LEVEL_COLOR_MAP class attribute.
    """

    # 日志级别对应的颜色映射
    LEVEL_COLOR_MAP = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.BLUE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    # 格式化日志记录，返回格式化后的字符串
    def format(self, record: logging.LogRecord) -> str:
        # 确保 `msg` 是一个字符串
        if not hasattr(record, "msg"):
            record.msg = ""
        elif not type(record.msg) is str:
            record.msg = str(record.msg)

        # 将日志级别名称左对齐，最小宽度为5个字符
        record.levelname = record.levelname.ljust(5)

        # 根据错误级别确定默认颜色
        level_color = ""
        if record.levelno in self.LEVEL_COLOR_MAP:
            level_color = self.LEVEL_COLOR_MAP[record.levelno]
            record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        # 确定消息的颜色
        color = getattr(record, "color", level_color)
        color_is_specified = hasattr(record, "color")

        # 除非显式指定颜色，否则不要给 INFO 消息上色
        if color and (record.levelno != logging.INFO or color_is_specified):
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        # 调用父类的 format 方法，返回格式化后的字符串
        return super().format(record)
```