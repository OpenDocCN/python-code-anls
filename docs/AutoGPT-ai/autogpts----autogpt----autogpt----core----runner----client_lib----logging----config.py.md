# `.\AutoGPT\autogpts\autogpt\autogpt\core\runner\client_lib\logging\config.py`

```py
# 导入 logging 模块，用于记录日志信息
import logging
# 导入 sys 模块，用于访问与 Python 解释器交互的变量
import sys
# 从 colorama 模块中导入 Fore 和 Style 类，用于控制输出文本的颜色和样式
from colorama import Fore, Style
# 从 openai._base_client 模块中导入 log 函数，用于记录 OpenAI 库的日志信息
from openai._base_client import log as openai_logger

# 定义简单的日志格式
SIMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s  %(message)s"
# 定义调试日志格式
DEBUG_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d  %(message)s"
)

# 配置根日志记录器
def configure_root_logger():
    # 创建自定义的控制台日志格式化器
    console_formatter = FancyConsoleFormatter(SIMPLE_LOG_FORMAT)

    # 创建输出到标准输出的日志处理器
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(logging.DEBUG)
    # 添加过滤器，只输出警告级别以下的日志信息
    stdout.addFilter(BelowLevelFilter(logging.WARNING))
    stdout.setFormatter(console_formatter)
    # 创建输出到标准错误的日志处理器
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(console_formatter)

    # 配置根日志记录器的日志级别和处理器
    logging.basicConfig(level=logging.DEBUG, handlers=[stdout, stderr])

    # 禁用 OpenAI 库的调试日志
    openai_logger.setLevel(logging.WARNING)

# 自定义的控制台日志格式化器类
class FancyConsoleFormatter(logging.Formatter):
    """
    A custom logging formatter designed for console output.

    This formatter enhances the standard logging output with color coding. The color
    coding is based on the level of the log message, making it easier to distinguish
    between different types of messages in the console output.

    The color for each level is defined in the LEVEL_COLOR_MAP class attribute.
    """

    # 定义日志级别与颜色的映射关系
    # 日志级别 -> (文本颜色, 标题颜色)
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

        # 根据错误级别确定默认颜色
        level_color = ""
        if record.levelno in self.LEVEL_COLOR_MAP:
            level_color = self.LEVEL_COLOR_MAP[record.levelno]
            record.levelname = f"{level_color}{record.levelname}{Style.RESET_ALL}"

        # 确定消息的颜色
        color = getattr(record, "color", level_color)
        color_is_specified = hasattr(record, "color")

        # 除非颜色被明确指定，否则不要给 INFO 消息上色
        if color and (record.levelno != logging.INFO or color_is_specified):
            record.msg = f"{color}{record.msg}{Style.RESET_ALL}"

        # 返回父类的格式化结果
        return super().format(record)
# 定义一个自定义的日志过滤器类，用于过滤低于特定级别的日志消息
class BelowLevelFilter(logging.Filter):
    """Filter for logging levels below a certain threshold."""

    # 初始化方法，接受一个整数参数作为低于的级别阈值
    def __init__(self, below_level: int):
        super().__init__()
        # 将传入的级别阈值保存在实例变量中
        self.below_level = below_level

    # 过滤方法，接受一个日志记录对象作为参数
    def filter(self, record: logging.LogRecord):
        # 返回一个布尔值，表示该日志记录的级别是否低于设定的阈值
        return record.levelno < self.below_level
```