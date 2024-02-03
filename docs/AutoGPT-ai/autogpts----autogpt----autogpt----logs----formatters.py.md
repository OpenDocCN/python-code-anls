# `.\AutoGPT\autogpts\autogpt\autogpt\logs\formatters.py`

```py
import logging
# 导入 logging 模块

from colorama import Style
# 从 colorama 模块中导入 Style

from google.cloud.logging_v2.handlers import CloudLoggingFilter, StructuredLogHandler
# 从 google.cloud.logging_v2.handlers 模块中导入 CloudLoggingFilter 和 StructuredLogHandler

from autogpt.core.runner.client_lib.logging import FancyConsoleFormatter
# 从 autogpt.core.runner.client_lib.logging 模块中导入 FancyConsoleFormatter

from .utils import remove_color_codes
# 从当前目录下的 utils 模块中导入 remove_color_codes 函数

class AutoGptFormatter(FancyConsoleFormatter):
    # 定义 AutoGptFormatter 类，继承自 FancyConsoleFormatter 类
    def __init__(self, *args, no_color: bool = False, **kwargs):
        # 初始化函数，接受可变参数和关键字参数
        super().__init__(*args, **kwargs)
        # 调用父类的初始化函数
        self.no_color = no_color
        # 设置是否使用颜色的标志

    def format(self, record: logging.LogRecord) -> str:
        # 定义 format 方法，接受 logging.LogRecord 对象，返回字符串
        # 确保 record.msg 是字符串
        if not hasattr(record, "msg"):
            record.msg = ""
        elif not type(record.msg) is str:
            record.msg = str(record.msg)

        # 剥离消息中的颜色代码，以防止颜色欺骗
        if record.msg and not getattr(record, "preserve_color", False):
            record.msg = remove_color_codes(record.msg)

        # 确定标题的颜色
        title = getattr(record, "title", "")
        title_color = getattr(record, "title_color", "") or self.LEVEL_COLOR_MAP.get(
            record.levelno, ""
        )
        if title and title_color:
            title = f"{title_color + Style.BRIGHT}{title}{Style.RESET_ALL}"
        # 确保 record.title 已设置，并在非空时用空格填充
        record.title = f"{title} " if title else ""

        if self.no_color:
            return remove_color_codes(super().format(record))
        else:
            return super().format(record)

class StructuredLoggingFormatter(StructuredLogHandler, logging.Formatter):
    # 定义 StructuredLoggingFormatter 类，继承自 StructuredLogHandler 和 logging.Formatter
    def __init__(self):
        # 初始化函数
        # 设置 CloudLoggingFilter 以向日志记录添加诊断信息
        self.cloud_logging_filter = CloudLoggingFilter()

        # 初始化 StructuredLogHandler
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        # 定义 format 方法，接受 logging.LogRecord 对象，返回字符串
        self.cloud_logging_filter.filter(record)
        # 使用 CloudLoggingFilter 过滤日志记录
        return super().format(record)
```