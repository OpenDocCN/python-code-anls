# `.\AutoGPT\autogpts\autogpt\autogpt\logs\filters.py`

```py
# 导入 logging 模块
import logging

# 定义一个自定义的日志过滤器类 BelowLevelFilter
class BelowLevelFilter(logging.Filter):
    """Filter for logging levels below a certain threshold."""

    # 初始化方法，接受一个整数参数 below_level 作为过滤的阈值
    def __init__(self, below_level: int):
        super().__init__()
        self.below_level = below_level

    # 过滤方法，接受一个 logging.LogRecord 对象 record 作为参数
    def filter(self, record: logging.LogRecord):
        # 返回是否记录的日志级别小于设定的阈值
        return record.levelno < self.below_level
```