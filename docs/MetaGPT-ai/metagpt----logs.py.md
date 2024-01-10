# `MetaGPT\metagpt\logs.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/1 12:41
@Author  : alexanderwu
@File    : logs.py
"""

# 导入所需的模块
import sys
from datetime import datetime
from functools import partial
# 导入日志模块
from loguru import logger as _logger
# 导入常量
from metagpt.const import METAGPT_ROOT

# 定义函数，调整日志级别
def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    """Adjust the log level to above level"""
    # 获取当前日期
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")

    # 移除之前的日志记录
    _logger.remove()
    # 添加控制台输出的日志记录
    _logger.add(sys.stderr, level=print_level)
    # 添加到文件的日志记录
    _logger.add(METAGPT_ROOT / f"logs/{formatted_date}.txt", level=logfile_level)
    return _logger

# 调用函数，定义日志记录器
logger = define_log_level()

# 定义函数，用于记录日志
def log_llm_stream(msg):
    _llm_stream_log(msg)

# 定义函数，设置日志记录的函数
def set_llm_stream_logfunc(func):
    global _llm_stream_log
    _llm_stream_log = func

# 设置默认的日志记录函数为 print
_llm_stream_log = partial(print, end="")

```