# `Bert-VITS2\tools\log.py`

```

"""
logger封装
"""
# 从loguru库中导入logger模块
from loguru import logger
# 导入sys模块

# 移除所有默认的处理器
# 移除所有默认的处理器，即移除所有已经存在的日志处理器
logger.remove()

# 自定义格式并添加到标准输出
# 自定义日志格式，并将日志输出到标准输出
log_format = (
    "<g>{time:MM-DD HH:mm:ss}</g> <lvl>{level:<9}</lvl>| {file}:{line} | {message}"
)

# 将自定义格式的日志添加到标准输出，包括格式、回溯信息和诊断信息
logger.add(sys.stdout, format=log_format, backtrace=True, diagnose=True)

```