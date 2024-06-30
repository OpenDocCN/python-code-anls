# `D:\src\scipysrc\sympy\sympy\series\tests\__init__.py`

```
# 导入标准库中的 logging 模块
import logging

# 定义名为 'logger' 的日志记录器对象
logger = logging.getLogger(__name__)

# 设置日志记录级别为 DEBUG
logger.setLevel(logging.DEBUG)

# 创建一个日志处理器，用于将日志消息发送到控制台
ch = logging.StreamHandler()

# 定义日志格式，包括时间、日志级别、日志来源模块、具体日志消息
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# 将格式应用到日志处理器中
ch.setFormatter(formatter)

# 将日志处理器添加到日志记录器中
logger.addHandler(ch)

# 使用日志记录器输出不同级别的日志消息
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
```