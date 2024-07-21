# `.\pytorch\torch\distributed\_shard\sharded_tensor\logger.py`

```
#!/usr/bin/env python3

# 引入日志模块
import logging
# 引入类型提示模块
from typing import List, Tuple
# 从 Torch 分布式模块中引入日志处理函数
from torch.distributed._shard.sharded_tensor.logging_handlers import _log_handlers

# 空列表，用于模块级别的 __all__ 属性
__all__: List[str] = []

# 获取或创建日志记录器的私有函数
def _get_or_create_logger() -> logging.Logger:
    # 调用 _get_logging_handler() 获取日志处理器和处理器名称
    logging_handler, log_handler_name = _get_logging_handler()
    # 创建以特定格式命名的日志记录器对象
    logger = logging.getLogger(f"sharding-spec-{log_handler_name}")
    # 设置日志记录器的日志级别为 DEBUG
    logger.setLevel(logging.DEBUG)
    # 设置日志消息的格式
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    # 将格式应用到日志处理器中
    logging_handler.setFormatter(formatter)
    # 禁止日志记录器向上传播日志消息
    logger.propagate = False
    # 将日志处理器添加到日志记录器中
    logger.addHandler(logging_handler)
    # 返回配置好的日志记录器对象
    return logger

# 获取日志处理器的私有函数
def _get_logging_handler(
    destination: str = "default",
) -> Tuple[logging.Handler, str]:
    # 从预定义的日志处理器字典中获取指定目标的日志处理器对象
    log_handler = _log_handlers[destination]
    # 获取日志处理器对象的类名作为处理器名称
    log_handler_name = type(log_handler).__name__
    # 返回日志处理器对象和处理器名称的元组
    return (log_handler, log_handler_name)
```