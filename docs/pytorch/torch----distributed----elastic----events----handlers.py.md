# `.\pytorch\torch\distributed\elastic\events\handlers.py`

```
#!/usr/bin/env python3

# 导入日志模块
import logging
# 导入类型提示模块中的字典类型
from typing import Dict

# 全局变量，存储不同日志处理器对象
_log_handlers: Dict[str, logging.Handler] = {
    "console": logging.StreamHandler(),  # 控制台日志处理器
    "dynamic_rendezvous": logging.NullHandler(),  # 空日志处理器（不做任何操作）
    "null": logging.NullHandler(),  # 空日志处理器（不做任何操作）
}

# 定义函数，根据目标返回相应的日志处理器对象
def get_logging_handler(destination: str = "null") -> logging.Handler:
    global _log_handlers  # 引用全局变量 _log_handlers
    return _log_handlers[destination]  # 返回指定目标的日志处理器对象
```