# `.\pytorch\torch\distributed\logging_handlers.py`

```py
#!/usr/bin/env python3

# 设置日志记录器的默认处理器为空处理器
# 这里创建了一个字典，用于存储日志处理器对象，键是处理器的名称，值是对应的处理器对象
# 初始情况下只定义了一个"default"键，对应的处理器是 logging.NullHandler()，用于防止日志输出
# 如果需要添加其他处理器，可以在这个字典中增加相应的键值对
import logging
from typing import Dict, List

__all__: List[str] = []  # 定义一个空列表，用于声明模块中的公开接口，初始为空

_log_handlers: Dict[str, logging.Handler] = {
    "default": logging.NullHandler(),  # 将"default"键对应的处理器设置为 NullHandler，即不输出任何日志
}
```