# `.\pytorch\torch\distributed\_shard\sharded_tensor\logging_handlers.py`

```
#!/usr/bin/env python3
# 指定 Python 解释器路径

# 引入日志模块
import logging
# 引入用于类型提示的工具
from typing import Dict, List

# 定义模块内可导出的符号列表为空
__all__: List[str] = []

# 定义日志处理器字典，初始包含一个默认键值对
_log_handlers: Dict[str, logging.Handler] = {
    "default": logging.NullHandler(),
}
```