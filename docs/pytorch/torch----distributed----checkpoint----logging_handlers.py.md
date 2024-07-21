# `.\pytorch\torch\distributed\checkpoint\logging_handlers.py`

```py
# 导入日志模块
import logging
# 导入列表类型的类型提示
from typing import List
# 从 torch 分布式日志处理程序中导入私有变量 _log_handlers
from torch.distributed.logging_handlers import _log_handlers

# 定义一个空列表，用于存放公开的变量名
__all__: List[str] = []

# 定义一个常量，用于表示日志记录器的名称为 "dcp_logger"
DCP_LOGGER_NAME = "dcp_logger"

# 更新 _log_handlers 字典，将 DCP_LOGGER_NAME 映射到一个空的日志处理程序 (NullHandler)
_log_handlers.update(
    {
        DCP_LOGGER_NAME: logging.NullHandler(),
    }
)
```