# `.\AutoGPT\autogpts\autogpt\autogpt\logs\__init__.py`

```py
# 从不同模块导入需要的函数和变量
from .config import configure_chat_plugins, configure_logging
from .helpers import user_friendly_output
from .log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    PROMPT_SUMMARY_FILE_NAME,
    PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME,
    SUMMARY_FILE_NAME,
    SUPERVISOR_FEEDBACK_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)

# 定义导出的所有函数和变量列表
__all__ = [
    "configure_logging",
    "configure_chat_plugins",
    "user_friendly_output",
    "CURRENT_CONTEXT_FILE_NAME",
    "NEXT_ACTION_FILE_NAME",
    "PROMPT_SUMMARY_FILE_NAME",
    "PROMPT_SUPERVISOR_FEEDBACK_FILE_NAME",
    "SUMMARY_FILE_NAME",
    "SUPERVISOR_FEEDBACK_FILE_NAME",
    "USER_INPUT_FILE_NAME",
    "LogCycleHandler",
]
```