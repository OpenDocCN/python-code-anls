# `.\DB-GPT-src\dbgpt\agent\core\schema.py`

```py
"""Schema definition for the agent."""
# 引入枚举模块
from enum import Enum

# 插件存储类型枚举
class PluginStorageType(Enum):
    """Plugin storage type."""
    # Git 存储类型
    Git = "git"
    # OSS 存储类型
    Oss = "oss"

# API 标签类型枚举
class ApiTagType(Enum):
    """API tag type."""
    # API 视图标签
    API_VIEW = "dbgpt_view"
    # API 调用标签
    API_CALL = "dbgpt_call"

# 任务状态枚举
class Status(Enum):
    """Status of a task."""
    # 待办状态
    TODO = "todo"
    # 运行中状态
    RUNNING = "running"
    # 等待中状态
    WAITING = "waiting"
    # 重试中状态
    RETRYING = "retrying"
    # 失败状态
    FAILED = "failed"
    # 完成状态
    COMPLETE = "complete"
```