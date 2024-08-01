# `.\DB-GPT-src\dbgpt\storage\chat_history\__init__.py`

```py
"""Module of chat history."""

# 导入聊天历史数据库相关类，包括Dao和实体类
from .chat_history_db import (
    ChatHistoryDao,                # 导入聊天历史Dao
    ChatHistoryEntity,             # 导入聊天历史实体类
    ChatHistoryMessageEntity,      # 导入聊天历史消息实体类
)

# 导入存储适配器相关类，包括消息存储项适配器和对话存储项适配器
from .storage_adapter import (
    DBMessageStorageItemAdapter,    # 导入数据库消息存储项适配器
    DBStorageConversationItemAdapter,  # 导入数据库对话存储项适配器
)

# 定义__ALL__列表，包含模块对外公开的类名
__ALL__ = [
    "ChatHistoryEntity",            # 聊天历史实体类
    "ChatHistoryMessageEntity",     # 聊天历史消息实体类
    "ChatHistoryDao",               # 聊天历史Dao
    "DBStorageConversationItemAdapter",  # 数据库对话存储项适配器
    "DBMessageStorageItemAdapter",   # 数据库消息存储项适配器
]
```