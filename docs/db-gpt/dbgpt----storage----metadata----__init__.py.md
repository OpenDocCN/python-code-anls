# `.\DB-GPT-src\dbgpt\storage\metadata\__init__.py`

```py
"""Module for handling metadata storage."""
# 导入基础数据访问对象类 BaseDao，用于元数据存储
from dbgpt.storage.metadata._base_dao import BaseDao  # noqa: F401
# 导入统一数据库管理工厂 UnifiedDBManagerFactory，用于元数据存储
from dbgpt.storage.metadata.db_factory import UnifiedDBManagerFactory  # noqa: F401
# 导入以下内容用于元数据存储：
# BaseModel: 数据库模型基类
# DatabaseManager: 数据库管理器
# Model: 数据库模型类
# create_model: 创建数据库模型函数
# db: 数据库对象
from dbgpt.storage.metadata.db_manager import (  # noqa: F401
    BaseModel,
    DatabaseManager,
    Model,
    create_model,
    db,
)

# 模块中可以被导入的所有符号列表
__ALL__ = [
    "db",
    "Model",
    "DatabaseManager",
    "create_model",
    "BaseModel",
    "BaseDao",
    "UnifiedDBManagerFactory",
]
```