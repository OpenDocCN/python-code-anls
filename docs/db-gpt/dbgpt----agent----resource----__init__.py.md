# `.\DB-GPT-src\dbgpt\agent\resource\__init__.py`

```py
"""Resource module for agent."""

# 导入代理资源模块的基础类和相关资源
from .base import (  # noqa: F401
    AgentResource,  # 代理资源基类
    Resource,  # 资源基类
    ResourceParameters,  # 资源参数类
    ResourceType,  # 资源类型枚举
)
# 导入数据库模块相关内容
from .database import (  # noqa: F401
    DBParameters,  # 数据库参数类
    DBResource,  # 数据库资源类
    RDBMSConnectorResource,  # 关系型数据库连接资源类
    SQLiteDBResource,  # SQLite数据库资源类
)
# 导入知识检索模块的资源类和参数类
from .knowledge import RetrieverResource, RetrieverResourceParameters  # noqa: F401
# 导入资源管理模块的相关内容
from .manage import (  # noqa: F401
    RegisterResource,  # 注册资源函数
    ResourceManager,  # 资源管理器类
    get_resource_manager,  # 获取资源管理器实例函数
    initialize_resource,  # 初始化资源函数
)
# 导入资源包模块的参数类和资源包类
from .pack import PackResourceParameters, ResourcePack  # noqa: F401
# 导入工具模块的基础工具类、功能工具类、工具参数类和工具装饰器
from .tool.base import BaseTool, FunctionTool, ToolParameter, tool  # noqa: F401
# 导入工具包模块的自动生成GPT插件工具包类和工具包类
from .tool.pack import AutoGPTPluginToolPack, ToolPack  # noqa: F401

# 导出所有公共的类、函数和常量
__all__ = [
    "AgentResource",  # 代理资源基类
    "Resource",  # 资源基类
    "ResourceParameters",  # 资源参数类
    "ResourceType",  # 资源类型枚举
    "DBParameters",  # 数据库参数类
    "DBResource",  # 数据库资源类
    "RDBMSConnectorResource",  # 关系型数据库连接资源类
    "SQLiteDBResource",  # SQLite数据库资源类
    "RetrieverResource",  # 知识检索资源类
    "RetrieverResourceParameters",  # 知识检索资源参数类
    "RegisterResource",  # 注册资源函数
    "ResourceManager",  # 资源管理器类
    "get_resource_manager",  # 获取资源管理器实例函数
    "initialize_resource",  # 初始化资源函数
    "PackResourceParameters",  # 资源包参数类
    "ResourcePack",  # 资源包类
    "BaseTool",  # 基础工具类
    "FunctionTool",  # 功能工具类
    "ToolParameter",  # 工具参数类
    "tool",  # 工具装饰器
    "AutoGPTPluginToolPack",  # 自动生成GPT插件工具包类
    "ToolPack",  # 工具包类
]
```