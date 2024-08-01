# `.\DB-GPT-src\dbgpt\core\awel\flow\compat.py`

```py
"""Compatibility mapping for flow classes."""

# 导入必要的模块和类
from dataclasses import dataclass
from typing import Dict, Optional

# 定义一个数据类，用于表示兼容性映射中的注册项
@dataclass
class _RegisterItem:
    """Register item for compatibility mapping."""

    old_module: str  # 旧模块名
    new_module: str  # 新模块名
    old_name: str  # 旧类名
    new_name: Optional[str] = None  # 新类名（可选）
    after: Optional[str] = None  # 版本信息（可选）

    def old_cls_key(self) -> str:
        """Get the old class key."""
        return f"{self.old_module}.{self.old_name}"  # 返回旧类的完整名称

    def new_cls_key(self) -> str:
        """Get the new class key."""
        return f"{self.new_module}.{self.new_name}"  # 返回新类的完整名称

# 兼容性映射的字典，用于存储旧类名到_RegisterItem对象的映射关系
_COMPAT_FLOW_MAPPING: Dict[str, _RegisterItem] = {}

# 定义旧模块的路径
_OLD_AGENT_RESOURCE_MODULE_1 = "dbgpt.serve.agent.team.layout.agent_operator_resource"
_OLD_AGENT_RESOURCE_MODULE_2 = "dbgpt.agent.plan.awel.agent_operator_resource"
# 定义新模块的路径
_NEW_AGENT_RESOURCE_MODULE = "dbgpt.agent.core.plan.awel.agent_operator_resource"

# 注册函数，将旧类名映射到新类名或注册项中
def _register(
    old_module: str,
    new_module: str,
    old_name: str,
    new_name: Optional[str] = None,
    after_version: Optional[str] = None,
):
    if not new_name:
        new_name = old_name
    # 创建_RegisterItem对象
    item = _RegisterItem(old_module, new_module, old_name, new_name, after_version)
    # 将旧类名作为键，注册项作为值存入兼容性映射字典中
    _COMPAT_FLOW_MAPPING[item.old_cls_key()] = item

# 获取新类名的函数，根据旧类名查询兼容性映射字典
def get_new_class_name(old_class_name: str) -> Optional[str]:
    """Get the new class name for the old class name."""
    if old_class_name not in _COMPAT_FLOW_MAPPING:
        return None
    item = _COMPAT_FLOW_MAPPING[old_class_name]
    return item.new_cls_key()

# 开始注册不同的旧类名到新类名的映射关系

# 注册 AwelAgentResource 类的映射关系
_register(
    _OLD_AGENT_RESOURCE_MODULE_1,
    _NEW_AGENT_RESOURCE_MODULE,
    "AwelAgentResource",
    "AWELAgentResource",
)

# 继续注册 AWELAgentResource 类的映射关系
_register(
    _OLD_AGENT_RESOURCE_MODULE_2,
    _NEW_AGENT_RESOURCE_MODULE,
    "AWELAgentResource",
)

# 注册 AwelAgentConfig 类的映射关系
_register(
    _OLD_AGENT_RESOURCE_MODULE_1,
    _NEW_AGENT_RESOURCE_MODULE,
    "AwelAgentConfig",
    "AWELAgentConfig",
)

# 继续注册 AWELAgentConfig 类的映射关系
_register(
    _OLD_AGENT_RESOURCE_MODULE_2,
    _NEW_AGENT_RESOURCE_MODULE,
    "AWELAgentConfig",
    "AWELAgentConfig",
)

# 注册 AwelAgent 类的映射关系
_register(
    _OLD_AGENT_RESOURCE_MODULE_1,
    _NEW_AGENT_RESOURCE_MODULE,
    "AwelAgent",
    "AWELAgent",
)

# 继续注册 AWELAgent 类的映射关系
_register(
    _OLD_AGENT_RESOURCE_MODULE_2,
    _NEW_AGENT_RESOURCE_MODULE,
    "AWELAgent",
    "AWELAgent",
)

# 注册 VectorStoreConnector 类的映射关系
_register(
    "dbgpt.storage.vector_store.connector",
    "dbgpt.serve.rag.connector",
    "VectorStoreConnector",
    after_version="v0.5.8",
)
```