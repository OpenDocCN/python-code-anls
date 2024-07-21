# `.\pytorch\torch\_export\pass_infra\node_metadata.py`

```
# 导入必要的类型定义模块
from typing import Any, Dict, Set

# 定义一个任意类型的节点元数据值
NodeMetadataValue = Any

# 定义一组受保护的键集合，这些键不允许被覆盖
PROTECTED_KEYS: Set[str] = {
    "val",
    "stack_trace",
    "nn_module_stack",
    "debug_handle",
    "tensor_meta",
}

# 定义节点元数据类
class NodeMetadata:
    # 初始化方法，接受一个字典作为数据源
    def __init__(self, data: Dict[str, Any]) -> None:
        # 使用深拷贝方式保存数据，避免直接引用原始数据
        self.data: Dict[str, Any] = data.copy()

    # 获取元数据的特定键对应的值
    def __getitem__(self, key: str) -> NodeMetadataValue:
        return self.data[key]

    # 设置元数据的特定键对应的值，如果键在受保护键集合中则抛出运行时错误
    def __setitem__(self, key: str, value: NodeMetadataValue) -> NodeMetadataValue:
        if key in PROTECTED_KEYS:
            raise RuntimeError(f"Could not override node key: {key}")
        self.data[key] = value

    # 检查元数据是否包含特定键
    def __contains__(self, key: str) -> bool:
        return key in self.data

    # 创建并返回当前元数据对象的深拷贝副本
    def copy(self) -> "NodeMetadata":
        return NodeMetadata(self.data.copy())
```