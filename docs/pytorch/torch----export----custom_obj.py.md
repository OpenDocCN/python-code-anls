# `.\pytorch\torch\export\custom_obj.py`

```
from dataclasses import dataclass
# 导入 dataclass 模块，用于创建数据类


__all__ = ["ScriptObjectMeta"]
# 定义当前模块中可以被外部访问的公开接口列表，只包含 ScriptObjectMeta


@dataclass
# 使用 dataclass 装饰器，简化创建数据类的过程
class ScriptObjectMeta:
    """
    Metadata which is stored on nodes representing ScriptObjects.
    """
    # 脚本对象元数据，存储在表示脚本对象的节点上的信息

    constant_name: str
    # 常量名，用于从常量表中检索真实的 ScriptObject

    class_fqn: str
    # 类的全限定名，表示这个元数据所关联的类的位置信息
```