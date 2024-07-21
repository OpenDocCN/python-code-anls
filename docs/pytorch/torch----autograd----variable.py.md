# `.\pytorch\torch\autograd\variable.py`

```
# 引入torch模块，用于深度学习任务
# 从torch._C中引入_ImperativeEngine作为ImperativeEngine
import torch
from torch._C import _ImperativeEngine as ImperativeEngine

# 定义模块的公开接口列表
__all__ = ["VariableMeta", "Variable"]

# 定义VariableMeta类，作为metaclass的元类
class VariableMeta(type):
    # 实现__instancecheck__方法，用于检查对象是否是torch.Tensor的实例
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)

# 定义Variable类，继承自torch._C._LegacyVariableBase，并使用VariableMeta作为其元类
class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    # 定义私有属性_execution_engine，并初始化为ImperativeEngine的实例
    _execution_engine = ImperativeEngine()
```