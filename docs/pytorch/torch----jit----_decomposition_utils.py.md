# `.\pytorch\torch\jit\_decomposition_utils.py`

```py
# 添加类型检查声明，允许未经类型定义的函数
# 导入 PyTorch 库
import torch
# 从 torch._ops 模块中导入 OpOverload 和 OpOverloadPacket 类
from torch._ops import OpOverload, OpOverloadPacket

# 定义一个函数，用于注册分解操作
def _register_decomposition(op: OpOverload, graph: torch._C.Graph):
    # 断言 op 参数不是 OpOverloadPacket 类的实例，而是 OpOverload 类的实例
    assert not isinstance(
        op, OpOverloadPacket
    ), f"Must pass specific op overload, not overload packet, found {op}"
    # 断言 op 参数是 OpOverload 类的实例
    assert isinstance(op, OpOverload)

    # 调用 PyTorch C++ 扩展函数，注册 op 的分解操作到指定的图中
    torch._C._jit_register_decomposition_for_schema(op._schema, graph)
```