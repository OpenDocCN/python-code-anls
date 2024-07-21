# `.\pytorch\torch\ao\quantization\fx\lower_to_fbgemm.py`

```
# 从 _lower_to_native_backend 模块中导入 _lower_to_native_backend 函数
# 这个函数用于将模型转换为本地后端的实现
from ._lower_to_native_backend import _lower_to_native_backend

# 从 ..qconfig 模块中导入 QConfigAny 类型
from ..qconfig import QConfigAny

# 从 torch.fx 模块中导入 GraphModule 类
from torch.fx import GraphModule

# 从 typing 模块中导入 Dict 和 Tuple 类型
from typing import Dict, Tuple

# 定义 __all__ 列表，用于在模块中声明导出的符号
__all__ = ['lower_to_fbgemm']

# 定义 lower_to_fbgemm 函数，接受三个参数并返回一个 GraphModule 对象
def lower_to_fbgemm(
    model: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]]
) -> GraphModule:
    """ 
    将一个量化参考模型（具有参考量化运算符模式）降低到 fbgemm 后端
    """
    # 调用 _lower_to_native_backend 函数，将模型转换为 fbgemm 后端的实现
    return _lower_to_native_backend(model, qconfig_map, node_name_to_scope)
```