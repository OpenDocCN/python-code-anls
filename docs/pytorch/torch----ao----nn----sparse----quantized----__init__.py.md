# `.\pytorch\torch\ao\nn\sparse\quantized\__init__.py`

```
# 导入torch.ao.nn.sparse.quantized模块中的dynamic符号
from torch.ao.nn.sparse.quantized import dynamic

# 从当前目录中导入linear.py文件中的Linear和LinearPackedParams类
from .linear import Linear
from .linear import LinearPackedParams

# 定义一个列表，包含要导出的公共符号名称
__all__ = [
    "dynamic",                # 将dynamic符号添加到导出列表中
    "Linear",                 # 将Linear类添加到导出列表中
    "LinearPackedParams",     # 将LinearPackedParams类添加到导出列表中
]
```