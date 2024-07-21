# `.\pytorch\torch\onnx\_internal\fx\analysis\__init__.py`

```py
# 从当前包中导入 unsupported_nodes 模块中的 UnsupportedFxNodesAnalysis 类
from .unsupported_nodes import UnsupportedFxNodesAnalysis

# 定义一个列表 __all__，指定在使用 `from module import *` 时导出的成员
__all__ = [
    "UnsupportedFxNodesAnalysis",  # 将 UnsupportedFxNodesAnalysis 类加入到导出列表中
]
```