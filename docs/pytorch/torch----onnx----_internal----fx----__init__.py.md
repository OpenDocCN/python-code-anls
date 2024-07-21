# `.\pytorch\torch\onnx\_internal\fx\__init__.py`

```
# 导入本地模块中的 ONNXTorchPatcher 类
from .patcher import ONNXTorchPatcher
# 导入本地模块中的 save_model_with_external_data 函数
from .serialization import save_model_with_external_data

# 定义一个列表，包含了模块中公开的所有符号（类、函数等）
__all__ = [
    "save_model_with_external_data",  # 将 save_model_with_external_data 添加到 __all__ 中
    "ONNXTorchPatcher",  # 将 ONNXTorchPatcher 添加到 __all__ 中
]
```