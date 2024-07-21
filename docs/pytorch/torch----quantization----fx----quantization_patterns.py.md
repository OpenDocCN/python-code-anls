# `.\pytorch\torch\quantization\fx\quantization_patterns.py`

```
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

# 从 torch.ao.quantization.fx.quantize_handler 模块导入以下类，用于量化处理
from torch.ao.quantization.fx.quantize_handler import (
    BatchNormQuantizeHandler,         # 批量归一化量化处理器
    BinaryOpQuantizeHandler,          # 二元操作量化处理器
    CatQuantizeHandler,               # 拼接量化处理器
    ConvReluQuantizeHandler,          # 卷积ReLU量化处理器
    CopyNodeQuantizeHandler,          # 复制节点量化处理器
    CustomModuleQuantizeHandler,      # 自定义模块量化处理器
    DefaultNodeQuantizeHandler,       # 默认节点量化处理器
    EmbeddingQuantizeHandler,         # 嵌入量化处理器
    FixedQParamsOpQuantizeHandler,    # 固定量化参数操作量化处理器
    GeneralTensorShapeOpQuantizeHandler,  # 通用张量形状操作量化处理器
    LinearReLUQuantizeHandler,        # 线性ReLU量化处理器
    QuantizeHandler,                  # 量化处理器基类
    RNNDynamicQuantizeHandler,        # 循环神经网络动态量化处理器
    StandaloneModuleQuantizeHandler,  # 独立模块量化处理器
)

# 设置每个类所在的模块，以确保正确导入
QuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
BinaryOpQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CatQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
ConvReluQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
LinearReLUQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
BatchNormQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
EmbeddingQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
RNNDynamicQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
DefaultNodeQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
FixedQParamsOpQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
CopyNodeQuantizeHandler.__module__ = "torch.ao.quantization.fx.quantization_patterns"
CustomModuleQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
GeneralTensorShapeOpQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
StandaloneModuleQuantizeHandler.__module__ = (
    "torch.ao.quantization.fx.quantization_patterns"
)
```