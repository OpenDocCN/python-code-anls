# `.\transformers\onnx\__init__.py`

```py
# 导入必要的模块和类型检查工具
from typing import TYPE_CHECKING
# 导入懒加载模块
from ..utils import _LazyModule

# 定义模块的导入结构
_import_structure = {
    "config": [
        "EXTERNAL_DATA_FORMAT_SIZE_LIMIT",  # 外部数据格式大小限制
        "OnnxConfig",                       # Onnx 配置
        "OnnxConfigWithPast",               # 带有过去信息的 Onnx 配置
        "OnnxSeq2SeqConfigWithPast",        # 带有过去信息的 Onnx 序列到序列配置
        "PatchingSpec",                     # 补丁规范
    ],
    "convert": ["export", "validate_model_outputs"],  # 转换相关函数
    "features": ["FeaturesManager"],                   # 特性管理器
    "utils": ["ParameterFormat", "compute_serialized_parameters_size"],  # 参数格式和计算序列化参数大小的工具函数
}

# 如果是类型检查，则导入特定的模块
if TYPE_CHECKING:
    from .config import (
        EXTERNAL_DATA_FORMAT_SIZE_LIMIT,
        OnnxConfig,
        OnnxConfigWithPast,
        OnnxSeq2SeqConfigWithPast,
        PatchingSpec,
    )
    from .convert import export, validate_model_outputs
    from .features import FeaturesManager
    from .utils import ParameterFormat, compute_serialized_parameters_size

# 如果不是类型检查，则进行懒加载
else:
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

```  
```