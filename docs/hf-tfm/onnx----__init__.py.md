# `.\onnx\__init__.py`

```
# 导入类型检查工具，用于检查类型注解的有效性
from typing import TYPE_CHECKING

# 导入延迟加载模块的工具函数
from ..utils import _LazyModule

# 定义模块的导入结构，包括各个子模块及其成员
_import_structure = {
    "config": [
        "EXTERNAL_DATA_FORMAT_SIZE_LIMIT",  # 外部数据格式大小限制
        "OnnxConfig",  # OnnxConfig 类型
        "OnnxConfigWithPast",  # 带有历史的 OnnxConfig 类型
        "OnnxSeq2SeqConfigWithPast",  # 带有历史的 OnnxSeq2SeqConfig 类型
        "PatchingSpec",  # 补丁规范类
    ],
    "convert": ["export", "validate_model_outputs"],  # 转换相关函数
    "features": ["FeaturesManager"],  # 特征管理器类
    "utils": ["ParameterFormat", "compute_serialized_parameters_size"],  # 参数格式及计算序列化参数大小函数
}

# 如果处于类型检查模式，则从各子模块导入特定类型
if TYPE_CHECKING:
    from .config import (
        EXTERNAL_DATA_FORMAT_SIZE_LIMIT,  # 外部数据格式大小限制
        OnnxConfig,  # OnnxConfig 类型
        OnnxConfigWithPast,  # 带有历史的 OnnxConfig 类型
        OnnxSeq2SeqConfigWithPast,  # 带有历史的 OnnxSeq2SeqConfig 类型
        PatchingSpec,  # 补丁规范类
    )
    from .convert import export, validate_model_outputs  # 导出和验证模型输出函数
    from .features import FeaturesManager  # 特征管理器类
    from .utils import ParameterFormat, compute_serialized_parameters_size  # 参数格式及计算序列化参数大小函数

# 如果不处于类型检查模式，则进行延迟加载模块设置
else:
    import sys
    
    # 将当前模块替换为延迟加载模块对象，使用给定的导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```