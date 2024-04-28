# `.\transformers\models\bit\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 引入自定义异常类，用于指示可选依赖项不可用
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构，包括配置和模型
_import_structure = {"configuration_bit": ["BIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BitConfig", "BitOnnxConfig"]}

# 检查是否可用torch库，若不可用则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则添加模型相关结构到导入结构中
    _import_structure["modeling_bit"] = [
        "BIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BitForImageClassification",
        "BitModel",
        "BitPreTrainedModel",
        "BitBackbone",
    ]

# 检查是否可用vision库，若不可用则引发自定义异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若vision可用，则添加图像处理相关结构到导入结构中
    _import_structure["image_processing_bit"] = ["BitImageProcessor"]

# 如果是类型检查阶段，则导入相关配置和模型类
if TYPE_CHECKING:
    from .configuration_bit import BIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BitConfig, BitOnnxConfig

    # 若torch可用，则导入模型相关类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bit import (
            BIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BitBackbone,
            BitForImageClassification,
            BitModel,
            BitPreTrainedModel,
        )

    # 若vision可用，则导入图像处理相关类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_bit import BitImageProcessor

# 如果不是类型检查阶段，则延迟导入相关模块
else:
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```  
```