# `.\transformers\models\oneformer\__init__.py`

```
# 依赖检查与导入模块声明
from typing import TYPE_CHECKING

# 导入异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构
_import_structure = {
    "configuration_oneformer": ["ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "OneFormerConfig"],
    "processing_oneformer": ["OneFormerProcessor"],
}

# 检查视觉相关库是否可用，若不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，添加视觉处理模块到导入结构中
    _import_structure["image_processing_oneformer"] = ["OneFormerImageProcessor"]

# 检查 torch 库是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，添加模型相关模块到导入结构中
    _import_structure["modeling_oneformer"] = [
        "ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OneFormerForUniversalSegmentation",
        "OneFormerModel",
        "OneFormerPreTrainedModel",
    ]

# 如果是类型检查模式，则导入相关模块
if TYPE_CHECKING:
    from .configuration_oneformer import ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, OneFormerConfig
    from .processing_oneformer import OneFormerProcessor

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_oneformer import OneFormerImageProcessor
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_oneformer import (
            ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            OneFormerForUniversalSegmentation,
            OneFormerModel,
            OneFormerPreTrainedModel,
        )

# 如果不是类型检查模式，将当前模块设置为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```