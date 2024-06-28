# `.\models\oneformer\__init__.py`

```py
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入相关的自定义异常和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义需要导入的模块结构
_import_structure = {
    "configuration_oneformer": ["ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "OneFormerConfig"],
    "processing_oneformer": ["OneFormerProcessor"],
}

# 检查视觉库是否可用，若不可用则引发自定义异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉库可用，则添加图像处理模块到导入结构中
    _import_structure["image_processing_oneformer"] = ["OneFormerImageProcessor"]

# 检查 Torch 库是否可用，若不可用则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 库可用，则添加模型处理模块到导入结构中
    _import_structure["modeling_oneformer"] = [
        "ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OneFormerForUniversalSegmentation",
        "OneFormerModel",
        "OneFormerPreTrainedModel",
    ]

# 如果是类型检查阶段，导入特定模块以供类型注解使用
if TYPE_CHECKING:
    from .configuration_oneformer import ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, OneFormerConfig
    from .processing_oneformer import OneFormerProcessor

    # 在视觉库可用时，导入图像处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_oneformer import OneFormerImageProcessor

    # 在 Torch 库可用时，导入模型处理模块
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

# 如果不是类型检查阶段，则定义 LazyModule 并将其绑定到当前模块
else:
    import sys

    # 创建 LazyModule 对象，用于延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```