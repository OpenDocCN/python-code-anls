# `.\transformers\models\owlv2\__init__.py`

```
# 导入必要的模块
from typing import TYPE_CHECKING
# 导入 LazyModule 类和检查 Torch 和 Vision 模块是否可用的函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

#定义要导入的结构
_import_structure = {
    "configuration_owlv2": [
        "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Owlv2Config",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "processing_owlv2": ["Owlv2Processor"],
}

# 检查 Vision 模块是否可用，不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# Vision 模块可用则增加 image_processing_owlv2 模块到导入结构
else:
    _import_structure["image_processing_owlv2"] = ["Owlv2ImageProcessor"]

# 检查 Torch 模块是否可用，不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# Torch 模块可用则增加 modeling_owlv2 模块到导入结构
else:
    _import_structure["modeling_owlv2"] = [
        "OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Owlv2Model",
        "Owlv2PreTrainedModel",
        "Owlv2TextModel",
        "Owlv2VisionModel",
        "Owlv2ForObjectDetection",
    ]

# 检查是否处于类型检查模式，导入相应模块
if TYPE_CHECKING:
    from .configuration_owlv2 import (
        OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Owlv2Config,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )
    from .processing_owlv2 import Owlv2Processor

    # 检查 Vision 模块是否可用，不可用则忽略导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_owlv2 import Owlv2ImageProcessor

    # 检查 Torch 模块是否可用，不可用则忽略导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_owlv2 import (
            OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Owlv2ForObjectDetection,
            Owlv2Model,
            Owlv2PreTrainedModel,
            Owlv2TextModel,
            Owlv2VisionModel,
        )

# 非类型检查模式下，设置当前模块为 LazyModule
else:
    import sys
    # 使用 LazyModule 对象代替当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```