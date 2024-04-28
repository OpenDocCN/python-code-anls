# `.\transformers\models\pix2struct\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_pix2struct": [
        "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pix2StructConfig",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "processing_pix2struct": ["Pix2StructProcessor"],
}

# 检查视觉模块是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加图像处理模块到导入结构中
    _import_structure["image_processing_pix2struct"] = ["Pix2StructImageProcessor"]

# 检查 Torch 模块是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型建模模块到导入结构中
    _import_structure["modeling_pix2struct"] = [
        "PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Pix2StructPreTrainedModel",
        "Pix2StructForConditionalGeneration",
        "Pix2StructVisionModel",
        "Pix2StructTextModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置、处理和模型模块
    from .configuration_pix2struct import (
        PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pix2StructConfig,
        Pix2StructTextConfig,
        Pix2StructVisionConfig,
    )
    from .processing_pix2struct import Pix2StructProcessor

    # 检查视觉模块是否可用，若不可用则抛出异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入图像处理模块
        from .image_processing_pix2struct import Pix2StructImageProcessor

    # 检查 Torch 模块是否可用，若不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型建模模块
        from .modeling_pix2struct import (
            PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Pix2StructForConditionalGeneration,
            Pix2StructPreTrainedModel,
            Pix2StructTextModel,
            Pix2StructVisionModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```