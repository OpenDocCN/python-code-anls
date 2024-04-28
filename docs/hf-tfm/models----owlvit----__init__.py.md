# `.\transformers\models\owlvit\__init__.py`

```
# 引入需要的模块和类
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


# 定义模块导入结构
_import_structure = {
    "configuration_owlvit": [
        "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "OwlViTConfig",
        "OwlViTOnnxConfig",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "processing_owlvit": ["OwlViTProcessor"],
}


# 检查是否可用视觉处理模块
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 如果可用，添加到 import_structure 字典中
    _import_structure["feature_extraction_owlvit"] = ["OwlViTFeatureExtractor"]
    _import_structure["image_processing_owlvit"] = ["OwlViTImageProcessor"]


# 检查是否可用 Torch 模块
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 如果可用，添加到 import_structure 字典中
    _import_structure["modeling_owlvit"] = [
        "OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OwlViTModel",
        "OwlViTPreTrainedModel",
        "OwlViTTextModel",
        "OwlViTVisionModel",
        "OwlViTForObjectDetection",
    ]


# 如果是类型检查模式，则导入相应的模块和类
if TYPE_CHECKING:
    from .configuration_owlvit import (
        OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OwlViTConfig,
        OwlViTOnnxConfig,
        OwlViTTextConfig,
        OwlViTVisionConfig,
    )
    from .processing_owlvit import OwlViTProcessor

    # 检查是否可用视觉处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:  # 如果可用，导入相应的模块和类
        from .feature_extraction_owlvit import OwlViTFeatureExtractor
        from .image_processing_owlvit import OwlViTImageProcessor

    # 检查是否可用 Torch 模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:  # 如果可用，导入相应的模块和类
        from .modeling_owlvit import (
            OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OwlViTForObjectDetection,
            OwlViTModel,
            OwlViTPreTrainedModel,
            OwlViTTextModel,
            OwlViTVisionModel,
        )


# 如果不是类型检查模式，则动态加载模块
else:
    import sys

    # 创建一个懒加载的模块对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```