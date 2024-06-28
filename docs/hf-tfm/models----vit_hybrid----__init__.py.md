# `.\models\vit_hybrid\__init__.py`

```
# 版权声明和许可证信息，指明代码版权归HuggingFace团队所有，使用Apache License, Version 2.0许可证
#
# 导入必要的类型检查工具
from typing import TYPE_CHECKING

# 导入自定义的异常和模块延迟加载工具，用于处理可能缺失的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构字典，包含配置和模型信息
_import_structure = {"configuration_vit_hybrid": ["VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTHybridConfig"]}

# 尝试导入torch，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则添加模型相关的导入信息到_import_structure字典
    _import_structure["modeling_vit_hybrid"] = [
        "VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTHybridForImageClassification",
        "ViTHybridModel",
        "ViTHybridPreTrainedModel",
    ]

# 尝试导入vision，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若vision可用，则添加图像处理相关的导入信息到_import_structure字典
    _import_structure["image_processing_vit_hybrid"] = ["ViTHybridImageProcessor"]

# 如果正在进行类型检查，导入具体的配置和模型类
if TYPE_CHECKING:
    from .configuration_vit_hybrid import VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTHybridConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vit_hybrid import (
            VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTHybridForImageClassification,
            ViTHybridModel,
            ViTHybridPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_vit_hybrid import ViTHybridImageProcessor

# 如果不是类型检查环境，则进行模块的延迟加载设置
else:
    import sys

    # 将当前模块替换为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```