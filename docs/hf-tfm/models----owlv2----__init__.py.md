# `.\models\owlv2\__init__.py`

```
# 版权声明和许可证信息，声明版权归 HuggingFace 团队所有
#
# 根据 Apache License, Version 2.0 进行许可
# 除非符合许可证的要求，否则不得使用此文件
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律不允许或书面同意，软件按"原样"分发
# 无任何明示或暗示的保证或条件
# 详细信息请查看许可证内容
from typing import TYPE_CHECKING

# 从 utils 中导入必要的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构，用于按需导入模块和类
_import_structure = {
    "configuration_owlv2": [
        "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Owlv2Config",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "processing_owlv2": ["Owlv2Processor"],
}

# 检查视觉处理是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则添加视觉处理相关的导入结构
    _import_structure["image_processing_owlv2"] = ["Owlv2ImageProcessor"]

# 检查 Torch 是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则添加模型处理相关的导入结构
    _import_structure["modeling_owlv2"] = [
        "OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Owlv2Model",
        "Owlv2PreTrainedModel",
        "Owlv2TextModel",
        "Owlv2VisionModel",
        "Owlv2ForObjectDetection",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从相关模块中导入必要的类和函数
    from .configuration_owlv2 import (
        OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Owlv2Config,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )
    from .processing_owlv2 import Owlv2Processor

    # 再次检查视觉处理是否可用，若不可用则忽略导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_owlv2 import Owlv2ImageProcessor

    # 再次检查 Torch 是否可用，若不可用则忽略导入
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

# 如果不是类型检查阶段，则将当前模块注册为 LazyModule
else:
    import sys

    # 使用 LazyModule 类来延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```