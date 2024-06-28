# `.\models\sam\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入所需的依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_sam": [
        "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "processing_sam": ["SamProcessor"],
}

# 检查是否可以导入 torch，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加相关模块到导入结构
    _import_structure["modeling_sam"] = [
        "SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SamModel",
        "SamPreTrainedModel",
    ]

# 检查是否可以导入 tensorflow，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加相关模块到导入结构
    _import_structure["modeling_tf_sam"] = [
        "TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSamModel",
        "TFSamPreTrainedModel",
    ]

# 检查是否可以导入视觉处理模块，若不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加相关模块到导入结构
    _import_structure["image_processing_sam"] = ["SamImageProcessor"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置相关的类
    from .configuration_sam import (
        SAM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SamConfig,
        SamMaskDecoderConfig,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )
    # 导入处理相关的类
    from .processing_sam import SamProcessor

    # 检查是否可以导入 torch，若不可用则跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 torch 模型相关的类
        from .modeling_sam import SAM_PRETRAINED_MODEL_ARCHIVE_LIST, SamModel, SamPreTrainedModel

    # 检查是否可以导入 tensorflow，若不可用则跳过
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 tensorflow 模型相关的类
        from .modeling_tf_sam import TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST, TFSamModel, TFSamPreTrainedModel

    # 检查是否可以导入视觉处理模块，若不可用则跳过
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入视觉处理模块相关的类
        from .image_processing_sam import SamImageProcessor

# 如果不是类型检查阶段，则进行懒加载模块的设置
else:
    import sys

    # 将当前模块设为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```