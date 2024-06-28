# `.\models\blip\__init__.py`

```py
# 导入类型检查标记
from typing import TYPE_CHECKING

# 导入必要的模块和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构字典
_import_structure = {
    "configuration_blip": [
        "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlipConfig",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "processing_blip": ["BlipProcessor"],
}

# 检查视觉模块是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加视觉处理模块到_import_structure字典中
    _import_structure["image_processing_blip"] = ["BlipImageProcessor"]

# 检查PyTorch模块是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加PyTorch模型相关模块到_import_structure字典中
    _import_structure["modeling_blip"] = [
        "BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BlipModel",
        "BlipPreTrainedModel",
        "BlipForConditionalGeneration",
        "BlipForQuestionAnswering",
        "BlipVisionModel",
        "BlipTextModel",
        "BlipForImageTextRetrieval",
    ]

# 检查TensorFlow模块是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加TensorFlow模型相关模块到_import_structure字典中
    _import_structure["modeling_tf_blip"] = [
        "TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBlipModel",
        "TFBlipPreTrainedModel",
        "TFBlipForConditionalGeneration",
        "TFBlipForQuestionAnswering",
        "TFBlipVisionModel",
        "TFBlipTextModel",
        "TFBlipForImageTextRetrieval",
    ]

# 如果在类型检查模式下，导入类型相关的类和常量
if TYPE_CHECKING:
    from .configuration_blip import BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP, BlipConfig, BlipTextConfig, BlipVisionConfig
    from .processing_blip import BlipProcessor

    # 检查视觉模块是否可用，若可用则导入视觉处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_blip import BlipImageProcessor

    # 检查PyTorch模块是否可用，若不可用则忽略导入相关模块
    # 如果前面的条件不满足，则执行以下代码块
    else:
        # 从当前包的模块中导入以下内容
        from .modeling_blip import (
            BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlipForConditionalGeneration,
            BlipForImageTextRetrieval,
            BlipForQuestionAnswering,
            BlipModel,
            BlipPreTrainedModel,
            BlipTextModel,
            BlipVisionModel,
        )

    # 尝试检查是否 TensorFlow 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 如果异常捕获到，则什么也不做，继续执行
        pass
    # 如果没有异常发生，则执行以下代码块
    else:
        # 从当前包的 TensorFlow 版本模块中导入以下内容
        from .modeling_tf_blip import (
            TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBlipForConditionalGeneration,
            TFBlipForImageTextRetrieval,
            TFBlipForQuestionAnswering,
            TFBlipModel,
            TFBlipPreTrainedModel,
            TFBlipTextModel,
            TFBlipVisionModel,
        )
else:
    # 导入 sys 模块，用于操作 Python 解释器的运行时环境
    import sys

    # 使用当前模块的名称作为键，将 _LazyModule 对象赋值给 sys.modules 中的对应条目
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```