# `.\transformers\models\blip\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从 utils 模块中导入必要的函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_blip": [
        "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlipConfig",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "processing_blip": ["BlipProcessor"],
}

# 检查是否有图像处理相关的库可用
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加图像处理相关的模块到导入结构中
    _import_structure["image_processing_blip"] = ["BlipImageProcessor"]

# 检查是否有 PyTorch 库可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 PyTorch 相关的模块到导入结构中
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

# 检查是否有 TensorFlow 库可用
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 TensorFlow 相关的模块到导入结构中
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

# 如果是类型检查阶段，则导入相关配置和处理模块
if TYPE_CHECKING:
    # 导入配置相关的类和变量
    from .configuration_blip import BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP, BlipConfig, BlipTextConfig, BlipVisionConfig
    # 导入处理模块
    from .processing_blip import BlipProcessor

    # 检查图像处理相关库是否可用
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入图像处理模块
        from .image_processing_blip import BlipImageProcessor

    # 检查是否有 PyTorch 库可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果导入失败，则尝试从当前目录下的modeling_blip模块中导入以下类和变量
    else:
        from .modeling_blip import (
            BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,  # BLIP预训练模型归档列表
            BlipForConditionalGeneration,  # 用于条件生成的Blip模型
            BlipForImageTextRetrieval,  # 用于图像文本检索的Blip模型
            BlipForQuestionAnswering,  # 用于问答的Blip模型
            BlipModel,  # Blip模型
            BlipPreTrainedModel,  # Blip预训练模型
            BlipTextModel,  # 文本模型
            BlipVisionModel,  # 视觉模型
        )

    # 尝试检查是否TensorFlow可用，如果不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果TensorFlow不可用，则不做任何操作，继续执行下一个代码块
        pass
    else:
        # 如果TensorFlow可用，则从当前目录下的modeling_tf_blip模块中导入以下类和变量
        from .modeling_tf_blip import (
            TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST,  # TF_BLIP预训练模型归档列表
            TFBlipForConditionalGeneration,  # 用于条件生成的TFBlip模型
            TFBlipForImageTextRetrieval,  # 用于图像文本检索的TFBlip模型
            TFBlipForQuestionAnswering,  # 用于问答的TFBlip模型
            TFBlipModel,  # TFBlip模型
            TFBlipPreTrainedModel,  # TFBlip预训练模型
            TFBlipTextModel,  # 文本模型
            TFBlipVisionModel,  # 视觉模型
        )
```  
# 否则，如果当前模块不是主程序，引入sys模块，该模块允许与Python解释器进行交互
import sys

# 将当前模块的命名空间中的__name__作为键，将_LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)作为值，
# 更新sys模块的模块字典，以便将当前模块替换为_LazyModule类的一个实例
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```