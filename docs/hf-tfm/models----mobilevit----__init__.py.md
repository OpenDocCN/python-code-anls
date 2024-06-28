# `.\models\mobilevit\__init__.py`

```
# 引入类型检查工具，用于类型检查
from typing import TYPE_CHECKING

# 从当前包中的工具模块导入相关依赖
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构，包含配置、模型和处理类
_import_structure = {
    "configuration_mobilevit": ["MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileViTConfig", "MobileViTOnnxConfig"],
}

# 检查视觉处理是否可用，若不可用则抛出可选依赖不可用的异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加特征提取和图像处理模块到导入结构
    _import_structure["feature_extraction_mobilevit"] = ["MobileViTFeatureExtractor"]
    _import_structure["image_processing_mobilevit"] = ["MobileViTImageProcessor"]

# 检查是否 Torch 可用，若不可用则抛出可选依赖不可用的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 Torch 版本的模型定义到导入结构
    _import_structure["modeling_mobilevit"] = [
        "MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileViTForImageClassification",
        "MobileViTForSemanticSegmentation",
        "MobileViTModel",
        "MobileViTPreTrainedModel",
    ]

# 检查是否 TensorFlow 可用，若不可用则抛出可选依赖不可用的异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 TensorFlow 版本的模型定义到导入结构
    _import_structure["modeling_tf_mobilevit"] = [
        "TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFMobileViTForImageClassification",
        "TFMobileViTForSemanticSegmentation",
        "TFMobileViTModel",
        "TFMobileViTPreTrainedModel",
    ]

# 如果是类型检查环境，从配置和模型模块中导入相关类和变量
if TYPE_CHECKING:
    from .configuration_mobilevit import MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileViTConfig, MobileViTOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果视觉处理可用，从特征提取和图像处理模块中导入相关类
        from .feature_extraction_mobilevit import MobileViTFeatureExtractor
        from .image_processing_mobilevit import MobileViTImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，从模型定义模块中导入相关类和变量
        from .modeling_mobilevit import (
            MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileViTForImageClassification,
            MobileViTForSemanticSegmentation,
            MobileViTModel,
            MobileViTPreTrainedModel,
        )
    # 尝试检查是否TensorFlow可用，如果不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获OptionalDependencyNotAvailable异常，不做任何处理
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有异常发生，则执行以下代码块
    else:
        # 从模块modeling_tf_mobilevit中导入以下内容：
        # TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST：移动ViT模型的预训练模型归档列表
        # TFMobileViTForImageClassification：用于图像分类的移动ViT模型
        # TFMobileViTForSemanticSegmentation：用于语义分割的移动ViT模型
        # TFMobileViTModel：移动ViT的基础模型
        # TFMobileViTPreTrainedModel：移动ViT的预训练模型基类
        from .modeling_tf_mobilevit import (
            TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMobileViTForImageClassification,
            TFMobileViTForSemanticSegmentation,
            TFMobileViTModel,
            TFMobileViTPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态修改模块属性
    import sys

    # 将当前模块(__name__)的引用指向一个自定义的 LazyModule 对象
    # LazyModule 是一个自定义的延迟加载模块，用于在需要时再加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```