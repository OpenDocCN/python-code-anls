# `.\transformers\models\mobilevit\__init__.py`

```
# 2022年版权声明

# 当前代码文件需遵守 Apache 许可证 2.0 版本
# 你可以在以下链接处获取该许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非被适用法律要求或在书面协议中同意，否则根据许可证分发的软件，都是基于"按原样"基础分发，没有任何明示或暗示的担保或条件。
# 请查看许可证获取有关特定语言管理权限和限制的详细信息。

# 引入类型检查库
from typing import TYPE_CHECKING

# 引入 LazyModule 类型，该类型可以延迟加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_mobilevit": ["MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileViTConfig", "MobileViTOnnxConfig"],
}

# 尝试导入视觉处理库，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_mobilevit"] = ["MobileViTFeatureExtractor"]
    _import_structure["image_processing_mobilevit"] = ["MobileViTImageProcessor"]

# 尝试导入 Torch 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mobilevit"] = [
        "MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileViTForImageClassification",
        "MobileViTForSemanticSegmentation",
        "MobileViTModel",
        "MobileViTPreTrainedModel",
    ]

# 尝试导入 TensorFlow 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_mobilevit"] = [
        "TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFMobileViTForImageClassification",
        "TFMobileViTForSemanticSegmentation",
        "TFMobileViTModel",
        "TFMobileViTPreTrainedModel",
    ]

# 如果是类型检查模式，则导入配置、特征提取、图片处理、模型相关库
if TYPE_CHECKING:
    from .configuration_mobilevit import MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileViTConfig, MobileViTOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_mobilevit import MobileViTFeatureExtractor
        from .image_processing_mobilevit import MobileViTImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mobilevit import (
            MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileViTForImageClassification,
            MobileViTForSemanticSegmentation,
            MobileViTModel,
            MobileViTPreTrainedModel,
        )
    # 尝试检查是否可用 TensorFlow，如果不可用则引发自定义的异常 OptionalDependencyNotAvailable
    try:
        # 如果 TensorFlow 不可用，则触发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，什么也不做，直接跳过
        pass
    # 如果没有发生异常，即 TensorFlow 可用，则执行下面的代码块
    else:
        # 从模块 modeling_tf_mobilevit 中导入以下内容：
        # TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST：MobileViT 预训练模型的存档列表
        # TFMobileViTForImageClassification：用于图像分类的 MobileViT 模型
        # TFMobileViTForSemanticSegmentation：用于语义分割的 MobileViT 模型
        # TFMobileViTModel：MobileViT 模型的基类
        # TFMobileViTPreTrainedModel：MobileViT 预训练模型的基类
        from .modeling_tf_mobilevit import (
            TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMobileViTForImageClassification,
            TFMobileViTForSemanticSegmentation,
            TFMobileViTModel,
            TFMobileViTPreTrainedModel,
        )
# 如果不在顶层作用域下，引入sys模块
else:
    import sys
    # 将当前模块添加到模块字典中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```