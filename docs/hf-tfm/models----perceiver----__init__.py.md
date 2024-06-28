# `.\models\perceiver\__init__.py`

```py
# 引入类型检查模块的类型检查功能
from typing import TYPE_CHECKING

# 从指定位置引入各种实用工具和依赖
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义一个字典结构，包含了要导入的模块和相应的成员
_import_structure = {
    "configuration_perceiver": ["PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP", "PerceiverConfig", "PerceiverOnnxConfig"],
    "tokenization_perceiver": ["PerceiverTokenizer"],
}

# 尝试检查视觉模块是否可用，如果不可用则抛出依赖不可用的异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉模块可用，则向_import_structure字典中添加特征提取和图像处理的相关成员
    _import_structure["feature_extraction_perceiver"] = ["PerceiverFeatureExtractor"]
    _import_structure["image_processing_perceiver"] = ["PerceiverImageProcessor"]

# 尝试检查torch模块是否可用，如果不可用则抛出依赖不可用的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch模块可用，则向_import_structure字典中添加模型相关的成员
    _import_structure["modeling_perceiver"] = [
        "PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PerceiverForImageClassificationConvProcessing",
        "PerceiverForImageClassificationFourier",
        "PerceiverForImageClassificationLearned",
        "PerceiverForMaskedLM",
        "PerceiverForMultimodalAutoencoding",
        "PerceiverForOpticalFlow",
        "PerceiverForSequenceClassification",
        "PerceiverLayer",
        "PerceiverModel",
        "PerceiverPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从配置模块中引入特定的配置映射、配置类和ONNX配置类
    from .configuration_perceiver import PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP, PerceiverConfig, PerceiverOnnxConfig
    # 从tokenization模块中引入PerceiverTokenizer类

    from .tokenization_perceiver import PerceiverTokenizer

    # 再次尝试检查视觉模块是否可用，如果不可用则抛出依赖不可用的异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果视觉模块可用，则从特征提取和图像处理模块中引入相应类
        from .feature_extraction_perceiver import PerceiverFeatureExtractor
        from .image_processing_perceiver import PerceiverImageProcessor

    # 再次尝试检查torch模块是否可用，如果不可用则抛出依赖不可用的异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模块中的特定内容，从.modeling_perceiver模块中
        from .modeling_perceiver import (
            PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的存档列表
            PerceiverForImageClassificationConvProcessing,  # 导入处理图像分类的Perceiver模型（使用卷积处理）
            PerceiverForImageClassificationFourier,  # 导入处理图像分类的Perceiver模型（使用傅里叶特征）
            PerceiverForImageClassificationLearned,  # 导入处理图像分类的Perceiver模型（学习特征）
            PerceiverForMaskedLM,  # 导入用于掩码语言模型任务的Perceiver模型
            PerceiverForMultimodalAutoencoding,  # 导入用于多模态自编码任务的Perceiver模型
            PerceiverForOpticalFlow,  # 导入用于光流处理任务的Perceiver模型
            PerceiverForSequenceClassification,  # 导入用于序列分类任务的Perceiver模型
            PerceiverLayer,  # 导入Perceiver的层定义
            PerceiverModel,  # 导入通用Perceiver模型
            PerceiverPreTrainedModel,  # 导入预训练的Perceiver模型
        )
else:
    # 如果前面的条件不满足，则执行以下代码块
    import sys
    # 导入 sys 模块，用于访问和操作与 Python 解释器相关的系统功能

    # 将当前模块注册为 LazyModule 的实例，作为当前模块的替代
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```