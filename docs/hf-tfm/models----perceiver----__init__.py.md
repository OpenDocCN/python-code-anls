# `.\transformers\models\perceiver\__init__.py`

```
# 引入 TYPE_CHECKING 以检查类型
from typing import TYPE_CHECKING
# 引入 utils 模块中的函数和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)
# 定义 _import_structure 字典，用于存储需要导入的模块和功能
_import_structure = {
    "configuration_perceiver": ["PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP", "PerceiverConfig", "PerceiverOnnxConfig"],
    "tokenization_perceiver": ["PerceiverTokenizer"],
}
# 检查是否安装了 vision 库，如果没有则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_perceiver"] = ["PerceiverFeatureExtractor"]
    _import_structure["image_processing_perceiver"] = ["PerceiverImageProcessor"]
# 检查是否安装了 torch 库，如果没有则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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
# 如果 TYPE_CHECKING 为真，则从相应模块中导入特定的类和功能
if TYPE_CHECKING:
    from .configuration_perceiver import PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP, PerceiverConfig, PerceiverOnnxConfig
    from .tokenization_perceiver import PerceiverTokenizer
    # 检查是否安装了 vision 库，如果没有则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_perceiver import PerceiverFeatureExtractor
        from .image_processing_perceiver import PerceiverImageProcessor
    # 检查是否安装了 torch 库，如果没有则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，从当前目录下的 model_perceiver 模块中导入以下内容：
    # - PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST: 预训练模型的存档列表
    # - PerceiverForImageClassificationConvProcessing: 用于图像分类的 Perceiver 模型（卷积处理）
    # - PerceiverForImageClassificationFourier: 用于图像分类的 Perceiver 模型（傅里叶处理）
    # - PerceiverForImageClassificationLearned: 用于图像分类的 Perceiver 模型（学习处理）
    # - PerceiverForMaskedLM: 用于遮蔽语言建模的 Perceiver 模型
    # - PerceiverForMultimodalAutoencoding: 用于多模态自编码的 Perceiver 模型
    # - PerceiverForOpticalFlow: 用于光流估计的 Perceiver 模型
    # - PerceiverForSequenceClassification: 用于序列分类的 Perceiver 模型
    # - PerceiverLayer: Perceiver 模型中的层
    # - PerceiverModel: Perceiver 模型
    # - PerceiverPreTrainedModel: Perceiver 预训练模型
    from .modeling_perceiver import (
        PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST,
        PerceiverForImageClassificationConvProcessing,
        PerceiverForImageClassificationFourier,
        PerceiverForImageClassificationLearned,
        PerceiverForMaskedLM,
        PerceiverForMultimodalAutoencoding,
        PerceiverForOpticalFlow,
        PerceiverForSequenceClassification,
        PerceiverLayer,
        PerceiverModel,
        PerceiverPreTrainedModel,
    )
# 如果上述 import 失败，则执行以下操作
else:
    # 引入 sys 模块
    import sys
    # 创建一个 _LazyModule 对象，并将其设置为当前模块的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```