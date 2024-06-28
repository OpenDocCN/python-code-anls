# `.\models\speecht5\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从工具模块中导入异常和延迟加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)

# 定义模块导入结构，包含不同子模块及其对应的类和常量
_import_structure = {
    "configuration_speecht5": [
        "SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP",
        "SpeechT5Config",
        "SpeechT5HifiGanConfig",
    ],
    "feature_extraction_speecht5": ["SpeechT5FeatureExtractor"],
    "processing_speecht5": ["SpeechT5Processor"],
}

# 尝试检查是否存在 SentencePiece 库，如果不存在则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，则将 tokenization_speecht5 模块添加到导入结构中
    _import_structure["tokenization_speecht5"] = ["SpeechT5Tokenizer"]

# 尝试检查是否存在 Torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则将 modeling_speecht5 模块添加到导入结构中
    _import_structure["modeling_speecht5"] = [
        "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SpeechT5ForSpeechToText",
        "SpeechT5ForSpeechToSpeech",
        "SpeechT5ForTextToSpeech",
        "SpeechT5Model",
        "SpeechT5PreTrainedModel",
        "SpeechT5HifiGan",
    ]

# 如果正在进行类型检查，则从子模块导入特定类和常量
if TYPE_CHECKING:
    from .configuration_speecht5 import (
        SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP,
        SpeechT5Config,
        SpeechT5HifiGanConfig,
    )
    from .feature_extraction_speecht5 import SpeechT5FeatureExtractor
    from .processing_speecht5 import SpeechT5Processor

    # 如果存在 SentencePiece 库，则从 tokenization_speecht5 导入 SpeechT5Tokenizer 类
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_speecht5 import SpeechT5Tokenizer

    # 如果存在 Torch 库，则从 modeling_speecht5 导入各个 SpeechT5 模型类和常量
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_speecht5 import (
            SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            SpeechT5ForSpeechToSpeech,
            SpeechT5ForSpeechToText,
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Model,
            SpeechT5PreTrainedModel,
        )

# 如果不是类型检查阶段，则使用延迟加载模块的 LazyModule 类进行模块的动态导入
else:
    import sys

    # 将当前模块设为 LazyModule 类型，动态导入相关子模块和类
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```