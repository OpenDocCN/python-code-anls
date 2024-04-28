# `.\transformers\models\speecht5\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入可选依赖未安装异常和延迟加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)

# 定义需要延迟加载的模块结构
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

# 检查是否安装了 sentencepiece
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_speecht5"] = ["SpeechT5Tokenizer"]

# 检查是否安装了 torch
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_speecht5"] = [
        "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SpeechT5ForSpeechToText",
        "SpeechT5ForSpeechToSpeech",
        "SpeechT5ForTextToSpeech",
        "SpeechT5Model",
        "SpeechT5PreTrainedModel",
        "SpeechT5HifiGan",
    ]

# 如果是类型检查，导入相应模块
if TYPE_CHECKING:
    from .configuration_speecht5 import (
        SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP,
        SpeechT5Config,
        SpeechT5HifiGanConfig,
    )
    from .feature_extraction_speecht5 import SpeechT5FeatureExtractor
    from .processing_speecht5 import SpeechT5Processor

    # 检查是否安装了 sentencepiece
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_speecht5 import SpeechT5Tokenizer

    # 检查是否安装了 torch
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

# 如果不是类型检查，将模块设为延迟加载
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```