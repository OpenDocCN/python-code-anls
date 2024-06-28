# `.\models\speech_to_text\__init__.py`

```
# 导入所需模块和函数，包括特定的异常处理和延迟加载模块
from typing import TYPE_CHECKING
# 从相对路径导入必要的模块和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构，包含不同子模块的导入映射
_import_structure = {
    "configuration_speech_to_text": ["SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "Speech2TextConfig"],
    "feature_extraction_speech_to_text": ["Speech2TextFeatureExtractor"],
    "processing_speech_to_text": ["Speech2TextProcessor"],
}

# 尝试导入句子处理模块，如果不可用则抛出异常并忽略
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_speech_to_text"] = ["Speech2TextTokenizer"]

# 尝试导入 TensorFlow 模块，如果不可用则抛出异常并忽略
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_speech_to_text"] = [
        "TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSpeech2TextForConditionalGeneration",
        "TFSpeech2TextModel",
        "TFSpeech2TextPreTrainedModel",
    ]

# 尝试导入 PyTorch 模块，如果不可用则抛出异常并忽略
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_speech_to_text"] = [
        "SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Speech2TextForConditionalGeneration",
        "Speech2TextModel",
        "Speech2TextPreTrainedModel",
    ]

# 如果在类型检查模式下，导入额外的模块和类来进行类型注解
if TYPE_CHECKING:
    from .configuration_speech_to_text import SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, Speech2TextConfig
    from .feature_extraction_speech_to_text import Speech2TextFeatureExtractor
    from .processing_speech_to_text import Speech2TextProcessor

    # 在类型检查模式下，尝试导入句子处理模块，如果不可用则忽略
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_speech_to_text import Speech2TextTokenizer

    # 在类型检查模式下，尝试导入 TensorFlow 模块，如果不可用则忽略
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_speech_to_text import (
            TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSpeech2TextForConditionalGeneration,
            TFSpeech2TextModel,
            TFSpeech2TextPreTrainedModel,
        )
    # 尝试检查是否 Torch 库可用
    try:
        # 如果 Torch 不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 如果 Torch 不可用，不进行任何操作，继续执行后续代码
        pass
    else:
        # 如果没有捕获异常，则导入语音到文本模型相关的模块和类
        from .modeling_speech_to_text import (
            SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2TextForConditionalGeneration,
            Speech2TextModel,
            Speech2TextPreTrainedModel,
        )
# 如果条件不成立，则导入 sys 模块
import sys
# 将当前模块注册到 sys.modules 中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```