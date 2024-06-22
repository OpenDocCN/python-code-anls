# `.\transformers\models\speech_to_text\__init__.py`

```py
# 版权声明及许可信息，注明代码版权和许可信息
# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入必要的依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_speech_to_text": ["SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "Speech2TextConfig"],
    "feature_extraction_speech_to_text": ["Speech2TextFeatureExtractor"],
    "processing_speech_to_text": ["Speech2TextProcessor"],
}

# 检查是否存在 sentencepiece 依赖，如果不存在则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_speech_to_text"] = ["Speech2TextTokenizer"]

# 检查是否存在 TensorFlow 依赖，如果不存在则引发异常
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

# 检查是否存在 PyTorch 依赖，如果不存在则引发异常
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

# 如果是类型检查，进一步引入相关模块
if TYPE_CHECKING:
    from .configuration_speech_to_text import SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, Speech2TextConfig
    from .feature_extraction_speech_to_text import Speech2TextFeatureExtractor
    from .processing_speech_to_text import Speech2TextProcessor

    # 检查是否存在 sentencepiece 依赖，如果不存在则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_speech_to_text import Speech2TextTokenizer

    # 检查是否存在 TensorFlow 依赖，如果不存在则引发异常
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
    # 尝试检查是否存在 torch 库
    try:
        # 如果 torch 库不存在，则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 什么都不做
        pass
    # 如果没有捕获到 OptionalDependencyNotAvailable 异常
    else:
        # 导入 modeling_speech_to_text 模块中的相关内容
        from .modeling_speech_to_text import (
            SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2TextForConditionalGeneration,
            Speech2TextModel,
            Speech2TextPreTrainedModel,
        )
# 否则（当不在一个包中时），导入sys模块
import sys

# 使用sys.modules字典将当前模块替换为_LazyModule类的实例
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```