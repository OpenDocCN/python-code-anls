# `.\transformers\models\speech_to_text_2\__init__.py`

```
# 版权声明以及开源许可证信息

# 判断是否为类型检查
from typing import TYPE_CHECKING

# 导入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)

# 定义待导入的结构
_import_structure = {
    "configuration_speech_to_text_2": ["SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Speech2Text2Config"],
    "processing_speech_to_text_2": ["Speech2Text2Processor"],
    "tokenization_speech_to_text_2": ["Speech2Text2Tokenizer"],
}

# 检查是否有torch依赖可用，否则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果有torch依赖可用，添加相应的模型结构
    _import_structure["modeling_speech_to_text_2"] = [
        "SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Speech2Text2ForCausalLM",
        "Speech2Text2PreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置、处理器和标记器模块
    from .configuration_speech_to_text_2 import SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Speech2Text2Config
    from .processing_speech_to_text_2 import Speech2Text2Processor
    from .tokenization_speech_to_text_2 import Speech2Text2Tokenizer

    # 检查是否有torch依赖可用，否则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型模块
        from .modeling_speech_to_text_2 import (
            SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2Text2ForCausalLM,
            Speech2Text2PreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```