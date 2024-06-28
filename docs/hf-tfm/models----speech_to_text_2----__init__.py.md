# `.\models\speech_to_text_2\__init__.py`

```py
# 版权声明和许可信息
#
# 本代码受版权保护，版权归 The HuggingFace Team 所有。
#
# 根据 Apache 许可证 2.0 版本授权使用本文件；
# 除非符合许可证的规定，否则不得使用本文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件按“原样”分发，不提供任何明示或暗示的保证或条件。
# 有关具体语言版本的详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的符号
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_speech_to_text_2": ["SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Speech2Text2Config"],
    "processing_speech_to_text_2": ["Speech2Text2Processor"],
    "tokenization_speech_to_text_2": ["Speech2Text2Tokenizer"],
}

# 检查是否可以导入 torch，如果不行则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 torch，则添加额外的模块到导入结构中
    _import_structure["modeling_speech_to_text_2"] = [
        "SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Speech2Text2ForCausalLM",
        "Speech2Text2PreTrainedModel",
    ]

# 如果是类型检查模式，则导入具体的类型
if TYPE_CHECKING:
    from .configuration_speech_to_text_2 import SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Speech2Text2Config
    from .processing_speech_to_text_2 import Speech2Text2Processor
    from .tokenization_speech_to_text_2 import Speech2Text2Tokenizer

    # 再次检查是否可以导入 torch，如果不行则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可以导入 torch，则导入额外的模块类型
        from .modeling_speech_to_text_2 import (
            SPEECH_TO_TEXT_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speech2Text2ForCausalLM,
            Speech2Text2PreTrainedModel,
        )

# 如果不是类型检查模式，则设置模块为 LazyModule，延迟加载
else:
    import sys

    # 设置当前模块为 LazyModule 形式，根据 _import_structure 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```