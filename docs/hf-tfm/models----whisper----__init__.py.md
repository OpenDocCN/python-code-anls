# `.\transformers\models\whisper\__init__.py`

```
# 版权声明和许可协议
# 版权声明：2022 年由 HuggingFace Team 版权所有
# 许可协议：基于 Apache 许可证 2.0 版本，除非符合许可协议，否则不得使用此文件。可以在以下网址获取许可协议的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可协议分发的软件将基于“AS IS”基础分发，无论是明示的还是暗示的保证或条件。请参阅许可协议以了解特定语言管理权限和限制

# 从类型提示模块导入模式 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从 utils 模块导入相关内容
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构的字典
_import_structure = {
    "configuration_whisper": ["WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP", "WhisperConfig", "WhisperOnnxConfig"],  # 导入配置的文件名
    "feature_extraction_whisper": ["WhisperFeatureExtractor"],  # 导入特征提取相关内容
    "processing_whisper": ["WhisperProcessor"],  # 导入处理相关内容
    "tokenization_whisper": ["WhisperTokenizer"],  # 导入标记化相关内容
}

# 异常处理：如果 tokenizers 不可用，则提升可选依赖项不可用
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 否则将导入结构的字典添加 tokenization_whisper_fast 相关内容
    _import_structure["tokenization_whisper_fast"] = ["WhisperTokenizerFast"]

# 异常处理：如果 torch 不可用，则提升可选依赖项不可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 否则将导入结构的字典添加 modeling_whisper 相关内容
    _import_structure["modeling_whisper"] = [
        "WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "WhisperForCausalLM",
        "WhisperForConditionalGeneration",
        "WhisperModel",
        "WhisperPreTrainedModel",
        "WhisperForAudioClassification",
    ]

# 异常处理：如果 tensorflow 不可用，则提升可选依赖项不可用
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 否则将导入结构的字典添加 modeling_tf_whisper 相关内容
    _import_structure["modeling_tf_whisper"] = [
        "TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFWhisperForConditionalGeneration",
        "TFWhisperModel",
        "TFWhisperPreTrainedModel",
    ]

# 异常处理：如果 flax 不可用，则提升���选依赖项不可用
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 否则将导入结构的字典添加 modeling_flax_whisper 相关内容
    _import_structure["modeling_flax_whisper"] = [
        "FlaxWhisperForConditionalGeneration",
        "FlaxWhisperModel",
        "FlaxWhisperPreTrainedModel",
        "FlaxWhisperForAudioClassification",
    ]

# 如果是类型检查模式，则导入相应模块内容
if TYPE_CHECKING:
    from .configuration_whisper import WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP, WhisperConfig, WhisperOnnxConfig
    from .feature_extraction_whisper import WhisperFeatureExtractor
    from .processing_whisper import WhisperProcessor
    from .tokenization_whisper import WhisperTokenizer

    # 异常处理：如果 tokenizers 不可用，则提升可选依赖项不可用
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:  # 否则将导入 tokenization_whisper_fast 相关内容
        from .tokenization_whisper_fast import WhisperTokenizerFast
    # 检查是否安装了 torch 库，如果没有则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 处理 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则执行下面的代码块
    else:
        # 导入 WHISPER 相关模块
        from .modeling_whisper import (
            WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            WhisperForAudioClassification,
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperModel,
            WhisperPreTrainedModel,
        )

    # 检查是否安装了 tensorflow 库，如果没有则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 处理 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则执行下面的代码块
    else:
        # 导入 TF_WHISPER 相关模块
        from .modeling_tf_whisper import (
            TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWhisperForConditionalGeneration,
            TFWhisperModel,
            TFWhisperPreTrainedModel,
        )

    # 检查是否安装了 flax 库，如果没有则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 处理 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则执行下面的代码块
    else:
        # 导入 FlaxWhisper 相关模块
        from .modeling_flax_whisper import (
            FlaxWhisperForAudioClassification,
            FlaxWhisperForConditionalGeneration,
            FlaxWhisperModel,
            FlaxWhisperPreTrainedModel,
        )
# 如果不满足前面的条件，则执行以下代码

# 导入 sys 模块
import sys

# 将当前模块添加到 sys 模块的 modules 列表中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```