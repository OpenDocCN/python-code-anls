# `.\models\whisper\__init__.py`

```
# 导入所需的模块和函数
from typing import TYPE_CHECKING

# 导入可能的异常处理类和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，每个模块对应其所需的类或函数列表
_import_structure = {
    "configuration_whisper": ["WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP", "WhisperConfig", "WhisperOnnxConfig"],
    "feature_extraction_whisper": ["WhisperFeatureExtractor"],
    "processing_whisper": ["WhisperProcessor"],
    "tokenization_whisper": ["WhisperTokenizer"],
}

# 检查是否存在 tokenizers 库，如果不存在则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，则将 tokenization_whisper_fast 模块添加到导入结构中
    _import_structure["tokenization_whisper_fast"] = ["WhisperTokenizerFast"]

# 检查是否存在 torch 库，如果不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，则将 modeling_whisper 模块添加到导入结构中
    _import_structure["modeling_whisper"] = [
        "WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "WhisperForCausalLM",
        "WhisperForConditionalGeneration",
        "WhisperModel",
        "WhisperPreTrainedModel",
        "WhisperForAudioClassification",
    ]

# 检查是否存在 tensorflow 库，如果不存在则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，则将 modeling_tf_whisper 模块添加到导入结构中
    _import_structure["modeling_tf_whisper"] = [
        "TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFWhisperForConditionalGeneration",
        "TFWhisperModel",
        "TFWhisperPreTrainedModel",
    ]

# 检查是否存在 flax 库，如果不存在则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，则将 modeling_flax_whisper 模块添加到导入结构中
    _import_structure["modeling_flax_whisper"] = [
        "FlaxWhisperForConditionalGeneration",
        "FlaxWhisperModel",
        "FlaxWhisperPreTrainedModel",
        "FlaxWhisperForAudioClassification",
    ]

# 如果是类型检查模式，导入相关的类型定义
if TYPE_CHECKING:
    from .configuration_whisper import WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP, WhisperConfig, WhisperOnnxConfig
    from .feature_extraction_whisper import WhisperFeatureExtractor
    from .processing_whisper import WhisperProcessor
    from .tokenization_whisper import WhisperTokenizer

    # 检查是否存在 tokenizers 库，如果可用，则导入 tokenization_whisper_fast 模块
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_whisper_fast import WhisperTokenizerFast
    # 检查是否安装了 Torch 库，如果没有则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，如果引发则不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Torch 版本的 Whisper 模型相关内容
        from .modeling_whisper import (
            WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            WhisperForAudioClassification,
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperModel,
            WhisperPreTrainedModel,
        )

    # 检查是否安装了 TensorFlow 库，如果没有则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，如果引发则不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 TensorFlow 版本的 Whisper 模型相关内容
        from .modeling_tf_whisper import (
            TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWhisperForConditionalGeneration,
            TFWhisperModel,
            TFWhisperPreTrainedModel,
        )

    # 检查是否安装了 Flax 库，如果没有则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，如果引发则不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Flax 版本的 Whisper 模型相关内容
        from .modeling_flax_whisper import (
            FlaxWhisperForAudioClassification,
            FlaxWhisperForConditionalGeneration,
            FlaxWhisperModel,
            FlaxWhisperPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态修改当前模块的属性
    import sys

    # 将当前模块注册到 sys.modules[__name__] 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```