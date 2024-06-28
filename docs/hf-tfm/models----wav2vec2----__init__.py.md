# `.\models\wav2vec2\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从内部模块中导入异常类和延迟加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义导入结构的字典，用于指定每个模块导入的内容
_import_structure = {
    "configuration_wav2vec2": ["WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Wav2Vec2Config"],
    "feature_extraction_wav2vec2": ["Wav2Vec2FeatureExtractor"],
    "processing_wav2vec2": ["Wav2Vec2Processor"],
    "tokenization_wav2vec2": ["Wav2Vec2CTCTokenizer", "Wav2Vec2Tokenizer"],
}

# 尝试导入 torch 相关模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，则添加 torch 版本的模型结构到 _import_structure 中
    _import_structure["modeling_wav2vec2"] = [
        "WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Wav2Vec2ForAudioFrameClassification",
        "Wav2Vec2ForCTC",
        "Wav2Vec2ForMaskedLM",
        "Wav2Vec2ForPreTraining",
        "Wav2Vec2ForSequenceClassification",
        "Wav2Vec2ForXVector",
        "Wav2Vec2Model",
        "Wav2Vec2PreTrainedModel",
    ]

# 尝试导入 tensorflow 相关模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，则添加 tensorflow 版本的模型结构到 _import_structure 中
    _import_structure["modeling_tf_wav2vec2"] = [
        "TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFWav2Vec2ForCTC",
        "TFWav2Vec2Model",
        "TFWav2Vec2PreTrainedModel",
        "TFWav2Vec2ForSequenceClassification",
    ]

# 尝试导入 flax 相关模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，则添加 flax 版本的模型结构到 _import_structure 中
    _import_structure["modeling_flax_wav2vec2"] = [
        "FlaxWav2Vec2ForCTC",
        "FlaxWav2Vec2ForPreTraining",
        "FlaxWav2Vec2Model",
        "FlaxWav2Vec2PreTrainedModel",
    ]

# 如果正在进行类型检查，导入类型检查所需的模块和类
if TYPE_CHECKING:
    from .configuration_wav2vec2 import WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Wav2Vec2Config
    from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
    from .processing_wav2vec2 import Wav2Vec2Processor
    from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer

    # 再次尝试导入 torch 相关模块，用于类型检查，如果不可用则跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关模块和预训练模型的存档列表（针对其他框架）
        from .modeling_wav2vec2 import (
            WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForCTC,
            Wav2Vec2ForMaskedLM,
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2ForXVector,
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

    try:
        # 检查是否可用 TensorFlow
        if not is_tf_available():
            # 如果 TensorFlow 不可用，则抛出异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，不做处理，继续执行
        pass
    else:
        # 导入 TensorFlow 版本的模型相关模块和预训练模型的存档列表
        from .modeling_tf_wav2vec2 import (
            TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWav2Vec2ForCTC,
            TFWav2Vec2ForSequenceClassification,
            TFWav2Vec2Model,
            TFWav2Vec2PreTrainedModel,
        )

    try:
        # 检查是否可用 Flax
        if not is_flax_available():
            # 如果 Flax 不可用，则抛出异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 Flax 不可用，不做处理，继续执行
        pass
    else:
        # 导入 Flax 版本的模型相关模块和预训练模型的存档列表
        from .modeling_tf_wav2vec2 import (
            FlaxWav2Vec2ForCTC,
            FlaxWav2Vec2ForPreTraining,
            FlaxWav2Vec2Model,
            FlaxWav2Vec2PreTrainedModel,
        )
else:
    # 导入系统模块 sys
    import sys
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```