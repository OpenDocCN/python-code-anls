# `.\transformers\models\wav2vec2\__init__.py`

```
# 导入必要的库和模块
from typing import TYPE_CHECKING
# 导入必要的异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的结构
_import_structure = {
    "configuration_wav2vec2": ["WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Wav2Vec2Config"],
    "feature_extraction_wav2vec2": ["Wav2Vec2FeatureExtractor"],
    "processing_wav2vec2": ["Wav2Vec2Processor"],
    "tokenization_wav2vec2": ["Wav2Vec2CTCTokenizer", "Wav2Vec2Tokenizer"],
}

# 检查是否torch可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则添加相关模型到模块结构中
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

# 检查是否tensorflow可用，如果不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果tensorflow可用，则添加相关模型到模块结构中
    _import_structure["modeling_tf_wav2vec2"] = [
        "TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFWav2Vec2ForCTC",
        "TFWav2Vec2Model",
        "TFWav2Vec2PreTrainedModel",
        "TFWav2Vec2ForSequenceClassification",
    ]

# 检查是否flax可用，如果不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果flax可用，则添加相关模型到模块结构中
    _import_structure["modeling_flax_wav2vec2"] = [
        "FlaxWav2Vec2ForCTC",
        "FlaxWav2Vec2ForPreTraining",
        "FlaxWav2Vec2Model",
        "FlaxWav2Vec2PreTrainedModel",
    ]

# 如果是类型检查阶段，则导入一些配置、特征提取、处理和分词模块的内容
if TYPE_CHECKING:
    from .configuration_wav2vec2 import WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP, Wav2Vec2Config
    from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
    from .processing_wav2vec2 import Wav2Vec2Processor
    from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2Tokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是之前定义的模块，则从modeling_wav2vec2中导入相应模块和变量
    else:
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

    # 尝试导入TensorFlow模块，如果不可用则抛出异常OptionalDependencyNotAvailable
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果成功导入TensorFlow模块，从modeling_tf_wav2vec2中导入相应模块和变量
    else:
        from .modeling_tf_wav2vec2 import (
            TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFWav2Vec2ForCTC,
            TFWav2Vec2ForSequenceClassification,
            TFWav2Vec2Model,
            TFWav2Vec2PreTrainedModel,
        )

    # 尝试导入Flax模块，如果不可用则抛出异常OptionalDependencyNotAvailable
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果成功导入Flax模块，从modeling_tf_wav2vec2中导入相应模块和变量
    else:
        from .modeling_tf_wav2vec2 import (
            FlaxWav2Vec2ForCTC,
            FlaxWav2Vec2ForPreTraining,
            FlaxWav2Vec2Model,
            FlaxWav2Vec2PreTrainedModel,
        )
# 如果以上条件都不满足，则执行以下代码块
import sys  # 导入 sys 模块

# 使用 _LazyModule 类创建一个模块对象，并将其赋值给当前模块的名称，同时传入指定的参数
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```