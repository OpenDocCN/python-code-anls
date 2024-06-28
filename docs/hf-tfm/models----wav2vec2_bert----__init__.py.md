# `.\models\wav2vec2_bert\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_wav2vec2_bert": [
        "WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2BertConfig",
    ],
    "processing_wav2vec2_bert": ["Wav2Vec2BertProcessor"],
}

# 检查是否存在 Torch 库，若不存在则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 存在，添加模型相关的导入结构
    _import_structure["modeling_wav2vec2_bert"] = [
        "WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Wav2Vec2BertForAudioFrameClassification",
        "Wav2Vec2BertForCTC",
        "Wav2Vec2BertForSequenceClassification",
        "Wav2Vec2BertForXVector",
        "Wav2Vec2BertModel",
        "Wav2Vec2BertPreTrainedModel",
    ]

# 如果是类型检查环境，进行类型导入
if TYPE_CHECKING:
    from .configuration_wav2vec2_bert import (
        WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2BertConfig,
    )
    from .processing_wav2vec2_bert import Wav2Vec2BertProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_wav2vec2_bert import (
            WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2BertForAudioFrameClassification,
            Wav2Vec2BertForCTC,
            Wav2Vec2BertForSequenceClassification,
            Wav2Vec2BertForXVector,
            Wav2Vec2BertModel,
            Wav2Vec2BertPreTrainedModel,
        )

# 非类型检查环境下，将当前模块替换为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```