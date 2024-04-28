# `.\transformers\models\wav2vec2_bert\__init__.py`

```py
# 版权声明和许可证
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下位置获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，不附带任何明示或暗示的保证或条件
# 详细了解许可证内容和限制
from typing import TYPE_CHECKING

# 从 ...utils 模块导入 OptionalDependencyNotAvailable, _LazyModule, is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


# 定义模块导入结构
_import_structure = {
    "configuration_wav2vec2_bert": [
        "WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2BertConfig",
    ],
    "processing_wav2vec2_bert": ["Wav2Vec2BertProcessor"],
}

# 尝试导入 torch 模块，如果不可用，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，扩展_import_structure，添加 modeling_wav2vec2_bert 模块的导入结构
    _import_structure["modeling_wav2vec2_bert"] = [
        "WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Wav2Vec2BertForAudioFrameClassification",
        "Wav2Vec2BertForCTC",
        "Wav2Vec2BertForSequenceClassification",
        "Wav2Vec2BertForXVector",
        "Wav2Vec2BertModel",
        "Wav2Vec2BertPreTrainedModel",
    ]

# 如果是类型检查，导入相关模块
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

# 如果不是类型检查，将该模块设为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```