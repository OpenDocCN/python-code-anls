# `.\models\wav2vec2_conformer\__init__.py`

```py
# 导入所需的依赖和模块
from typing import TYPE_CHECKING  # 导入类型检查模块

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available  # 导入自定义工具函数和类


_import_structure = {
    "configuration_wav2vec2_conformer": [
        "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "Wav2Vec2ConformerConfig",  # Wav2Vec2Conformer 的配置类
    ],
}

# 检查是否有 torch 可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，添加下列模块到导入结构中
    _import_structure["modeling_wav2vec2_conformer"] = [
        "WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "Wav2Vec2ConformerForAudioFrameClassification",  # 用于音频帧分类的 Wav2Vec2Conformer 模型
        "Wav2Vec2ConformerForCTC",  # 用于 CTC 的 Wav2Vec2Conformer 模型
        "Wav2Vec2ConformerForPreTraining",  # 用于预训练的 Wav2Vec2Conformer 模型
        "Wav2Vec2ConformerForSequenceClassification",  # 用于序列分类的 Wav2Vec2Conformer 模型
        "Wav2Vec2ConformerForXVector",  # 用于 XVector 的 Wav2Vec2Conformer 模型
        "Wav2Vec2ConformerModel",  # Wav2Vec2Conformer 模型
        "Wav2Vec2ConformerPreTrainedModel",  # 预训练的 Wav2Vec2Conformer 模型
    ]

# 如果是类型检查模式，从相应的模块中导入特定的符号
if TYPE_CHECKING:
    from .configuration_wav2vec2_conformer import (
        WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置映射
        Wav2Vec2ConformerConfig,  # Wav2Vec2Conformer 的配置类
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_wav2vec2_conformer import (
            WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            Wav2Vec2ConformerForAudioFrameClassification,  # 用于音频帧分类的 Wav2Vec2Conformer 模型
            Wav2Vec2ConformerForCTC,  # 用于 CTC 的 Wav2Vec2Conformer 模型
            Wav2Vec2ConformerForPreTraining,  # 用于预训练的 Wav2Vec2Conformer 模型
            Wav2Vec2ConformerForSequenceClassification,  # 用于序列分类的 Wav2Vec2Conformer 模型
            Wav2Vec2ConformerForXVector,  # 用于 XVector 的 Wav2Vec2Conformer 模型
            Wav2Vec2ConformerModel,  # Wav2Vec2Conformer 模型
            Wav2Vec2ConformerPreTrainedModel,  # 预训练的 Wav2Vec2Conformer 模型
        )

# 如果不是类型检查模式，将当前模块设为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```