# `.\transformers\models\wav2vec2_conformer\__init__.py`

```
# 版权声明和许可信息
#
# 2022 年由 HuggingFace 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本授权，
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件根据“原样”基础分发，
# 没有任何明示或默示的担保或条件。
# 有关特定语言确定权限和限制，请参见许可证。
from typing import TYPE_CHECKING

# 导入自定义的模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_wav2vec2_conformer": [
        "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2ConformerConfig",
    ],
}

# 检查是否有 torch 库可用，如果没有则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果有 torch 库可用，则添加下列模块到导入结构
    _import_structure["modeling_wav2vec2_conformer"] = [
        "WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Wav2Vec2ConformerForAudioFrameClassification",
        "Wav2Vec2ConformerForCTC",
        "Wav2Vec2ConformerForPreTraining",
        "Wav2Vec2ConformerForSequenceClassification",
        "Wav2Vec2ConformerForXVector",
        "Wav2Vec2ConformerModel",
        "Wav2Vec2ConformerPreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 从 configuration_wav2vec2_conformer 模块中导入特定项
    from .configuration_wav2vec2_conformer import (
        WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2ConformerConfig,
    )

    # 再次检查是否有 torch 库可用，如果没有则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 modeling_wav2vec2_conformer 模块中导入特定项
        from .modeling_wav2vec2_conformer import (
            WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            Wav2Vec2ConformerForAudioFrameClassification,
            Wav2Vec2ConformerForCTC,
            Wav2Vec2ConformerForPreTraining,
            Wav2Vec2ConformerForSequenceClassification,
            Wav2Vec2ConformerForXVector,
            Wav2Vec2ConformerModel,
            Wav2Vec2ConformerPreTrainedModel,
        )

# 否则，以 LazyModule 形式将引入的模块导入当前模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```