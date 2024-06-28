# `.\models\musicgen_melody\__init__.py`

```py
# 版权声明和许可证信息
# 版权所有 2024 年 HuggingFace 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”分发，
# 无任何明示或暗示的保证或条件。
# 有关更多信息，请参见许可证。
from typing import TYPE_CHECKING

# 导入必要的模块和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_torchaudio_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_musicgen_melody": [
        "MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MusicgenMelodyConfig",
        "MusicgenMelodyDecoderConfig",
    ],
}

# 检查是否存在 torch，如果不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch，添加相关的模型配置和类到导入结构中
    _import_structure["modeling_musicgen_melody"] = [
        "MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MusicgenMelodyForConditionalGeneration",
        "MusicgenMelodyForCausalLM",
        "MusicgenMelodyModel",
        "MusicgenMelodyPreTrainedModel",
    ]

# 检查是否存在 torchaudio，如果不存在则抛出异常
try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torchaudio，添加相关的特征提取和处理器类到导入结构中
    _import_structure["feature_extraction_musicgen_melody"] = ["MusicgenMelodyFeatureExtractor"]
    _import_structure["processing_musicgen_melody"] = ["MusicgenMelodyProcessor"]

# 如果当前环境支持类型检查，进行额外的导入
if TYPE_CHECKING:
    from .configuration_musicgen_melody import (
        MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MusicgenMelodyConfig,
        MusicgenMelodyDecoderConfig,
    )

    # 如果存在 torch，导入相关模型类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_musicgen_melody import (
            MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST,
            MusicgenMelodyForCausalLM,
            MusicgenMelodyForConditionalGeneration,
            MusicgenMelodyModel,
            MusicgenMelodyPreTrainedModel,
        )

    # 如果存在 torchaudio，导入相关特征提取和处理器类
    try:
        if not is_torchaudio_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_musicgen_melody import MusicgenMelodyFeatureExtractor
        from .processing_musicgen_melody import MusicgenMelodyProcessor

# 如果当前环境不支持类型检查，定义 LazyModule 来延迟加载模块
else:
    import sys

    # 将当前模块设置为 LazyModule 类型，用于按需加载导入结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```