# `.\models\audio_spectrogram_transformer\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义异常和模块延迟加载工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构字典，包含配置、特征提取和模型相关内容
_import_structure = {
    "configuration_audio_spectrogram_transformer": [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ASTConfig",
    ],
    "feature_extraction_audio_spectrogram_transformer": ["ASTFeatureExtractor"],
}

# 检查是否存在torch库，若不存在则引发自定义的依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则添加模型相关的导入结构到_import_structure字典中
    _import_structure["modeling_audio_spectrogram_transformer"] = [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ASTForAudioClassification",
        "ASTModel",
        "ASTPreTrainedModel",
    ]

# 如果是类型检查模式，则从各自的模块导入特定的符号
if TYPE_CHECKING:
    from .configuration_audio_spectrogram_transformer import (
        AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ASTConfig,
    )
    from .feature_extraction_audio_spectrogram_transformer import ASTFeatureExtractor

    # 同样地，检查是否存在torch库，若不存在则引发自定义的依赖不可用异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果torch可用，则从模型模块中导入特定的符号
        from .modeling_audio_spectrogram_transformer import (
            AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ASTForAudioClassification,
            ASTModel,
            ASTPreTrainedModel,
        )

# 如果不是类型检查模式，则将当前模块设为一个LazyModule，用于延迟加载相关依赖
else:
    import sys

    # 设置当前模块的sys.modules，使其变为一个延迟加载的模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```