# `.\transformers\models\audio_spectrogram_transformer\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_audio_spectrogram_transformer": [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ASTConfig",
    ],
    "feature_extraction_audio_spectrogram_transformer": ["ASTFeatureExtractor"],
}

# 检查是否存在 torch 模块，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 模块，则添加以下模块到导入结构中
    _import_structure["modeling_audio_spectrogram_transformer"] = [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ASTForAudioClassification",
        "ASTModel",
        "ASTPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置和特征提取模块
    from .configuration_audio_spectrogram_transformer import (
        AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ASTConfig,
    )
    from .feature_extraction_audio_spectrogram_transformer import ASTFeatureExtractor

    # 再次检查是否存在 torch 模块，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关模块
        from .modeling_audio_spectrogram_transformer import (
            AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            ASTForAudioClassification,
            ASTModel,
            ASTPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```