# `.\models\fastspeech2_conformer\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    # 配置文件相关导入
    "configuration_fastspeech2_conformer": [
        "FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    # FastSpeech2 Conformer模型的tokenization模块导入
    "tokenization_fastspeech2_conformer": ["FastSpeech2ConformerTokenizer"],
}

# 检查是否可用Torch，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若Torch可用，则增加模型相关导入
    _import_structure["modeling_fastspeech2_conformer"] = [
        "FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FastSpeech2ConformerWithHifiGan",
        "FastSpeech2ConformerHifiGan",
        "FastSpeech2ConformerModel",
        "FastSpeech2ConformerPreTrainedModel",
    ]

# 若为类型检查环境
if TYPE_CHECKING:
    # 导入配置文件相关内容
    from .configuration_fastspeech2_conformer import (
        FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerWithHifiGanConfig,
    )
    # 导入tokenization模块内容
    from .tokenization_fastspeech2_conformer import FastSpeech2ConformerTokenizer

    # 再次检查Torch是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若Torch可用，则导入模型相关内容
        from .modeling_fastspeech2_conformer import (
            FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            FastSpeech2ConformerHifiGan,
            FastSpeech2ConformerModel,
            FastSpeech2ConformerPreTrainedModel,
            FastSpeech2ConformerWithHifiGan,
        )

# 若不在类型检查环境中，则将当前模块替换为_LazyModule对象
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```