# `.\models\unispeech_sat\__init__.py`

```
# 导入必要的模块和函数，包括依赖检查和异常处理
from typing import TYPE_CHECKING

# 导入自定义的异常类和延迟加载模块函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和模型相关内容
_import_structure = {
    "configuration_unispeech_sat": ["UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "UniSpeechSatConfig"],
}

# 检查是否可以导入 torch，如果不行则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 torch，则添加与模型相关的导入内容
    _import_structure["modeling_unispeech_sat"] = [
        "UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "UniSpeechSatForAudioFrameClassification",
        "UniSpeechSatForCTC",
        "UniSpeechSatForPreTraining",
        "UniSpeechSatForSequenceClassification",
        "UniSpeechSatForXVector",
        "UniSpeechSatModel",
        "UniSpeechSatPreTrainedModel",
    ]

# 如果是类型检查阶段，导入配置和模型相关内容的具体类
if TYPE_CHECKING:
    from .configuration_unispeech_sat import UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP, UniSpeechSatConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_unispeech_sat import (
            UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            UniSpeechSatForAudioFrameClassification,
            UniSpeechSatForCTC,
            UniSpeechSatForPreTraining,
            UniSpeechSatForSequenceClassification,
            UniSpeechSatForXVector,
            UniSpeechSatModel,
            UniSpeechSatPreTrainedModel,
        )

# 如果不是类型检查阶段，则将模块注册为懒加载模块，以延迟加载相关内容
else:
    import sys

    # 将当前模块注册为懒加载模块，用于实现模块的延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```