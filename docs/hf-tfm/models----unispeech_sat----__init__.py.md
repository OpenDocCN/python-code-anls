# `.\transformers\models\unispeech_sat\__init__.py`

```py
# 导入类型检查模块，用于检查是否支持类型检查
from typing import TYPE_CHECKING

# 导入必要的工具函数和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构，包含配置和模型
_import_structure = {
    "configuration_unispeech_sat": ["UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "UniSpeechSatConfig"],
}

# 检查是否支持 PyTorch，若不支持则引发可选依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若支持 PyTorch，则添加 UniSpeechSat 相关模型到导入结构中
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

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入 UNISPEECH_SAT 配置相关内容
    from .configuration_unispeech_sat import UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP, UniSpeechSatConfig

    # 再次检查是否支持 PyTorch，若不支持则引发可选依赖不可用异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 UNISPEECH_SAT 模型相关内容
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

# 如果不是类型检查环境，则进行惰性加载
else:
    import sys

    # 将当前模块替换为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

```  
```