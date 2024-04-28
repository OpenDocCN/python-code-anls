# `.\transformers\models\unispeech\__init__.py`

```py
# 导入必要的模块和依赖项，包括类型检查和延迟加载模块
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {"configuration_unispeech": ["UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP", "UniSpeechConfig"]}

# 检查是否有可用的 torch 库，如果没有则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果有 torch 库，则添加相应的模型配置和模型类到导入结构中
    _import_structure["modeling_unispeech"] = [
        "UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST",
        "UniSpeechForCTC",
        "UniSpeechForPreTraining",
        "UniSpeechForSequenceClassification",
        "UniSpeechModel",
        "UniSpeechPreTrainedModel",
    ]

# 如果类型检查为真，则从相应模块导入特定类
if TYPE_CHECKING:
    from .configuration_unispeech import UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP, UniSpeechConfig
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_unispeech import (
            UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST,
            UniSpeechForCTC,
            UniSpeechForPreTraining,
            UniSpeechForSequenceClassification,
            UniSpeechModel,
            UniSpeechPreTrainedModel,
        )
# 如果类型检查为假，则将当前模块设为一个延迟加载模块
else:
    import sys
    # 动态设置当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```