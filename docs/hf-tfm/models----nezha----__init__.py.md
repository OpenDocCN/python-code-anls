# `.\models\nezha\__init__.py`

```
# 导入必要的模块和函数，包括自定义的异常和延迟加载模块
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构，包含配置和模型类名称
_import_structure = {
    "configuration_nezha": ["NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP", "NezhaConfig"],
}

# 检查是否有 torch 库可用，若不可用则引发自定义的依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 可用，则添加模型相关的导入结构
    _import_structure["modeling_nezha"] = [
        "NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NezhaForNextSentencePrediction",
        "NezhaForMaskedLM",
        "NezhaForPreTraining",
        "NezhaForMultipleChoice",
        "NezhaForQuestionAnswering",
        "NezhaForSequenceClassification",
        "NezhaForTokenClassification",
        "NezhaModel",
        "NezhaPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入配置和模型类名
if TYPE_CHECKING:
    from .configuration_nezha import NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP, NezhaConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_nezha import (
            NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST,
            NezhaForMaskedLM,
            NezhaForMultipleChoice,
            NezhaForNextSentencePrediction,
            NezhaForPreTraining,
            NezhaForQuestionAnswering,
            NezhaForSequenceClassification,
            NezhaForTokenClassification,
            NezhaModel,
            NezhaPreTrainedModel,
        )

# 如果不是类型检查阶段，则进行模块的延迟加载和替换
else:
    import sys

    # 将当前模块替换为 LazyModule 实例，进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```