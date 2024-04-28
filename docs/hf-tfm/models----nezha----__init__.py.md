# `.\transformers\models\nezha\__init__.py`

```py
# 导入需要的模块和类型
# 首先导入必要的模块和类型，包括了自定义的utils模块、判断tokenizers是否可用的函数、判断torch是否可用的函数、TYPE_CHECKING类型
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
# 定义一个字典来存储模块的导入结构
_import_structure = {
    "configuration_nezha": ["NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP", "NezhaConfig"],
}

# 检查torch是否可用
# 如果torch不可用，则抛出OptionalDependencyNotAvailable异常
# 否则，将NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST, NezhaForNextSentencePrediction, NezhaForMaskedLM, NezhaForPreTraining等模块添加到_import_structure字典中
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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


# 类型检查
# 如果TYPE_CHECKING为True，则导入模块的相关类
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


# 如果TYPE_CHECKING不为True，则使用_LazyModule创建模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```