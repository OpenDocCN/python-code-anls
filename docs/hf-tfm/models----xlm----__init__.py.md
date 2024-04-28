# `.\transformers\models\xlm\__init__.py`

```
# 导入TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 导入OptionalDependencyNotAvailable异常、_LazyModule对象和一些检查库是否可用的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available


# 准备一个字典，用于保存需要导入的模块和它们的属性
_import_structure = {
    "configuration_xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMOnnxConfig"], # XLM模型的配置
    "tokenization_xlm": ["XLMTokenizer"], # XLM模型的tokenizer
}

# 检查torch是否可用，如果不可用则引发OptionalDependencyNotAvailable异常
# 如果可用，则将一些torch模型相关的属性添加到_import_structure字典中
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_xlm"] = [
        "XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMForMultipleChoice",
        "XLMForQuestionAnswering",
        "XLMForQuestionAnsweringSimple",
        "XLMForSequenceClassification",
        "XLMForTokenClassification",
        "XLMModel",
        "XLMPreTrainedModel",
        "XLMWithLMHeadModel",
    ]

# 检查tensorflow是否可用，如果不可用则引发OptionalDependencyNotAvailable异常
# 如果可用，则将一些tensorflow模型相关的属性添加到_import_structure字典中
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_xlm"] = [
        "TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXLMForMultipleChoice",
        "TFXLMForQuestionAnsweringSimple",
        "TFXLMForSequenceClassification",
        "TFXLMForTokenClassification",
        "TFXLMMainLayer",
        "TFXLMModel",
        "TFXLMPreTrainedModel",
        "TFXLMWithLMHeadModel",
    ]


# 如果是类型检查模式，则导入一些模块和它们的属性，并定义一些类型别名
if TYPE_CHECKING:
    from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMOnnxConfig # XLM模型的配置类相关属性
    from .tokenization_xlm import XLMTokenizer # XLM模型的tokenizer类属性

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_xlm import (
            XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMForMultipleChoice,
            XLMForQuestionAnswering,
            XLMForQuestionAnsweringSimple,
            XLMForSequenceClassification,
            XLMForTokenClassification,
            XLMModel,
            XLMPreTrainedModel,
            XLMWithLMHeadModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件不成立，从当前包中引入以下模块
    from .modeling_tf_xlm import (
        TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST
        TFXLMForMultipleChoice,  # 导入TFXLMForMultipleChoice
        TFXLMForQuestionAnsweringSimple,  # 导入TFXLMForQuestionAnsweringSimple
        TFXLMForSequenceClassification,  # 导入TFXLMForSequenceClassification
        TFXLMForTokenClassification,  # 导入TFXLMForTokenClassification
        TFXLMMainLayer,  # 导入TFXLMMainLayer
        TFXLMModel,  # 导入TFXLMModel
        TFXLMPreTrainedModel,  # 导入TFXLMPreTrainedModel
        TFXLMWithLMHeadModel,  # 导入TFXLMWithLMHeadModel
    )
# 如果不是在一个包中，导入 sys 模块
import sys
# 将当前模块注册到 sys 模块的字典中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```