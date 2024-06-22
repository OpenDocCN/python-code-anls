# `.\transformers\models\led\__init__.py`

```py
# 版权声明和许可
# 版权声明和许可信息，详细规定了如何使用该代码
# 引入类型检查
from typing import TYPE_CHECKING
# 引入相关的依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 设置导入结构
_import_structure = {
    "configuration_led": ["LED_PRETRAINED_CONFIG_ARCHIVE_MAP", "LEDConfig"],
    "tokenization_led": ["LEDTokenizer"],
}

# 检查是否 tokenizers 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_led_fast"] = ["LEDTokenizerFast"]

# 检查是否 torch 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_led"] = [
        "LED_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LEDForConditionalGeneration",
        "LEDForQuestionAnswering",
        "LEDForSequenceClassification",
        "LEDModel",
        "LEDPreTrainedModel",
    ]

# 检查是否 tf 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_led"] = ["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 从相关模块引入所需的类和常量
    from .configuration_led import LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LEDConfig
    from .tokenization_led import LEDTokenizer

    # 检查是否 tokenizers 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_led_fast import LEDTokenizerFast

    # 检查是否 torch 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_led import (
            LED_PRETRAINED_MODEL_ARCHIVE_LIST,
            LEDForConditionalGeneration,
            LEDForQuestionAnswering,
            LEDForSequenceClassification,
            LEDModel,
            LEDPreTrainedModel,
        )

    # 检查是否 tf 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_led import TFLEDForConditionalGeneration, TFLEDModel, TFLEDPreTrainedModel

# 如果不是类型检查环境
else:
    import sys
    # 将当前模块注册到sys.modules字典中，以当前模块名作为键，_LazyModule对象作为值
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```