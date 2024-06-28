# `.\models\led\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构的字典，包含LED模型配置和标记化
_import_structure = {
    "configuration_led": ["LED_PRETRAINED_CONFIG_ARCHIVE_MAP", "LEDConfig"],
    "tokenization_led": ["LEDTokenizer"],
}

# 检查是否可用标记器，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将LED快速标记化添加到导入结构中
    _import_structure["tokenization_led_fast"] = ["LEDTokenizerFast"]

# 检查是否可用PyTorch，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将LED模型相关类添加到导入结构中
    _import_structure["modeling_led"] = [
        "LED_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LEDForConditionalGeneration",
        "LEDForQuestionAnswering",
        "LEDForSequenceClassification",
        "LEDModel",
        "LEDPreTrainedModel",
    ]

# 检查是否可用TensorFlow，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将TensorFlow版LED模型相关类添加到导入结构中
    _import_structure["modeling_tf_led"] = ["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"]

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 从相应模块导入LED模型配置和标记化类
    from .configuration_led import LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LEDConfig
    from .tokenization_led import LEDTokenizer

    # 检查是否可用标记器，若不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从快速标记化模块导入LED快速标记化类
        from .tokenization_led_fast import LEDTokenizerFast

    # 检查是否可用PyTorch，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从LED建模模块导入相关类
        from .modeling_led import (
            LED_PRETRAINED_MODEL_ARCHIVE_LIST,
            LEDForConditionalGeneration,
            LEDForQuestionAnswering,
            LEDForSequenceClassification,
            LEDModel,
            LEDPreTrainedModel,
        )

    # 检查是否可用TensorFlow，若不可用则忽略
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从TensorFlow版LED建模模块导入相关类
        from .modeling_tf_led import TFLEDForConditionalGeneration, TFLEDModel, TFLEDPreTrainedModel

# 如果不是类型检查阶段，导入sys模块
else:
    import sys
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```