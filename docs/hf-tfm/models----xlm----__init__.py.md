# `.\models\xlm\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义模块和异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义导入结构字典，包含不同模块和对应的类/函数列表
_import_structure = {
    "configuration_xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMOnnxConfig"],
    "tokenization_xlm": ["XLMTokenizer"],
}

# 检查是否 Torch 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加 Torch 版本的 XLM 模型相关类到导入结构中
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

# 检查是否 TensorFlow 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则添加 TensorFlow 版本的 XLM 模型相关类到导入结构中
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

# 如果是类型检查模式，导入相应模块的类型和类
if TYPE_CHECKING:
    from .configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMOnnxConfig
    from .tokenization_xlm import XLMTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Torch 版本的 XLM 模型相关类
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
    else:
        # 导入相对当前目录下的 .modeling_tf_xlm 模块中的特定内容
        from .modeling_tf_xlm import (
            TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入 TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST 变量
            TFXLMForMultipleChoice,  # 导入 TFXLMForMultipleChoice 类
            TFXLMForQuestionAnsweringSimple,  # 导入 TFXLMForQuestionAnsweringSimple 类
            TFXLMForSequenceClassification,  # 导入 TFXLMForSequenceClassification 类
            TFXLMForTokenClassification,  # 导入 TFXLMForTokenClassification 类
            TFXLMMainLayer,  # 导入 TFXLMMainLayer 类
            TFXLMModel,  # 导入 TFXLMModel 类
            TFXLMPreTrainedModel,  # 导入 TFXLMPreTrainedModel 类
            TFXLMWithLMHeadModel,  # 导入 TFXLMWithLMHeadModel 类
        )
else:
    # 导入系统模块 sys
    import sys
    # 将当前模块（__name__）的映射关系指向一个 LazyModule 实例，以延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```