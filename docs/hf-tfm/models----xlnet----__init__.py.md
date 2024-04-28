# `.\transformers\models\xlnet\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从 utils 模块中导入所需的工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义需要导入的模块和函数的结构
_import_structure = {"configuration_xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"]}

# 检查是否存在 sentencepiece，若不存在则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果存在则导入相关模块和函数
else:
    _import_structure["tokenization_xlnet"] = ["XLNetTokenizer"]

# 检查是否存在 tokenizers，若不存在则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果存在则导入相关模块和函数
else:
    _import_structure["tokenization_xlnet_fast"] = ["XLNetTokenizerFast"]

# 检查是否存在 torch，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果存在则导入相关模块和函数
else:
    _import_structure["modeling_xlnet"] = [
        "XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLNetForMultipleChoice",
        "XLNetForQuestionAnswering",
        "XLNetForQuestionAnsweringSimple",
        "XLNetForSequenceClassification",
        "XLNetForTokenClassification",
        "XLNetLMHeadModel",
        "XLNetModel",
        "XLNetPreTrainedModel",
        "load_tf_weights_in_xlnet",
    ]

# 检查是否存在 tensorflow，若不存在则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果存在则导入相关模块和函数
else:
    _import_structure["modeling_tf_xlnet"] = [
        "TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXLNetForMultipleChoice",
        "TFXLNetForQuestionAnsweringSimple",
        "TFXLNetForSequenceClassification",
        "TFXLNetForTokenClassification",
        "TFXLNetLMHeadModel",
        "TFXLNetMainLayer",
        "TFXLNetModel",
        "TFXLNetPreTrainedModel",
    ]

# 如果是 TYPE_CHECKING 模式，则导入对应的模块和函数
if TYPE_CHECKING:
    from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xlnet import XLNetTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xlnet_fast import XLNetTokenizerFast
    # 尝试检查是否存在 torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何处理
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出任何异常，则执行下面的语句
    else:
        # 从 modeling_xlnet 模块导入相关函数和类
        from .modeling_xlnet import (
            XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLNetForMultipleChoice,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForSequenceClassification,
            XLNetForTokenClassification,
            XLNetLMHeadModel,
            XLNetModel,
            XLNetPreTrainedModel,
            load_tf_weights_in_xlnet,
        )

    # 尝试检查是否存在 tensorflow 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何处理
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有抛出任何异常，则执行下面的语句
    else:
        # 从 modeling_tf_xlnet 模块导入相关函数和类
        from .modeling_tf_xlnet import (
            TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLNetForMultipleChoice,
            TFXLNetForQuestionAnsweringSimple,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetLMHeadModel,
            TFXLNetMainLayer,
            TFXLNetModel,
            TFXLNetPreTrainedModel,
        )
else:
    # 导入 sys 模块
    import sys

    # 将当前模块加入到 sys.modules 中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```