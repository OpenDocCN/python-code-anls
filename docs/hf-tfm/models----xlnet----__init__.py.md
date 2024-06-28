# `.\models\xlnet\__init__.py`

```py
# 导入必要的模块和函数，包括类型检查功能
from typing import TYPE_CHECKING

# 从工具包中导入相关模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构字典，用于组织需要导入的模块和函数
_import_structure = {"configuration_xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"]}

# 检查是否安装了 sentencepiece 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 sentencepiece 库，则添加相关的 XLNetTokenizer 到导入结构中
    _import_structure["tokenization_xlnet"] = ["XLNetTokenizer"]

# 检查是否安装了 tokenizers 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 tokenizers 库，则添加相关的 XLNetTokenizerFast 到导入结构中
    _import_structure["tokenization_xlnet_fast"] = ["XLNetTokenizerFast"]

# 检查是否安装了 torch 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 torch 库，则添加相关的 XLNet 模型组件到导入结构中
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

# 检查是否安装了 tensorflow 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 tensorflow 库，则添加相关的 TensorFlow XLNet 模型组件到导入结构中
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

# 如果是类型检查模式，则导入相关的类型定义
if TYPE_CHECKING:
    from .configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 sentencepiece 库，则从 tokenization_xlnet 模块中导入 XLNetTokenizer
        from .tokenization_xlnet import XLNetTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 tokenizers 库，则从 tokenization_xlnet_fast 模块中导入 XLNetTokenizerFast
        from .tokenization_xlnet_fast import XLNetTokenizerFast
    # 尝试检查是否存在 Torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 Torch 库不可用
    except OptionalDependencyNotAvailable:
        pass
    # 如果 Torch 库可用，则执行以下代码块
    else:
        # 从当前目录下的 modeling_xlnet 模块导入以下符号
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

    # 尝试检查是否存在 TensorFlow 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 TensorFlow 库不可用
    except OptionalDependencyNotAvailable:
        pass
    # 如果 TensorFlow 库可用，则执行以下代码块
    else:
        # 从当前目录下的 modeling_tf_xlnet 模块导入以下符号
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
    # 如果不在以上的情况下，则导入 sys 模块
    import sys
    
    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    # __name__ 是当前模块的名称，__file__ 是当前文件的路径
    # _import_structure 是导入的结构信息，module_spec=__spec__ 是模块的规范信息
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```