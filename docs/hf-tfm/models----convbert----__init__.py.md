# `.\models\convbert\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从 utils 模块中导入所需函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包含各模块对应的导入内容列表
_import_structure = {
    "configuration_convbert": ["CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvBertConfig", "ConvBertOnnxConfig"],
    "tokenization_convbert": ["ConvBertTokenizer"],
}

# 检查 tokenizers 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 tokenization_convbert_fast 模块的导入内容列表
    _import_structure["tokenization_convbert_fast"] = ["ConvBertTokenizerFast"]

# 检查 torch 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_convbert 模块的导入内容列表
    _import_structure["modeling_convbert"] = [
        "CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvBertForMaskedLM",
        "ConvBertForMultipleChoice",
        "ConvBertForQuestionAnswering",
        "ConvBertForSequenceClassification",
        "ConvBertForTokenClassification",
        "ConvBertLayer",
        "ConvBertModel",
        "ConvBertPreTrainedModel",
        "load_tf_weights_in_convbert",
    ]

# 检查 tensorflow 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_tf_convbert 模块的导入内容列表
    _import_structure["modeling_tf_convbert"] = [
        "TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFConvBertForMaskedLM",
        "TFConvBertForMultipleChoice",
        "TFConvBertForQuestionAnswering",
        "TFConvBertForSequenceClassification",
        "TFConvBertForTokenClassification",
        "TFConvBertLayer",
        "TFConvBertModel",
        "TFConvBertPreTrainedModel",
    ]

# 若为类型检查模式，则进行更详细的导入结构定义
if TYPE_CHECKING:
    # 导入 configuration_convbert 模块的特定类和常量
    from .configuration_convbert import CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig, ConvBertOnnxConfig
    # 导入 tokenization_convbert 模块的特定类
    from .tokenization_convbert import ConvBertTokenizer

    # 再次检查 tokenizers 是否可用，若不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入 tokenization_convbert_fast 模块的特定类
        from .tokenization_convbert_fast import ConvBertTokenizerFast

    # 再次检查 torch 是否可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果当前环境支持 TensorFlow，则导入 TensorFlow 版的 ConvBERT 模型和相关内容
        from .modeling_convbert import (
            CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvBertForMaskedLM,
            ConvBertForMultipleChoice,
            ConvBertForQuestionAnswering,
            ConvBertForSequenceClassification,
            ConvBertForTokenClassification,
            ConvBertLayer,
            ConvBertModel,
            ConvBertPreTrainedModel,
            load_tf_weights_in_convbert,
        )

    try:
        # 检查是否存在 TensorFlow 的依赖
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，则忽略错误
        pass
    else:
        # 如果 TensorFlow 可用，则导入 TensorFlow 版的 ConvBERT 模型和相关内容
        from .modeling_tf_convbert import (
            TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFConvBertForMaskedLM,
            TFConvBertForMultipleChoice,
            TFConvBertForQuestionAnswering,
            TFConvBertForSequenceClassification,
            TFConvBertForTokenClassification,
            TFConvBertLayer,
            TFConvBertModel,
            TFConvBertPreTrainedModel,
        )
else:
    # 导入 sys 模块
    import sys

    # 将当前模块的名称注册到 sys.modules 中
    # 使用 _LazyModule 类来延迟加载模块内容
    # __name__ 表示当前模块的名称
    # globals()["__file__"] 获取当前模块的文件路径
    # _import_structure 包含要导入的模块结构信息
    # module_spec=__spec__ 指定模块的规范对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```