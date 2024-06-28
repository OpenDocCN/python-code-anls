# `.\models\rembert\__init__.py`

```
# 引入类型检查模块，用于检查类型相关的导入
from typing import TYPE_CHECKING

# 从工具模块中导入所需内容：可选依赖不可用异常、延迟加载模块、判断是否有句子分词模块可用、判断是否有 TensorFlow 模块可用、判断是否有 Tokenizers 模块可用、判断是否有 PyTorch 模块可用
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构字典，用于存储不同模块的导入结构
_import_structure = {
    "configuration_rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig", "RemBertOnnxConfig"]
}

# 尝试检查是否句子分词模块可用，如果不可用则抛出可选依赖不可用异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果句子分词模块不可用，继续执行

# 如果句子分词模块可用，则将 RemBertTokenizer 添加到导入结构字典中
else:
    _import_structure["tokenization_rembert"] = ["RemBertTokenizer"]

# 尝试检查是否 Tokenizers 模块可用，如果不可用则抛出可选依赖不可用异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果 Tokenizers 模块不可用，继续执行

# 如果 Tokenizers 模块可用，则将 RemBertTokenizerFast 添加到导入结构字典中
else:
    _import_structure["tokenization_rembert_fast"] = ["RemBertTokenizerFast"]

# 尝试检查是否 PyTorch 模块可用，如果不可用则抛出可选依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果 PyTorch 模块不可用，继续执行

# 如果 PyTorch 模块可用，则将 RemBert 相关模块添加到导入结构字典中
else:
    _import_structure["modeling_rembert"] = [
        "REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RemBertForCausalLM",
        "RemBertForMaskedLM",
        "RemBertForMultipleChoice",
        "RemBertForQuestionAnswering",
        "RemBertForSequenceClassification",
        "RemBertForTokenClassification",
        "RemBertLayer",
        "RemBertModel",
        "RemBertPreTrainedModel",
        "load_tf_weights_in_rembert",
    ]

# 尝试检查是否 TensorFlow 模块可用，如果不可用则抛出可选依赖不可用异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果 TensorFlow 模块不可用，继续执行

# 如果 TensorFlow 模块可用，则将 TFRemBert 相关模块添加到导入结构字典中
else:
    _import_structure["modeling_tf_rembert"] = [
        "TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRemBertForCausalLM",
        "TFRemBertForMaskedLM",
        "TFRemBertForMultipleChoice",
        "TFRemBertForQuestionAnswering",
        "TFRemBertForSequenceClassification",
        "TFRemBertForTokenClassification",
        "TFRemBertLayer",
        "TFRemBertModel",
        "TFRemBertPreTrainedModel",
    ]

# 如果当前环境在类型检查模式下，从配置模块中导入所需内容：预训练配置存档映射、RemBertConfig、RemBertOnnxConfig
if TYPE_CHECKING:
    from .configuration_rembert import REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RemBertConfig, RemBertOnnxConfig

    # 尝试检查是否句子分词模块可用，如果不可用则抛出可选依赖不可用异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 如果句子分词模块不可用，继续执行

    # 如果句子分词模块可用，则从分词模块中导入 RemBertTokenizer
    else:
        from .tokenization_rembert import RemBertTokenizer
    # 检查是否安装了 tokenizers 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做处理继续执行后续代码
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 tokenizers 库可用，则从本地模块导入 RemBertTokenizerFast 类
        from .tokenization_rembert_fast import RemBertTokenizerFast

    # 检查是否安装了 torch 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做处理继续执行后续代码
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 库可用，则从本地模块导入以下类和函数
        from .modeling_rembert import (
            REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RemBertForCausalLM,
            RemBertForMaskedLM,
            RemBertForMultipleChoice,
            RemBertForQuestionAnswering,
            RemBertForSequenceClassification,
            RemBertForTokenClassification,
            RemBertLayer,
            RemBertModel,
            RemBertPreTrainedModel,
            load_tf_weights_in_rembert,
        )

    # 检查是否安装了 TensorFlow 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做处理继续执行后续代码
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 TensorFlow 库可用，则从本地模块导入以下类和函数
        from .modeling_tf_rembert import (
            TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRemBertForCausalLM,
            TFRemBertForMaskedLM,
            TFRemBertForMultipleChoice,
            TFRemBertForQuestionAnswering,
            TFRemBertForSequenceClassification,
            TFRemBertForTokenClassification,
            TFRemBertLayer,
            TFRemBertModel,
            TFRemBertPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于管理 Python 解释器的运行时环境
    import sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 类进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```