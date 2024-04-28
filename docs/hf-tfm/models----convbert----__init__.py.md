# `.\models\convbert\__init__.py`

```
# 导入模块所需的类型检查功能
from typing import TYPE_CHECKING

# 导入必要的工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个包含模块导入结构的字典
_import_structure = {
    "configuration_convbert": ["CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvBertConfig", "ConvBertOnnxConfig"],
    "tokenization_convbert": ["ConvBertTokenizer"],
}

# 检查是否有可选的依赖包，如果没有则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果有可选依赖包，则添加对应的模块导入结构
else:
    _import_structure["tokenization_convbert_fast"] = ["ConvBertTokenizerFast"]

# 检查是否有可选的依赖包，如果没有则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果有可选依赖包，则添加对应的模块导入结构
else:
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

# 检查是否有可选的依赖包，如果没有则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 如果有可选依赖包，则添加对应的模块导入结构
else:
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

# 如果是类型检查模式，则导入相应的模块
if TYPE_CHECKING:
    from .configuration_convbert import CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig, ConvBertOnnxConfig
    from .tokenization_convbert import ConvBertTokenizer

    # 检查是否有可选的依赖包，如果没有则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果有可选依赖包，则添加对应的模块导入结构
    else:
        from .tokenization_convbert_fast import ConvBertTokenizerFast

    # 检查是否有可选的依赖包，如果没有则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，导入convbert模块下的相关模型和函数
    else:
        from .modeling_convbert import (
            CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入convbert预训练模型的存档列表
            ConvBertForMaskedLM,  # 导入用于masked语言模型任务的ConvBert模型
            ConvBertForMultipleChoice,  # 导入用于多项选择任务的ConvBert模型
            ConvBertForQuestionAnswering,  # 导入用于问答任务的ConvBert模型
            ConvBertForSequenceClassification,  # 导入用于序列分类任务的ConvBert模型
            ConvBertForTokenClassification,  # 导入用于标记分类的ConvBert模型
            ConvBertLayer,  # 导入ConvBert的层
            ConvBertModel,  # 导入ConvBert模型
            ConvBertPreTrainedModel,  # 导入ConvBert预训练模型
            load_tf_weights_in_convbert,  # 导入加载tensorflow权重到ConvBert模型的函数
        )

    # 尝试检查是否已安装TensorFlow，若未安装则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果引发了OptionalDependencyNotAvailable异常，则忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果未引发异常，则导入tf_convbert模块下的相关模型和函数
    else:
        from .modeling_tf_convbert import (
            TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入tf_convbert预训练模型的存档列表
            TFConvBertForMaskedLM,  # 导入用于masked语言模型任务的tf_convbert模型
            TFConvBertForMultipleChoice,  # 导入用于多项选择任务的tf_convbert模型
            TFConvBertForQuestionAnswering,  # 导入用于问答任务的tf_convbert模型
            TFConvBertForSequenceClassification,  # 导入用于序列分类任务的tf_convbert模型
            TFConvBertForTokenClassification,  # 导入用于标记分类的tf_convbert模型
            TFConvBertLayer,  # 导入tf_convbert的层
            TFConvBertModel,  # 导入tf_convbert模型
            TFConvBertPreTrainedModel,  # 导入tf_convbert预训练模型
        )
# 如果条件不成立
else:
    # 导入 sys 模块
    import sys
    # 将当前模块添加到 sys 模块的 modules 属性里
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```