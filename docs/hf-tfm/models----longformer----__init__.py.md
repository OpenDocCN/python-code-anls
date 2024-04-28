# `.\transformers\models\longformer\__init__.py`

```py
# 导入类型检查模块，用于检查类型
from typing import TYPE_CHECKING
# 导入必要的工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包含不同模块的名称和其对应的导入内容
_import_structure = {
    "configuration_longformer": [
        "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LongformerConfig",
        "LongformerOnnxConfig",
    ],
    "tokenization_longformer": ["LongformerTokenizer"],
}

# 尝试导入 tokenizers 模块，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_longformer_fast 模块到导入结构中
    _import_structure["tokenization_longformer_fast"] = ["LongformerTokenizerFast"]

# 尝试导入 torch 模块，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_longformer 模块到导入结构中
    _import_structure["modeling_longformer"] = [
        "LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LongformerForMaskedLM",
        "LongformerForMultipleChoice",
        "LongformerForQuestionAnswering",
        "LongformerForSequenceClassification",
        "LongformerForTokenClassification",
        "LongformerModel",
        "LongformerPreTrainedModel",
        "LongformerSelfAttention",
    ]

# 尝试导入 tensorflow 模块，若不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_tf_longformer 模块到导入结构中
    _import_structure["modeling_tf_longformer"] = [
        "TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLongformerForMaskedLM",
        "TFLongformerForMultipleChoice",
        "TFLongformerForQuestionAnswering",
        "TFLongformerForSequenceClassification",
        "TFLongformerForTokenClassification",
        "TFLongformerModel",
        "TFLongformerPreTrainedModel",
        "TFLongformerSelfAttention",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置和标记器模块，并声明它们的类型
    from .configuration_longformer import (
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LongformerConfig,
        LongformerOnnxConfig,
    )
    from .tokenization_longformer import LongformerTokenizer

    # 尝试导入 tokenizers 模块，若不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 tokenization_longformer_fast 模块
        from .tokenization_longformer_fast import LongformerTokenizerFast

    # 尝试导入 torch 模块，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入可选依赖模块
        # 如果可选依赖不可用，则捕获 OptionalDependencyNotAvailable 异常
        except OptionalDependencyNotAvailable:
            # 如果可选依赖不可用，则不执行任何操作
            pass
        # 如果没有捕获到 OptionalDependencyNotAvailable 异常，则执行以下代码块
        else:
            # 导入模块中的特定内容，包括模型和模型相关的部分
            from .modeling_longformer import (
                # 导入长形式模型的预训练模型列表
                LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
                # 导入用于填充掩码语言模型的长形式模型
                LongformerForMaskedLM,
                # 导入多项选择任务的长形式模型
                LongformerForMultipleChoice,
                # 导入问答任务的长形式模型
                LongformerForQuestionAnswering,
                # 导入序列分类任务的长形式模型
                LongformerForSequenceClassification,
                # 导入标记分类任务的长形式模型
                LongformerForTokenClassification,
                # 导入长形式模型
                LongformerModel,
                # 导入长形式预训练模型的基类
                LongformerPreTrainedModel,
                # 导入长形式自注意力机制
                LongformerSelfAttention,
            )
    
    # 尝试导入 TensorFlow 版本的长形式模型
        # 如果 TensorFlow 不可用，则捕获 OptionalDependencyNotAvailable 异常
        try:
            # 如果 TensorFlow 不可用，则抛出异常
            if not is_tf_available():
                raise OptionalDependencyNotAvailable()
        # 如果捕获到 OptionalDependencyNotAvailable 异常，则执行以下代码块
        except OptionalDependencyNotAvailable:
            # 如果 TensorFlow 不可用，则不执行任何操作
            pass
        # 如果没有捕获到 OptionalDependencyNotAvailable 异常，则执行以下代码块
        else:
            # 导入 TensorFlow 版本的长形式模型中的特定内容，包括模型和模型相关的部分
            from .modeling_tf_longformer import (
                # 导入 TensorFlow 版本的长形式模型的预训练模型列表
                TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
                # 导入用于填充掩码语言模型的 TensorFlow 版本的长形式模型
                TFLongformerForMaskedLM,
                # 导入多项选择任务的 TensorFlow 版本的长形式模型
                TFLongformerForMultipleChoice,
                # 导入问答任务的 TensorFlow 版本的长形式模型
                TFLongformerForQuestionAnswering,
                # 导入序列分类任务的 TensorFlow 版本的长形式模型
                TFLongformerForSequenceClassification,
                # 导入标记分类任务的 TensorFlow 版本的长形式模型
                TFLongformerForTokenClassification,
                # 导入 TensorFlow 版本的长形式模型
                TFLongformerModel,
                # 导入 TensorFlow 版本的长形式预训练模型的基类
                TFLongformerPreTrainedModel,
                # 导入 TensorFlow 版本的长形式自注意力机制
                TFLongformerSelfAttention,
            )
# 如果条件不成立，即不满足之前的条件，执行以下代码块
else:
    # 导入 sys 模块
    import sys
    # 使用 LazyModule 对象替换当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```