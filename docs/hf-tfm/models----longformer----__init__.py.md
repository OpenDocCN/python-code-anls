# `.\models\longformer\__init__.py`

```
# 引入类型检查标记，用于在类型检查时导入不同的模块和类
from typing import TYPE_CHECKING

# 从本地包中导入所需的工具和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典，描述了不同模块和类的导入结构
_import_structure = {
    "configuration_longformer": [
        "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LongformerConfig",
        "LongformerOnnxConfig",
    ],
    "tokenization_longformer": ["LongformerTokenizer"],
}

# 检查是否可用 Tokenizers 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入 tokenization_longformer_fast 模块到导入结构中
    _import_structure["tokenization_longformer_fast"] = ["LongformerTokenizerFast"]

# 检查是否可用 Torch 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入 modeling_longformer 模块到导入结构中
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

# 检查是否可用 TensorFlow 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入 modeling_tf_longformer 模块到导入结构中
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

# 如果在类型检查模式下，则导入特定的配置和分词器类
if TYPE_CHECKING:
    from .configuration_longformer import (
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LongformerConfig,
        LongformerOnnxConfig,
    )
    from .tokenization_longformer import LongformerTokenizer

    # 在类型检查模式下，再次检查 Tokenizers 库是否可用，若可用则导入快速分词器类
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_longformer_fast import LongformerTokenizerFast

    # 在类型检查模式下，再次检查 Torch 库是否可用，若可用则导入模型相关类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入长模型相关的依赖项，如果依赖项不可用则跳过
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入长模型相关的Python文件中的模块
        from .modeling_longformer import (
            LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            LongformerForMaskedLM,
            LongformerForMultipleChoice,
            LongformerForQuestionAnswering,
            LongformerForSequenceClassification,
            LongformerForTokenClassification,
            LongformerModel,
            LongformerPreTrainedModel,
            LongformerSelfAttention,
        )

    # 尝试检查是否TensorFlow可用，如果不可用则跳过
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入TensorFlow长模型相关的Python文件中的模块
        from .modeling_tf_longformer import (
            TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLongformerForMaskedLM,
            TFLongformerForMultipleChoice,
            TFLongformerForQuestionAnswering,
            TFLongformerForSequenceClassification,
            TFLongformerForTokenClassification,
            TFLongformerModel,
            TFLongformerPreTrainedModel,
            TFLongformerSelfAttention,
        )
else:
    # 导入sys模块，用于对当前模块进行操作
    import sys

    # 将当前模块（__name__）的模块对象映射到_LazyModule的实例，实现懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```