# `.\models\reformer\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入所需的工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构字典，包含相关配置和类名
_import_structure = {"configuration_reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"]}

# 检查是否存在 sentencepiece 库，若不可用则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入 tokenization_reformer 模块到导入结构中
    _import_structure["tokenization_reformer"] = ["ReformerTokenizer"]

# 检查是否存在 tokenizers 库，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入 tokenization_reformer_fast 模块到导入结构中
    _import_structure["tokenization_reformer_fast"] = ["ReformerTokenizerFast"]

# 检查是否存在 torch 库，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入 modeling_reformer 模块到导入结构中，包含多个类和常量
    _import_structure["modeling_reformer"] = [
        "REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ReformerAttention",
        "ReformerForMaskedLM",
        "ReformerForQuestionAnswering",
        "ReformerForSequenceClassification",
        "ReformerLayer",
        "ReformerModel",
        "ReformerModelWithLMHead",
        "ReformerPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 configuration_reformer 模块中的指定内容
    from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig

    try:
        # 检查是否存在 sentencepiece 库，若不可用则抛出异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入 tokenization_reformer 模块中的 ReformerTokenizer
        from .tokenization_reformer import ReformerTokenizer

    try:
        # 检查是否存在 tokenizers 库，若不可用则抛出异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入 tokenization_reformer_fast 模块中的 ReformerTokenizerFast
        from .tokenization_reformer_fast import ReformerTokenizerFast

    try:
        # 检查是否存在 torch 库，若不可用则抛出异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模块中的一系列符号和类
        from .modeling_reformer import (
            REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入预训练模型的存档列表
            ReformerAttention,                      # 导入Reformer模型中的Attention类
            ReformerForMaskedLM,                    # 导入用于Masked Language Modeling的Reformer模型类
            ReformerForQuestionAnswering,           # 导入用于问答任务的Reformer模型类
            ReformerForSequenceClassification,      # 导入用于序列分类任务的Reformer模型类
            ReformerLayer,                          # 导入Reformer模型的一个层类
            ReformerModel,                          # 导入Reformer模型类
            ReformerModelWithLMHead,                # 导入带有LM头的Reformer模型类
            ReformerPreTrainedModel,                # 导入预训练的Reformer模型类
        )
else:
    # 导入 sys 模块，用于动态修改当前模块的属性
    import sys

    # 使用 sys.modules[__name__] 将当前模块替换为 LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```