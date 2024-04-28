# `.\transformers\models\reformer\__init__.py`

```
# 版权声明及许可声明
# 从 ...utils 模块中导入相关函数和类
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构
_import_structure = {"configuration_reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"]}

# 检查 sentencepiece 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入导入结构
    _import_structure["tokenization_reformer"] = ["ReformerTokenizer"]

# 检查 tokenizers 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入导入结构
    _import_structure["tokenization_reformer_fast"] = ["ReformerTokenizerFast"]

# 检查 torch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则加入导入结构
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

# 检查是否处于类型检查模式，若是，则导入相关模块
if TYPE_CHECKING:
    from .configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_reformer import ReformerTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_reformer_fast import ReformerTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是第一次导入该模块，则从相对路径中导入以下模块
    from .modeling_reformer import (
        # 导入 REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST 常量，包含所有预训练模型的名称列表
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        # 导入 ReformerAttention 类，用于 Reformer 模型的注意力机制
        ReformerAttention,
        # 导入 ReformerForMaskedLM 类，用于执行遮蔽语言建模任务的 Reformer 模型
        ReformerForMaskedLM,
        # 导入 ReformerForQuestionAnswering 类，用于执行问答任务的 Reformer 模型
        ReformerForQuestionAnswering,
        # 导入 ReformerForSequenceClassification 类，用于执行序列分类任务的 Reformer 模型
        ReformerForSequenceClassification,
        # 导入 ReformerLayer 类，表示 Reformer 模型的一个层
        ReformerLayer,
        # 导入 ReformerModel 类，用于构建基本的 Reformer 模型
        ReformerModel,
        # 导入 ReformerModelWithLMHead 类，表示带有语言建模头部的 Reformer 模型
        ReformerModelWithLMHead,
        # 导入 ReformerPreTrainedModel 类，表示 Reformer 模型的预训练模型基类
        ReformerPreTrainedModel,
    )
# 如果不是以上条件，则导入 sys 模块
import sys
# 将当前模块的命名空间赋值给 __name__ 对应的模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```