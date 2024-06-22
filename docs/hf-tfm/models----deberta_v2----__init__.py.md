# `.\models\deberta_v2\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入可选依赖异常模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义需要导入的模块结构
_import_structure = {
    "configuration_deberta_v2": ["DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaV2Config", "DebertaV2OnnxConfig"],
    "tokenization_deberta_v2": ["DebertaV2Tokenizer"],
}

# 检查 tokenizers 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 若可用则添加 tokenization_deberta_v2_fast 到 _import_structure 中
    _import_structure["tokenization_deberta_v2_fast"] = ["DebertaV2TokenizerFast"]

# 检查 tensorflow 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 若可用则添加 modeling_tf_deberta_v2 到 _import_structure 中
    _import_structure["modeling_tf_deberta_v2"] = [
        "TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDebertaV2ForMaskedLM",
        "TFDebertaV2ForQuestionAnswering",
        "TFDebertaV2ForMultipleChoice",
        "TFDebertaV2ForSequenceClassification",
        "TFDebertaV2ForTokenClassification",
        "TFDebertaV2Model",
        "TFDebertaV2PreTrainedModel",
    ]

# 检查 pytorch 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 若可用则添加 modeling_deberta_v2 到 _import_structure 中
    _import_structure["modeling_deberta_v2"] = [
        "DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DebertaV2ForMaskedLM",
        "DebertaV2ForMultipleChoice",
        "DebertaV2ForQuestionAnswering",
        "DebertaV2ForSequenceClassification",
        "DebertaV2ForTokenClassification",
        "DebertaV2Model",
        "DebertaV2PreTrainedModel",
    ]

# 若为类型检查环境，则添加相应模块到导入结构中
if TYPE_CHECKING:
    from .configuration_deberta_v2 import (
        DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DebertaV2Config,
        DebertaV2OnnxConfig,
    )
    from .tokenization_deberta_v2 import DebertaV2Tokenizer

    # 检查 tokenizers 是否可用，若可用则添加 tokenization_deberta_v2_fast 到导入结构中
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_deberta_v2_fast import DebertaV2TokenizerFast

    # 检查 tensorflow 是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是使用Torch，则从模型的TF版本中导入相关内容
    else:
        from .modeling_tf_deberta_v2 import (
            TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST,  # TF版本的预训练模型文件列表
            TFDebertaV2ForMaskedLM,  # TF版本的用于Masked Language Modeling的DeBERTaV2模型
            TFDebertaV2ForMultipleChoice,  # TF版本的用于多项选择任务的DeBERTaV2模型
            TFDebertaV2ForQuestionAnswering,  # TF版本的用于问答任务的DeBERTaV2模型
            TFDebertaV2ForSequenceClassification,  # TF版本的用于序列分类任务的DeBERTaV2模型
            TFDebertaV2ForTokenClassification,  # TF版本的用于标记分类任务的DeBERTaV2模型
            TFDebertaV2Model,  # TF版本的DeBERTaV2模型
            TFDebertaV2PreTrainedModel,  # TF版本的DeBERTaV2预训练模型
        )

    # 尝试检查是否导入了Torch，若未导入则引发OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果OptionalDependencyNotAvailable异常被引发，则忽略
        pass
    # 如果没有异常被引发，则执行以下内容
    else:
        # 从模型的PyTorch版本中导入相关内容
        from .modeling_deberta_v2 import (
            DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST,  # PyTorch版本的预训练模型文件列表
            DebertaV2ForMaskedLM,  # PyTorch版本的用于Masked Language Modeling的DeBERTaV2模型
            DebertaV2ForMultipleChoice,  # PyTorch版本的用于多项选择任务的DeBERTaV2模型
            DebertaV2ForQuestionAnswering,  # PyTorch版本的用于问答任务的DeBERTaV2模型
            DebertaV2ForSequenceClassification,  # PyTorch版本的用于序列分类任务的DeBERTaV2模型
            DebertaV2ForTokenClassification,  # PyTorch版本的用于标记分类任务的DeBERTaV2模型
            DebertaV2Model,  # PyTorch版本的DeBERTaV2模型
            DebertaV2PreTrainedModel,  # PyTorch版本的DeBERTaV2预训练模型
        )
# 如果条件不成立，则导入sys模块
import sys
# 将当前模块注册到sys.modules字典中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```