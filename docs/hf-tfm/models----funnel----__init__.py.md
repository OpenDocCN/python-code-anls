# `.\models\funnel\__init__.py`

```py
# 导入类型检查标记
from typing import TYPE_CHECKING

# 导入工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_funnel": ["FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP", "FunnelConfig"],
    "convert_funnel_original_tf_checkpoint_to_pytorch": [],
    "tokenization_funnel": ["FunnelTokenizer"],
}

# 检查是否有 Tokenizers 库可用，若无则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_funnel_fast 到导入结构
    _import_structure["tokenization_funnel_fast"] = ["FunnelTokenizerFast"]

# 检查是否有 PyTorch 库可用，若无则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_funnel 到导入结构
    _import_structure["modeling_funnel"] = [
        "FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FunnelBaseModel",
        "FunnelForMaskedLM",
        "FunnelForMultipleChoice",
        "FunnelForPreTraining",
        "FunnelForQuestionAnswering",
        "FunnelForSequenceClassification",
        "FunnelForTokenClassification",
        "FunnelModel",
        "FunnelPreTrainedModel",
        "load_tf_weights_in_funnel",
    ]

# 检查是否有 TensorFlow 库可用，若无则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_tf_funnel 到导入结构
    _import_structure["modeling_tf_funnel"] = [
        "TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFFunnelBaseModel",
        "TFFunnelForMaskedLM",
        "TFFunnelForMultipleChoice",
        "TFFunnelForPreTraining",
        "TFFunnelForQuestionAnswering",
        "TFFunnelForSequenceClassification",
        "TFFunnelForTokenClassification",
        "TFFunnelModel",
        "TFFunnelPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 FunnelConfig 和 FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP 类型
    from .configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
    # 导入 FunnelTokenizer 类型
    from .tokenization_funnel import FunnelTokenizer

    # 检查是否有 Tokenizers 库可用，若无则抛出异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 FunnelTokenizerFast 类型
        from .tokenization_funnel_fast import FunnelTokenizerFast

    # 检查是否有 PyTorch 库可用，若无则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，从本地导入Funnel模型相关的模块和变量
    from .modeling_funnel import (
        FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,   # 导入预训练模型的存档列表
        FunnelBaseModel,                       # 导入Funnel基础模型类
        FunnelForMaskedLM,                     # 导入用于MLM的Funnel模型类
        FunnelForMultipleChoice,               # 导入用于多项选择任务的Funnel模型类
        FunnelForPreTraining,                  # 导入用于预训练的Funnel模型类
        FunnelForQuestionAnswering,            # 导入用于问答任务的Funnel模型类
        FunnelForSequenceClassification,       # 导入用于序列分类任务的Funnel模型类
        FunnelForTokenClassification,          # 导入用于标记分类任务的Funnel模型类
        FunnelModel,                           # 导入Funnel模型类
        FunnelPreTrainedModel,                 # 导入Funnel预训练模型类
        load_tf_weights_in_funnel,             # 导入加载TensorFlow权重的函数
    )

    # 尝试检查是否TensorFlow可用，如果不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获OptionalDependencyNotAvailable异常并忽略
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 否则，从本地导入TensorFlow版Funnel模型相关的模块和变量
        from .modeling_tf_funnel import (
            TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,     # 导入TensorFlow版预训练模型的存档列表
            TFFunnelBaseModel,                           # 导入TensorFlow版Funnel基础模型类
            TFFunnelForMaskedLM,                         # 导入用于MLM的TensorFlow版Funnel模型类
            TFFunnelForMultipleChoice,                   # 导入用于多项选择任务的TensorFlow版Funnel模型类
            TFFunnelForPreTraining,                      # 导入用于预训练的TensorFlow版Funnel模型类
            TFFunnelForQuestionAnswering,                # 导入用于问答任务的TensorFlow版Funnel模型类
            TFFunnelForSequenceClassification,           # 导入用于序列分类任务的TensorFlow版Funnel模型类
            TFFunnelForTokenClassification,              # 导入用于标记分类任务的TensorFlow版Funnel模型类
            TFFunnelModel,                               # 导入TensorFlow版Funnel模型类
            TFFunnelPreTrainedModel,                     # 导入TensorFlow版Funnel预训练模型类
        )
else:
    # 导入 sys 模块，用于处理模块对象和引用
    import sys

    # 将当前模块添加到 sys.modules 中，通过 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```