# `.\models\mpnet\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入相关依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_mpnet": ["MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "MPNetConfig"],
    "tokenization_mpnet": ["MPNetTokenizer"],
}

# 检查是否可用 tokenizers，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_mpnet_fast 到导入结构
    _import_structure["tokenization_mpnet_fast"] = ["MPNetTokenizerFast"]

# 检查是否可用 torch，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_mpnet 到导入结构
    _import_structure["modeling_mpnet"] = [
        "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MPNetForMaskedLM",
        "MPNetForMultipleChoice",
        "MPNetForQuestionAnswering",
        "MPNetForSequenceClassification",
        "MPNetForTokenClassification",
        "MPNetLayer",
        "MPNetModel",
        "MPNetPreTrainedModel",
    ]

# 检查是否可用 tensorflow，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_tf_mpnet 到导入结构
    _import_structure["modeling_tf_mpnet"] = [
        "TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFMPNetEmbeddings",
        "TFMPNetForMaskedLM",
        "TFMPNetForMultipleChoice",
        "TFMPNetForQuestionAnswering",
        "TFMPNetForSequenceClassification",
        "TFMPNetForTokenClassification",
        "TFMPNetMainLayer",
        "TFMPNetModel",
        "TFMPNetPreTrainedModel",
    ]

# 如果是类型检查阶段，导入配置和 tokenizer 模块
if TYPE_CHECKING:
    from .configuration_mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig
    from .tokenization_mpnet import MPNetTokenizer

    # 检查是否可用 tokenizers，如果不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 tokenization_mpnet_fast 模块
        from .tokenization_mpnet_fast import MPNetTokenizerFast

    # 检查是否可用 torch，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果在当前环境中找不到 TensorFlow 库，则抛出 OptionalDependencyNotAvailable 异常
    else:
        # 导入 MPNet 模型相关模块，这些模块通常用于处理自然语言处理任务
        from .modeling_mpnet import (
            MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            MPNetForMaskedLM,
            MPNetForMultipleChoice,
            MPNetForQuestionAnswering,
            MPNetForSequenceClassification,
            MPNetForTokenClassification,
            MPNetLayer,
            MPNetModel,
            MPNetPreTrainedModel,
        )

    # 尝试检查是否 TensorFlow 可用，如果不可用则捕获 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果 TensorFlow 不可用，处理捕获到的 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 什么都不做，继续执行后续逻辑
        pass
    # 如果 TensorFlow 可用
    else:
        # 导入 TensorFlow 版本的 MPNet 模型相关模块
        from .modeling_tf_mpnet import (
            TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMPNetEmbeddings,
            TFMPNetForMaskedLM,
            TFMPNetForMultipleChoice,
            TFMPNetForQuestionAnswering,
            TFMPNetForSequenceClassification,
            TFMPNetForTokenClassification,
            TFMPNetMainLayer,
            TFMPNetModel,
            TFMPNetPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于操作 Python 解释器的系统功能
    import sys

    # 将当前模块注册到 sys.modules 中，使其可以通过当前模块的名称访问
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```