# `.\models\roberta\__init__.py`

```
# 引入需要的模块和函数
from typing import TYPE_CHECKING

# 从工具包中导入依赖项
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
    "configuration_roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig", "RobertaOnnxConfig"],
    "tokenization_roberta": ["RobertaTokenizer"],
}

# 检查是否存在 tokenizers 库，若不存在则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers 库，则导入快速 tokenization 模块
    _import_structure["tokenization_roberta_fast"] = ["RobertaTokenizerFast"]

# 检查是否存在 torch 库，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则导入 PyTorch 的相关模型和工具
    _import_structure["modeling_roberta"] = [
        "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RobertaForCausalLM",
        "RobertaForMaskedLM",
        "RobertaForMultipleChoice",
        "RobertaForQuestionAnswering",
        "RobertaForSequenceClassification",
        "RobertaForTokenClassification",
        "RobertaModel",
        "RobertaPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，若不存在则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow 库，则导入 TensorFlow 的相关模型和工具
    _import_structure["modeling_tf_roberta"] = [
        "TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRobertaForCausalLM",
        "TFRobertaForMaskedLM",
        "TFRobertaForMultipleChoice",
        "TFRobertaForQuestionAnswering",
        "TFRobertaForSequenceClassification",
        "TFRobertaForTokenClassification",
        "TFRobertaMainLayer",
        "TFRobertaModel",
        "TFRobertaPreTrainedModel",
    ]

# 检查是否存在 flax 库，若不存在则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 flax 库，则导入 Flax 的相关模型和工具
    _import_structure["modeling_flax_roberta"] = [
        "FlaxRobertaForCausalLM",
        "FlaxRobertaForMaskedLM",
        "FlaxRobertaForMultipleChoice",
        "FlaxRobertaForQuestionAnswering",
        "FlaxRobertaForSequenceClassification",
        "FlaxRobertaForTokenClassification",
        "FlaxRobertaModel",
        "FlaxRobertaPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入特定的配置和类型定义
if TYPE_CHECKING:
    from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig, RobertaOnnxConfig
    # 导入来自当前目录的 tokenization_roberta 模块中的 RobertaTokenizer 类
    from .tokenization_roberta import RobertaTokenizer

    # 检查是否已安装 tokenizers，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被引发，不做任何处理
        pass
    else:
        # 如果没有异常被引发，则从当前目录导入 tokenization_roberta_fast 模块中的 RobertaTokenizerFast 类
        from .tokenization_roberta_fast import RobertaTokenizerFast

    # 检查是否已安装 torch，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被引发，不做任何处理
        pass
    else:
        # 如果没有异常被引发，则从当前目录导入 modeling_roberta 模块中列出的一系列类和常量
        from .modeling_roberta import (
            ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            RobertaForCausalLM,
            RobertaForMaskedLM,
            RobertaForMultipleChoice,
            RobertaForQuestionAnswering,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
            RobertaModel,
            RobertaPreTrainedModel,
        )

    # 检查是否已安装 TensorFlow，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被引发，不做任何处理
        pass
    else:
        # 如果没有异常被引发，则从当前目录导入 modeling_tf_roberta 模块中列出的一系列类和常量
        from .modeling_tf_roberta import (
            TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRobertaForCausalLM,
            TFRobertaForMaskedLM,
            TFRobertaForMultipleChoice,
            TFRobertaForQuestionAnswering,
            TFRobertaForSequenceClassification,
            TFRobertaForTokenClassification,
            TFRobertaMainLayer,
            TFRobertaModel,
            TFRobertaPreTrainedModel,
        )

    # 检查是否已安装 Flax，若未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被引发，不做任何处理
        pass
    else:
        # 如果没有异常被引发，则从当前目录导入 modeling_flax_roberta 模块中列出的一系列类和常量
        from .modeling_flax_roberta import (
            FlaxRobertaForCausalLM,
            FlaxRobertaForMaskedLM,
            FlaxRobertaForMultipleChoice,
            FlaxRobertaForQuestionAnswering,
            FlaxRobertaForSequenceClassification,
            FlaxRobertaForTokenClassification,
            FlaxRobertaModel,
            FlaxRobertaPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于操作 Python 解释器运行时的系统环境
    import sys

    # 将当前模块注册到 sys.modules 字典中，使用 _LazyModule 对象作为值
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```