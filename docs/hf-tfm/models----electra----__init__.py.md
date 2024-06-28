# `.\models\electra\__init__.py`

```
# 引入类型检查模块，用于静态类型检查
from typing import TYPE_CHECKING

# 从工具模块中引入所需函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构，包含不同模块及其对应的导入内容列表
_import_structure = {
    "configuration_electra": ["ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraConfig", "ElectraOnnxConfig"],
    "tokenization_electra": ["ElectraTokenizer"],
}

# 检查是否存在 tokenizers 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers 库，则将 ElectraTokenizerFast 添加到导入结构中
    _import_structure["tokenization_electra_fast"] = ["ElectraTokenizerFast"]

# 检查是否存在 torch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则将相关的 Electra 模型导入添加到导入结构中
    _import_structure["modeling_electra"] = [
        "ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ElectraForCausalLM",
        "ElectraForMaskedLM",
        "ElectraForMultipleChoice",
        "ElectraForPreTraining",
        "ElectraForQuestionAnswering",
        "ElectraForSequenceClassification",
        "ElectraForTokenClassification",
        "ElectraModel",
        "ElectraPreTrainedModel",
        "load_tf_weights_in_electra",
    ]

# 检查是否存在 tensorflow 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow 库，则将相关的 TFElectra 模型导入添加到导入结构中
    _import_structure["modeling_tf_electra"] = [
        "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFElectraForMaskedLM",
        "TFElectraForMultipleChoice",
        "TFElectraForPreTraining",
        "TFElectraForQuestionAnswering",
        "TFElectraForSequenceClassification",
        "TFElectraForTokenClassification",
        "TFElectraModel",
        "TFElectraPreTrainedModel",
    ]

# 检查是否存在 flax 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 flax 库，则将相关的 FlaxElectra 模型导入添加到导入结构中
    _import_structure["modeling_flax_electra"] = [
        "FlaxElectraForCausalLM",
        "FlaxElectraForMaskedLM",
        "FlaxElectraForMultipleChoice",
        "FlaxElectraForPreTraining",
        "FlaxElectraForQuestionAnswering",
        "FlaxElectraForSequenceClassification",
        "FlaxElectraForTokenClassification",
        "FlaxElectraModel",
        "FlaxElectraPreTrainedModel",
    ]

# 如果在类型检查环境下
if TYPE_CHECKING:
    # 空语句，因为在类型检查环境下不需要执行额外的代码
    pass
    # 从当前目录中导入以下模块和变量，分别是预训练配置映射、ElectraConfig 类和 ElectraOnnxConfig 类
    from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraOnnxConfig
    # 导入 ElectraTokenizer 类，用于处理 Electra 模型的分词器
    
    # 检查是否安装了 tokenizers 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 tokenizers 可用，则从当前目录中导入 ElectraTokenizerFast 类，用于更快速的分词操作
        from .tokenization_electra_fast import ElectraTokenizerFast
    
    # 检查是否安装了 PyTorch 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 PyTorch 可用，则从当前目录中导入以下 Electra 相关类和函数
        from .modeling_electra import (
            ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            ElectraForCausalLM,
            ElectraForMaskedLM,
            ElectraForMultipleChoice,
            ElectraForPreTraining,
            ElectraForQuestionAnswering,
            ElectraForSequenceClassification,
            ElectraForTokenClassification,
            ElectraModel,
            ElectraPreTrainedModel,
            load_tf_weights_in_electra,
        )
    
    # 检查是否安装了 TensorFlow 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 TensorFlow 可用，则从当前目录中导入以下 TF-Electra 相关类和函数
        from .modeling_tf_electra import (
            TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFElectraForMaskedLM,
            TFElectraForMultipleChoice,
            TFElectraForPreTraining,
            TFElectraForQuestionAnswering,
            TFElectraForSequenceClassification,
            TFElectraForTokenClassification,
            TFElectraModel,
            TFElectraPreTrainedModel,
        )
    
    # 检查是否安装了 Flax 库，若未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 Flax 可用，则从当前目录中导入以下 Flax-Electra 相关类和函数
        from .modeling_flax_electra import (
            FlaxElectraForCausalLM,
            FlaxElectraForMaskedLM,
            FlaxElectraForMultipleChoice,
            FlaxElectraForPreTraining,
            FlaxElectraForQuestionAnswering,
            FlaxElectraForSequenceClassification,
            FlaxElectraForTokenClassification,
            FlaxElectraModel,
            FlaxElectraPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态设置当前模块为懒加载模块
    import sys

    # 使用 sys.modules 来将当前模块设置为懒加载模块的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```