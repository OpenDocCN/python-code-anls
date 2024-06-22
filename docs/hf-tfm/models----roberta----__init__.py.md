# `.\transformers\models\roberta\__init__.py`

```py
# 版权声明和许可协议信息

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入依赖包和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构
_import_structure = {
    "configuration_roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig", "RobertaOnnxConfig"],
    "tokenization_roberta": ["RobertaTokenizer"],
}

# 检查是否有tokenizers可用
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_roberta_fast"] = ["RobertaTokenizerFast"]

# 检查是否有torch可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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

# 检查是否有tensorflow可用
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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

# 检查是否有flax可用
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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

# 如果是类型检查模式，则导入特定模块
if TYPE_CHECKING:
    from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig, RobertaOnnxConfig
    # 从 tokenization_roberta 模块导入 RobertaTokenizer 类
    from .tokenization_roberta import RobertaTokenizer

    # 检查 tokenizers 库是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 tokenizers 库可用，则从 tokenization_roberta_fast 模块导入 RobertaTokenizerFast 类
        from .tokenization_roberta_fast import RobertaTokenizerFast

    # 检查 torch 库是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 库可用，则从 modeling_roberta 模块导入一系列类和常量
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

    # 检查 tensorflow 库是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 tensorflow 库可用，则从 modeling_tf_roberta 模块导入一系列类和常量
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

    # 检查 flax 库是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 flax 库可用，则从 modeling_flax_roberta 模块导入一系列类
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
# 如果不在之前的条件分支中，导入 sys 模块
import sys
# 将当前模块（__name__）的属性设置为 LazyModule 类的实例，实现延迟加载
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```