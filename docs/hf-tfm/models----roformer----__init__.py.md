# `.\transformers\models\roformer\__init__.py`

```
# 版权声明和许可证信息
# 从相应的模块中导入需要的依赖
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_roformer": ["ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoFormerConfig", "RoFormerOnnxConfig"],
    "tokenization_roformer": ["RoFormerTokenizer"],
}

# 检查是否有tokenizers可用，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_roformer_fast"] = ["RoFormerTokenizerFast"]

# 检查是否有torch可用，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_roformer"] = [
        "ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RoFormerForCausalLM",
        "RoFormerForMaskedLM",
        "RoFormerForMultipleChoice",
        "RoFormerForQuestionAnswering",
        "RoFormerForSequenceClassification",
        "RoFormerForTokenClassification",
        "RoFormerLayer",
        "RoFormerModel",
        "RoFormerPreTrainedModel",
        "load_tf_weights_in_roformer",
    ]

# 检查是否有tensorflow可用，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_roformer"] = [
        "TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRoFormerForCausalLM",
        "TFRoFormerForMaskedLM",
        "TFRoFormerForMultipleChoice",
        "TFRoFormerForQuestionAnswering",
        "TFRoFormerForSequenceClassification",
        "TFRoFormerForTokenClassification",
        "TFRoFormerLayer",
        "TFRoFormerModel",
        "TFRoFormerPreTrainedModel",
    ]

# 检查是否有flax可用，如果没有则引发OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_roformer"] = [
        "FLAX_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlaxRoFormerForMaskedLM",
        "FlaxRoFormerForMultipleChoice",
        "FlaxRoFormerForQuestionAnswering",
        "FlaxRoFormerForSequenceClassification",
        "FlaxRoFormerForTokenClassification",
        "FlaxRoFormerModel",
        "FlaxRoFormerPreTrainedModel",
    ]

# 如果是类型检查，则...
if TYPE_CHECKING:
    # 从configuration_roformer.py文件中导入ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP、RoFormerConfig和RoFormerOnnxConfig
    from .configuration_roformer import ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, RoFormerConfig, RoFormerOnnxConfig
    # 从tokenization_roformer.py文件中导入RoFormerTokenizer
    from .tokenization_roformer import RoFormerTokenizer

    # 尝试检查是否安装了 tokenizers 库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若出现异常则不做任何操作
        pass
    else:
        # 若无异常则从tokenization_roformer_fast.py文件中导入RoFormerTokenizerFast
        from .tokenization_roformer_fast import RoFormerTokenizerFast

    # 尝试检查是否安装了 torch 库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若出现异常则不做任何操作
        pass
    else:
        # 若无异常则从modeling_roformer.py文件中导入相关模块
        from .modeling_roformer import (
            ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            RoFormerForCausalLM,
            RoFormerForMaskedLM,
            RoFormerForMultipleChoice,
            RoFormerForQuestionAnswering,
            RoFormerForSequenceClassification,
            RoFormerForTokenClassification,
            RoFormerLayer,
            RoFormerModel,
            RoFormerPreTrainedModel,
            load_tf_weights_in_roformer,
        )

    # 尝试检查是否安装了 tensorflow 库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若出现异常则不做任何操作
        pass
    else:
        # 若无异常则从modeling_tf_roformer.py文件中导入相关模块
        from .modeling_tf_roformer import (
            TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRoFormerForCausalLM,
            TFRoFormerForMaskedLM,
            TFRoFormerForMultipleChoice,
            TFRoFormerForQuestionAnswering,
            TFRoFormerForSequenceClassification,
            TFRoFormerForTokenClassification,
            TFRoFormerLayer,
            TFRoFormerModel,
            TFRoFormerPreTrainedModel,
        )

    # 尝试检查是否安装了 flax 库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若出现异常则不做任何操作
        pass
    else:
        # 若无异常则从modeling_flax_roformer.py文件中导入相关模块
        from .modeling_flax_roformer import (
            FLAX_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaxRoFormerForMaskedLM,
            FlaxRoFormerForMultipleChoice,
            FlaxRoFormerForQuestionAnswering,
            FlaxRoFormerForSequenceClassification,
            FlaxRoFormerForTokenClassification,
            FlaxRoFormerModel,
            FlaxRoFormerPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于操作解释器相关的功能
    import sys

    # 将当前模块替换为 LazyModule 对象，延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```