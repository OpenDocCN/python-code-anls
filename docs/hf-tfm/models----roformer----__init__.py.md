# `.\models\roformer\__init__.py`

```py
# 导入必要的模块和函数来检查当前环境中是否可用特定的依赖项
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典，表示需要导入的模块结构
_import_structure = {
    "configuration_roformer": ["ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoFormerConfig", "RoFormerOnnxConfig"],
    "tokenization_roformer": ["RoFormerTokenizer"],
}

# 检查是否可用tokenizers，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将RoFormerTokenizerFast加入导入结构字典
    _import_structure["tokenization_roformer_fast"] = ["RoFormerTokenizerFast"]

# 检查是否可用torch，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将模型相关的torch模块加入导入结构字典
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

# 检查是否可用tensorflow，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将模型相关的tensorflow模块加入导入结构字典
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

# 检查是否可用flax，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将模型相关的flax模块加入导入结构字典
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

# 如果是类型检查阶段，处理完成
if TYPE_CHECKING:
    pass
    # 导入 RoFormer 相关配置文件和类
    from .configuration_roformer import ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, RoFormerConfig, RoFormerOnnxConfig
    # 导入 RoFormer 的 Tokenizer 类
    from .tokenization_roformer import RoFormerTokenizer
    
    # 检查是否安装了 tokenizers 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 tokenizers 库，则导入 RoFormer 的快速 Tokenizer 类
        from .tokenization_roformer_fast import RoFormerTokenizerFast
    
    # 检查是否安装了 PyTorch 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 PyTorch 库，则导入 RoFormer 的相关模型和函数
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
    
    # 检查是否安装了 TensorFlow 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 TensorFlow 库，则导入 TensorFlow 版本的 RoFormer 模型和函数
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
    
    # 检查是否安装了 Flax 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 Flax 库，则导入 Flax 版本的 RoFormer 模型和函数
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
    # 导入 sys 模块，用于操作 Python 解释器的系统功能
    import sys
    
    # 将当前模块添加到 sys.modules 中，以 LazyModule 的形式延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```