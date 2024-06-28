# `.\models\roberta_prelayernorm\__init__.py`

```
# 引入类型检查依赖，用于在类型检查环境下做条件导入
from typing import TYPE_CHECKING

# 从工具模块中导入相关工具和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构的字典，用于存储模块路径和名称
_import_structure = {
    "configuration_roberta_prelayernorm": [
        "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaPreLayerNormConfig",
        "RobertaPreLayerNormOnnxConfig",
    ],
}

# 检查是否支持 Torch 库，若不支持则引发依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Torch 库，则添加相关模型的路径和名称到导入结构字典
    _import_structure["modeling_roberta_prelayernorm"] = [
        "ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RobertaPreLayerNormForCausalLM",
        "RobertaPreLayerNormForMaskedLM",
        "RobertaPreLayerNormForMultipleChoice",
        "RobertaPreLayerNormForQuestionAnswering",
        "RobertaPreLayerNormForSequenceClassification",
        "RobertaPreLayerNormForTokenClassification",
        "RobertaPreLayerNormModel",
        "RobertaPreLayerNormPreTrainedModel",
    ]

# 检查是否支持 TensorFlow 库，若不支持则引发依赖不可用异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 TensorFlow 库，则添加相关模型的路径和名称到导入结构字典
    _import_structure["modeling_tf_roberta_prelayernorm"] = [
        "TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRobertaPreLayerNormForCausalLM",
        "TFRobertaPreLayerNormForMaskedLM",
        "TFRobertaPreLayerNormForMultipleChoice",
        "TFRobertaPreLayerNormForQuestionAnswering",
        "TFRobertaPreLayerNormForSequenceClassification",
        "TFRobertaPreLayerNormForTokenClassification",
        "TFRobertaPreLayerNormMainLayer",
        "TFRobertaPreLayerNormModel",
        "TFRobertaPreLayerNormPreTrainedModel",
    ]

# 检查是否支持 Flax 库，若不支持则引发依赖不可用异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Flax 库，则添加相关模型的路径和名称到导入结构字典
    _import_structure["modeling_flax_roberta_prelayernorm"] = [
        "FlaxRobertaPreLayerNormForCausalLM",
        "FlaxRobertaPreLayerNormForMaskedLM",
        "FlaxRobertaPreLayerNormForMultipleChoice",
        "FlaxRobertaPreLayerNormForQuestionAnswering",
        "FlaxRobertaPreLayerNormForSequenceClassification",
        "FlaxRobertaPreLayerNormForTokenClassification",
        "FlaxRobertaPreLayerNormModel",
        "FlaxRobertaPreLayerNormPreTrainedModel",
    ]

# 如果在类型检查环境下
if TYPE_CHECKING:
    # 导入 RoBERTa 预训练模型配置文件映射和配置类
    from .configuration_roberta_prelayernorm import (
        ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaPreLayerNormConfig,
        RobertaPreLayerNormOnnxConfig,
    )
    
    # 检查是否存在 Torch 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 RoBERTa 预训练模型相关类（基于 Torch）
        from .modeling_roberta_prelayernorm import (
            ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST,
            RobertaPreLayerNormForCausalLM,
            RobertaPreLayerNormForMaskedLM,
            RobertaPreLayerNormForMultipleChoice,
            RobertaPreLayerNormForQuestionAnswering,
            RobertaPreLayerNormForSequenceClassification,
            RobertaPreLayerNormForTokenClassification,
            RobertaPreLayerNormModel,
            RobertaPreLayerNormPreTrainedModel,
        )
    
    # 检查是否存在 TensorFlow 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 RoBERTa 预训练模型相关类（基于 TensorFlow）
        from .modeling_tf_roberta_prelayernorm import (
            TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRobertaPreLayerNormForCausalLM,
            TFRobertaPreLayerNormForMaskedLM,
            TFRobertaPreLayerNormForMultipleChoice,
            TFRobertaPreLayerNormForQuestionAnswering,
            TFRobertaPreLayerNormForSequenceClassification,
            TFRobertaPreLayerNormForTokenClassification,
            TFRobertaPreLayerNormMainLayer,
            TFRobertaPreLayerNormModel,
            TFRobertaPreLayerNormPreTrainedModel,
        )
    
    # 检查是否存在 Flax 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 RoBERTa 预训练模型相关类（基于 Flax）
        from .modeling_flax_roberta_prelayernorm import (
            FlaxRobertaPreLayerNormForCausalLM,
            FlaxRobertaPreLayerNormForMaskedLM,
            FlaxRobertaPreLayerNormForMultipleChoice,
            FlaxRobertaPreLayerNormForQuestionAnswering,
            FlaxRobertaPreLayerNormForSequenceClassification,
            FlaxRobertaPreLayerNormForTokenClassification,
            FlaxRobertaPreLayerNormModel,
            FlaxRobertaPreLayerNormPreTrainedModel,
        )
else:
    # 导入系统模块 sys
    import sys

    # 将当前模块的名字作为键，将一个特定的 _LazyModule 对象作为值，存入 sys.modules 字典中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```