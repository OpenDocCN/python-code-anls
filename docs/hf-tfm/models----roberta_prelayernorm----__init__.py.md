# `.\transformers\models\roberta_prelayernorm\__init__.py`

```
# 版权声明和许可证信息
# 版权 2022 年由 HuggingFace 团队所有。保留所有权利
# 根据 Apache 许可证 2.0 版本许可
# 在遵守此许可证的前提下可以使用此文件
# 您可以获取许可证的副本
# 请访问 http://www.apache.org/licenses/LICENSE-2.0
# 未经有关法律规定或协议书面同意
# 根据许可证分发的软件以 "原样" 分发
# 没有任何明示或暗示的保证或条件
# 查看许可证以获取具体语言的规定和限制

# 引入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义需要导入的结构
_import_structure = {
    "configuration_roberta_prelayernorm": [
        "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaPreLayerNormConfig",
        "RobertaPreLayerNormOnnxConfig",
    ],
}

# 检查 PyTorch 是否可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 可用则导入相关模块
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

# 检查 TensorFlow 是否可用
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用则导入相关模块
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

# 检查 Flax 是否可用
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Flax 可用则导入相关模块
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


# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入所需的模块和类
    from .configuration_roberta_prelayernorm import (
        ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaPreLayerNormConfig,
        RobertaPreLayerNormOnnxConfig,
    )
    
    # 尝试检测是否安装了 torch 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果能成功导入 torch 库，则执行以下代码
    else:
        # 导入相关的模型定义和类
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
    
    # 尝试检测是否安装了 tensorflow 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果能成功导入 tensorflow 库，则执行以下代码
    else:
        # 导入相关的模型定义和类
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
    
    # 尝试检测是否安装了 flax 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果能成功导入 flax 库，则执行以下代码
    else:
        # 导入相关的模型定义和类
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
# 如果前面的条件不满足，执行这个分支
else:
    # 导入 Python 内置的 sys 模块
    import sys

    # 使用 _LazyModule 对象替换当前模块的 sys.modules 条目
    # 这样可以实现懒加载，即在实际使用该模块时才完全加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```