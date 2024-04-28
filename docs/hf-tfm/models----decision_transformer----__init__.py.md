# `.\models\decision_transformer\__init__.py`

```
# 导入必要的模块和依赖
from typing import TYPE_CHECKING
# 导入自定义的异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义需要导入的模块和对应的结构
_import_structure = {
    "configuration_decision_transformer": [
        "DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DecisionTransformerConfig",
    ],
}

# 检查是否已安装torch，若未安装则抛出自定义的OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了torch，则添加需要导入的模块和对应的结构
    _import_structure["modeling_decision_transformer"] = [
        "DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DecisionTransformerGPT2Model",
        "DecisionTransformerGPT2PreTrainedModel",
        "DecisionTransformerModel",
        "DecisionTransformerPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入模型配置和结构
    from .configuration_decision_transformer import (
        DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DecisionTransformerConfig,
    )
    # 检查是否已安装torch，若未安装则抛出自定义的OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型结构
        from .modeling_decision_transformer import (
            DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DecisionTransformerGPT2Model,
            DecisionTransformerGPT2PreTrainedModel,
            DecisionTransformerModel,
            DecisionTransformerPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入sys模块
    import sys
    # 将当前模块设置为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```