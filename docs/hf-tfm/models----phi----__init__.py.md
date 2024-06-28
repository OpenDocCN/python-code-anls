# `.\models\phi\__init__.py`

```
# 导入类型检查模块 TYPE_CHECKING，用于在类型检查时导入特定模块
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需的异常和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，用于延迟加载模块
_import_structure = {
    "configuration_phi": ["PHI_PRETRAINED_CONFIG_ARCHIVE_MAP", "PhiConfig"],
}

# 检查是否支持 Torch，如果不支持则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Torch，则添加 modeling_phi 模块到导入结构中
    _import_structure["modeling_phi"] = [
        "PHI_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PhiPreTrainedModel",
        "PhiModel",
        "PhiForCausalLM",
        "PhiForSequenceClassification",
        "PhiForTokenClassification",
    ]

# 如果正在进行类型检查，则从 configuration_phi 和 modeling_phi 模块导入特定内容
if TYPE_CHECKING:
    from .configuration_phi import PHI_PRETRAINED_CONFIG_ARCHIVE_MAP, PhiConfig

    # 再次检查 Torch 是否可用，如果不可用则不导入 modeling_phi 模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_phi import (
            PHI_PRETRAINED_MODEL_ARCHIVE_LIST,
            PhiForCausalLM,
            PhiForSequenceClassification,
            PhiForTokenClassification,
            PhiModel,
            PhiPreTrainedModel,
        )

# 如果不是在类型检查模式下，则进行模块的懒加载处理
else:
    import sys

    # 使用 _LazyModule 类封装当前模块，以实现懒加载和模块属性的动态设置
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```