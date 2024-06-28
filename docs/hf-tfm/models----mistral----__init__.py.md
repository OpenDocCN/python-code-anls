# `.\models\mistral\__init__.py`

```
# 引入需要的模块和函数
from typing import TYPE_CHECKING

# 引入自定义异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_mistral": ["MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MistralConfig"],
}

# 检查是否可用 Torch，若不可用则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 Mistral 模型相关的导入结构
    _import_structure["modeling_mistral"] = [
        "MistralForCausalLM",
        "MistralModel",
        "MistralPreTrainedModel",
        "MistralForSequenceClassification",
    ]

# 检查是否可用 Flax，若不可用则引发自定义异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 FlaxMistral 模型相关的导入结构
    _import_structure["modeling_flax_mistral"] = [
        "FlaxMistralForCausalLM",
        "FlaxMistralModel",
        "FlaxMistralPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 Mistral 配置相关的类和函数
    from .configuration_mistral import MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MistralConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Mistral 模型相关的类和函数
        from .modeling_mistral import (
            MistralForCausalLM,
            MistralForSequenceClassification,
            MistralModel,
            MistralPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 FlaxMistral 模型相关的类和函数
        from .modeling_flax_mistral import (
            FlaxMistralForCausalLM,
            FlaxMistralModel,
            FlaxMistralPreTrainedModel,
        )

# 如果不是类型检查模式，则进行延迟加载模块的设置
else:
    import sys

    # 将当前模块设置为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```