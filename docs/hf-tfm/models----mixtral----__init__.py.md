# `.\models\mixtral\__init__.py`

```
# 导入需要的模块和函数
from typing import TYPE_CHECKING
# 从项目内部工具中导入必要的异常和工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构，指定哪些类和函数可以被外部导入
_import_structure = {
    "configuration_mixtral": ["MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MixtralConfig"],
}

# 检查是否 Torch 可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加额外的模型类到导入结构中
    _import_structure["modeling_mixtral"] = [
        "MixtralForCausalLM",
        "MixtralModel",
        "MixtralPreTrainedModel",
        "MixtralForSequenceClassification",
    ]

# 如果是类型检查阶段，则从配置和模型模块导入特定类和常量
if TYPE_CHECKING:
    from .configuration_mixtral import MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MixtralConfig

    # 再次检查 Torch 是否可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则从模型模块导入额外的模型类
        from .modeling_mixtral import (
            MixtralForCausalLM,
            MixtralForSequenceClassification,
            MixtralModel,
            MixtralPreTrainedModel,
        )

# 如果不是类型检查阶段，则将当前模块注册为一个 LazyModule
else:
    import sys

    # 动态将当前模块替换为 LazyModule 对象，这样在导入时会延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```