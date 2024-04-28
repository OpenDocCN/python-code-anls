# `.\transformers\models\persimmon\__init__.py`

```py
# 引入所需的模块和函数
from typing import TYPE_CHECKING
# 引入自定义的异常类和LazyModule
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和模型
_import_structure = {
    "configuration_persimmon": ["PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP", "PersimmonConfig"],
}

# 检查是否存在 torch 库，如果不存在则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加模型部分到导入结构中
    _import_structure["modeling_persimmon"] = [
        "PersimmonForCausalLM",
        "PersimmonModel",
        "PersimmonPreTrainedModel",
        "PersimmonForSequenceClassification",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从配置模块中导入特定的类和变量
    from .configuration_persimmon import PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP, PersimmonConfig
    # 检查是否存在 torch 库，如果不存在则引发自定义异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则从模型模块中导入特定的类
        from .modeling_persimmon import (
            PersimmonForCausalLM,
            PersimmonForSequenceClassification,
            PersimmonModel,
            PersimmonPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设为延迟加载模块，使用 LazyModule 类
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```