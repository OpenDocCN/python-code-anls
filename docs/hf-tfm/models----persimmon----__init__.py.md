# `.\models\persimmon\__init__.py`

```py
# 导入需要的模块和类型检查
from typing import TYPE_CHECKING

# 导入必要的异常和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和模型
_import_structure = {
    "configuration_persimmon": ["PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP", "PersimmonConfig"],
}

# 尝试检查是否存在 Torch 模块，如果不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 存在，添加模型相关的导入结构
    _import_structure["modeling_persimmon"] = [
        "PersimmonForCausalLM",
        "PersimmonModel",
        "PersimmonPreTrainedModel",
        "PersimmonForSequenceClassification",
    ]

# 如果类型检查为真，则导入配置和模型
if TYPE_CHECKING:
    from .configuration_persimmon import PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP, PersimmonConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_persimmon import (
            PersimmonForCausalLM,
            PersimmonForSequenceClassification,
            PersimmonModel,
            PersimmonPreTrainedModel,
        )

# 如果不是类型检查阶段，则设置模块为懒加载模式
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```