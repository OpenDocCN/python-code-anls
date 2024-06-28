# `.\models\starcoder2\__init__.py`

```py
# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入必要的依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_starcoder2": ["STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Starcoder2Config"],
}

# 检查是否存在 Torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，添加模型相关的导入结构
    _import_structure["modeling_starcoder2"] = [
        "Starcoder2ForCausalLM",
        "Starcoder2Model",
        "Starcoder2PreTrainedModel",
        "Starcoder2ForSequenceClassification",
    ]

# 如果类型检查开启
if TYPE_CHECKING:
    # 导入配置相关的模块和类
    from .configuration_starcoder2 import STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP, Starcoder2Config

    # 再次检查 Torch 是否可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的模块和类
        from .modeling_starcoder2 import (
            Starcoder2ForCausalLM,
            Starcoder2ForSequenceClassification,
            Starcoder2Model,
            Starcoder2PreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块替换为 LazyModule，用于惰性加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```