# `.\models\deprecated\van\__init__.py`

```py
# 版权声明和许可信息，声明代码版权及许可协议
#
# 依赖导入：从不同位置导入必要的模块和函数
from typing import TYPE_CHECKING

# 引入自定义的异常类和模块，用于在依赖不可用时触发异常
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构的字典，用于懒加载模块和函数
_import_structure = {"configuration_van": ["VAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "VanConfig"]}

# 检查是否可用 Torch 库，如果不可用则抛出自定义的依赖异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加以下模块到导入结构字典中
    _import_structure["modeling_van"] = [
        "VAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VanForImageClassification",
        "VanModel",
        "VanPreTrainedModel",
    ]

# 如果当前环境支持类型检查，则从特定模块导入相关配置和模型类
if TYPE_CHECKING:
    from .configuration_van import VAN_PRETRAINED_CONFIG_ARCHIVE_MAP, VanConfig

    # 再次检查 Torch 是否可用，不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 modeling_van 模块导入以下类和常量
        from .modeling_van import (
            VAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            VanForImageClassification,
            VanModel,
            VanPreTrainedModel,
        )

# 如果不是类型检查环境，则直接将当前模块注册为懒加载模块
else:
    import sys

    # 将当前模块注册为懒加载模块，用 _LazyModule 进行包装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```