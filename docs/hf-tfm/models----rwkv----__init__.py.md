# `.\transformers\models\rwkv\__init__.py`

```py
# 导入所需的类型检查模块
from typing import TYPE_CHECKING

# 导入所需的工具函数、异常类和懒加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义需要导入的模块结构，包括配置和建模两个部分
_import_structure = {
    "configuration_rwkv": ["RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP", "RwkvConfig", "RwkvOnnxConfig"],
}

# 检查是否已经安装了 Torch 库
try:
    if not is_torch_available():
        # 如果未安装，则抛出异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果已安装 Torch，则导入建模部分的模块
    _import_structure["modeling_rwkv"] = [
        "RWKV_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RwkvForCausalLM",
        "RwkvModel",
        "RwkvPreTrainedModel",
    ]

# 如果当前环境支持类型检查
if TYPE_CHECKING:
    # 导入配置部分的模块和类
    from .configuration_rwkv import RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP, RwkvConfig, RwkvOnnxConfig

    # 再次检查是否已安装 Torch
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入建模部分的模块和类
        from .modeling_rwkv import (
            RWKV_PRETRAINED_MODEL_ARCHIVE_LIST,
            RwkvForCausalLM,
            RwkvModel,
            RwkvPreTrainedModel,
        )
# 如果不支持类型检查
else:
    import sys

    # 动态创建一个懒加载模块，并将其设置为当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```