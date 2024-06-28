# `.\models\vitmatte\__init__.py`

```
# 版权声明和保留所有权利的声明
# 根据 Apache 许可证 2.0 版本授权，许可证详细信息可以通过给定的 URL 获取
from typing import TYPE_CHECKING

# 从特定的路径导入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构，包含配置和模型定义
_import_structure = {"configuration_vitmatte": ["VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitMatteConfig"]}

# 检查视觉处理模块是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将图像处理模块导入结构中
    _import_structure["image_processing_vitmatte"] = ["VitMatteImageProcessor"]

# 检查 Torch 是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，将模型处理模块导入结构中
    _import_structure["modeling_vitmatte"] = [
        "VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VitMattePreTrainedModel",
        "VitMatteForImageMatting",
    ]

# 如果处于类型检查模式，导入特定的配置和模型类
if TYPE_CHECKING:
    from .configuration_vitmatte import VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP, VitMatteConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果视觉处理可用，导入图像处理器类
        from .image_processing_vitmatte import VitMatteImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，导入模型相关类
        from .modeling_vitmatte import (
            VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitMatteForImageMatting,
            VitMattePreTrainedModel,
        )

# 如果不处于类型检查模式，使用 LazyModule 来处理模块的延迟导入
else:
    import sys

    # 将当前模块替换为 LazyModule，实现按需导入功能
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```