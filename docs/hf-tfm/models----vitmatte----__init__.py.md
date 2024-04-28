# `.\transformers\models\vitmatte\__init__.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {"configuration_vitmatte": ["VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitMatteConfig"]}

# 检查视觉处理模块是否可用，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_vitmatte"] = ["VitMatteImageProcessor"]

# 检查 Torch 模块是否可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vitmatte"] = [
        "VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VitMattePreTrainedModel",
        "VitMatteForImageMatting",
    ]

# 如果是类型检查模式，则导入特定模块
if TYPE_CHECKING:
    from .configuration_vitmatte import VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP, VitMatteConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_vitmatte import VitMatteImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vitmatte import (
            VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitMatteForImageMatting,
            VitMattePreTrainedModel,
        )

# 如果不是类型检查模式，则将模块定义为 LazyModule
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```