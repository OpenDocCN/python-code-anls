# `.\models\superpoint\__init__.py`

```py
# 版权声明和许可证信息
# 版权所有 2024 年 HuggingFace 团队保留所有权利。
# 
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则依据许可证分发的软件是基于“按原样”提供的，无任何明示或暗示的担保或条件。
# 请参阅许可证获取具体的语言和权限。
from typing import TYPE_CHECKING

# 从 HuggingFace 的 utils 模块导入相关内容
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构字典
_import_structure = {
    "configuration_superpoint": [
        "SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SuperPointConfig",
    ]
}

# 检查视觉模块是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 SuperPointImageProcessor 导入到 image_processing_superpoint 模块中
    _import_structure["image_processing_superpoint"] = ["SuperPointImageProcessor"]

# 检查 torch 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将一些 SuperPoint 模型相关的类和常量导入到 modeling_superpoint 模块中
    _import_structure["modeling_superpoint"] = [
        "SUPERPOINT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SuperPointForKeypointDetection",
        "SuperPointPreTrainedModel",
    ]

# 如果是类型检查模式，导入配置和模型相关内容
if TYPE_CHECKING:
    from .configuration_superpoint import (
        SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SuperPointConfig,
    )

    # 如果视觉模块可用，导入图像处理相关的内容
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_superpoint import SuperPointImageProcessor

    # 如果 torch 可用，导入模型相关的内容
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_superpoint import (
            SUPERPOINT_PRETRAINED_MODEL_ARCHIVE_LIST,
            SuperPointForKeypointDetection,
            SuperPointPreTrainedModel,
        )

# 如果不是类型检查模式，设置模块为 LazyModule
else:
    import sys

    # 将当前模块设为 LazyModule，支持按需导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```