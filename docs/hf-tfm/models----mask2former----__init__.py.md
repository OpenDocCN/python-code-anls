# `.\transformers\models\mask2former\__init__.py`

```
# 版权声明，版权所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本 (以下简称"许可证") 授权
# 除非你遵守许可证，否则你不得使用此文件
# 你可以从以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则在"按原样"的基础上发布的软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制
from typing import TYPE_CHECKING

# 导入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构
_import_structure = {
    "configuration_mask2former": [
        "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Mask2FormerConfig",
    ],
}

# 检查视觉库是否可用
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_mask2former"] = ["Mask2FormerImageProcessor"]

# 检查 torch 库是否可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mask2former"] = [
        "MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Mask2FormerForUniversalSegmentation",
        "Mask2FormerModel",
        "Mask2FormerPreTrainedModel",
    ]

# 如果是类型检查模式，导入配置和模型
if TYPE_CHECKING:
    from .configuration_mask2former import MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, Mask2FormerConfig
    # 检查视觉库是否可用，导入图像处理依赖
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_mask2former import Mask2FormerImageProcessor
    # 检查 torch 库是否可用，导入模型依赖
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mask2former import (
            MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )
# 如果不是类型检查模式，将当前模块设为延迟加载模块
else:
    import sys
    # 创建延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```