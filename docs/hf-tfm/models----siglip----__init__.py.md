# `.\transformers\models\siglip\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_siglip": [
        "SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SiglipConfig",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
    "processing_siglip": ["SiglipProcessor"],
    "tokenization_siglip": ["SiglipTokenizer"],
}

# 检查视觉模块是否可用，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_siglip"] = ["SiglipImageProcessor"]

# 检查 Torch 模块是否可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_siglip"] = [
        "SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SiglipModel",
        "SiglipPreTrainedModel",
        "SiglipTextModel",
        "SiglipVisionModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置相关模块
    from .configuration_siglip import (
        SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SiglipConfig,
        SiglipTextConfig,
        SiglipVisionConfig,
    )
    # 导入处理相关模块
    from .processing_siglip import SiglipProcessor
    # 导入标记化相关模块
    from .tokenization_siglip import SiglipTokenizer

    # 检查视觉模块是否可用，如果不可用则引发异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入图像处理相关模块
        from .image_processing_siglip import SiglipImageProcessor

    # 检查 Torch 模块是否可用，如果不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入建模相关模块
        from .modeling_siglip import (
            SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            SiglipModel,
            SiglipPreTrainedModel,
            SiglipTextModel,
            SiglipVisionModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设置为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```