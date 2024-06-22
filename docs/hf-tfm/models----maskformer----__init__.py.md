# `.\transformers\models\maskformer\__init__.py`

```py
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_maskformer": ["MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "MaskFormerConfig"],
    "configuration_maskformer_swin": ["MaskFormerSwinConfig"],
}

# 检查视觉模块是否可用，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉模块可用，则添加以下模块到导入结构中
    _import_structure["feature_extraction_maskformer"] = ["MaskFormerFeatureExtractor"]
    _import_structure["image_processing_maskformer"] = ["MaskFormerImageProcessor"]

# 检查 Torch 模块是否可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 模块可用，则添加以下模块到导入结构中
    _import_structure["modeling_maskformer"] = [
        "MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MaskFormerForInstanceSegmentation",
        "MaskFormerModel",
        "MaskFormerPreTrainedModel",
    ]
    _import_structure["modeling_maskformer_swin"] = [
        "MaskFormerSwinBackbone",
        "MaskFormerSwinModel",
        "MaskFormerSwinPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置和模型相关模块
    from .configuration_maskformer import MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, MaskFormerConfig
    from .configuration_maskformer_swin import MaskFormerSwinConfig

    # 检查视觉模块是否可用，如果不可用则引发异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果视觉模块可用，则导入以下模块
        from .feature_extraction_maskformer import MaskFormerFeatureExtractor
        from .image_processing_maskformer import MaskFormerImageProcessor
    # 检查 Torch 模块是否可用，如果不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 模块可用，则导入以下模块
        from .modeling_maskformer import (
            MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            MaskFormerForInstanceSegmentation,
            MaskFormerModel,
            MaskFormerPreTrainedModel,
        )
        from .modeling_maskformer_swin import (
            MaskFormerSwinBackbone,
            MaskFormerSwinModel,
            MaskFormerSwinPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```