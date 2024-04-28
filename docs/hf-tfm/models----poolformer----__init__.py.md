# `.\transformers\models\poolformer\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_poolformer": [
        "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PoolFormerConfig",
        "PoolFormerOnnxConfig",
    ]
}

# 检查视觉模块是否可用，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉模块可用，则添加特征提取和图像处理模块到导入结构中
    _import_structure["feature_extraction_poolformer"] = ["PoolFormerFeatureExtractor"]
    _import_structure["image_processing_poolformer"] = ["PoolFormerImageProcessor"]

# 检查 Torch 模块是否可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 模块可用，则添加模型建模模块到导入结构中
    _import_structure["modeling_poolformer"] = [
        "POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PoolFormerForImageClassification",
        "PoolFormerModel",
        "PoolFormerPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置模块中的特定类和变量
    from .configuration_poolformer import (
        POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PoolFormerConfig,
        PoolFormerOnnxConfig,
    )

    # 检查视觉模块是否可用，如果可用则导入特征提取和图像处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_poolformer import PoolFormerFeatureExtractor
        from .image_processing_poolformer import PoolFormerImageProcessor

    # 检查 Torch 模块是否可用，如��可用则导入模型建模模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_poolformer import (
            POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            PoolFormerForImageClassification,
            PoolFormerModel,
            PoolFormerPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```