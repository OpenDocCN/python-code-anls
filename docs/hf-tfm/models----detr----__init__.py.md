# `.\models\detr\__init__.py`

```
# 版权声明和许可信息，指明代码版权及使用许可条件
# The HuggingFace 团队，版权所有 © 2020
#
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的担保或条件。
# 有关许可的详细信息，请参阅许可证。
#
# 引入类型检查模块
from typing import TYPE_CHECKING

# 从...utils 中引入必要的依赖项，包括自定义异常和延迟加载模块的工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构字典
_import_structure = {"configuration_detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig", "DetrOnnxConfig"]}

# 检查视觉处理模块是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果模块可用，添加相关特征提取和图像处理的导入结构
    _import_structure["feature_extraction_detr"] = ["DetrFeatureExtractor"]
    _import_structure["image_processing_detr"] = ["DetrImageProcessor"]

# 检查 Torch 模块是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 模块可用，添加相关模型建模的导入结构
    _import_structure["modeling_detr"] = [
        "DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DetrForObjectDetection",
        "DetrForSegmentation",
        "DetrModel",
        "DetrPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从配置模块导入相关类和映射
    from .configuration_detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig, DetrOnnxConfig

    try:
        # 检查视觉处理模块是否可用，如果不可用则忽略
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入特征提取和图像处理模块
        from .feature_extraction_detr import DetrFeatureExtractor
        from .image_processing_detr import DetrImageProcessor

    try:
        # 检查 Torch 模块是否可用，如果不可用则忽略
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型建模相关类
        from .modeling_detr import (
            DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块指定为一个延迟加载模块，用于动态导入定义的结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```