# `.\models\imagegpt\__init__.py`

```
# 版权声明和许可信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_imagegpt": ["IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ImageGPTConfig", "ImageGPTOnnxConfig"]
}

# 检查视觉模块是否可用，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉模块可用，则添加以下模块到导入结构中
    _import_structure["feature_extraction_imagegpt"] = ["ImageGPTFeatureExtractor"]
    _import_structure["image_processing_imagegpt"] = ["ImageGPTImageProcessor"]

# 检查 Torch 模块是否可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 模块可用，则添加以下模块到导入结构中
    _import_structure["modeling_imagegpt"] = [
        "IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ImageGPTForCausalImageModeling",
        "ImageGPTForImageClassification",
        "ImageGPTModel",
        "ImageGPTPreTrainedModel",
        "load_tf_weights_in_imagegpt",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关模块
    from .configuration_imagegpt import IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, ImageGPTConfig, ImageGPTOnnxConfig

    # 检查视觉模块是否可用，如果不可用则忽略
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果视觉模块可用，则导入以下模块
        from .feature_extraction_imagegpt import ImageGPTFeatureExtractor
        from .image_processing_imagegpt import ImageGPTImageProcessor

    # 检查 Torch 模块是否可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 模块可用，则导入以下模块
        from .modeling_imagegpt import (
            IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ImageGPTForCausalImageModeling,
            ImageGPTForImageClassification,
            ImageGPTModel,
            ImageGPTPreTrainedModel,
            load_tf_weights_in_imagegpt,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```