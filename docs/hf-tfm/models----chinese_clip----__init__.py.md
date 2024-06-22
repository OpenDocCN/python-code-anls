# `.\transformers\models\chinese_clip\__init__.py`

```py
# 版权声明和许可信息
# 版权归 OFA-Sys 团队作者和 HuggingFace 团队所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_chinese_clip": [
        "CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ChineseCLIPConfig",
        "ChineseCLIPOnnxConfig",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "processing_chinese_clip": ["ChineseCLIPProcessor"],
}

# 检查视觉库是否可用，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉库可用，则添加特征提取和图像处理模块到导入结构中
    _import_structure["feature_extraction_chinese_clip"] = ["ChineseCLIPFeatureExtractor"]
    _import_structure["image_processing_chinese_clip"] = ["ChineseCLIPImageProcessor"]

# 检查 Torch 库是否可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 库可用，则添加模型建模模块到导入结构中
    _import_structure["modeling_chinese_clip"] = [
        "CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ChineseCLIPModel",
        "ChineseCLIPPreTrainedModel",
        "ChineseCLIPTextModel",
        "ChineseCLIPVisionModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置和处理模块
    from .configuration_chinese_clip import (
        CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ChineseCLIPConfig,
        ChineseCLIPOnnxConfig,
        ChineseCLIPTextConfig,
        ChineseCLIPVisionConfig,
    )
    from .processing_chinese_clip import ChineseCLIPProcessor

    # 检查视觉库是否可用，如果不可用则忽略
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入特征提取和图像处理模块
        from .feature_extraction_chinese_clip import ChineseCLIPFeatureExtractor, ChineseCLIPImageProcessor

    # 检查 Torch 库是否可用，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型建模模块
        from .modeling_chinese_clip import (
            CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            ChineseCLIPModel,
            ChineseCLIPPreTrainedModel,
            ChineseCLIPTextModel,
            ChineseCLIPVisionModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```