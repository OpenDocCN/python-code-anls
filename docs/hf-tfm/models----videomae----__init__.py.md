# `.\transformers\models\videomae\__init__.py`

```py
# 引入需要的模块和函数
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块结构（导入列表）
_import_structure = {
    "configuration_videomae": ["VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VideoMAEConfig"],
}

# 如果 torch 可用，则导入相关模块和函数
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_videomae"] = [
        "VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VideoMAEForPreTraining",
        "VideoMAEModel",
        "VideoMAEPreTrainedModel",
        "VideoMAEForVideoClassification",
    ]

# 如果 vision 可用，则导入相关模块和函数
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_videomae"] = ["VideoMAEFeatureExtractor"]
    _import_structure["image_processing_videomae"] = ["VideoMAEImageProcessor"]

# 如果是类型检查，则导入相关模块和函数
if TYPE_CHECKING:
    from .configuration_videomae import VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP, VideoMAEConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_videomae import (
            VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST,
            VideoMAEForPreTraining,
            VideoMAEForVideoClassification,
            VideoMAEModel,
            VideoMAEPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_videomae import VideoMAEFeatureExtractor
        from .image_processing_videomae import VideoMAEImageProcessor

# 如果不是类型检查，则将当前模块替换为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```