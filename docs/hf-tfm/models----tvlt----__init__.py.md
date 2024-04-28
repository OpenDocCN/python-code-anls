# `.\transformers\models\tvlt\__init__.py`

```
# flake8: noqa
# 在这个模块中，无法忽略 "F401 '...' imported but unused" 警告，但要保留其他警告。所以，不检查这个模块。

# 2023 年版权归 HuggingFace 团队所有。
# 
# 根据 Apache 许可 2.0 版本（“许可证”）获得许可；
# 您不得使用此文件，除非遵守许可证。
# 您可以在以下位置获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 无论是明示还是暗示的，都没有任何形式的担保或条件。
# 有关特定语言的权限，请参阅许可证。

from typing import TYPE_CHECKING

# 从工具包中导入相关功能和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构
_import_structure = {
    "configuration_tvlt": ["TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP", "TvltConfig"],
    "feature_extraction_tvlt": ["TvltFeatureExtractor"],
    "processing_tvlt": ["TvltProcessor"],
}

# 检查是否存在 torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则导入相关模型
    _import_structure["modeling_tvlt"] = [
        "TVLT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TvltModel",
        "TvltForPreTraining",
        "TvltForAudioVisualClassification",
        "TvltPreTrainedModel",
    ]

# 检查是否存在 vision 库
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 vision 库，则导入图像处理模块
    _import_structure["image_processing_tvlt"] = ["TvltImageProcessor"]

# 如果是类型检查，则导入相关类
if TYPE_CHECKING:
    from .configuration_tvlt import TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP, TvltConfig
    from .processing_tvlt import TvltProcessor
    from .feature_extraction_tvlt import TvltFeatureExtractor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tvlt import (
            TVLT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TvltForAudioVisualClassification,
            TvltForPreTraining,
            TvltModel,
            TvltPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_tvlt import TvltImageProcessor

# 如果不是类型检查，则将该模块设置为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```