# `.\models\dpt\__init__.py`

```
# 版权声明和许可证信息
# 版权所有 2022 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件以“原样”分发，
# 不附带任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。

# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入 LazyModule 和依赖检查函数
from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available, is_vision_available
from ...utils import OptionalDependencyNotAvailable

# 定义模块导入结构
_import_structure = {"configuration_dpt": ["DPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPTConfig"]}

# 检查视觉处理依赖是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加相应的导入结构
    _import_structure["feature_extraction_dpt"] = ["DPTFeatureExtractor"]
    _import_structure["image_processing_dpt"] = ["DPTImageProcessor"]

# 检查 PyTorch 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加相应的导入结构
    _import_structure["modeling_dpt"] = [
        "DPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPTForDepthEstimation",
        "DPTForSemanticSegmentation",
        "DPTModel",
        "DPTPreTrainedModel",
    ]

# 如果是类型检查模式，则导入特定的类和函数
if TYPE_CHECKING:
    from .configuration_dpt import DPT_PRETRAINED_CONFIG_ARCHIVE_MAP, DPTConfig

    # 检查视觉处理依赖是否可用，如果可用则导入相应的类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_dpt import DPTFeatureExtractor
        from .image_processing_dpt import DPTImageProcessor

    # 检查 PyTorch 是否可用，如果可用则导入相应的类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dpt import (
            DPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPTForDepthEstimation,
            DPTForSemanticSegmentation,
            DPTModel,
            DPTPreTrainedModel,
        )

# 如果不是类型检查模式，则配置 LazyModule 并设置当前模块
else:
    import sys

    # 将当前模块配置为 LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```