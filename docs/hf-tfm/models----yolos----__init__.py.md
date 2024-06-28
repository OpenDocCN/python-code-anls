# `.\models\yolos\__init__.py`

```py
# 版权声明和许可信息
# 2022 年由 HuggingFace 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权
# 您除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件根据“原样”分发，
# 没有任何明示或暗示的担保或条件。
# 请查阅许可证以了解具体的法律信息和限制。
from typing import TYPE_CHECKING

# 从 utils 中导入所需的模块和异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 指定需要导入的结构
_import_structure = {"configuration_yolos": ["YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP", "YolosConfig", "YolosOnnxConfig"]}

# 检查视觉相关依赖是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果依赖可用，则添加 feature_extraction_yolos 和 image_processing_yolos 到导入结构
    _import_structure["feature_extraction_yolos"] = ["YolosFeatureExtractor"]
    _import_structure["image_processing_yolos"] = ["YolosImageProcessor"]

# 检查 Torch 相关依赖是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果依赖可用，则添加 modeling_yolos 到导入结构
    _import_structure["modeling_yolos"] = [
        "YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "YolosForObjectDetection",
        "YolosModel",
        "YolosPreTrainedModel",
    ]

# 如果在 TYPE_CHECKING 模式下
if TYPE_CHECKING:
    # 从 configuration_yolos 模块导入特定的类和常量
    from .configuration_yolos import YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP, YolosConfig, YolosOnnxConfig

    # 检查视觉相关依赖是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果依赖可用，则从 feature_extraction_yolos 和 image_processing_yolos 模块导入相应的类
        from .feature_extraction_yolos import YolosFeatureExtractor
        from .image_processing_yolos import YolosImageProcessor

    # 检查 Torch 相关依赖是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果依赖可用，则从 modeling_yolos 模块导入相应的类和常量
        from .modeling_yolos import (
            YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST,
            YolosForObjectDetection,
            YolosModel,
            YolosPreTrainedModel,
        )

# 如果不在 TYPE_CHECKING 模式下，则将当前模块替换为延迟加载模块
else:
    import sys

    # 使用 _LazyModule 类进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```