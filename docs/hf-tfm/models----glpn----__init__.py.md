# `.\models\glpn\__init__.py`

```py
# 版权声明和许可信息，标明此代码版权归 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律要求或书面同意，软件按“原样”分发，不提供任何明示或暗示的担保或条件。
# 请查阅许可证以获取特定语言的详细信息。
from typing import TYPE_CHECKING

# 导入 OptionalDependencyNotAvailable 异常类、_LazyModule 类、is_torch_available 和 is_vision_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构的字典
_import_structure = {"configuration_glpn": ["GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP", "GLPNConfig"]}

# 检查视觉库是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉库可用，则添加特征提取和图像处理到导入结构字典
    _import_structure["feature_extraction_glpn"] = ["GLPNFeatureExtractor"]
    _import_structure["image_processing_glpn"] = ["GLPNImageProcessor"]

# 检查 Torch 库是否可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 库可用，则添加建模相关类到导入结构字典
    _import_structure["modeling_glpn"] = [
        "GLPN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GLPNForDepthEstimation",
        "GLPNLayer",
        "GLPNModel",
        "GLPNPreTrainedModel",
    ]

# 如果是类型检查模式，则进行类型检查导入
if TYPE_CHECKING:
    # 从 configuration_glpn 模块导入特定符号
    from .configuration_glpn import GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP, GLPNConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 feature_extraction_glpn 模块导入特定符号
        from .feature_extraction_glpn import GLPNFeatureExtractor
        # 从 image_processing_glpn 模块导入特定符号
        from .image_processing_glpn import GLPNImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 modeling_glpn 模块导入特定符号
        from .modeling_glpn import (
            GLPN_PRETRAINED_MODEL_ARCHIVE_LIST,
            GLPNForDepthEstimation,
            GLPNLayer,
            GLPNModel,
            GLPNPreTrainedModel,
        )

# 如果不是类型检查模式，则使用 LazyModule 懒加载模式导入模块
else:
    import sys

    # 将当前模块替换为 LazyModule 对象，用于按需导入模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```