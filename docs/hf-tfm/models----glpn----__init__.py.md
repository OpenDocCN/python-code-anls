# `.\models\glpn\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义 _import_structure 字典，包含配置和模型的导入结构
_import_structure = {"configuration_glpn": ["GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP", "GLPNConfig"]}

# 检查视觉模块是否可用，若不可用则产生 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_glpn"] = ["GLPNFeatureExtractor"]  # 添加特征提取模块
    _import_structure["image_processing_glpn"] = ["GLPNImageProcessor"]  # 添加图像处理模块

# 检查 torch 模块是否可用，若不可用则产生 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_glpn"] = [  # 添加 GLPN 模型相关组件
        "GLPN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GLPNForDepthEstimation",
        "GLPNLayer",
        "GLPNModel",
        "GLPNPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从配置模块导入相关内容
    from .configuration_glpn import GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP, GLPNConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从特征提取模块和图像处理模块导入相关内容
        from .feature_extraction_glpn import GLPNFeatureExtractor
        from .image_processing_glpn import GLPNImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型模块导入相关内容
        from .modeling_glpn import (
            GLPN_PRETRAINED_MODEL_ARCHIVE_LIST,
            GLPNForDepthEstimation,
            GLPNLayer,
            GLPNModel,
            GLPNPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    import sys

    # 改写当前模块为 LazyModule 类，用于惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```