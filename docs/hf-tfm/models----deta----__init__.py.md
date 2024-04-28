# `.\models\deta\__init__.py`

```py
# 版权声明
#
# 该程序代码的版权属于 2022 年的 HuggingFace 团队，保留所有权利。
# 
# 根据 Apache 2.0 许可证授权
# 你可以遵循许可证使用该文件。你只能在遵守许可证的情况下使用该文件。
# 你可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律要求或书面同意，软件在 “按原样” 基础上分发，
# 没有任何类型的担保或条件，无论是明示的还是隐含的。
# 有关限制和指定语言的权限，请查看许可证。

# 导入类型检查的模块
from typing import TYPE_CHECKING

# 导入相应的实用工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 模块的导入结构
_import_structure = {
    "configuration_deta": ["DETA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetaConfig"],
}

# 如果视觉模块可用，导入图像处理模块
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_deta"] = ["DetaImageProcessor"]

# 如果 Torch 模块可用，导入 Deta 模型模块
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deta"] = [
        "DETA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DetaForObjectDetection",
        "DetaModel",
        "DetaPreTrainedModel",
    ]

# 如果是类型检查模式，导入相应的配置和模型模块
if TYPE_CHECKING:
    from .configuration_deta import DETA_PRETRAINED_CONFIG_ARCHIVE_MAP, DetaConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_deta import DetaImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deta import (
            DETA_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetaForObjectDetection,
            DetaModel,
            DetaPreTrainedModel,
        )

# 如果不是类型检查模式，将 LazyModule 添加到 sys.modules
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```