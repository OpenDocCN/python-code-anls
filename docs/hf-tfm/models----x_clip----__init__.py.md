# `.\models\x_clip\__init__.py`

```py
# 导入类型检查模块，用于检查类型是否符合预期
from typing import TYPE_CHECKING

# 导入自定义的异常和模块惰性加载相关的工具函数和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包含配置、处理和建模相关的内容
_import_structure = {
    "configuration_x_clip": [
        "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XCLIPConfig",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "processing_x_clip": ["XCLIPProcessor"],
}

# 检查是否有torch可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，增加模型相关的导入内容到_import_structure
    _import_structure["modeling_x_clip"] = [
        "XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XCLIPModel",
        "XCLIPPreTrainedModel",
        "XCLIPTextModel",
        "XCLIPVisionModel",
    ]

# 如果是类型检查模式，导入具体的配置和建模内容
if TYPE_CHECKING:
    from .configuration_x_clip import (
        XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XCLIPConfig,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    from .processing_x_clip import XCLIPProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_x_clip import (
            XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            XCLIPModel,
            XCLIPPreTrainedModel,
            XCLIPTextModel,
            XCLIPVisionModel,
        )

# 如果不是类型检查模式，将当前模块置为懒加载模块
else:
    import sys

    # 将当前模块替换为懒加载模块_LazyModule，以延迟加载依赖模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```