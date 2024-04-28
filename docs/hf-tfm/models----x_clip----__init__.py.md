# `.\transformers\models\x_clip\__init__.py`

```
# 引入类型检查模块
from typing import TYPE_CHECKING
# 引入 OptionalDependencyNotAvailable 异常和 _LazyModule 工具类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 组织依赖结构的字典
_import_structure = {
    "configuration_x_clip": [
        "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XCLIPConfig",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "processing_x_clip": ["XCLIPProcessor"],
}

# 检查是否存在 torch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
# 若存在 torch 库，则添加 modeling_x_clip 模块到 _import_structure 字典
else:
    _import_structure["modeling_x_clip"] = [
        "XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XCLIPModel",
        "XCLIPPreTrainedModel",
        "XCLIPTextModel",
        "XCLIPVisionModel",
    ]

# 若为类型检查模式，则导入相应模块
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

# 若非类型检查模式，则使用 LazyModule 来延迟加载相应模块
else:
    import sys

    # 将 LazyModule 设置为当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```