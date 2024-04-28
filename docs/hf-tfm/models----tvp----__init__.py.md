# `.\transformers\models\tvp\__init__.py`

```
# 定义代码的字符编码格式为 UTF-8
# 版权声明
# 导入所需的类型检查模块
from typing import TYPE_CHECKING
# 导入依赖检查和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义需要导入的结构
_import_structure = {
    "configuration_tvp": [
        "TVP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TvpConfig",
    ],
    "processing_tvp": ["TvpProcessor"],
}

# 如果视觉处理功能不可用，则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉处理功能可用，则导入图像处理模块
    _import_structure["image_processing_tvp"] = ["TvpImageProcessor"]

# 如果Torch框架不可用，则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果Torch框架可用，则导入模型处理模块
    _import_structure["modeling_tvp"] = [
        "TVP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TvpModel",
        "TvpPreTrainedModel",
        "TvpForVideoGrounding",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入所需的类型检查模块
    from .configuration_tvp import (
        TVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TvpConfig,
    )
    from .processing_tvp import TvpProcessor
    # 如果视觉处理功能可用，则导入视觉图像处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_tvp import TvpImageProcessor
    # 如果Torch框架可用，则导入模型处理模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tvp import (
            TVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            TvpForVideoGrounding,
            TvpModel,
            TvpPreTrainedModel,
        )
# 否则
else:
    import sys
    # 将当前模块存储为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```