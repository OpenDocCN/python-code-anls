# `.\models\seggpt\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从项目内的 utils 模块导入自定义异常 OptionalDependencyNotAvailable 和 _LazyModule
from ...utils import OptionalDependencyNotAvailable, _LazyModule
# 从 utils 模块导入检查函数 is_torch_available 和 is_vision_available
from ...utils import is_torch_available, is_vision_available

# 定义一个字典 _import_structure，用于存储模块导入结构
_import_structure = {
    "configuration_seggpt": ["SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "SegGptConfig", "SegGptOnnxConfig"]
}

# 检查是否可以导入 torch，如果不可以则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 torch，则将相关模块添加到 _import_structure 中
    _import_structure["modeling_seggpt"] = [
        "SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SegGptModel",
        "SegGptPreTrainedModel",
        "SegGptForImageSegmentation",
    ]

# 检查是否可以导入 vision 相关模块，如果不可以则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 vision 相关模块，则将其添加到 _import_structure 中
    _import_structure["image_processing_seggpt"] = ["SegGptImageProcessor"]

# 如果当前处于类型检查模式（TYPE_CHECKING 为 True）
if TYPE_CHECKING:
    # 从本地模块中导入特定的类和常量
    from .configuration_seggpt import SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, SegGptConfig, SegGptOnnxConfig

    try:
        # 再次检查是否可以导入 torch，如果不可以则忽略异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可以导入 torch，则从本地模块中导入相关模块
        from .modeling_seggpt import (
            SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            SegGptForImageSegmentation,
            SegGptModel,
            SegGptPreTrainedModel,
        )

    try:
        # 再次检查是否可以导入 vision 相关模块，如果不可以则忽略异常
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可以导入 vision 相关模块，则从本地模块中导入相关模块
        from .image_processing_seggpt import SegGptImageProcessor

# 如果不处于类型检查模式，则执行以下代码块
else:
    import sys

    # 将当前模块替换为懒加载模块 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```