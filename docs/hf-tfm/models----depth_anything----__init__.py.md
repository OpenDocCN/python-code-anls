# `.\models\depth_anything\__init__.py`

```
# 导入所需模块和函数
from typing import TYPE_CHECKING
# 从文件工具中导入懒加载模块和是否可用torch的函数
from ...file_utils import _LazyModule, is_torch_available
# 导入可选依赖未安装的异常
from ...utils import OptionalDependencyNotAvailable

# 定义模块导入结构字典
_import_structure = {
    "configuration_depth_anything": ["DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP", "DepthAnythingConfig"]
}

# 检查torch是否可用，若不可用则抛出可选依赖未安装的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则添加模型相关的导入结构到_import_structure中
    _import_structure["modeling_depth_anything"] = [
        "DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DepthAnythingForDepthEstimation",
        "DepthAnythingPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从配置模块中导入所需的符号
    from .configuration_depth_anything import DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP, DepthAnythingConfig

    # 再次检查torch是否可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若torch可用，则从模型模块中导入所需的符号
        from .modeling_depth_anything import (
            DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST,
            DepthAnythingForDepthEstimation,
            DepthAnythingPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    # 导入sys模块
    import sys

    # 将当前模块设为懒加载模块的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```