# `.\models\deprecated\van\__init__.py`

```py
# 版权声明和许可证信息
# The HuggingFace Team 版权所有
# 根据 Apache 许可证 Version 2.0 发布
# 您只能在遵守许可证的情况下使用此文件
# 您可以从以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发
# 没有任何种类的明示或暗示的担保或条件
# 请参阅许可证以了解具体语言规定的权限和限制
from typing import TYPE_CHECKING

from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义需要导入的模块结构
_import_structure = {"configuration_van": ["VAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "VanConfig"]}

# 检查是否有 torch 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_van"] = [
        "VAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VanForImageClassification",
        "VanModel",
        "VanPreTrainedModel",
    ]

# 如果当前为类型检查阶段
if TYPE_CHECKING:
    # 导入模型配置相关信息
    from .configuration_van import VAN_PRETRAINED_CONFIG_ARCHIVE_MAP, VanConfig

    # 检查是否有 torch 库可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关信息
        from .modeling_van import (
            VAN_PRETRAINED_MODEL_ARCHIVE_LIST,
            VanForImageClassification,
            VanModel,
            VanPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块替换为懒加载模块的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```