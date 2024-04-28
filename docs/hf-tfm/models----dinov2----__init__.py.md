# `.\models\dinov2\__init__.py`

```
# 2023年 HuggingFace 团队版权所有
#
# 根据 Apache 许可证 2.0 版授权使用此文件；
# 您除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的拷贝
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得基于许可证分发软件
# 按"原样"基础分发，没有任何担保或条款，无论是明示的还是暗示的。
# 查看许可证以了解特定语言下的权限和限制
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义_import_structure字典用于记录模块间依赖关系
_import_structure = {
    "configuration_dinov2": ["DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Dinov2Config", "Dinov2OnnxConfig"]
}

# 检查是否存在 torch 库，如果不存在则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则更新_import_structure字典，记录额外的模块间依赖关系
    _import_structure["modeling_dinov2"] = [
        "DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Dinov2ForImageClassification",
        "Dinov2Model",
        "Dinov2PreTrainedModel",
        "Dinov2Backbone",
    ]

if TYPE_CHECKING:
    # 如果是类型检查阶段，则导入相关模块
    from .configuration_dinov2 import DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Dinov2Config, Dinov2OnnxConfig
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch 库，则导入相关模块
        from .modeling_dinov2 import (
            DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Dinov2Backbone,
            Dinov2ForImageClassification,
            Dinov2Model,
            Dinov2PreTrainedModel,
        )

else:
    import sys
    # 如果不是类型检查阶段，则将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```