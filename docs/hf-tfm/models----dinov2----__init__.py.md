# `.\models\dinov2\__init__.py`

```
# 版权声明及许可信息
#
# 版权所有 2023 年 HuggingFace 团队。保留所有权利。
# 
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 中导入必要的模块和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义导入结构，指定 Dinov2 模块的组织结构
_import_structure = {
    "configuration_dinov2": ["DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Dinov2Config", "Dinov2OnnxConfig"]
}

# 检查是否存在 Torch，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则导入 modeling_dinov2 模块相关内容
    _import_structure["modeling_dinov2"] = [
        "DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Dinov2ForImageClassification",
        "Dinov2Model",
        "Dinov2PreTrainedModel",
        "Dinov2Backbone",
    ]

# 如果是类型检查阶段，导入具体的配置和模型类
if TYPE_CHECKING:
    from .configuration_dinov2 import DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Dinov2Config, Dinov2OnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dinov2 import (
            DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Dinov2Backbone,
            Dinov2ForImageClassification,
            Dinov2Model,
            Dinov2PreTrainedModel,
        )

# 如果不是类型检查阶段，则将当前模块替换为一个懒加载模块，根据 _import_structure 中的定义懒加载相关模块
else:
    import sys

    # 使用 _LazyModule 创建一个懒加载模块，替换当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```