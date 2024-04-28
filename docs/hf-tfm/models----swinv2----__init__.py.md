# `.\transformers\models\swinv2\__init__.py`

```
# 版权声明
# 2022年版权归 HuggingFace 团队所有。
# 根据 Apache 许可证 2.0 版本授权；
# 您不得使用本文件，除非符合许可证的规定。
# 您可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”基础分发软件，
# 没有任何明示或暗示的担保或条件。
# 请参阅许可证，了解具体的语言规定和限制。

# 导入必要的依赖
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义 import_structure
_import_structure = {
    "configuration_swinv2": ["SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swinv2Config"],
}

# 尝试导入 torch，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_swinv2 到 import_structure
    _import_structure["modeling_swinv2"] = [
        "SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Swinv2ForImageClassification",
        "Swinv2ForMaskedImageModeling",
        "Swinv2Model",
        "Swinv2PreTrainedModel",
        "Swinv2Backbone",
    ]

# 如果是类型检查模式，则导入特定模块
if TYPE_CHECKING:
    from .configuration_swinv2 import SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Swinv2Config

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_swinv2 import (
            SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swinv2Backbone,
            Swinv2ForImageClassification,
            Swinv2ForMaskedImageModeling,
            Swinv2Model,
            Swinv2PreTrainedModel,
        )

# 如果不是类型检查模式，则创建懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```