# `.\models\convnextv2\__init__.py`

```py
# flake8: noqa
# 禁用 flake8 对当前模块的检查，忽略未使用导入的警告

# 版权声明
# 本模块版权归 HuggingFace 团队所有

# 根据 Apache 许可 2.0，除非符合许可，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发
# 没有任何明示或暗示的担保或条件，查看特定语言规定的权限和限制

from typing import TYPE_CHECKING
# 引用 isort 来合并导入

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_tf_available,
)

# 导入结构
_import_structure = {
    "configuration_convnextv2": [
        "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConvNextV2Config",
    ]
}

try:
    # 如果 torch 不可用则抛出 OptionalDependencyNotAvailable 异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_convnextv2"] = [
        "CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvNextV2ForImageClassification",
        "ConvNextV2Model",
        "ConvNextV2PreTrainedModel",
        "ConvNextV2Backbone",
    ]

try:
    # 如果 tf 不可用则抛出 OptionalDependencyNotAvailable 异常
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_convnextv2"] = [
        "TFConvNextV2ForImageClassification",
        "TFConvNextV2Model",
        "TFConvNextV2PreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    from .configuration_convnextv2 import (
        CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConvNextV2Config,
    )
    # 如果 torch 可用，则导入以下模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_convnextv2 import (
            CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextV2Backbone,
            ConvNextV2ForImageClassification,
            ConvNextV2Model,
            ConvNextV2PreTrainedModel,
        )
    # 如果 tf 可用，则导入以下模块
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_convnextv2 import (
            TFConvNextV2ForImageClassification,
            TFConvNextV2Model,
            TFConvNextV2PreTrainedModel,
        )
# 如果不是类型检查环境，将本模块设置为懒加载模块
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```