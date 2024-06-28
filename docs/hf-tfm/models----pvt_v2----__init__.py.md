# `.\models\pvt_v2\__init__.py`

```py
# coding=utf-8
# 版权所有 2023 作者：Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao 和 HuggingFace Inc. 团队。
# 保留所有权利。
#
# 根据Apache许可证2.0版许可
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何形式的明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构
_import_structure = {
    "configuration_pvt_v2": ["PVT_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtV2Config"],
}

try:
    # 如果torch不可用，引发OptionalDependencyNotAvailable异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果OptionalDependencyNotAvailable异常被引发，则不执行任何操作
    pass
else:
    # 如果torch可用，则添加以下模块到导入结构
    _import_structure["modeling_pvt_v2"] = [
        "PVT_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PvtV2ForImageClassification",
        "PvtV2Model",
        "PvtV2PreTrainedModel",
        "PvtV2Backbone",
    ]


if TYPE_CHECKING:
    # 如果当前在类型检查模式下
    from .configuration_pvt_v2 import PVT_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtV2Config

    try:
        # 如果torch不可用，引发OptionalDependencyNotAvailable异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果OptionalDependencyNotAvailable异常被引发，则不执行任何操作
        pass
    else:
        # 如果torch可用，则导入以下模块到当前命名空间
        from .modeling_pvt_v2 import (
            PVT_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            PvtV2Backbone,
            PvtV2ForImageClassification,
            PvtV2Model,
            PvtV2PreTrainedModel,
        )

else:
    # 如果不在类型检查模式下，则导入延迟模块_LazyModule
    import sys

    # 将当前模块注册为_LazyModule类型，使用指定的导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```