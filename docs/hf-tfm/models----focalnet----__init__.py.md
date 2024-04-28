# `.\models\focalnet\__init__.py`

```
# 版权声明
# 版权声明，版权所有
# 根据 Apache 许可证 2.0 版本（“许可证”）授权
# 除非符合许可证，否则不得使用此文件
# 您可以从以下位置获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或经书面同意，否则不得分发软件
# 分发的软件是按“现状”基础分发的，没有任何明示或默示的担保或条件
# 请参阅许可证以获得关于特定语言的权限和限制
from typing import TYPE_CHECKING

# 依赖于 isort 合并导入
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 导入结构
_import_structure = {"configuration_focalnet": ["FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FocalNetConfig"]}

# 检查是否 torch 可用
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_focalnet"] = [
        "FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FocalNetForImageClassification",
        "FocalNetForMaskedImageModeling",
        "FocalNetBackbone",
        "FocalNetModel",
        "FocalNetPreTrainedModel",
    ]

# 如果类型检查开启
if TYPE_CHECKING:
    # 导入配置文件相关内容，条件依赖于 torch 是否可用
    from .configuration_focalnet import FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FocalNetConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关内容，条件依赖于 torch 是否可用
        from .modeling_focalnet import (
            FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            FocalNetBackbone,
            FocalNetForImageClassification,
            FocalNetForMaskedImageModeling,
            FocalNetModel,
            FocalNetPreTrainedModel,
        )

# 如果类型检查未开启
else:
    import sys

    # 设置模块指定的属性
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```