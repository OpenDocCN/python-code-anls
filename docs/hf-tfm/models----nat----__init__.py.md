# `.\models\nat\__init__.py`

```py
# 版权声明及许可信息
# 2022 年版权归 HuggingFace 团队所有。保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证的详细信息，请参阅许可证。

# 引入 TYPE_CHECKING 类型检查工具
from typing import TYPE_CHECKING

# 引入 OptionalDependencyNotAvailable 异常和 _LazyModule、is_torch_available 工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义预期的导入结构
_import_structure = {"configuration_nat": ["NAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "NatConfig"]}

# 检查是否可用 Torch
try:
    if not is_torch_available():
        # 如果 Torch 不可用，引发 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到 OptionalDependencyNotAvailable 异常，则继续执行后续代码
    pass
else:
    # 如果 Torch 可用，则添加相关模块到导入结构中
    _import_structure["modeling_nat"] = [
        "NAT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NatForImageClassification",
        "NatModel",
        "NatPreTrainedModel",
        "NatBackbone",
    ]

# 如果 TYPE_CHECKING 为 True，则导入所需的配置和模型
if TYPE_CHECKING:
    from .configuration_nat import NAT_PRETRAINED_CONFIG_ARCHIVE_MAP, NatConfig

    try:
        if not is_torch_available():
            # 如果 Torch 不可用，引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果捕获到 OptionalDependencyNotAvailable 异常，则继续执行后续代码
        pass
    else:
        # 如果 Torch 可用，则从 modeling_nat 模块中导入所需的类
        from .modeling_nat import (
            NAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            NatBackbone,
            NatForImageClassification,
            NatModel,
            NatPreTrainedModel,
        )

# 如果 TYPE_CHECKING 为 False，则将当前模块设为 LazyModule，并指定导入结构
else:
    import sys

    # 使用 _LazyModule 将当前模块设为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```