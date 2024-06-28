# `.\models\sew_d\__init__.py`

```
# 版权声明及许可声明，指明版权归 The HuggingFace Team 所有，依照 Apache License, Version 2.0 许可
#
# 在遵循许可的前提下，你可以使用本文件。你可以从以下链接获取许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件根据“原样”分发，无任何明示或暗示的担保或条件。
# 请查阅许可文件以了解详细信息。
from typing import TYPE_CHECKING

# 从相对路径引入工具函数和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构字典
_import_structure = {"configuration_sew_d": ["SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP", "SEWDConfig"]}

# 检查是否有 torch 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，将下列模块添加到导入结构中
    _import_structure["modeling_sew_d"] = [
        "SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SEWDForCTC",
        "SEWDForSequenceClassification",
        "SEWDModel",
        "SEWDPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从配置模块中导入指定符号
    from .configuration_sew_d import SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWDConfig

    # 再次检查 torch 是否可用，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 torch 可用，则从模型模块中导入指定符号
        from .modeling_sew_d import (
            SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST,
            SEWDForCTC,
            SEWDForSequenceClassification,
            SEWDModel,
            SEWDPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为 LazyModule，延迟加载所需的子模块和符号
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```