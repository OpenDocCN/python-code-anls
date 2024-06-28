# `.\models\patchtst\__init__.py`

```py
# 版权声明和版权信息，声明此代码的版权归 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可证规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 不附带任何明示或暗示的保证或条件。请参阅许可证了解具体的法律条款和限制。
from typing import TYPE_CHECKING

# 从模块中导入必要的异常和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_patchtst": [
        "PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PatchTSTConfig",
    ],
}

# 检查是否导入了 torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，将模型相关的内容添加到导入结构中
    _import_structure["modeling_patchtst"] = [
        "PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PatchTSTModel",
        "PatchTSTPreTrainedModel",
        "PatchTSTForPrediction",
        "PatchTSTForPretraining",
        "PatchTSTForRegression",
        "PatchTSTForClassification",
    ]

# 如果是类型检查阶段，导入相关类型检查需要的内容
if TYPE_CHECKING:
    from .configuration_patchtst import PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP, PatchTSTConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_patchtst import (
            PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST,
            PatchTSTForClassification,
            PatchTSTForPrediction,
            PatchTSTForPretraining,
            PatchTSTForRegression,
            PatchTSTModel,
            PatchTSTPreTrainedModel,
        )

# 如果不是类型检查阶段，则将当前模块注册为 LazyModule
else:
    import sys

    # 使用 LazyModule 将当前模块注册到 sys.modules 中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```