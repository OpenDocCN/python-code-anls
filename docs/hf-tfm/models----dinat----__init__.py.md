# `.\models\dinat\__init__.py`

```py
# 版权声明
# 版权所有2022年HuggingFace团队
#
# 根据Apache许可证2.0版本（"许可证"）许可；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，软件是基于"按原样"分发的，
# 没有任何形式的明示或暗示的担保或条件。
# 查看许可证以获取特定语言规定的权利和限制
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义要导入的模块结构
_import_structure = {"configuration_dinat": ["DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DinatConfig"]}

# 如果torch不可用，则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，添加另一个模块到导入结构
    _import_structure["modeling_dinat"] = [
        "DINAT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DinatForImageClassification",
        "DinatModel",
        "DinatPreTrainedModel",
        "DinatBackbone",
    ]

# 如果是类型检查，从相应模块导入所需内容
if TYPE_CHECKING:
    from .configuration_dinat import DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP, DinatConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dinat import (
            DINAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DinatBackbone,
            DinatForImageClassification,
            DinatModel,
            DinatPreTrainedModel,
        )

# 如果不是类型检查，将懒加载模块注册到sys.modules中
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```