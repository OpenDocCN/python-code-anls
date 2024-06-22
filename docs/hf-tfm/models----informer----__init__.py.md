# `.\models\informer\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入类型检查模块
from typing import TYPE_CHECKING

# 依赖于 isort 来合并导入
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_informer": [
        "INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "InformerConfig",
    ],
}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加以下模块到导入结构中
    _import_structure["modeling_informer"] = [
        "INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "InformerForPrediction",
        "InformerModel",
        "InformerPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置和模型相关模块
    from .configuration_informer import INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, InformerConfig

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关模块
        from .modeling_informer import (
            INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            InformerForPrediction,
            InformerModel,
            InformerPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule 类型，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```