# `.\models\graphormer\__init__.py`

```
# 版权声明及许可信息
# 版权所有 2020 年 HuggingFace 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以获取许可证的副本，详见
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”分发，
# 没有任何明示或暗示的担保或条件。
# 有关具体语言版本下的权限，请参阅许可证。
from typing import TYPE_CHECKING

# 从...utils 中导入 OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_graphormer": ["GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "GraphormerConfig"],
}

# 检查是否支持 torch 库，若不支持则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 torch，则添加 modeling_graphormer 模块的导入结构
    _import_structure["modeling_graphormer"] = [
        "GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GraphormerForGraphClassification",
        "GraphormerModel",
        "GraphormerPreTrainedModel",
    ]

# 如果是类型检查模式，则导入 GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP 和 GraphormerConfig 类
if TYPE_CHECKING:
    from .configuration_graphormer import GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, GraphormerConfig

    # 检查是否支持 torch 库，若不支持则忽略异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果支持 torch，则导入 modeling_graphormer 模块的相关类和变量
        from .modeling_graphormer import (
            GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            GraphormerForGraphClassification,
            GraphormerModel,
            GraphormerPreTrainedModel,
        )

# 如果不是类型检查模式，则将当前模块定义为延迟加载模块
else:
    import sys

    # 将当前模块替换为 _LazyModule 实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```