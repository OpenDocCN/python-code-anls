# `.\models\graphormer\__init__.py`

```
# 版权声明和许可证信息
# 版权声明和许可证信息，指定了代码的版权和许可证信息
# 根据 Apache 许可证版本 2.0 许可使用此文件
# 获取许可证的副本
# 如果不符合许可证要求，则不得使用此文件
# 根据适用法律或书面协议，分发的软件基于“原样”基础分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_graphormer": ["GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "GraphormerConfig"],
}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则添加以下模块到导入结构中
    _import_structure["modeling_graphormer"] = [
        "GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GraphormerForGraphClassification",
        "GraphormerModel",
        "GraphormerPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置和模型相关的内容
    from .configuration_graphormer import GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, GraphormerConfig

    # 检查是否存在 torch 库，如果不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch 库，则导入模型相关的内容
        from .modeling_graphormer import (
            GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            GraphormerForGraphClassification,
            GraphormerModel,
            GraphormerPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为 LazyModule，延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```