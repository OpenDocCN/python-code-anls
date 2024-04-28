# `.\transformers\models\mistral\__init__.py`

```py
# 版权声明
# 版权归 Mistral AI 和 The HuggingFace Inc. 团队所有，保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证的副本:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，软件分发将基于"按原样"的基础，
# 没有任何形式的明示或暗示的担保或条件。
# 请查阅许可证以了解特定语言下的权限和限制。
from typing import TYPE_CHECKING

# 从 utils 中导入必要的模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_mistral": ["MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MistralConfig"],
}

# 尝试导入 torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入成功，则添加到导入结构中
    _import_structure["modeling_mistral"] = [
        "MistralForCausalLM",
        "MistralModel",
        "MistralPreTrainedModel",
        "MistralForSequenceClassification",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从 configuration_mistral 中导入指定的类
    from .configuration_mistral import MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MistralConfig

    # 类型检查阶段尝试导入 torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 在类型检查阶段，从 modeling_mistral 中导入指定的类
        from .modeling_mistral import (
            MistralForCausalLM,
            MistralForSequenceClassification,
            MistralModel,
            MistralPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块的命名空间指向 LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```