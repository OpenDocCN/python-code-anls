# `.\transformers\models\mixtral\__init__.py`

```py
# 版权声明
# 版权所有©2023年Mixtral AI和The HuggingFace Inc.团队。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的条款，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得基于"AS IS"的基础上分发软件，
# 没有任何种类的明示或默示的担保或条件。
# 请参阅许可证以获取特定语言的权限和限制。
# 类型检查
from typing import TYPE_CHECKING

# 从utils模块中导入相关内容
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 指定导入结构
_import_structure = {
    "configuration_mixtral": ["MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MixtralConfig"],
}

# 捕获可能发生的依赖不可用的异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果依赖可用，增加导入结构中的模型相关内容
    _import_structure["modeling_mixtral"] = [
        "MixtralForCausalLM",
        "MixtralModel",
        "MixtralPreTrainedModel",
        "MixtralForSequenceClassification",
    ]

# 如果类型是检查的，则根据类型导入相关内容
if TYPE_CHECKING:
    from .configuration_mixtral import MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MixtralConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mixtral import (
            MixtralForCausalLM,
            MixtralForSequenceClassification,
            MixtralModel,
            MixtralPreTrainedModel,
        )

# 如果类型不是检查的，则将其设置为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```