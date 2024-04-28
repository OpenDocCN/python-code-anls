# `.\transformers\models\opt\__init__.py`

```
# 版权声明及许可信息
# 版权声明告知此代码的版权信息及受限制的使用方式
# 根据 Apache 许可证版本 2.0 进行许可
# 你不得在没有遵守许可证的情况下使用此文件
# 你可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则分发的软件应基于"原样"分发，
# 没有任何明示或暗示的担保或条件
# 请查阅许可证以了解特定语言的权限和限制

# 导入需要的模块和类型
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {"configuration_opt": ["OPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OPTConfig"]}

# 检查是否存在 torch，若不存在则报错
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_opt"] = [
        "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OPTForCausalLM",
        "OPTModel",
        "OPTPreTrainedModel",
        "OPTForSequenceClassification",
        "OPTForQuestionAnswering",
    ]

# 检查是否存在 tf，若不存在则报错
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_opt"] = ["TFOPTForCausalLM", "TFOPTModel", "TFOPTPreTrainedModel"]

# 检查是否存在 flax，若不存在则报错
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_opt"] = [
        "FlaxOPTForCausalLM",
        "FlaxOPTModel",
        "FlaxOPTPreTrainedModel",
    ]

# 若处于类型检查阶段
if TYPE_CHECKING:
    # 导入类型检查所需要的模块
    from .configuration_opt import OPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OPTConfig
    # 检查是否存在 torch，若不存在则报错
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关类型检查所需的模块
        from .modeling_opt import (
            OPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OPTForCausalLM,
            OPTForQuestionAnswering,
            OPTForSequenceClassification,
            OPTModel,
            OPTPreTrainedModel,
        )
    # 检查是否存在 tf，若不存在则报错
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关类型检查所需的模块
        from .modeling_tf_opt import TFOPTForCausalLM, TFOPTModel, TFOPTPreTrainedModel
    # 检查是否存在 flax，若不存在则报错
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关类型检查所需的模块
        from .modeling_flax_opt import FlaxOPTForCausalLM, FlaxOPTModel, FlaxOPTPreTrainedModel

# 若不处于类型检查阶段
else:
    import sys
    # 创建懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```