# `.\models\gptj\__init__.py`

```py
# 导入需要的模块和类型
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需的函数和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {"configuration_gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig", "GPTJOnnxConfig"]}

# 检查是否 Torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，添加 modeling_gptj 模块的导入结构
    _import_structure["modeling_gptj"] = [
        "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTJForCausalLM",
        "GPTJForQuestionAnswering",
        "GPTJForSequenceClassification",
        "GPTJModel",
        "GPTJPreTrainedModel",
    ]

# 检查是否 TensorFlow 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，添加 modeling_tf_gptj 模块的导入结构
    _import_structure["modeling_tf_gptj"] = [
        "TFGPTJForCausalLM",
        "TFGPTJForQuestionAnswering",
        "TFGPTJForSequenceClassification",
        "TFGPTJModel",
        "TFGPTJPreTrainedModel",
    ]

# 检查是否 Flax 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Flax 可用，添加 modeling_flax_gptj 模块的导入结构
    _import_structure["modeling_flax_gptj"] = [
        "FlaxGPTJForCausalLM",
        "FlaxGPTJModel",
        "FlaxGPTJPreTrainedModel",
    ]

# 如果是类型检查模式，则进一步导入相关模块和类型
if TYPE_CHECKING:
    from .configuration_gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig, GPTJOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 modeling_gptj 模块的类型
        from .modeling_gptj import (
            GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 modeling_tf_gptj 模块的类型
        from .modeling_tf_gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()

... (接下去的代码超出了最大长度限制，无法继续完整注释)
    # 捕获 OptionalDependencyNotAvailable 异常，如果发生则跳过执行下面的语句
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则导入以下模块
    else:
        # 从当前目录下的 modeling_flax_gptj 模块中导入 FlaxGPTJForCausalLM、FlaxGPTJModel、FlaxGPTJPreTrainedModel 类
        from .modeling_flax_gptj import FlaxGPTJForCausalLM, FlaxGPTJModel, FlaxGPTJPreTrainedModel
# 如果上述条件都不满足，则执行以下代码块
# 导入 sys 模块
import sys
# 将当前模块的导入从 sys.modules 中移除，并替换为 LazyModule 实例，其中包含了惰性导入所需的信息
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```