# `.\models\gpt_neo\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_gpt_neo": ["GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoConfig", "GPTNeoOnnxConfig"],
}

# 检查是否存在torch库，如果不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型GPT Neo所需的模型文件和函数到导入结构中
    _import_structure["modeling_gpt_neo"] = [
        "GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTNeoForCausalLM",
        "GPTNeoForQuestionAnswering",
        "GPTNeoForSequenceClassification",
        "GPTNeoForTokenClassification",
        "GPTNeoModel",
        "GPTNeoPreTrainedModel",
        "load_tf_weights_in_gpt_neo",
    ]

# 检查是否存在flax库，如果不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加Flax模型GPT Neo所需的模型文件和函数到导入结构中
    _import_structure["modeling_flax_gpt_neo"] = [
        "FlaxGPTNeoForCausalLM",
        "FlaxGPTNeoModel",
        "FlaxGPTNeoPreTrainedModel",
    ]

# 如果是类型检查模式，则在此处导入相应的模块
if TYPE_CHECKING:
    from .configuration_gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig, GPTNeoOnnxConfig

    # 导入torch模型GPT Neo所需的模型文件和函数
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gpt_neo import (
            GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoForCausalLM,
            GPTNeoForQuestionAnswering,
            GPTNeoForSequenceClassification,
            GPTNeoForTokenClassification,
            GPTNeoModel,
            GPTNeoPreTrainedModel,
            load_tf_weights_in_gpt_neo,
        )

    # 导入flax模型GPT Neo所需的模型文件和函数
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_gpt_neo import FlaxGPTNeoForCausalLM, FlaxGPTNeoModel, FlaxGPTNeoPreTrainedModel

# 如果不处于类型检查模式，则将LazyModule作为当前模块的代理
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```