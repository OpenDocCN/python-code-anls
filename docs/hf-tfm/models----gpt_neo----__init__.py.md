# `.\models\gpt_neo\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 导入自定义异常类，用于指示可选依赖项不可用
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available

# 定义导入结构，包括配置和模型的名称列表
_import_structure = {
    "configuration_gpt_neo": ["GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoConfig", "GPTNeoOnnxConfig"],
}

# 检查是否有 torch 可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则添加 GPT Neo 模型相关的名称列表到导入结构中
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

# 检查是否有 flax 可用，若不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则添加 Flax GPT Neo 模型相关的名称列表到导入结构中
    _import_structure["modeling_flax_gpt_neo"] = [
        "FlaxGPTNeoForCausalLM",
        "FlaxGPTNeoModel",
        "FlaxGPTNeoPreTrainedModel",
    ]

# 如果类型检查被启用，导入具体的配置和模型类
if TYPE_CHECKING:
    from .configuration_gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig, GPTNeoOnnxConfig

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

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_gpt_neo import FlaxGPTNeoForCausalLM, FlaxGPTNeoModel, FlaxGPTNeoPreTrainedModel

# 如果类型检查未启用，则将当前模块设置为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```