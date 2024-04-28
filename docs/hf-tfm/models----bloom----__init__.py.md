# `.\transformers\models\bloom\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入可选依赖相关模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_bloom": ["BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP", "BloomConfig", "BloomOnnxConfig"],
}
# 检查 tokenizers 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 BloomTokenizerFast 添加到导入结构中
    _import_structure["tokenization_bloom_fast"] = ["BloomTokenizerFast"]

# 检查 torch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 modeling_bloom 中的模块添加到导入结构中
    _import_structure["modeling_bloom"] = [
        "BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BloomForCausalLM",
        "BloomModel",
        "BloomPreTrainedModel",
        "BloomForSequenceClassification",
        "BloomForTokenClassification",
        "BloomForQuestionAnswering",
    ]

# 检查 flax 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 modeling_flax_bloom 中的模块添加到导入结构中
    _import_structure["modeling_flax_bloom"] = [
        "FlaxBloomForCausalLM",
        "FlaxBloomModel",
        "FlaxBloomPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入相关模块
    from .configuration_bloom import BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP, BloomConfig, BloomOnnxConfig

    try:
        # 检查 tokenizers 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 tokenization_bloom_fast 中的模块
        from .tokenization_bloom_fast import BloomTokenizerFast

    try:
        # 检查 torch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 modeling_bloom 中的模块
        from .modeling_bloom import (
            BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST,
            BloomForCausalLM,
            BloomForQuestionAnswering,
            BloomForSequenceClassification,
            BloomForTokenClassification,
            BloomModel,
            BloomPreTrainedModel,
        )

    try:
        # 检查 flax 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 modeling_flax_bloom 中的模块
        from .modeling_flax_bloom import FlaxBloomForCausalLM, FlaxBloomModel, FlaxBloomPreTrainedModel
# 如果不是类型检查阶段
else:
    # 直接导入 sys 模块
    import sys
    # 将当前模块注册到sys.modules字典中，使用_LazyModule延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```