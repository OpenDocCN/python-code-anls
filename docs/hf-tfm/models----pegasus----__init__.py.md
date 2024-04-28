# `.\transformers\models\pegasus\__init__.py`

```py
# 版权声明，版权属于 The HuggingFace Team，并受 Apache 2.0 许可证保护
# 本文件仅可在遵守许可证的前提下使用
from typing import TYPE_CHECKING

# 导入相关模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块结构，包含配置和模型的导入结构
_import_structure = {"configuration_pegasus": ["PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusConfig"]}

# 检查是否存在 sentencepiece 库，未安装则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_pegasus"] = ["PegasusTokenizer"]

# 检查是否存在 tokenizers 库，未安装则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_pegasus_fast"] = ["PegasusTokenizerFast"]

# 检查是否存在 torch 库，未安装则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_pegasus"] = [
        "PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PegasusForCausalLM",
        "PegasusForConditionalGeneration",
        "PegasusModel",
        "PegasusPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，未安装则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_pegasus"] = [
        "TFPegasusForConditionalGeneration",
        "TFPegasusModel",
        "TFPegasusPreTrainedModel",
    ]

# 检查是否存在 flax 库，未安装则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_pegasus"] = [
        "FlaxPegasusForConditionalGeneration",
        "FlaxPegasusModel",
        "FlaxPegasusPreTrainedModel",
    ]

# 如果是类型检查状态，引入相关配置和模块
if TYPE_CHECKING:
    from .configuration_pegasus import PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusConfig

    # 检查是否存在 sentencepiece 库，未安装则抛出异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_pegasus import PegasusTokenizer

    # 检查是否存在 tokenizers 库，未安装则抛出异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_pegasus_fast import PegasusTokenizerFast
    try:
        # 检查是否有可选依赖torch，若无则抛出异常OptionalDependencyNotAvailable
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若没有torch的可选依赖，则直接跳过此部分代码
        pass
    else:
        # 导入相关模块和类
        from .modeling_pegasus import (
            PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST,
            PegasusForCausalLM,
            PegasusForConditionalGeneration,
            PegasusModel,
            PegasusPreTrainedModel,
        )

    try:
        # 检查是否有可选依赖tensorflow，若无则抛出异常OptionalDependencyNotAvailable
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若没有tensorflow的可选依赖，则直接跳过此部分代码
        pass
    else:
        # 导入相关模块和类
        from .modeling_tf_pegasus import TFPegasusForConditionalGeneration, TFPegasusModel, TFPegasusPreTrainedModel

    try:
        # 检查是否有可选依赖flax，若无则抛出异常OptionalDependencyNotAvailable
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 若没有flax的可选依赖，则直接跳过此部分代码
        pass
    else:
        # 导入相关模块和类
        from .modeling_flax_pegasus import (
            FlaxPegasusForConditionalGeneration,
            FlaxPegasusModel,
            FlaxPegasusPreTrainedModel,
        )
# 如果不在主模块中，则导入sys模块
else:
    # 导入sys模块
    import sys
    # 将当前模块的属性添加到sys模块的modules字典中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```