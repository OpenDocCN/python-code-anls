# `.\models\pegasus\__init__.py`

```
# 导入必要的模块和函数，包括类型检查相关内容
from typing import TYPE_CHECKING

# 从特定路径导入必要的工具和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典，描述导入的结构
_import_structure = {"configuration_pegasus": ["PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusConfig"]}

# 尝试导入 PegasusTokenizer，如果 sentencepiece 不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_pegasus"] = ["PegasusTokenizer"]

# 尝试导入 PegasusTokenizerFast，如果 tokenizers 不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_pegasus_fast"] = ["PegasusTokenizerFast"]

# 尝试导入 Pegasus 相关的模型，如果 torch 不可用则引发异常
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

# 尝试导入 TF Pegasus 相关的模型，如果 TensorFlow 不可用则引发异常
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

# 尝试导入 Flax Pegasus 相关的模型，如果 Flax 不可用则引发异常
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

# 如果是类型检查阶段，则进一步导入相应的配置和工具
if TYPE_CHECKING:
    from .configuration_pegasus import PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusConfig

    # 在类型检查阶段，如果 sentencepiece 可用，则导入 PegasusTokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_pegasus import PegasusTokenizer

    # 在类型检查阶段，如果 tokenizers 可用，则导入 PegasusTokenizerFast
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_pegasus_fast import PegasusTokenizerFast
    try:
        # 检查是否安装了 PyTorch 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，不做任何处理，继续执行下面的代码
        pass
    else:
        # 如果未引发异常，则从相应模块导入必要的类和变量
        from .modeling_pegasus import (
            PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST,
            PegasusForCausalLM,
            PegasusForConditionalGeneration,
            PegasusModel,
            PegasusPreTrainedModel,
        )

    try:
        # 检查是否安装了 TensorFlow 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，不做任何处理，继续执行下面的代码
        pass
    else:
        # 如果未引发异常，则从相应模块导入必要的类和变量
        from .modeling_tf_pegasus import TFPegasusForConditionalGeneration, TFPegasusModel, TFPegasusPreTrainedModel

    try:
        # 检查是否安装了 Flax 库，如果未安装则引发 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，不做任何处理，继续执行下面的代码
        pass
    else:
        # 如果未引发异常，则从相应模块导入必要的类和变量
        from .modeling_flax_pegasus import (
            FlaxPegasusForConditionalGeneration,
            FlaxPegasusModel,
            FlaxPegasusPreTrainedModel,
        )
else:
    # 如果不是以上情况，则执行以下代码
    import sys
    # 导入系统模块 sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 封装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```