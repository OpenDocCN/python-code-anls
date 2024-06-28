# `.\models\blenderbot\__init__.py`

```
# 导入类型检查工具，用于检查类型是否存在
from typing import TYPE_CHECKING

# 导入依赖的模块和异常类
# _LazyModule: 惰性加载模块
# is_flax_available: 检查是否存在Flax库
# is_tf_available: 检查是否存在TensorFlow库
# is_tokenizers_available: 检查是否存在Tokenizers库
# is_torch_available: 检查是否存在PyTorch库
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构字典，列出各个模块的相关导入内容
_import_structure = {
    "configuration_blenderbot": [
        "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlenderbotConfig",
        "BlenderbotOnnxConfig",
    ],
    "tokenization_blenderbot": ["BlenderbotTokenizer"],
}

# 检查是否存在Tokenizers库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在Tokenizers库，则添加tokenization_blenderbot_fast模块到_import_structure
    _import_structure["tokenization_blenderbot_fast"] = ["BlenderbotTokenizerFast"]

# 检查是否存在PyTorch库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在PyTorch库，则添加modeling_blenderbot模块到_import_structure
    _import_structure["modeling_blenderbot"] = [
        "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BlenderbotForCausalLM",
        "BlenderbotForConditionalGeneration",
        "BlenderbotModel",
        "BlenderbotPreTrainedModel",
    ]

# 检查是否存在TensorFlow库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在TensorFlow库，则添加modeling_tf_blenderbot模块到_import_structure
    _import_structure["modeling_tf_blenderbot"] = [
        "TFBlenderbotForConditionalGeneration",
        "TFBlenderbotModel",
        "TFBlenderbotPreTrainedModel",
    ]

# 检查是否存在Flax库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在Flax库，则添加modeling_flax_blenderbot模块到_import_structure
    _import_structure["modeling_flax_blenderbot"] = [
        "FlaxBlenderbotForConditionalGeneration",
        "FlaxBlenderbotModel",
        "FlaxBlenderbotPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从configuration_blenderbot模块导入指定内容
    from .configuration_blenderbot import (
        BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlenderbotConfig,
        BlenderbotOnnxConfig,
    )
    # 从tokenization_blenderbot模块导入指定内容
    from .tokenization_blenderbot import BlenderbotTokenizer

    # 检查是否存在Tokenizers库，若不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在Tokenizers库，则从tokenization_blenderbot_fast模块导入指定内容
        from .tokenization_blenderbot_fast import BlenderbotTokenizerFast

    # 检查是否存在PyTorch库，若不存在则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件不成立，则导入以下 Blenderbot 模型相关的内容
    else:
        from .modeling_blenderbot import (
            BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotForCausalLM,
            BlenderbotForConditionalGeneration,
            BlenderbotModel,
            BlenderbotPreTrainedModel,
        )

    try:
        # 检查 TensorFlow 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，不做任何处理，继续执行后续代码
        pass
    else:
        # 如果 TensorFlow 可用，则导入以下 TensorFlow 版本的 Blenderbot 模型相关内容
        from .modeling_tf_blenderbot import (
            TFBlenderbotForConditionalGeneration,
            TFBlenderbotModel,
            TFBlenderbotPreTrainedModel,
        )

    try:
        # 检查 Flax 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 Flax 不可用，不做任何处理，继续执行后续代码
        pass
    else:
        # 如果 Flax 可用，则导入以下 Flax 版本的 Blenderbot 模型相关内容
        from .modeling_flax_blenderbot import (
            FlaxBlenderbotForConditionalGeneration,
            FlaxBlenderbotModel,
            FlaxBlenderbotPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于对当前模块进行动态修改
    import sys
    # 使用 sys.modules[__name__] 将当前模块的引用指向 _LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```