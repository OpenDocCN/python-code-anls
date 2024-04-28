# `.\transformers\models\blenderbot\__init__.py`

```py
# 引入必要的类型检查模块
from typing import TYPE_CHECKING
# 引入自定义的异常和模块结构工具
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_blenderbot": [
        "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlenderbotConfig",
        "BlenderbotOnnxConfig",
    ],
    "tokenization_blenderbot": ["BlenderbotTokenizer"],
}

# 检查是否可用 tokenizers 库
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_blenderbot_fast"] = ["BlenderbotTokenizerFast"]

# 检查是否可用 torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_blenderbot"] = [
        "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BlenderbotForCausalLM",
        "BlenderbotForConditionalGeneration",
        "BlenderbotModel",
        "BlenderbotPreTrainedModel",
    ]

# 检查是否可用 tensorflow 库
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_blenderbot"] = [
        "TFBlenderbotForConditionalGeneration",
        "TFBlenderbotModel",
        "TFBlenderbotPreTrainedModel",
    ]

# 检查是否可用 flax 库
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_blenderbot"] = [
        "FlaxBlenderbotForConditionalGeneration",
        "FlaxBlenderbotModel",
        "FlaxBlenderbotPreTrainedModel",
    ]

# 如果是类型检查模式，则进行必要的导入
if TYPE_CHECKING:
    from .configuration_blenderbot import (
        BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlenderbotConfig,
        BlenderbotOnnxConfig,
    )
    from .tokenization_blenderbot import BlenderbotTokenizer

    # 检查是否可用 tokenizers 库
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_blenderbot_fast import BlenderbotTokenizerFast

    # 检查是否可用 torch 库
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是在顶层模块中，则从modeling_blenderbot模块中导入相关内容
    else:
        from .modeling_blenderbot import (
            BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotForCausalLM,
            BlenderbotForConditionalGeneration,
            BlenderbotModel,
            BlenderbotPreTrainedModel,
        )

    # 尝试检查是否TensorFlow可用，如果不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果TensorFlow可用，则从modeling_tf_blenderbot模块中导入相关内容
    else:
        from .modeling_tf_blenderbot import (
            TFBlenderbotForConditionalGeneration,
            TFBlenderbotModel,
            TFBlenderbotPreTrainedModel,
        )

    # 尝试检查是否Flax可用，如果不可用则引发OptionalDependencyNotAvailable异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果Flax可用，则从modeling_flax_blenderbot模块中导入相关内容
    else:
        from .modeling_flax_blenderbot import (
            FlaxBlenderbotForConditionalGeneration,
            FlaxBlenderbotModel,
            FlaxBlenderbotPreTrainedModel,
        )
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```