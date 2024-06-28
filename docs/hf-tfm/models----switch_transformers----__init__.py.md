# `.\models\switch_transformers\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从 utils 模块导入所需的异常和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和模型相关内容
_import_structure = {
    "configuration_switch_transformers": [
        "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SwitchTransformersConfig",
        "SwitchTransformersOnnxConfig",
    ]
}

# 检查是否 Torch 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则导入模型相关内容到模块导入结构中
    _import_structure["modeling_switch_transformers"] = [
        "SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwitchTransformersEncoderModel",
        "SwitchTransformersForConditionalGeneration",
        "SwitchTransformersModel",
        "SwitchTransformersPreTrainedModel",
        "SwitchTransformersTop1Router",
        "SwitchTransformersSparseMLP",
    ]

# 如果是类型检查阶段，则从相应的模块导入配置和模型内容
if TYPE_CHECKING:
    from .configuration_switch_transformers import (
        SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwitchTransformersConfig,
        SwitchTransformersOnnxConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_switch_transformers import (
            SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )

# 如果不是类型检查阶段，则将当前模块设为延迟加载模块
else:
    import sys

    # 使用 _LazyModule 将当前模块设为延迟加载模块，确保按需导入内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```