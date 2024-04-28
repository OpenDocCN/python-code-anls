# `.\transformers\models\switch_transformers\__init__.py`

```py
# 版权声明和许可信息
# 此处是关于HuggingFace团队的版权声明和许可信息，告知使用者在遵守许可协议的情况下才能使用此文件
#
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入所需的工具函数和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_switch_transformers": [
        "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SwitchTransformersConfig",
        "SwitchTransformersOnnxConfig",
    ]
}

# 检查torch包是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则添加modeling_switch_transformers模块到导入结构中
    _import_structure["modeling_switch_transformers"] = [
        "SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwitchTransformersEncoderModel",
        "SwitchTransformersForConditionalGeneration",
        "SwitchTransformersModel",
        "SwitchTransformersPreTrainedModel",
        "SwitchTransformersTop1Router",
        "SwitchTransformersSparseMLP",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置类和模型类
    from .configuration_switch_transformers import (
        SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwitchTransformersConfig,
        SwitchTransformersOnnxConfig,
    )

    # 检查torch包是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型类
        from .modeling_switch_transformers import (
            SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块设置为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```