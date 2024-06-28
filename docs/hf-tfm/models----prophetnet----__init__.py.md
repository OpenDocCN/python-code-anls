# `.\models\prophetnet\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从工具包中导入自定义异常和模块延迟加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括各个子模块及其导入的类和变量
_import_structure = {
    "configuration_prophetnet": ["PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ProphetNetConfig"],
    "tokenization_prophetnet": ["ProphetNetTokenizer"],
}

# 检查是否可以导入 torch，若不可用则抛出自定义的依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，添加 modeling_prophetnet 子模块及其导入的类和变量
    _import_structure["modeling_prophetnet"] = [
        "PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ProphetNetDecoder",
        "ProphetNetEncoder",
        "ProphetNetForCausalLM",
        "ProphetNetForConditionalGeneration",
        "ProphetNetModel",
        "ProphetNetPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从子模块中导入特定的类和变量，用于类型检查
    from .configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
    from .tokenization_prophetnet import ProphetNetTokenizer

    # 再次检查是否可以导入 torch，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 modeling_prophetnet 子模块中导入特定的类和变量，用于类型检查
        from .modeling_prophetnet import (
            PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            ProphetNetDecoder,
            ProphetNetEncoder,
            ProphetNetForCausalLM,
            ProphetNetForConditionalGeneration,
            ProphetNetModel,
            ProphetNetPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    import sys

    # 将当前模块替换为 LazyModule 对象，用于延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```