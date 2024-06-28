# `.\models\biogpt\__init__.py`

```py
# 引入必要的模块和函数来处理依赖性检查和延迟加载
from typing import TYPE_CHECKING

# 从工具包中导入自定义异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_biogpt": ["BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BioGptConfig"],
    "tokenization_biogpt": ["BioGptTokenizer"],
}

# 检查是否有torch可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则添加相关的模型定义到导入结构中
    _import_structure["modeling_biogpt"] = [
        "BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BioGptForCausalLM",
        "BioGptForTokenClassification",
        "BioGptForSequenceClassification",
        "BioGptModel",
        "BioGptPreTrainedModel",
    ]

# 如果是类型检查模式，则从相关模块导入类型信息
if TYPE_CHECKING:
    from .configuration_biogpt import BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BioGptConfig
    from .tokenization_biogpt import BioGptTokenizer

    # 在torch可用的情况下，导入模型相关的类型信息
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_biogpt import (
            BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BioGptForCausalLM,
            BioGptForSequenceClassification,
            BioGptForTokenClassification,
            BioGptModel,
            BioGptPreTrainedModel,
        )

# 如果不是类型检查模式，则设置模块为延迟加载模式
else:
    import sys

    # 设置当前模块为延迟加载模式，通过_LazyModule类实现
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```