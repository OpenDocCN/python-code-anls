# `.\transformers\models\xlm_prophetnet\__init__.py`

```py
# 这是一个 Python 文件，包含了 XLM-ProphetNet 模型相关的定义和导入。
# 该模型是一种预训练的语言生成模型，由 Hugging Face 团队开发。

# 这些是需要导入的类和变量，根据是否有可选依赖来动态导入。
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available

# 定义需要导入的模块结构
_import_structure = {
    "configuration_xlm_prophetnet": ["XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMProphetNetConfig"],
}

# 如果没有安装 sentencepiece 库，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_xlm_prophetnet"] = ["XLMProphetNetTokenizer"]

# 如果没有安装 PyTorch 库，则引发 OptionalDependencyNotAvailable 异常  
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_xlm_prophetnet"] = [
        "XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMProphetNetDecoder",
        "XLMProphetNetEncoder",
        "XLMProphetNetForCausalLM",
        "XLMProphetNetForConditionalGeneration",
        "XLMProphetNetModel",
        "XLMProphetNetPreTrainedModel",
    ]

# 如果是类型检查，则导入相应的类和变量
if TYPE_CHECKING:
    from .configuration_xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xlm_prophetnet import XLMProphetNetTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_xlm_prophetnet import (
            XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMProphetNetDecoder,
            XLMProphetNetEncoder,
            XLMProphetNetForCausalLM,
            XLMProphetNetForConditionalGeneration,
            XLMProphetNetModel,
            XLMProphetNetPreTrainedModel,
        )

# 如果不是类型检查，则使用 _LazyModule 进行延迟导入
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```