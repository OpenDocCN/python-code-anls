# `.\models\gpt_neox\__init__.py`

```
# 导入需要的依赖库和模块
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
from ...utils import OptionalDependencyNotAvailable

# 定义模块间的导入结构
_import_structure = {"configuration_gpt_neox": ["GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXConfig"]}

# 检查并导入tokenizers库
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gpt_neox_fast"] = ["GPTNeoXTokenizerFast"]

# 检查并导入torch库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gpt_neox"] = [
        "GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTNeoXForCausalLM",
        "GPTNeoXForQuestionAnswering",
        "GPTNeoXForSequenceClassification",
        "GPTNeoXForTokenClassification",
        "GPTNeoXLayer",
        "GPTNeoXModel",
        "GPTNeoXPreTrainedModel",
    ]

# 如果是类型检查，导入相关的模块和类定义
if TYPE_CHECKING:
    from .configuration_gpt_neox import GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXConfig
    
    # 检查并导入tokenizers库
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
    
    # 检查并导入torch库
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gpt_neox import (
            GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
            GPTNeoXLayer,
            GPTNeoXModel,
            GPTNeoXPreTrainedModel,
        )

# 如果不是类型检查，则设置模块延迟导入的方式
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```