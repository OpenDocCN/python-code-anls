# `.\models\bert_generation\__init__.py`

```py
# 引入必要的模块和依赖项
from typing import TYPE_CHECKING
# 从相对路径导入工具函数和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available

# 定义模块的导入结构，包含一个字典，用于按需导入不同的模块
_import_structure = {"configuration_bert_generation": ["BertGenerationConfig"]}

# 检查是否存在SentencePiece库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，将tokenization_bert_generation模块添加到导入结构中
    _import_structure["tokenization_bert_generation"] = ["BertGenerationTokenizer"]

# 检查是否存在Torch库，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果库可用，将modeling_bert_generation模块添加到导入结构中
    _import_structure["modeling_bert_generation"] = [
        "BertGenerationDecoder",
        "BertGenerationEncoder",
        "BertGenerationPreTrainedModel",
        "load_tf_weights_in_bert_generation",
    ]

# 如果当前环境是类型检查模式，导入额外的模块
if TYPE_CHECKING:
    from .configuration_bert_generation import BertGenerationConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_bert_generation import BertGenerationTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bert_generation import (
            BertGenerationDecoder,
            BertGenerationEncoder,
            BertGenerationPreTrainedModel,
            load_tf_weights_in_bert_generation,
        )

# 如果不是类型检查模式，则将LazyModule注册为当前模块，用于按需导入
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```