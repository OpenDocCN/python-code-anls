# `.\models\mt5\__init__.py`

```
# 引入类型检查相关模块
from typing import TYPE_CHECKING

# 引入必要的依赖项和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 如果sentencepiece可用，则使用T5Tokenizer来自../t5/tokenization_t5模块
if is_sentencepiece_available():
    from ..t5.tokenization_t5 import T5Tokenizer
else:
    # 否则，使用dummy_sentencepiece_objects中的T5Tokenizer作为替代
    from ...utils.dummy_sentencepiece_objects import T5Tokenizer

# 定义MT5Tokenizer为T5Tokenizer
MT5Tokenizer = T5Tokenizer

# 如果tokenizers可用，则使用T5TokenizerFast来自../t5/tokenization_t5_fast模块
if is_tokenizers_available():
    from ..t5.tokenization_t5_fast import T5TokenizerFast
else:
    # 否则，使用dummy_tokenizers_objects中的T5TokenizerFast作为替代
    from ...utils.dummy_tokenizers_objects import T5TokenizerFast

# 定义MT5TokenizerFast为T5TokenizerFast
MT5TokenizerFast = T5TokenizerFast

# 定义模块导入结构_import_structure，包含MT5Config和MT5OnnxConfig
_import_structure = {"configuration_mt5": ["MT5Config", "MT5OnnxConfig"]}

# 尝试导入torch相关模块，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，定义modeling_mt5结构包含各种MT5模型和类
    _import_structure["modeling_mt5"] = [
        "MT5EncoderModel",
        "MT5ForConditionalGeneration",
        "MT5ForQuestionAnswering",
        "MT5ForSequenceClassification",
        "MT5ForTokenClassification",
        "MT5Model",
        "MT5PreTrainedModel",
        "MT5Stack",
    ]

# 尝试导入tensorflow相关模块，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，定义modeling_tf_mt5结构包含各种TFMT5模型和类
    _import_structure["modeling_tf_mt5"] = ["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"]

# 尝试导入flax相关模块，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，定义modeling_flax_mt5结构包含各种FlaxMT5模型和类
    _import_structure["modeling_flax_mt5"] = ["FlaxMT5EncoderModel", "FlaxMT5ForConditionalGeneration", "FlaxMT5Model"]

# 如果在类型检查模式下，导入MT5Config和MT5OnnxConfig配置
if TYPE_CHECKING:
    from .configuration_mt5 import MT5Config, MT5OnnxConfig

    # 尝试导入torch相关MT5模块，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入modeling_mt5中各种MT5模型和类
        from .modeling_mt5 import (
            MT5EncoderModel,
            MT5ForConditionalGeneration,
            MT5ForQuestionAnswering,
            MT5ForSequenceClassification,
            MT5ForTokenClassification,
            MT5Model,
            MT5PreTrainedModel,
            MT5Stack,
        )

    # 尝试导入tensorflow相关MT5模块，如果不可用则忽略
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果未导入模块，则从当前包导入 TensorFlow 版的 MT5 模型相关类
    else:
        from .modeling_tf_mt5 import TFMT5EncoderModel, TFMT5ForConditionalGeneration, TFMT5Model

    # 尝试检查是否可用 Flax，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 如果 OptionalDependencyNotAvailable 异常被抛出，则捕获并忽略
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有异常被抛出，则导入 Flax 版的 MT5 模型相关类
    else:
        from .modeling_flax_mt5 import FlaxMT5EncoderModel, FlaxMT5ForConditionalGeneration, FlaxMT5Model
else:
    # 导入 sys 模块，用于动态操作模块对象
    import sys

    # 将当前模块的名称映射到 _LazyModule 类的实例，并设置相关属性
    sys.modules[__name__] = _LazyModule(
        __name__,  # 模块的名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 导入结构
        extra_objects={"MT5Tokenizer": MT5Tokenizer, "MT5TokenizerFast": MT5TokenizerFast},  # 额外的对象映射
        module_spec=__spec__,  # 模块的规范
    )
```