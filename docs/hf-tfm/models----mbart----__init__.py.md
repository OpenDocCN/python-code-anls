# `.\transformers\models\mbart\__init__.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {"configuration_mbart": ["MBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "MBartConfig", "MBartOnnxConfig"]}

# 检查是否存在 sentencepiece 库，如果不存在则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mbart"] = ["MBartTokenizer"]

# 检查是否存在 tokenizers 库，如果不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mbart_fast"] = ["MBartTokenizerFast"]

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mbart"] = [
        "MBART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MBartForCausalLM",
        "MBartForConditionalGeneration",
        "MBartForQuestionAnswering",
        "MBartForSequenceClassification",
        "MBartModel",
        "MBartPreTrainedModel",
    ]

# 检查是否存在 tensorflow 库，如果不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_mbart"] = [
        "TFMBartForConditionalGeneration",
        "TFMBartModel",
        "TFMBartPreTrainedModel",
    ]

# 检查是否存在 flax 库，如果不存在则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_mbart"] = [
        "FlaxMBartForConditionalGeneration",
        "FlaxMBartForQuestionAnswering",
        "FlaxMBartForSequenceClassification",
        "FlaxMBartModel",
        "FlaxMBartPreTrainedModel",
    ]

# 如果是类型检查模式，则导入相关模块
if TYPE_CHECKING:
    from .configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig, MBartOnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_mbart import MBartTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入 MBartTokenizerFast 类，如果 OptionalDependencyNotAvailable 异常发生则跳过
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_mbart_fast import MBartTokenizerFast

    # 尝试检查是否有 torch 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常并跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 MBART 相关模型类和常量
        from .modeling_mbart import (
            MBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )

    # 尝试检查是否有 tensorflow 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常并跳过
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 TFMBart 相关模型类
        from .modeling_tf_mbart import TFMBartForConditionalGeneration, TFMBartModel, TFMBartPreTrainedModel

    # 尝试检查是否有 flax 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常并跳过
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 FlaxMBart 相关模型类
        from .modeling_flax_mbart import (
            FlaxMBartForConditionalGeneration,
            FlaxMBartForQuestionAnswering,
            FlaxMBartForSequenceClassification,
            FlaxMBartModel,
            FlaxMBartPreTrainedModel,
        )
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```