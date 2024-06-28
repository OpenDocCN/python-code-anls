# `.\models\mbart\__init__.py`

```py
# 版权声明和许可证信息
#
# 版权所有 2020 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）获得许可；
# 除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发软件，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需函数和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构字典，用于存储不同条件下的导入列表
_import_structure = {"configuration_mbart": ["MBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "MBartConfig", "MBartOnnxConfig"]}

# 检查是否有 sentencepiece 库可用，若无则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则向导入结构字典添加 tokenization_mbart 模块的导入列表
    _import_structure["tokenization_mbart"] = ["MBartTokenizer"]

# 检查是否有 tokenizers 库可用，若无则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则向导入结构字典添加 tokenization_mbart_fast 模块的导入列表
    _import_structure["tokenization_mbart_fast"] = ["MBartTokenizerFast"]

# 检查是否有 torch 库可用，若无则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则向导入结构字典添加 modeling_mbart 模块的导入列表
    _import_structure["modeling_mbart"] = [
        "MBART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MBartForCausalLM",
        "MBartForConditionalGeneration",
        "MBartForQuestionAnswering",
        "MBartForSequenceClassification",
        "MBartModel",
        "MBartPreTrainedModel",
    ]

# 检查是否有 tensorflow 库可用，若无则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则向导入结构字典添加 modeling_tf_mbart 模块的导入列表
    _import_structure["modeling_tf_mbart"] = [
        "TFMBartForConditionalGeneration",
        "TFMBartModel",
        "TFMBartPreTrainedModel",
    ]

# 检查是否有 flax 库可用，若无则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则向导入结构字典添加 modeling_flax_mbart 模块的导入列表
    _import_structure["modeling_flax_mbart"] = [
        "FlaxMBartForConditionalGeneration",
        "FlaxMBartForQuestionAnswering",
        "FlaxMBartForSequenceClassification",
        "FlaxMBartModel",
        "FlaxMBartPreTrainedModel",
    ]

# 如果在类型检查模式下，导入相关模块
if TYPE_CHECKING:
    from .configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig, MBartOnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 tokenization_mbart 模块导入 MBartTokenizer
        from .tokenization_mbart import MBartTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入 MBartTokenizerFast 类，如果 OptionalDependencyNotAvailable 异常发生则跳过
    try:
        from .tokenization_mbart_fast import MBartTokenizerFast
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果未发生异常，则导入以下模块
    else:
        from .modeling_mbart import (
            MBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )

    # 尝试检查 Torch 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果未发生异常，则导入以下模块
    else:
        from .modeling_mbart import (
            MBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )

    # 尝试检查 TensorFlow 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果未发生异常，则导入以下模块
    else:
        from .modeling_tf_mbart import (
            TFMBartForConditionalGeneration,
            TFMBartModel,
            TFMBartPreTrainedModel,
        )

    # 尝试检查 Flax 是否可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何操作
    except OptionalDependencyNotAvailable:
        pass
    # 如果未发生异常，则导入以下模块
    else:
        from .modeling_flax_mbart import (
            FlaxMBartForConditionalGeneration,
            FlaxMBartForQuestionAnswering,
            FlaxMBartForSequenceClassification,
            FlaxMBartModel,
            FlaxMBartPreTrainedModel,
        )
else:
    # 如果条件不成立，即导入模块失败时执行以下操作

    # 导入 sys 模块，用于管理 Python 解释器的运行时环境
    import sys

    # 将当前模块的字典 (__name__) 更新为一个 LazyModule 对象，
    # LazyModule 是一个自定义类，用于延迟加载模块的实现
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```