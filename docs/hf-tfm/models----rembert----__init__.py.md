# `.\transformers\models\rembert\__init__.py`

```
# 版权声明
# 版权归 The HuggingFace 团队所有
#
# 根据 Apache 许可证，版本 2.0 授权使用此文件；
# 除非获得许可证，否则不得使用此文件
# 可以在以下链接获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，软件分发基于“按现有状况”分发，
# 没有任何明示或暗示的保证或条件
# 请查阅许可证以了解明确语言的权限和限制

# 导入需要使用的模块
from typing import TYPE_CHECKING
# 导入 LazyModule 模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 按模块结构组织依赖关系
_import_structure = {
    "configuration_rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig", "RemBertOnnxConfig"]
}

# 检查是否存在 SentencePiece 库
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_rembert"] = ["RemBertTokenizer"]

# 检查是否存在 Tokenizers 库
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_rembert_fast"] = ["RemBertTokenizerFast"]

# 检查是否存在 PyTorch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rembert"] = [
        "REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RemBertForCausalLM",
        "RemBertForMaskedLM",
        "RemBertForMultipleChoice",
        "RemBertForQuestionAnswering",
        "RemBertForSequenceClassification",
        "RemBertForTokenClassification",
        "RemBertLayer",
        "RemBertModel",
        "RemBertPreTrainedModel",
        "load_tf_weights_in_rembert",
    ]

# 检查是否存在 TensorFlow 库
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_rembert"] = [
        "TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST", 
        "TFRemBertForCausalLM",
        "TFRemBertForMaskedLM",
        "TFRemBertForMultipleChoice",
        "TFRemBertForQuestionAnswering",
        "TFRemBertForSequenceClassification", 
        "TFRemBertForTokenClassification",
        "TFRemBertLayer",
        "TFRemBertModel",
        "TFRemBertPreTrainedModel",
    ]

# 如果为类型检查，导入配置信息和 Tokenizer 模块
if TYPE_CHECKING:
    from .configuration_rembert import REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RemBertConfig, RemBertOnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_rembert import RemBertTokenizer
    # 尝试检查是否安装了 tokenizers 库
    try:
        # 如果 tokenizers 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果引发了 OptionalDependencyNotAvailable 异常，则忽略，继续执行后续代码
        pass
    else:
        # 如果 tokenizers 库可用，则导入 RemBertTokenizerFast 类
        from .tokenization_rembert_fast import RemBertTokenizerFast

    # 尝试检查是否安装了 PyTorch 库
    try:
        # 如果 PyTorch 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果引发了 OptionalDependencyNotAvailable 异常，则忽略，继续执行后续代码
        pass
    else:
        # 如果 PyTorch 库可用，则导入 RemBert 相关的模型和类
        from .modeling_rembert import (
            REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RemBertForCausalLM,
            RemBertForMaskedLM,
            RemBertForMultipleChoice,
            RemBertForQuestionAnswering,
            RemBertForSequenceClassification,
            RemBertForTokenClassification,
            RemBertLayer,
            RemBertModel,
            RemBertPreTrainedModel,
            load_tf_weights_in_rembert,
        )

    # 尝试检查是否安装了 TensorFlow 库
    try:
        # 如果 TensorFlow 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果引发了 OptionalDependencyNotAvailable 异常，则忽略，继续执行后续代码
        pass
    else:
        # 如果 TensorFlow 库可用，则导入 TFRemBert 相关的模型和类
        from .modeling_tf_rembert import (
            TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRemBertForCausalLM,
            TFRemBertForMaskedLM,
            TFRemBertForMultipleChoice,
            TFRemBertForQuestionAnswering,
            TFRemBertForSequenceClassification,
            TFRemBertForTokenClassification,
            TFRemBertLayer,
            TFRemBertModel,
            TFRemBertPreTrainedModel,
        )
# 如果不在主程序中，则导入sys模块
import sys
# 将当前模块的名称赋值给sys.modules字典中的键[__name__]对应的值
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```