# `.\models\deberta\__init__.py`

```py
# 版权声明和许可信息
#
# 版权所有 (c) 2020 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以获取许可证的副本，请参阅
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅许可证。
#

# 导入类型检查工具
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需的工具和异常
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_deberta": ["DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaConfig", "DebertaOnnxConfig"],
    "tokenization_deberta": ["DebertaTokenizer"],
}

# 尝试导入 tokenizers_deberta_fast 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_deberta_fast"] = ["DebertaTokenizerFast"]

# 尝试导入 modeling_deberta 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_deberta"] = [
        "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DebertaForMaskedLM",
        "DebertaForQuestionAnswering",
        "DebertaForSequenceClassification",
        "DebertaForTokenClassification",
        "DebertaModel",
        "DebertaPreTrainedModel",
    ]

# 尝试导入 modeling_tf_deberta 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_deberta"] = [
        "TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDebertaForMaskedLM",
        "TFDebertaForQuestionAnswering",
        "TFDebertaForSequenceClassification",
        "TFDebertaForTokenClassification",
        "TFDebertaModel",
        "TFDebertaPreTrainedModel",
    ]

# 如果在类型检查模式下，则导入特定的模块和符号
if TYPE_CHECKING:
    from .configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig, DebertaOnnxConfig
    from .tokenization_deberta import DebertaTokenizer

    # 尝试导入 tokenization_deberta_fast 模块，如果不可用则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_deberta_fast import DebertaTokenizerFast

    # 尝试导入 torch 模块，如果不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，导入以下模块来自模型定义的Deberta相关类和预训练模型列表
    from .modeling_deberta import (
        DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        DebertaForMaskedLM,
        DebertaForQuestionAnswering,
        DebertaForSequenceClassification,
        DebertaForTokenClassification,
        DebertaModel,
        DebertaPreTrainedModel,
    )

try:
    # 检查是否没有可用的TensorFlow，如果是则抛出OptionalDependencyNotAvailable异常
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果OptionalDependencyNotAvailable异常被抛出，不做任何操作
    pass
else:
    # 否则，导入以下TensorFlow版本的Deberta相关类和预训练模型列表
    from .modeling_tf_deberta import (
        TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFDebertaForMaskedLM,
        TFDebertaForQuestionAnswering,
        TFDebertaForSequenceClassification,
        TFDebertaForTokenClassification,
        TFDebertaModel,
        TFDebertaPreTrainedModel,
    )
else:
    # 导入 sys 模块，用于处理模块操作
    import sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```