# `.\models\distilbert\__init__.py`

```py
# 导入`
# 版权声明和许可证信息，指明代码版权和使用许可
# 详细描述了此代码的版权所有者和许可证（Apache License, Version 2.0）
# 提供了 Apache License, Version 2.0 的网址链接，以便查阅
# 如果符合许可证的条件，允许按“原样”分发和使用此代码
# 详细说明了在适用法律或书面同意的情况下，此软件是按“原样”分发的
# 详细说明了此软件是按“原样”分发，不带任何明示或暗示的担保或条件
# 提供了 Apache License, Version 2.0 的网址链接，以便查阅

from typing import TYPE_CHECKING

# 导入 LazyModule、检查各种库是否可用等工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构，列出了各模块所需的配置、类和函数
_import_structure = {
    "configuration_distilbert": [
        "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DistilBertConfig",
        "DistilBertOnnxConfig",
    ],
    "tokenization_distilbert": ["DistilBertTokenizer"],
}

# 检查 tokenizers 库是否可用，不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_distilbert_fast 到导入结构
    _import_structure["tokenization_distilbert_fast"] = ["DistilBertTokenizerFast"]

# 检查 torch 库是否可用，不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_distilbert 到导入结构
    _import_structure["modeling_distilbert"] = [
        "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DistilBertForMaskedLM",
        "DistilBertForMultipleChoice",
        "DistilBertForQuestionAnswering",
        "DistilBertForSequenceClassification",
        "DistilBertForTokenClassification",
        "DistilBertModel",
        "DistilBertPreTrainedModel",
    ]

# 检查 tensorflow 库是否可用，不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_tf_distilbert 到导入结构
    _import_structure["modeling_tf_distilbert"] = [
        "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDistilBertForMaskedLM",
        "TFDistilBertForMultipleChoice",
        "TFDistilBertForQuestionAnswering",
        "TFDistilBertForSequenceClassification",
        "TFDistilBertForTokenClassification",
        "TFDistilBertMainLayer",
        "TFDistilBertModel",
        "TFDistilBertPreTrainedModel",
    ]

# 检查 flax 库是否可用，不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_flax_distilbert 到导入结构
    _import_structure["modeling_flax_distilbert"] = [
        "FlaxDistilBertForMaskedLM",
        "FlaxDistilBertForMultipleChoice",
        "FlaxDistilBertForQuestionAnswering",
        "FlaxDistilBertForSequenceClassification",
        "FlaxDistilBertForTokenClassification",
        "FlaxDistilBertModel",
        "FlaxDistilBertPreTrainedModel",
    ]


if TYPE_CHECKING:
    # 如果是类型检查阶段，这里可能会有类型相关的导入或代码
    # 导入DistilBERT预训练模型的配置映射、配置类和ONNX配置类
    from .configuration_distilbert import (
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DistilBertConfig,
        DistilBertOnnxConfig,
    )
    
    # 导入DistilBERT的标记器
    from .tokenization_distilbert import DistilBertTokenizer
    
    # 检查是否安装了tokenizers库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果tokenizers库可用，则导入DistilBERT的快速标记器
        from .tokenization_distilbert_fast import DistilBertTokenizerFast
    
    # 检查是否安装了torch库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果torch库可用，则导入DistilBERT的模型相关模块
        from .modeling_distilbert import (
            DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DistilBertForMaskedLM,
            DistilBertForMultipleChoice,
            DistilBertForQuestionAnswering,
            DistilBertForSequenceClassification,
            DistilBertForTokenClassification,
            DistilBertModel,
            DistilBertPreTrainedModel,
        )
    
    # 检查是否安装了tensorflow库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果tensorflow库可用，则导入TF版本的DistilBERT模型相关模块
        from .modeling_tf_distilbert import (
            TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDistilBertForMaskedLM,
            TFDistilBertForMultipleChoice,
            TFDistilBertForQuestionAnswering,
            TFDistilBertForSequenceClassification,
            TFDistilBertForTokenClassification,
            TFDistilBertMainLayer,
            TFDistilBertModel,
            TFDistilBertPreTrainedModel,
        )
    
    # 检查是否安装了flax库，若未安装则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果flax库可用，则导入Flax版本的DistilBERT模型相关模块
        from .modeling_flax_distilbert import (
            FlaxDistilBertForMaskedLM,
            FlaxDistilBertForMultipleChoice,
            FlaxDistilBertForQuestionAnswering,
            FlaxDistilBertForSequenceClassification,
            FlaxDistilBertForTokenClassification,
            FlaxDistilBertModel,
            FlaxDistilBertPreTrainedModel,
        )
else:
    # 如果不是以上情况，即需要延迟加载模块
    import sys
    # 导入系统模块 sys

    # 将当前模块注册为一个延迟加载模块，并将其设置为当前模块的引用
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```