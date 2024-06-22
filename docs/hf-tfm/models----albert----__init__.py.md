# `.\transformers\models\albert\__init__.py`

```py
# 版权声明和许可证信息
# 从必要的模块中导入所需的函数和类
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
_import_structure = {
    "configuration_albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig", "AlbertOnnxConfig"],
}

# 检查是否安装了 sentencepiece 库，若未安装则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了 sentencepiece 库，则添加 tokenization_albert 模块到导入结构中
    _import_structure["tokenization_albert"] = ["AlbertTokenizer"]

# 检查是否安装了 tokenizers 库，若未安装则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了 tokenizers 库，则添加 tokenization_albert_fast 模块到导入结构中
    _import_structure["tokenization_albert_fast"] = ["AlbertTokenizerFast"]

# 检查是否安装了 torch 库，若未安装则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了 torch 库，则添加 modeling_albert 模块到导入结构中
    _import_structure["modeling_albert"] = [
        "ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AlbertForMaskedLM",
        "AlbertForMultipleChoice",
        "AlbertForPreTraining",
        "AlbertForQuestionAnswering",
        "AlbertForSequenceClassification",
        "AlbertForTokenClassification",
        "AlbertModel",
        "AlbertPreTrainedModel",
        "load_tf_weights_in_albert",
    ]

# 检查是否安装了 tensorflow 库，若未安装则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了 tensorflow 库，则添加 modeling_tf_albert 模块到导入结构中
    _import_structure["modeling_tf_albert"] = [
        "TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFAlbertForMaskedLM",
        "TFAlbertForMultipleChoice",
        "TFAlbertForPreTraining",
        "TFAlbertForQuestionAnswering",
        "TFAlbertForSequenceClassification",
        "TFAlbertForTokenClassification",
        "TFAlbertMainLayer",
        "TFAlbertModel",
        "TFAlbertPreTrainedModel",
    ]

# 检查是否安装了 flax 库，若未安装则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若安装了 flax 库，则继续处理
    # 将模块名 "modeling_flax_albert" 关联到该模块中的类名列表
    _import_structure["modeling_flax_albert"] = [
        "FlaxAlbertForMaskedLM",  # FlaxAlbertForMaskedLM 类，用于掩码语言建模任务
        "FlaxAlbertForMultipleChoice",  # FlaxAlbertForMultipleChoice 类，用于多项选择任务
        "FlaxAlbertForPreTraining",  # FlaxAlbertForPreTraining 类，用于预训练任务
        "FlaxAlbertForQuestionAnswering",  # FlaxAlbertForQuestionAnswering 类，用于问答任务
        "FlaxAlbertForSequenceClassification",  # FlaxAlbertForSequenceClassification 类，用于序列分类任务
        "FlaxAlbertForTokenClassification",  # FlaxAlbertForTokenClassification 类，用于标记分类任务
        "FlaxAlbertModel",  # FlaxAlbertModel 类，ALBERT 模型的主类
        "FlaxAlbertPreTrainedModel",  # FlaxAlbertPreTrainedModel 类，ALBERT 预训练模型的基类
    ]
# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入相关的配置和类
    from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig, AlbertOnnxConfig

    # 尝试检查是否安装了 sentencepiece 库，如果没有则抛出异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 AlbertTokenizer 类
        from .tokenization_albert import AlbertTokenizer

    # 尝试检查是否安装了 tokenizers 库，如果没有则抛出异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 AlbertTokenizerFast 类
        from .tokenization_albert_fast import AlbertTokenizerFast

    # 尝试检查是否安装了 torch 库，如果没有则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关的类和函数
        from .modeling_albert import (
            ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            AlbertForMaskedLM,
            AlbertForMultipleChoice,
            AlbertForPreTraining,
            AlbertForQuestionAnswering,
            AlbertForSequenceClassification,
            AlbertForTokenClassification,
            AlbertModel,
            AlbertPreTrainedModel,
            load_tf_weights_in_albert,
        )

    # 尝试检查是否安装了 tensorflow 库，如果没有则抛出异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关的类和函数
        from .modeling_tf_albert import (
            TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFAlbertForMaskedLM,
            TFAlbertForMultipleChoice,
            TFAlbertForPreTraining,
            TFAlbertForQuestionAnswering,
            TFAlbertForSequenceClassification,
            TFAlbertForTokenClassification,
            TFAlbertMainLayer,
            TFAlbertModel,
            TFAlbertPreTrainedModel,
        )

    # 尝试检查是否安装了 flax 库，如果没有则抛出异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入相关的类
        from .modeling_flax_albert import (
            FlaxAlbertForMaskedLM,
            FlaxAlbertForMultipleChoice,
            FlaxAlbertForPreTraining,
            FlaxAlbertForQuestionAnswering,
            FlaxAlbertForSequenceClassification,
            FlaxAlbertForTokenClassification,
            FlaxAlbertModel,
            FlaxAlbertPreTrainedModel,
        )
# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为 LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```