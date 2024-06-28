# `.\models\albert\__init__.py`

```
# 导入所需的类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖项和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构字典
_import_structure = {
    "configuration_albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig", "AlbertOnnxConfig"],
}

# 检查是否安装了 sentencepiece，若未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 sentencepiece，则将 AlbertTokenizer 添加到导入结构中
    _import_structure["tokenization_albert"] = ["AlbertTokenizer"]

# 检查是否安装了 tokenizers，若未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 tokenizers，则将 AlbertTokenizerFast 添加到导入结构中
    _import_structure["tokenization_albert_fast"] = ["AlbertTokenizerFast"]

# 检查是否安装了 torch，若未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 torch，则将 Albert 相关模块添加到导入结构中
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

# 检查是否安装了 TensorFlow，若未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 TensorFlow，则将 TFAlbert 相关模块添加到导入结构中
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

# 检查是否安装了 Flax，若未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 Flax，继续添加相关模块（此处未完整给出，应根据实际情况补充）
    pass
    # 将一组模块名称添加到_import_structure字典中，以便后续导入
    _import_structure["modeling_flax_albert"] = [
        "FlaxAlbertForMaskedLM",                    # 添加FlaxAlbertForMaskedLM模块名
        "FlaxAlbertForMultipleChoice",              # 添加FlaxAlbertForMultipleChoice模块名
        "FlaxAlbertForPreTraining",                 # 添加FlaxAlbertForPreTraining模块名
        "FlaxAlbertForQuestionAnswering",           # 添加FlaxAlbertForQuestionAnswering模块名
        "FlaxAlbertForSequenceClassification",      # 添加FlaxAlbertForSequenceClassification模块名
        "FlaxAlbertForTokenClassification",         # 添加FlaxAlbertForTokenClassification模块名
        "FlaxAlbertModel",                          # 添加FlaxAlbertModel模块名
        "FlaxAlbertPreTrainedModel",                # 添加FlaxAlbertPreTrainedModel模块名
    ]
# 如果 TYPE_CHECKING 为真，则导入以下模块和类
if TYPE_CHECKING:
    # 导入 ALBERT 相关的配置映射和配置类
    from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig, AlbertOnnxConfig
    
    # 尝试检查是否安装了 sentencepiece，若未安装则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若安装了 sentencepiece，则导入 AlbertTokenizer
        from .tokenization_albert import AlbertTokenizer
    
    # 尝试检查是否安装了 tokenizers，若未安装则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若安装了 tokenizers，则导入 AlbertTokenizerFast
        from .tokenization_albert_fast import AlbertTokenizerFast
    
    # 尝试检查是否安装了 torch，若未安装则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若安装了 torch，则导入以下 Albert 相关模块和类
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
    
    # 尝试检查是否安装了 tensorflow，若未安装则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若安装了 tensorflow，则导入以下 TFAlbert 相关模块和类
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
    
    # 尝试检查是否安装了 flax，若未安装则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若安装了 flax，则导入以下 FlaxAlbert 相关模块和类
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
# 如果 TYPE_CHECKING 为假，则导入 sys 模块，并将当前模块设为懒加载模块
else:
    import sys
    
    # 使用 _LazyModule 类将当前模块设置为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```