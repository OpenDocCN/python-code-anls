# `.\models\camembert\__init__.py`

```
# 引入类型检查模块，用于检查类型相关的导入
from typing import TYPE_CHECKING

# 引入依赖项检查函数和相关模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构字典，包含Camembert相关配置和模型
_import_structure = {
    "configuration_camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig", "CamembertOnnxConfig"],
}

# 检查是否支持sentencepiece，若不支持则引发OptionalDependencyNotAvailable异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 支持时将CamembertTokenizer模块添加到导入结构中
    _import_structure["tokenization_camembert"] = ["CamembertTokenizer"]

# 检查是否支持tokenizers，若不支持则引发OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 支持时将CamembertTokenizerFast模块添加到导入结构中
    _import_structure["tokenization_camembert_fast"] = ["CamembertTokenizerFast"]

# 检查是否支持torch，若不支持则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 支持时将Camembert相关模型添加到导入结构中
    _import_structure["modeling_camembert"] = [
        "CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CamembertForCausalLM",
        "CamembertForMaskedLM",
        "CamembertForMultipleChoice",
        "CamembertForQuestionAnswering",
        "CamembertForSequenceClassification",
        "CamembertForTokenClassification",
        "CamembertModel",
        "CamembertPreTrainedModel",
    ]

# 检查是否支持tensorflow，若不支持则引发OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 支持时将TensorFlow版Camembert模型添加到导入结构中
    _import_structure["modeling_tf_camembert"] = [
        "TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFCamembertForCausalLM",
        "TFCamembertForMaskedLM",
        "TFCamembertForMultipleChoice",
        "TFCamembertForQuestionAnswering",
        "TFCamembertForSequenceClassification",
        "TFCamembertForTokenClassification",
        "TFCamembertModel",
        "TFCamembertPreTrainedModel",
    ]


# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从configuration_camembert模块导入特定配置和类定义
    from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig, CamembertOnnxConfig

    # 检查是否支持sentencepiece，若不支持则不导入CamembertTokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 支持时从tokenization_camembert模块导入CamembertTokenizer
        from .tokenization_camembert import CamembertTokenizer

    # 检查是否支持tokenizers，若不支持则不导入任何内容
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()

        pass
    else:
        # 支持时继续导入相关内容，这里不包含在示例中的代码部分
        pass
    # 尝试导入 CamembertTokenizerFast，如果 OptionalDependencyNotAvailable 异常发生则跳过
    try:
        from .tokenization_camembert_fast import CamembertTokenizerFast
    # 如果 OptionalDependencyNotAvailable 异常发生，则什么也不做，直接跳过
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有异常发生，则成功导入了 CamembertTokenizerFast
    
    # 尝试检查是否 Torch 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果 OptionalDependencyNotAvailable 异常发生，则什么也不做，直接跳过
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有异常发生，则 Torch 库可用，导入相关 Camembert 模型和工具类
    
    else:
        from .modeling_camembert import (
            CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CamembertForCausalLM,
            CamembertForMaskedLM,
            CamembertForMultipleChoice,
            CamembertForQuestionAnswering,
            CamembertForSequenceClassification,
            CamembertForTokenClassification,
            CamembertModel,
            CamembertPreTrainedModel,
        )
    
    # 尝试检查是否 TensorFlow 库可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果 OptionalDependencyNotAvailable 异常发生，则什么也不做，直接跳过
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有异常发生，则 TensorFlow 库可用，导入相关 TensorFlow 版本的 Camembert 模型和工具类
    else:
        from .modeling_tf_camembert import (
            TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFCamembertForCausalLM,
            TFCamembertForMaskedLM,
            TFCamembertForMultipleChoice,
            TFCamembertForQuestionAnswering,
            TFCamembertForSequenceClassification,
            TFCamembertForTokenClassification,
            TFCamembertModel,
            TFCamembertPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态设置当前模块的属性
    import sys

    # 使用 sys.modules 和 __name__ 将当前模块名指定为 _LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```