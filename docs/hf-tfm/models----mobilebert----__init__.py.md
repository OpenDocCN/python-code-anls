# `.\models\mobilebert\__init__.py`

```py
# 引入依赖类型检查模块
from typing import TYPE_CHECKING

# 引入内部工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构字典，用于按需加载模块
_import_structure = {
    "configuration_mobilebert": [
        "MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # MobileBERT预训练配置文件映射表
        "MobileBertConfig",  # MobileBERT配置类
        "MobileBertOnnxConfig",  # MobileBERT ONNX配置类
    ],
    "tokenization_mobilebert": ["MobileBertTokenizer"],  # MobileBERT分词器类
}

# 检查是否存在tokenizers库，若不存在则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mobilebert_fast"] = ["MobileBertTokenizerFast"]  # 引入快速分词器类

# 检查是否存在torch库，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 引入MobileBERT的PyTorch模块
    _import_structure["modeling_mobilebert"] = [
        "MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # MobileBERT预训练模型归档列表
        "MobileBertForMaskedLM",  # 用于Masked Language Modeling的MobileBERT模型
        "MobileBertForMultipleChoice",  # 用于多项选择任务的MobileBERT模型
        "MobileBertForNextSentencePrediction",  # 用于下一句预测任务的MobileBERT模型
        "MobileBertForPreTraining",  # MobileBERT预训练模型
        "MobileBertForQuestionAnswering",  # 用于问答任务的MobileBERT模型
        "MobileBertForSequenceClassification",  # 用于序列分类任务的MobileBERT模型
        "MobileBertForTokenClassification",  # 用于标记分类任务的MobileBERT模型
        "MobileBertLayer",  # MobileBERT的层模块
        "MobileBertModel",  # MobileBERT模型
        "MobileBertPreTrainedModel",  # MobileBERT预训练模型基类
        "load_tf_weights_in_mobilebert",  # 加载MobileBERT的TensorFlow权重
    ]

# 检查是否存在tensorflow库，若不存在则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 引入MobileBERT的TensorFlow模块
    _import_structure["modeling_tf_mobilebert"] = [
        "TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # MobileBERT预训练模型归档列表（TensorFlow）
        "TFMobileBertForMaskedLM",  # 用于Masked Language Modeling的MobileBERT模型（TensorFlow）
        "TFMobileBertForMultipleChoice",  # 用于多项选择任务的MobileBERT模型（TensorFlow）
        "TFMobileBertForNextSentencePrediction",  # 用于下一句预测任务的MobileBERT模型（TensorFlow）
        "TFMobileBertForPreTraining",  # MobileBERT预训练模型（TensorFlow）
        "TFMobileBertForQuestionAnswering",  # 用于问答任务的MobileBERT模型（TensorFlow）
        "TFMobileBertForSequenceClassification",  # 用于序列分类任务的MobileBERT模型（TensorFlow）
        "TFMobileBertForTokenClassification",  # 用于标记分类任务的MobileBERT模型（TensorFlow）
        "TFMobileBertMainLayer",  # MobileBERT的主层模块（TensorFlow）
        "TFMobileBertModel",  # MobileBERT模型（TensorFlow）
        "TFMobileBertPreTrainedModel",  # MobileBERT预训练模型基类（TensorFlow）
    ]

# 如果是类型检查模式，引入必要的MobileBERT配置和分词器类
if TYPE_CHECKING:
    from .configuration_mobilebert import (
        MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileBERT预训练配置文件映射表
        MobileBertConfig,  # MobileBERT配置类
        MobileBertOnnxConfig,  # MobileBERT ONNX配置类
    )
    from .tokenization_mobilebert import MobileBertTokenizer  # MobileBERT分词器类

    # 再次检查是否存在tokenizers库，若不存在则忽略
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_mobilebert_fast import MobileBertTokenizerFast
    ```
    # 如果上面的条件不成立，即没有从 .tokenization_mobilebert_fast 导入 MobileBertTokenizerFast

    ```    
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 Torch 不可用，则抛出 OptionalDependencyNotAvailable 异常
        pass
    else:
        # 如果 Torch 可用，则从 .modeling_mobilebert 导入以下模块
        from .modeling_mobilebert import (
            MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileBertForMaskedLM,
            MobileBertForMultipleChoice,
            MobileBertForNextSentencePrediction,
            MobileBertForPreTraining,
            MobileBertForQuestionAnswering,
            MobileBertForSequenceClassification,
            MobileBertForTokenClassification,
            MobileBertLayer,
            MobileBertModel,
            MobileBertPreTrainedModel,
            load_tf_weights_in_mobilebert,
        )
    ```

    ```
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，则抛出 OptionalDependencyNotAvailable 异常
        pass
    else:
        # 如果 TensorFlow 可用，则从 .modeling_tf_mobilebert 导入以下模块
        from .modeling_tf_mobilebert import (
            TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFMobileBertForMaskedLM,
            TFMobileBertForMultipleChoice,
            TFMobileBertForNextSentencePrediction,
            TFMobileBertForPreTraining,
            TFMobileBertForQuestionAnswering,
            TFMobileBertForSequenceClassification,
            TFMobileBertForTokenClassification,
            TFMobileBertMainLayer,
            TFMobileBertModel,
            TFMobileBertPreTrainedModel,
        )
    ```
else:
    # 导入 sys 模块，用于操作 Python 解释器相关的功能
    import sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 封装，使模块在需要时被延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```