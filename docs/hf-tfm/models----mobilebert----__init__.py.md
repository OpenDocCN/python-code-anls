# `.\transformers\models\mobilebert\__init__.py`

```
# 导入需要的类型检查模块
from typing import TYPE_CHECKING

# 导入 HuggingFace 库中的实用工具和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构字典，包含 MobileBERT 配置、模型和 tokenizer 的导入结构
_import_structure = {
    "configuration_mobilebert": [
        "MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # MobileBERT 预训练配置文件映射
        "MobileBertConfig",  # MobileBERT 配置类
        "MobileBertOnnxConfig",  # MobileBERT ONNX 配置类
    ],
    "tokenization_mobilebert": ["MobileBertTokenizer"],  # MobileBERT tokenizer

# 尝试导入 tokenizers 包，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_mobilebert_fast"] = ["MobileBertTokenizerFast"]  # 快速版本的 MobileBERT tokenizer

# 尝试导入 torch 包，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mobilebert"] = [
        "MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # MobileBERT 预训练模型存档列表
        "MobileBertForMaskedLM",  # 用于遮蔽语言模型任务的 MobileBERT 模型
        "MobileBertForMultipleChoice",  # 用于多项选择任务的 MobileBERT 模型
        "MobileBertForNextSentencePrediction",  # 用于下一个句子预测任务的 MobileBERT 模型
        "MobileBertForPreTraining",  # 用于预训练任务的 MobileBERT 模型
        "MobileBertForQuestionAnswering",  # 用于问答任务的 MobileBERT 模型
        "MobileBertForSequenceClassification",  # 用于序列分类任务的 MobileBERT 模型
        "MobileBertForTokenClassification",  # 用于令牌分类任务的 MobileBERT 模型
        "MobileBertLayer",  # MobileBERT 模型层
        "MobileBertModel",  # MobileBERT 模型
        "MobileBertPreTrainedModel",  # MobileBERT 预训练模型基类
        "load_tf_weights_in_mobilebert",  # 用于加载 TensorFlow 权重到 MobileBERT 模型的函数
    ]

# 尝试导入 TensorFlow 包，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_mobilebert"] = [
        "TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # TensorFlow 版本的 MobileBERT 预训练模型存档列表
        "TFMobileBertForMaskedLM",  # 用于遮蔽语言模型任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertForMultipleChoice",  # 用于多项选择任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertForNextSentencePrediction",  # 用于下一个句子预测任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertForPreTraining",  # 用于预训练任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertForQuestionAnswering",  # 用于问答任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertForSequenceClassification",  # 用于序列分类任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertForTokenClassification",  # 用于令牌分类任务的 TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertMainLayer",  # TensorFlow 版本的 MobileBERT 主层
        "TFMobileBertModel",  # TensorFlow 版本的 MobileBERT 模型
        "TFMobileBertPreTrainedModel",  # TensorFlow 版本的 MobileBERT 预训练模型基类
    ]

# 如果处于类型检查模式，导入所需的配置、tokenizer 和模型类
if TYPE_CHECKING:
    from .configuration_mobilebert import (
        MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileBERT 预训练配置文件映射
        MobileBertConfig,  # MobileBERT 配置类
        MobileBertOnnxConfig,  # MobileBERT ONNX 配置类
    )
    from .tokenization_mobilebert import MobileBertTokenizer  # MobileBERT tokenizer
    # 如果 MobileBertTokenizerFast 可用，则从 .tokenization_mobilebert_fast 模块中导入
    else:
        from .tokenization_mobilebert_fast import MobileBertTokenizerFast
    
    # 尝试导入 PyTorch 相关模块
    try:
        # 如果 PyTorch 不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果引发 OptionalDependencyNotAvailable 异常，则什么也不做
    except OptionalDependencyNotAvailable:
        pass
    # 如果 PyTorch 可用，则从 .modeling_mobilebert 模块中导入相关类
    else:
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
    
    # 尝试导入 TensorFlow 相关模块
    try:
        # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果引发 OptionalDependencyNotAvailable 异常，则什么也不做
    except OptionalDependencyNotAvailable:
        pass
    # 如果 TensorFlow 可用，则从 .modeling_tf_mobilebert 模块中导入相关类
    else:
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
# 如果不在指定的条件下，则导入sys模块
import sys
# 将当前模块的名称和相关信息存储到sys.modules中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```