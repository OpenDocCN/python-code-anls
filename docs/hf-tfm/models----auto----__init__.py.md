# `.\models\auto\__init__.py`

```
# 引入类型检查标记，用于条件检查时的类型提示
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需内容
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构，包含不同子模块及其对应的导入项列表
_import_structure = {
    "auto_factory": ["get_values"],
    "configuration_auto": ["ALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"],
    "feature_extraction_auto": ["FEATURE_EXTRACTOR_MAPPING", "AutoFeatureExtractor"],
    "image_processing_auto": ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"],
    "processing_auto": ["PROCESSOR_MAPPING", "AutoProcessor"],
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
}

# 尝试检查是否 Torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果 Torch 不可用，则不进行任何操作
else:
    # 如果 Torch 可用，则继续执行以下代码段（未提供完整代码，此处应补充具体操作）
    pass

# 尝试检查是否 TensorFlow 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 如果 TensorFlow 不可用，则不进行任何操作
else:
    # 如果 TensorFlow 可用，则继续执行以下代码段（未提供完整代码，此处应补充具体操作）
    pass
    # 将"modeling_tf_auto"键添加到_import_structure字典中，其对应的值是包含多个字符串的列表
    _import_structure["modeling_tf_auto"] = [
        "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",  # 包含音频分类模型映射的字符串
        "TF_MODEL_FOR_CAUSAL_LM_MAPPING",  # 包含因果语言模型映射的字符串
        "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",  # 包含图像分类模型映射的字符串
        "TF_MODEL_FOR_MASK_GENERATION_MAPPING",  # 包含生成掩码模型映射的字符串
        "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",  # 包含掩码图像建模模型映射的字符串
        "TF_MODEL_FOR_MASKED_LM_MAPPING",  # 包含掩码语言模型映射的字符串
        "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",  # 包含多选题模型映射的字符串
        "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",  # 包含下一句预测模型映射的字符串
        "TF_MODEL_FOR_PRETRAINING_MAPPING",  # 包含预训练模型映射的字符串
        "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",  # 包含问答模型映射的字符串
        "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",  # 包含文档问答模型映射的字符串
        "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",  # 包含语义分割模型映射的字符串
        "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",  # 包含序列到序列因果语言模型映射的字符串
        "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",  # 包含序列分类模型映射的字符串
        "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",  # 包含语音序列到序列模型映射的字符串
        "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",  # 包含表格问答模型映射的字符串
        "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",  # 包含文本编码模型映射的字符串
        "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",  # 包含标记分类模型映射的字符串
        "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",  # 包含视觉到序列模型映射的字符串
        "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",  # 包含零样本图像分类模型映射的字符串
        "TF_MODEL_MAPPING",  # 包含通用模型映射的字符串
        "TF_MODEL_WITH_LM_HEAD_MAPPING",  # 包含带语言模型头部模型映射的字符串
        "TFAutoModel",  # 自动选择模型的通用类
        "TFAutoModelForAudioClassification",  # 自动选择音频分类模型的类
        "TFAutoModelForCausalLM",  # 自动选择因果语言模型的类
        "TFAutoModelForImageClassification",  # 自动选择图像分类模型的类
        "TFAutoModelForMaskedImageModeling",  # 自动选择掩码图像建模模型的类
        "TFAutoModelForMaskedLM",  # 自动选择掩码语言模型的类
        "TFAutoModelForMaskGeneration",  # 自动选择生成掩码模型的类
        "TFAutoModelForMultipleChoice",  # 自动选择多选题模型的类
        "TFAutoModelForNextSentencePrediction",  # 自动选择下一句预测模型的类
        "TFAutoModelForPreTraining",  # 自动选择预训练模型的类
        "TFAutoModelForDocumentQuestionAnswering",  # 自动选择文档问答模型的类
        "TFAutoModelForQuestionAnswering",  # 自动选择问答模型的类
        "TFAutoModelForSemanticSegmentation",  # 自动选择语义分割模型的类
        "TFAutoModelForSeq2SeqLM",  # 自动选择序列到序列语言模型的类
        "TFAutoModelForSequenceClassification",  # 自动选择序列分类模型的类
        "TFAutoModelForSpeechSeq2Seq",  # 自动选择语音序列到序列模型的类
        "TFAutoModelForTableQuestionAnswering",  # 自动选择表格问答模型的类
        "TFAutoModelForTextEncoding",  # 自动选择文本编码模型的类
        "TFAutoModelForTokenClassification",  # 自动选择标记分类模型的类
        "TFAutoModelForVision2Seq",  # 自动选择视觉到序列模型的类
        "TFAutoModelForZeroShotImageClassification",  # 自动选择零样本图像分类模型的类
        "TFAutoModelWithLMHead",  # 自动选择带语言模型头部模型的类
    ]
try:
    # 检查是否可用 Flax 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 捕获 OptionalDependencyNotAvailable 异常，不做任何处理
    pass
else:
    # 如果 Flax 可用，则定义 Flax 模型的导入结构
    _import_structure["modeling_flax_auto"] = [
        "FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",
        "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_MASKED_LM_MAPPING",
        "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "FLAX_MODEL_FOR_PRETRAINING_MAPPING",
        "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
        "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",
        "FLAX_MODEL_MAPPING",
        "FlaxAutoModel",
        "FlaxAutoModelForCausalLM",
        "FlaxAutoModelForImageClassification",
        "FlaxAutoModelForMaskedLM",
        "FlaxAutoModelForMultipleChoice",
        "FlaxAutoModelForNextSentencePrediction",
        "FlaxAutoModelForPreTraining",
        "FlaxAutoModelForQuestionAnswering",
        "FlaxAutoModelForSeq2SeqLM",
        "FlaxAutoModelForSequenceClassification",
        "FlaxAutoModelForSpeechSeq2Seq",
        "FlaxAutoModelForTokenClassification",
        "FlaxAutoModelForVision2Seq",
    ]

if TYPE_CHECKING:
    # 若为类型检查模式，则从相应模块导入所需符号
    from .auto_factory import get_values
    from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    from .feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
    from .image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
    from .processing_auto import PROCESSOR_MAPPING, AutoProcessor
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

    try:
        # 检查是否可用 Torch 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 捕获 OptionalDependencyNotAvailable 异常，不做任何处理
        pass

    try:
        # 检查是否可用 TensorFlow 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 捕获 OptionalDependencyNotAvailable 异常，不做任何处理
        pass
    # 如果不是Flax可用状态，则引发OptionalDependencyNotAvailable异常
    else:
        # 从当前目录下的modeling_tf_auto模块导入多个TF模型映射和TF模型类
        from .modeling_tf_auto import (
            TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_MASK_GENERATION_MAPPING,
            TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_TEXT_ENCODING_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForAudioClassification,
            TFAutoModelForCausalLM,
            TFAutoModelForDocumentQuestionAnswering,
            TFAutoModelForImageClassification,
            TFAutoModelForMaskedImageModeling,
            TFAutoModelForMaskedLM,
            TFAutoModelForMaskGeneration,
            TFAutoModelForMultipleChoice,
            TFAutoModelForNextSentencePrediction,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSemanticSegmentation,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForSpeechSeq2Seq,
            TFAutoModelForTableQuestionAnswering,
            TFAutoModelForTextEncoding,
            TFAutoModelForTokenClassification,
            TFAutoModelForVision2Seq,
            TFAutoModelForZeroShotImageClassification,
            TFAutoModelWithLMHead,
        )

    # 尝试检测是否Flax可用，如果不可用则捕获异常并忽略
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果导入模块失败，则尝试从当前包的子模块中导入多个符号和名称
    else:
        from .modeling_flax_auto import (
            FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,  # 导入音频分类模型映射
            FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,            # 导入因果语言模型映射
            FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,  # 导入图像分类模型映射
            FLAX_MODEL_FOR_MASKED_LM_MAPPING,            # 导入遮蔽语言模型映射
            FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,      # 导入多选题模型映射
            FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,  # 导入下一句预测模型映射
            FLAX_MODEL_FOR_PRETRAINING_MAPPING,          # 导入预训练模型映射
            FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,   # 导入问答模型映射
            FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, # 导入序列到序列因果语言模型映射
            FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,  # 导入序列分类模型映射
            FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,     # 导入语音序列到序列模型映射
            FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, # 导入标记分类模型映射
            FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,         # 导入视觉到序列模型映射
            FLAX_MODEL_MAPPING,                          # 导入通用模型映射
            FlaxAutoModel,                               # 导入通用 Flax 自动模型
            FlaxAutoModelForCausalLM,                    # 导入因果语言模型的 Flax 自动模型
            FlaxAutoModelForImageClassification,         # 导入图像分类的 Flax 自动模型
            FlaxAutoModelForMaskedLM,                    # 导入遮蔽语言模型的 Flax 自动模型
            FlaxAutoModelForMultipleChoice,              # 导入多选题的 Flax 自动模型
            FlaxAutoModelForNextSentencePrediction,      # 导入下一句预测的 Flax 自动模型
            FlaxAutoModelForPreTraining,                 # 导入预训练的 Flax 自动模型
            FlaxAutoModelForQuestionAnswering,           # 导入问答的 Flax 自动模型
            FlaxAutoModelForSeq2SeqLM,                   # 导入序列到序列语言模型的 Flax 自动模型
            FlaxAutoModelForSequenceClassification,      # 导入序列分类的 Flax 自动模型
            FlaxAutoModelForSpeechSeq2Seq,               # 导入语音序列到序列的 Flax 自动模型
            FlaxAutoModelForTokenClassification,         # 导入标记分类的 Flax 自动模型
            FlaxAutoModelForVision2Seq,                  # 导入视觉到序列的 Flax 自动模型
        )
else:
    # 导入 sys 模块，用于动态配置当前模块
    import sys

    # 将当前模块的名称和其他相关信息交给 _LazyModule 类处理，并赋值给 sys.modules 中的当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```