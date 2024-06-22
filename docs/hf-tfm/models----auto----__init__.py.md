# `.\transformers\models\auto\__init__.py`

```py
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可证要求，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "auto_factory": ["get_values"],
    "configuration_auto": ["ALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"],
    "feature_extraction_auto": ["FEATURE_EXTRACTOR_MAPPING", "AutoFeatureExtractor"],
    "image_processing_auto": ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"],
    "processing_auto": ["PROCESSOR_MAPPING", "AutoProcessor"],
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
}

# 检查是否存在 torch 库，如果不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则执行以下代码块
    # 这里可以添加针对 torch 库的操作

# 检查是否存在 tensorflow 库，如果不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow 库，则执行以下代码块
    # 这里可以添加针对 tensorflow 库的操作
    # 将模型文件名与模型类的映射添加到_import_structure字典中，用于自动导入
    _import_structure["modeling_tf_auto"] = [
        "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",  # TF模型用于音频分类的映射
        "TF_MODEL_FOR_CAUSAL_LM_MAPPING",  # TF模型用于因果语言模型的映射
        "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",  # TF模型用于图像分类的映射
        "TF_MODEL_FOR_MASK_GENERATION_MAPPING",  # TF模型用于掩码生成的映射
        "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",  # TF模型用于掩码图像建模的映射
        "TF_MODEL_FOR_MASKED_LM_MAPPING",  # TF模型用于掩码语言模型的映射
        "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",  # TF模型用于多项选择的映射
        "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",  # TF模型用于下一句预测的映射
        "TF_MODEL_FOR_PRETRAINING_MAPPING",  # TF模型用于预训练的映射
        "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",  # TF模型用于问答的映射
        "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",  # TF模型用于文档问答的映射
        "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",  # TF模型用于语义分割的映射
        "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",  # TF模型用于序列到序列因果语言模型的映射
        "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",  # TF模型用于序列分类的映射
        "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",  # TF模型用于语音序列到序列的映射
        "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",  # TF模型用于表格问答的映射
        "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",  # TF模型用于文本编码的映射
        "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",  # TF模型用于标记分类的映射
        "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",  # TF模型用于视觉到序列的映射
        "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",  # TF模型用于零样本图像分类的映射
        "TF_MODEL_MAPPING",  # TF模型映射
        "TF_MODEL_WITH_LM_HEAD_MAPPING",  # 带有语言模型头部的TF模型映射
        "TFAutoModel",  # TF自动模型
        "TFAutoModelForAudioClassification",  # 用于音频分类的TF自动模型
        "TFAutoModelForCausalLM",  # 用于因果语言模型的TF自动模型
        "TFAutoModelForImageClassification",  # 用于图像分类的TF自动模型
        "TFAutoModelForMaskedImageModeling",  # 用于掩码图像建模的TF自动模型
        "TFAutoModelForMaskedLM",  # 用于掩码语言模型的TF自动模型
        "TFAutoModelForMaskGeneration",  # 用于掩码生成的TF自动模型
        "TFAutoModelForMultipleChoice",  # 用于多项选择的TF自动模型
        "TFAutoModelForNextSentencePrediction",  # 用于下一句预测的TF自动模型
        "TFAutoModelForPreTraining",  # 用于预训练的TF自动模型
        "TFAutoModelForDocumentQuestionAnswering",  # 用于文档问答的TF自动模型
        "TFAutoModelForQuestionAnswering",  # 用于问答的TF自动模型
        "TFAutoModelForSemanticSegmentation",  # 用于语义分割的TF自动模型
        "TFAutoModelForSeq2SeqLM",  # 用于序列到序列语言模型的TF自动模型
        "TFAutoModelForSequenceClassification",  # 用于序列分类的TF自动模型
        "TFAutoModelForSpeechSeq2Seq",  # 用于语音序列到序列的TF自动模型
        "TFAutoModelForTableQuestionAnswering",  # 用于表格问答的TF自动模型
        "TFAutoModelForTextEncoding",  # 用于文本编码的TF自动模型
        "TFAutoModelForTokenClassification",  # 用于标记分类的TF自动模型
        "TFAutoModelForVision2Seq",  # 用于视觉到序列的TF自动模型
        "TFAutoModelForZeroShotImageClassification",  # 用于零样本图像分类的TF自动模型
        "TFAutoModelWithLMHead",  # 带有语言模型头部的TF自动模型
    ]
# 尝试导入可选依赖项，并在导入失败时引发异常 OptionalDependencyNotAvailable
try:
    # 如果不可用，即 is_flax_available() 返回 False
    if not is_flax_available():
        # 抛出可选依赖项不可用的异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果发生 OptionalDependencyNotAvailable 异常，则执行此块代码
    pass
else:
    # 如果没有引发异常，则执行此块代码
    # 将模型导入结构映射添加到 _import_structure 字典中
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


# 如果是类型检查（静态类型检查），则执行以下代码块
if TYPE_CHECKING:
    # 从 .auto_factory 模块中导入 get_values 函数
    from .auto_factory import get_values
    # 从 .configuration_auto 模块中导入以下内容
    from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    # 从 .feature_extraction_auto 模块中导入以下内容
    from .feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING, AutoFeatureExtractor
    # 从 .image_processing_auto 模块中导入以下内容
    from .image_processing_auto import IMAGE_PROCESSOR_MAPPING, AutoImageProcessor
    # 从 .processing_auto 模块中导入以下内容
    from .processing_auto import PROCESSOR_MAPPING, AutoProcessor
    # 从 .tokenization_auto 模块中导入以下内容
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

    # 尝试导入 Torch 库，如果不可用，则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 Torch 不可用，则执行此块代码
        pass
    
    # 尝试导入 TensorFlow 库，如果不可用，则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，则执行此块代码
        pass
    # 如果不是第一个条件，导入以下模块
    else:
        from .modeling_tf_auto import (
            TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,  # 导入 TF 模型用于音频分类的映射
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,  # 导入 TF 模型用于因果语言建模的映射
            TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,  # 导入 TF 模型用于文档问答的映射
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,  # 导入 TF 模型用于图像分类的映射
            TF_MODEL_FOR_MASK_GENERATION_MAPPING,  # 导入 TF 模型用于掩码生成的映射
            TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,  # 导入 TF 模型用于掩码图像建模的映射
            TF_MODEL_FOR_MASKED_LM_MAPPING,  # 导入 TF 模型用于掩码语言建模的映射
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,  # 导入 TF 模型用于多选题的映射
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,  # 导入 TF 模型用于下一个句子预测的映射
            TF_MODEL_FOR_PRETRAINING_MAPPING,  # 导入 TF 模型用于预训练的映射
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,  # 导入 TF 模型用于问答的映射
            TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,  # 导入 TF 模型用于语义分割的映射
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,  # 导入 TF 模型用于序列到序列因果语言建模的映射
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,  # 导入 TF 模型用于序列分类的映射
            TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,  # 导入 TF 模型用于语音序列到序列的映射
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,  # 导入 TF 模型用于表格问答的映射
            TF_MODEL_FOR_TEXT_ENCODING_MAPPING,  # 导入 TF 模型用于文本编码的映射
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,  # 导入 TF 模型用于标记分类的映射
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING,  # 导入 TF 模型用于视觉到序列的映射
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,  # 导入 TF 模型用于零样本图像分类的映射
            TF_MODEL_MAPPING,  # 导入 TF 模型映射
            TF_MODEL_WITH_LM_HEAD_MAPPING,  # 导入带有语言模型头的 TF 模型映射
            TFAutoModel,  # 导入 TF 自动模型
            TFAutoModelForAudioClassification,  # 导入 TF 自动音频分类模型
            TFAutoModelForCausalLM,  # 导入 TF 自动因果语言建模模型
            TFAutoModelForDocumentQuestionAnswering,  # 导入 TF 自动文档问答模型
            TFAutoModelForImageClassification,  # 导入 TF 自动图像分类模型
            TFAutoModelForMaskedImageModeling,  # 导入 TF 自动掩码图像建模模型
            TFAutoModelForMaskedLM,  # 导入 TF 自动掩码语言建模模型
            TFAutoModelForMaskGeneration,  # 导入 TF 自动掩码生成模型
            TFAutoModelForMultipleChoice,  # 导入 TF 自动多选题模型
            TFAutoModelForNextSentencePrediction,  # 导入 TF 自动下一个句子预测模型
            TFAutoModelForPreTraining,  # 导入 TF 自动预训练模型
            TFAutoModelForQuestionAnswering,  # 导入 TF 自动问答模型
            TFAutoModelForSemanticSegmentation,  # 导入 TF 自动语义分割模型
            TFAutoModelForSeq2SeqLM,  # 导入 TF 自动序列到序列语言建模模型
            TFAutoModelForSequenceClassification,  # 导入 TF 自动序列分类模型
            TFAutoModelForSpeechSeq2Seq,  # 导入 TF 自动语音序列到序列模型
            TFAutoModelForTableQuestionAnswering,  # 导入 TF 自动表格问答模型
            TFAutoModelForTextEncoding,  # 导入 TF 自动文本编码模型
            TFAutoModelForTokenClassification,  # 导入 TF 自动标记分类模型
            TFAutoModelForVision2Seq,  # 导入 TF 自动视觉到序列模型
            TFAutoModelForZeroShotImageClassification,  # 导入 TF 自动零样本图像分类模型
            TFAutoModelWithLMHead,  # 导入带有语言模型头的 TF 自动模型
        )

    # 尝试检查是否存在 Flax 库，如果不存在则引发异常并捕获
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，从当前目录下的.modeling_flax_auto模块导入以下模块和映射关系
    from .modeling_flax_auto import (
        FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,  # 音频分类模型的映射
        FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,  # 因果语言模型的映射
        FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,  # 图像分类模型的映射
        FLAX_MODEL_FOR_MASKED_LM_MAPPING,  # 掩码语言模型的映射
        FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,  # 多项选择模型的映射
        FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,  # 下一个句子预测模型的映射
        FLAX_MODEL_FOR_PRETRAINING_MAPPING,  # 预训练模型的映射
        FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,  # 问答模型的映射
        FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,  # 序列到序列因果语言模型的映射
        FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,  # 序列分类模型的映射
        FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,  # 语音序列到序列模型的映射
        FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,  # 标记分类模型的映射
        FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,  # 视觉到序列模型的映射
        FLAX_MODEL_MAPPING,  # 模型映射
        FlaxAutoModel,  # FLAX自动模型
        FlaxAutoModelForCausalLM,  # 因果语言模型的FLAX自动模型
        FlaxAutoModelForImageClassification,  # 图像分类的FLAX自动模型
        FlaxAutoModelForMaskedLM,  # 掩码语言模型的FLAX自动模型
        FlaxAutoModelForMultipleChoice,  # 多项选择的FLAX自动模型
        FlaxAutoModelForNextSentencePrediction,  # 下一个句子预测的FLAX自动模型
        FlaxAutoModelForPreTraining,  # 预训练的FLAX自动模型
        FlaxAutoModelForQuestionAnswering,  # 问答的FLAX自动模型
        FlaxAutoModelForSeq2SeqLM,  # 序列到序列的FLAX自动模型
        FlaxAutoModelForSequenceClassification,  # 序列分类的FLAX自动模型
        FlaxAutoModelForSpeechSeq2Seq,  # 语音序列到序列的FLAX自动模型
        FlaxAutoModelForTokenClassification,  # 标记分类的FLAX自动模型
        FlaxAutoModelForVision2Seq,  # 视觉到序列的FLAX自动模型
    )
```  
# 否则，如果进入这个分支，即当前模块不是一个包，而是一个单独的模块

# 导入 sys 模块，用于访问和操作与 Python 解释器相关的变量和函数

import sys

# 将当前模块的命名空间中的 "__name__" 键对应的值设置为 _LazyModule 类的一个实例，
# 这个实例用于惰性地加载模块

sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```