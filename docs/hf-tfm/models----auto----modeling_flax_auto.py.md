# `.\models\auto\modeling_flax_auto.py`

```py
# 导入必要的模块和函数
from collections import OrderedDict
# 导入日志记录器
from ...utils import logging
# 导入自动模型工厂相关类和函数
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
# 导入自动配置映射名称
from .configuration_auto import CONFIG_MAPPING_NAMES

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义模型名称到类的映射字典，用OrderedDict确保顺序
FLAX_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # 基础模型映射
        ("albert", "FlaxAlbertModel"),
        ("bart", "FlaxBartModel"),
        ("beit", "FlaxBeitModel"),
        ("bert", "FlaxBertModel"),
        ("big_bird", "FlaxBigBirdModel"),
        ("blenderbot", "FlaxBlenderbotModel"),
        ("blenderbot-small", "FlaxBlenderbotSmallModel"),
        ("bloom", "FlaxBloomModel"),
        ("clip", "FlaxCLIPModel"),
        ("distilbert", "FlaxDistilBertModel"),
        ("electra", "FlaxElectraModel"),
        ("gemma", "FlaxGemmaModel"),
        ("gpt-sw3", "FlaxGPT2Model"),
        ("gpt2", "FlaxGPT2Model"),
        ("gpt_neo", "FlaxGPTNeoModel"),
        ("gptj", "FlaxGPTJModel"),
        ("llama", "FlaxLlamaModel"),
        ("longt5", "FlaxLongT5Model"),
        ("marian", "FlaxMarianModel"),
        ("mbart", "FlaxMBartModel"),
        ("mistral", "FlaxMistralModel"),
        ("mt5", "FlaxMT5Model"),
        ("opt", "FlaxOPTModel"),
        ("pegasus", "FlaxPegasusModel"),
        ("regnet", "FlaxRegNetModel"),
        ("resnet", "FlaxResNetModel"),
        ("roberta", "FlaxRobertaModel"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormModel"),
        ("roformer", "FlaxRoFormerModel"),
        ("t5", "FlaxT5Model"),
        ("vision-text-dual-encoder", "FlaxVisionTextDualEncoderModel"),
        ("vit", "FlaxViTModel"),
        ("wav2vec2", "FlaxWav2Vec2Model"),
        ("whisper", "FlaxWhisperModel"),
        ("xglm", "FlaxXGLMModel"),
        ("xlm-roberta", "FlaxXLMRobertaModel"),
    ]
)

# 定义用于预训练任务的模型名称到类的映射字典，初始化为空OrderedDict
FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # 预训练模型到 Flax 模型类的映射关系列表
    
        # ("albert", "FlaxAlbertForPreTraining") 表示将 "albert" 映射到 Flax 中的 FlaxAlbertForPreTraining 类
        ("albert", "FlaxAlbertForPreTraining"),
    
        # ("bart", "FlaxBartForConditionalGeneration") 表示将 "bart" 映射到 Flax 中的 FlaxBartForConditionalGeneration 类
        ("bart", "FlaxBartForConditionalGeneration"),
    
        # ("bert", "FlaxBertForPreTraining") 表示将 "bert" 映射到 Flax 中的 FlaxBertForPreTraining 类
        ("bert", "FlaxBertForPreTraining"),
    
        # ("big_bird", "FlaxBigBirdForPreTraining") 表示将 "big_bird" 映射到 Flax 中的 FlaxBigBirdForPreTraining 类
        ("big_bird", "FlaxBigBirdForPreTraining"),
    
        # ("electra", "FlaxElectraForPreTraining") 表示将 "electra" 映射到 Flax 中的 FlaxElectraForPreTraining 类
        ("electra", "FlaxElectraForPreTraining"),
    
        # ("longt5", "FlaxLongT5ForConditionalGeneration") 表示将 "longt5" 映射到 Flax 中的 FlaxLongT5ForConditionalGeneration 类
        ("longt5", "FlaxLongT5ForConditionalGeneration"),
    
        # ("mbart", "FlaxMBartForConditionalGeneration") 表示将 "mbart" 映射到 Flax 中的 FlaxMBartForConditionalGeneration 类
        ("mbart", "FlaxMBartForConditionalGeneration"),
    
        # ("mt5", "FlaxMT5ForConditionalGeneration") 表示将 "mt5" 映射到 Flax 中的 FlaxMT5ForConditionalGeneration 类
        ("mt5", "FlaxMT5ForConditionalGeneration"),
    
        # ("roberta", "FlaxRobertaForMaskedLM") 表示将 "roberta" 映射到 Flax 中的 FlaxRobertaForMaskedLM 类
        ("roberta", "FlaxRobertaForMaskedLM"),
    
        # ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMaskedLM") 表示将 "roberta-prelayernorm" 映射到 Flax 中的 FlaxRobertaPreLayerNormForMaskedLM 类
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMaskedLM"),
    
        # ("roformer", "FlaxRoFormerForMaskedLM") 表示将 "roformer" 映射到 Flax 中的 FlaxRoFormerForMaskedLM 类
        ("roformer", "FlaxRoFormerForMaskedLM"),
    
        # ("t5", "FlaxT5ForConditionalGeneration") 表示将 "t5" 映射到 Flax 中的 FlaxT5ForConditionalGeneration 类
        ("t5", "FlaxT5ForConditionalGeneration"),
    
        # ("wav2vec2", "FlaxWav2Vec2ForPreTraining") 表示将 "wav2vec2" 映射到 Flax 中的 FlaxWav2Vec2ForPreTraining 类
        ("wav2vec2", "FlaxWav2Vec2ForPreTraining"),
    
        # ("whisper", "FlaxWhisperForConditionalGeneration") 表示将 "whisper" 映射到 Flax 中的 FlaxWhisperForConditionalGeneration 类
        ("whisper", "FlaxWhisperForConditionalGeneration"),
    
        # ("xlm-roberta", "FlaxXLMRobertaForMaskedLM") 表示将 "xlm-roberta" 映射到 Flax 中的 FlaxXLMRobertaForMaskedLM 类
        ("xlm-roberta", "FlaxXLMRobertaForMaskedLM"),
    ]
# 带有模型名称到对应 Flax 模型类的映射字典，用于 Masked LM 模型
FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # 模型为 Masked LM 时的映射
        ("albert", "FlaxAlbertForMaskedLM"),
        ("bart", "FlaxBartForConditionalGeneration"),
        ("bert", "FlaxBertForMaskedLM"),
        ("big_bird", "FlaxBigBirdForMaskedLM"),
        ("distilbert", "FlaxDistilBertForMaskedLM"),
        ("electra", "FlaxElectraForMaskedLM"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("roberta", "FlaxRobertaForMaskedLM"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMaskedLM"),
        ("roformer", "FlaxRoFormerForMaskedLM"),
        ("xlm-roberta", "FlaxXLMRobertaForMaskedLM"),
    ]
)

# 带有模型名称到对应 Flax 模型类的映射字典，用于 Seq2Seq Causal LM 模型
FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # 模型为 Seq2Seq Causal LM 时的映射
        ("bart", "FlaxBartForConditionalGeneration"),
        ("blenderbot", "FlaxBlenderbotForConditionalGeneration"),
        ("blenderbot-small", "FlaxBlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "FlaxEncoderDecoderModel"),
        ("longt5", "FlaxLongT5ForConditionalGeneration"),
        ("marian", "FlaxMarianMTModel"),
        ("mbart", "FlaxMBartForConditionalGeneration"),
        ("mt5", "FlaxMT5ForConditionalGeneration"),
        ("pegasus", "FlaxPegasusForConditionalGeneration"),
        ("t5", "FlaxT5ForConditionalGeneration"),
    ]
)

# 带有模型名称到对应 Flax 模型类的映射字典，用于图像分类模型
FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 图像分类模型的映射
        ("beit", "FlaxBeitForImageClassification"),
        ("regnet", "FlaxRegNetForImageClassification"),
        ("resnet", "FlaxResNetForImageClassification"),
        ("vit", "FlaxViTForImageClassification"),
    ]
)

# 带有模型名称到对应 Flax 模型类的映射字典，用于 Vision 2 Seq 模型
FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("vision-encoder-decoder", "FlaxVisionEncoderDecoderModel"),
    ]
)

# 带有模型名称到对应 Flax 模型类的映射字典，用于 Causal LM 模型
FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # 模型为 Causal LM 时的映射
        ("bart", "FlaxBartForCausalLM"),
        ("bert", "FlaxBertForCausalLM"),
        ("big_bird", "FlaxBigBirdForCausalLM"),
        ("bloom", "FlaxBloomForCausalLM"),
        ("electra", "FlaxElectraForCausalLM"),
        ("gemma", "FlaxGemmaForCausalLM"),
        ("gpt-sw3", "FlaxGPT2LMHeadModel"),
        ("gpt2", "FlaxGPT2LMHeadModel"),
        ("gpt_neo", "FlaxGPTNeoForCausalLM"),
        ("gptj", "FlaxGPTJForCausalLM"),
        ("llama", "FlaxLlamaForCausalLM"),
        ("mistral", "FlaxMistralForCausalLM"),
        ("opt", "FlaxOPTForCausalLM"),
        ("roberta", "FlaxRobertaForCausalLM"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForCausalLM"),
        ("xglm", "FlaxXGLMForCausalLM"),
        ("xlm-roberta", "FlaxXLMRobertaForCausalLM"),
    ]
)

# 带有模型名称到对应 Flax 模型类的映射字典，用于序列分类模型
FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 定义了一系列元组，每个元组包含两个字符串：
        # 第一个字符串是模型的名称，第二个字符串是用于该模型的序列分类任务的类名
        ("albert", "FlaxAlbertForSequenceClassification"),
        ("bart", "FlaxBartForSequenceClassification"),
        ("bert", "FlaxBertForSequenceClassification"),
        ("big_bird", "FlaxBigBirdForSequenceClassification"),
        ("distilbert", "FlaxDistilBertForSequenceClassification"),
        ("electra", "FlaxElectraForSequenceClassification"),
        ("mbart", "FlaxMBartForSequenceClassification"),
        ("roberta", "FlaxRobertaForSequenceClassification"),
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForSequenceClassification"),
        ("roformer", "FlaxRoFormerForSequenceClassification"),
        ("xlm-roberta", "FlaxXLMRobertaForSequenceClassification"),
    ]
# 定义用于问题回答的模型名称映射字典
FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # 将 "albert" 映射到 FlaxAlbertForQuestionAnswering
        ("albert", "FlaxAlbertForQuestionAnswering"),
        # 将 "bart" 映射到 FlaxBartForQuestionAnswering
        ("bart", "FlaxBartForQuestionAnswering"),
        # 将 "bert" 映射到 FlaxBertForQuestionAnswering
        ("bert", "FlaxBertForQuestionAnswering"),
        # 将 "big_bird" 映射到 FlaxBigBirdForQuestionAnswering
        ("big_bird", "FlaxBigBirdForQuestionAnswering"),
        # 将 "distilbert" 映射到 FlaxDistilBertForQuestionAnswering
        ("distilbert", "FlaxDistilBertForQuestionAnswering"),
        # 将 "electra" 映射到 FlaxElectraForQuestionAnswering
        ("electra", "FlaxElectraForQuestionAnswering"),
        # 将 "mbart" 映射到 FlaxMBartForQuestionAnswering
        ("mbart", "FlaxMBartForQuestionAnswering"),
        # 将 "roberta" 映射到 FlaxRobertaForQuestionAnswering
        ("roberta", "FlaxRobertaForQuestionAnswering"),
        # 将 "roberta-prelayernorm" 映射到 FlaxRobertaPreLayerNormForQuestionAnswering
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForQuestionAnswering"),
        # 将 "roformer" 映射到 FlaxRoFormerForQuestionAnswering
        ("roformer", "FlaxRoFormerForQuestionAnswering"),
        # 将 "xlm-roberta" 映射到 FlaxXLMRobertaForQuestionAnswering
        ("xlm-roberta", "FlaxXLMRobertaForQuestionAnswering"),
    ]
)

# 定义用于标记分类的模型名称映射字典
FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 将 "albert" 映射到 FlaxAlbertForTokenClassification
        ("albert", "FlaxAlbertForTokenClassification"),
        # 将 "bert" 映射到 FlaxBertForTokenClassification
        ("bert", "FlaxBertForTokenClassification"),
        # 将 "big_bird" 映射到 FlaxBigBirdForTokenClassification
        ("big_bird", "FlaxBigBirdForTokenClassification"),
        # 将 "distilbert" 映射到 FlaxDistilBertForTokenClassification
        ("distilbert", "FlaxDistilBertForTokenClassification"),
        # 将 "electra" 映射到 FlaxElectraForTokenClassification
        ("electra", "FlaxElectraForTokenClassification"),
        # 将 "roberta" 映射到 FlaxRobertaForTokenClassification
        ("roberta", "FlaxRobertaForTokenClassification"),
        # 将 "roberta-prelayernorm" 映射到 FlaxRobertaPreLayerNormForTokenClassification
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForTokenClassification"),
        # 将 "roformer" 映射到 FlaxRoFormerForTokenClassification
        ("roformer", "FlaxRoFormerForTokenClassification"),
        # 将 "xlm-roberta" 映射到 FlaxXLMRobertaForTokenClassification
        ("xlm-roberta", "FlaxXLMRobertaForTokenClassification"),
    ]
)

# 定义用于多项选择的模型名称映射字典
FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # 将 "albert" 映射到 FlaxAlbertForMultipleChoice
        ("albert", "FlaxAlbertForMultipleChoice"),
        # 将 "bert" 映射到 FlaxBertForMultipleChoice
        ("bert", "FlaxBertForMultipleChoice"),
        # 将 "big_bird" 映射到 FlaxBigBirdForMultipleChoice
        ("big_bird", "FlaxBigBirdForMultipleChoice"),
        # 将 "distilbert" 映射到 FlaxDistilBertForMultipleChoice
        ("distilbert", "FlaxDistilBertForMultipleChoice"),
        # 将 "electra" 映射到 FlaxElectraForMultipleChoice
        ("electra", "FlaxElectraForMultipleChoice"),
        # 将 "roberta" 映射到 FlaxRobertaForMultipleChoice
        ("roberta", "FlaxRobertaForMultipleChoice"),
        # 将 "roberta-prelayernorm" 映射到 FlaxRobertaPreLayerNormForMultipleChoice
        ("roberta-prelayernorm", "FlaxRobertaPreLayerNormForMultipleChoice"),
        # 将 "roformer" 映射到 FlaxRoFormerForMultipleChoice
        ("roformer", "FlaxRoFormerForMultipleChoice"),
        # 将 "xlm-roberta" 映射到 FlaxXLMRobertaForMultipleChoice
        ("xlm-roberta", "FlaxXLMRobertaForMultipleChoice"),
    ]
)

# 定义用于下一个句子预测的模型名称映射字典
FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        # 将 "bert" 映射到 FlaxBertForNextSentencePrediction
        ("bert", "FlaxBertForNextSentencePrediction"),
    ]
)

# 定义用于语音序列到序列的模型名称映射字典
FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        # 将 "speech-encoder-decoder" 映射到 FlaxSpeechEncoderDecoderModel
        ("speech-encoder-decoder", "FlaxSpeechEncoderDecoderModel"),
        # 将 "whisper" 映射到 FlaxWhisperForConditionalGeneration
        ("whisper", "FlaxWhisperForConditionalGeneration"),
    ]
)

# 定义用于音频分类的模型名称映射字典
FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 将 "whisper" 映射到 FlaxWhisperForAudioClassification
        ("whisper", "FlaxWhisperForAudioClassification"),
    ]
)

# 定义 Flax 模型映射对象，通过 LazyAutoMapping 进行自动映射
FLAX_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_MAPPING_NAMES)
# 定义用于预训练的 Flax 模型映射对象
FLAX_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
# 定义用于遮盖语言模型的 Flax 模型映射对象
FLAX_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MASKED_LM_MAPPING_NAMES)
# 定义用于序列到序列因果语言模型的 Flax 模型映射对象
FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
    # 导入两个变量：CONFIG_MAPPING_NAMES 和 FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING 映射
FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING 映射
FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_CAUSAL_LM_MAPPING 映射
FLAX_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING 映射
FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING 映射
FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING 映射
FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING 映射
FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING 映射
FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING 映射
FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING 映射
FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)

# 定义 FlaxAutoModel 类，并将 _model_mapping 设置为 FLAX_MODEL_MAPPING
class FlaxAutoModel(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModel
FlaxAutoModel = auto_class_update(FlaxAutoModel)

# 定义 FlaxAutoModelForPreTraining 类，并将 _model_mapping 设置为 FLAX_MODEL_FOR_PRETRAINING_MAPPING
class FlaxAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_PRETRAINING_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModelForPreTraining，并设置头部文档为 "pretraining"
FlaxAutoModelForPreTraining = auto_class_update(FlaxAutoModelForPreTraining, head_doc="pretraining")

# 定义 FlaxAutoModelForCausalLM 类，并将 _model_mapping 设置为 FLAX_MODEL_FOR_CAUSAL_LM_MAPPING
class FlaxAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModelForCausalLM，并设置头部文档为 "causal language modeling"
FlaxAutoModelForCausalLM = auto_class_update(FlaxAutoModelForCausalLM, head_doc="causal language modeling")

# 定义 FlaxAutoModelForMaskedLM 类，并将 _model_mapping 设置为 FLAX_MODEL_FOR_MASKED_LM_MAPPING
class FlaxAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_MASKED_LM_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModelForMaskedLM，并设置头部文档为 "masked language modeling"
FlaxAutoModelForMaskedLM = auto_class_update(FlaxAutoModelForMaskedLM, head_doc="masked language modeling")

# 定义 FlaxAutoModelForSeq2SeqLM 类，并将 _model_mapping 设置为 FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
class FlaxAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModelForSeq2SeqLM，并设置头部文档为 "sequence-to-sequence language modeling"，以及示例检查点为 "google-t5/t5-base"
FlaxAutoModelForSeq2SeqLM = auto_class_update(
    FlaxAutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google-t5/t5-base",
)

# 定义 FlaxAutoModelForSequenceClassification 类，并将 _model_mapping 设置为 FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
class FlaxAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModelForSequenceClassification，并设置头部文档为 "sequence classification"
FlaxAutoModelForSequenceClassification = auto_class_update(
    FlaxAutoModelForSequenceClassification, head_doc="sequence classification"
)

# 定义 FlaxAutoModelForQuestionAnswering 类，并将 _model_mapping 设置为 FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING
class FlaxAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING

# 使用 auto_class_update 函数更新 FlaxAutoModelForQuestionAnswering，并设置头部文档为 "question answering"
FlaxAutoModelForQuestionAnswering = auto_class_update(FlaxAutoModelForQuestionAnswering, head_doc="question answering")
# 定义用于标记分类任务的自动化模型类
class FlaxAutoModelForTokenClassification(_BaseAutoModelClass):
    # 指定模型映射到标记分类任务的类别
    _model_mapping = FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


# 更新标记分类任务模型类，添加头部文档说明为"token classification"
FlaxAutoModelForTokenClassification = auto_class_update(
    FlaxAutoModelForTokenClassification, head_doc="token classification"
)


# 定义用于多项选择任务的自动化模型类
class FlaxAutoModelForMultipleChoice(_BaseAutoModelClass):
    # 指定模型映射到多项选择任务的类别
    _model_mapping = FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


# 更新多项选择任务模型类，添加头部文档说明为"multiple choice"
FlaxAutoModelForMultipleChoice = auto_class_update(FlaxAutoModelForMultipleChoice, head_doc="multiple choice")


# 定义用于下一句预测任务的自动化模型类
class FlaxAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    # 指定模型映射到下一句预测任务的类别
    _model_mapping = FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


# 更新下一句预测任务模型类，添加头部文档说明为"next sentence prediction"
FlaxAutoModelForNextSentencePrediction = auto_class_update(
    FlaxAutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


# 定义用于图像分类任务的自动化模型类
class FlaxAutoModelForImageClassification(_BaseAutoModelClass):
    # 指定模型映射到图像分类任务的类别
    _model_mapping = FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


# 更新图像分类任务模型类，添加头部文档说明为"image classification"
FlaxAutoModelForImageClassification = auto_class_update(
    FlaxAutoModelForImageClassification, head_doc="image classification"
)


# 定义用于视觉到文本建模任务的自动化模型类
class FlaxAutoModelForVision2Seq(_BaseAutoModelClass):
    # 指定模型映射到视觉到文本建模任务的类别
    _model_mapping = FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING


# 更新视觉到文本建模任务模型类，添加头部文档说明为"vision-to-text modeling"
FlaxAutoModelForVision2Seq = auto_class_update(FlaxAutoModelForVision2Seq, head_doc="vision-to-text modeling")


# 定义用于语音序列到序列建模任务的自动化模型类
class FlaxAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    # 指定模型映射到语音序列到序列建模任务的类别
    _model_mapping = FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


# 更新语音序列到序列建模任务模型类，添加头部文档说明为"sequence-to-sequence speech-to-text modeling"
FlaxAutoModelForSpeechSeq2Seq = auto_class_update(
    FlaxAutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)
```