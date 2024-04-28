# `.\transformers\models\auto\modeling_auto.py`

```
# 指定编码为 UTF-8，确保代码中的 Unicode 字符正确解析
# 版权声明，声明代码版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 无论明示或暗示，软件不附带任何保证或条件
# 请参阅许可证以获取特定语言的权限

# 导入警告模块，用于处理警告信息
import warnings
# 导入有序字典模块，用于定义有序的映射关系
from collections import OrderedDict

# 导入日志记录工具
from ...utils import logging
# 导入自动模型工厂相关模块
from .auto_factory import (
    _BaseAutoBackboneClass,
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
)
# 导入自动模型配置相关模块
from .configuration_auto import CONFIG_MAPPING_NAMES

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义模型映射名称的有序字典
MODEL_MAPPING_NAMES = OrderedDict(
    # 未定义任何内容，保留为空
)

# 定义用于预训练模型的映射名称的有序字典
MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    # 未定义任何内容，保留为空
)

# 定义带有语言模型头部的模型映射名称的有序字典
MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    # 未定义任何内容，保留为空
)

# 定义用于因果语言建模的模型映射名称的有序字典
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    # 未定义任何内容，保留为空
)
    [
        # 定义了不同模型名称和对应的 Causal LM 类型
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("code_llama", "LlamaForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        ("ernie", "ErnieForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("git", "GitForCausalLM"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("mega", "MegaForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("open-llama", "OpenLlamaForCausalLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qdqbert", "QDQBertLMHeadModel"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text_2", "Speech2Text2ForCausalLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForCausalLM"),
    ]
# 定义模型名称到模型类的映射字典，用于掩蔽图像建模
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "DeiTForMaskedImageModeling"),
        ("focalnet", "FocalNetForMaskedImageModeling"),
        ("swin", "SwinForMaskedImageModeling"),
        ("swinv2", "Swinv2ForMaskedImageModeling"),
        ("vit", "ViTForMaskedImageModeling"),
    ]
)

# 定义模型名称到模型类的映射字典，用于因果图像建模
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    # Model for Causal Image Modeling mapping
    [
        ("imagegpt", "ImageGPTForCausalImageModeling"),
    ]
)

# 定义模型名称到模型类的映射字典，用于图像分类
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("beit", "BeitForImageClassification"),
        ("bit", "BitForImageClassification"),
        ("convnext", "ConvNextForImageClassification"),
        ("convnextv2", "ConvNextV2ForImageClassification"),
        ("cvt", "CvtForImageClassification"),
        ("data2vec-vision", "Data2VecVisionForImageClassification"),
        (
            "deit",
            ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher"),
        ),
        ("dinat", "DinatForImageClassification"),
        ("dinov2", "Dinov2ForImageClassification"),
        (
            "efficientformer",
            (
                "EfficientFormerForImageClassification",
                "EfficientFormerForImageClassificationWithTeacher",
            ),
        ),
        ("efficientnet", "EfficientNetForImageClassification"),
        ("focalnet", "FocalNetForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),
        ("mobilenet_v1", "MobileNetV1ForImageClassification"),
        ("mobilenet_v2", "MobileNetV2ForImageClassification"),
        ("mobilevit", "MobileViTForImageClassification"),
        ("mobilevitv2", "MobileViTV2ForImageClassification"),
        ("nat", "NatForImageClassification"),
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
            ),
        ),
        ("poolformer", "PoolFormerForImageClassification"),
        ("pvt", "PvtForImageClassification"),
        ("regnet", "RegNetForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("swiftformer", "SwiftFormerForImageClassification"),
        ("swin", "SwinForImageClassification"),
        ("swinv2", "Swinv2ForImageClassification"),
        ("van", "VanForImageClassification"),
        ("vit", "ViTForImageClassification"),
        ("vit_hybrid", "ViTHybridForImageClassification"),
        ("vit_msn", "ViTMSNForImageClassification"),
    ]
)

# 定义模型名称到模型类的映射字典，用于图像分割
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 不要在这里添加新模型，此类将来会被弃用。
        # 用于图像分割映射的模型
        ("detr", "DetrForSegmentation"),
    ]
# 定义一个有序字典，用于将模型名称映射到语义分割任务的模型类名
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 语义分割模型映射
        ("beit", "BeitForSemanticSegmentation"),  # Beit 模型的语义分割版本
        ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),  # Data2VecVision 模型的语义分割版本
        ("dpt", "DPTForSemanticSegmentation"),  # DPT 模型的语义分割版本
        ("mobilenet_v2", "MobileNetV2ForSemanticSegmentation"),  # MobileNetV2 模型的语义分割版本
        ("mobilevit", "MobileViTForSemanticSegmentation"),  # MobileViT 模型的语义分割版本
        ("mobilevitv2", "MobileViTV2ForSemanticSegmentation"),  # MobileViTV2 模型的语义分割版本
        ("segformer", "SegformerForSemanticSegmentation"),  # Segformer 模型的语义分割版本
        ("upernet", "UperNetForSemanticSegmentation"),  # UperNet 模型的语义分割版本
    ]
)

# 定义一个有序字典，用于将模型名称映射到实例分割任务的模型类名
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 实例分割模型映射
        # 在版本 v5 中可以将 MaskFormerForInstanceSegmentation 从此映射中移除
        ("maskformer", "MaskFormerForInstanceSegmentation"),  # MaskFormer 的实例分割版本
    ]
)

# 定义一个有序字典，用于将模型名称映射到通用分割任务的模型类名
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 通用分割模型映射
        ("detr", "DetrForSegmentation"),  # Detr 模型的分割版本
        ("mask2former", "Mask2FormerForUniversalSegmentation"),  # Mask2Former 模型的通用分割版本
        ("maskformer", "MaskFormerForInstanceSegmentation"),  # MaskFormer 模型的实例分割版本
        ("oneformer", "OneFormerForUniversalSegmentation"),  # OneFormer 模型的通用分割版本
    ]
)

# 定义一个有序字典，用于将模型名称映射到视频分类任务的模型类名
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 视频分类模型映射
        ("timesformer", "TimesformerForVideoClassification"),  # Timesformer 模型的视频分类版本
        ("videomae", "VideoMAEForVideoClassification"),  # VideoMAE 模型的视频分类版本
        ("vivit", "VivitForVideoClassification"),  # Vivit 模型的视频分类版本
    ]
)

# 定义一个有序字典，用于将模型名称映射到视觉到序列任务的模型类名
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        # 视觉到序列模型映射
        ("blip", "BlipForConditionalGeneration"),  # Blip 模型的条件生成版本
        ("blip-2", "Blip2ForConditionalGeneration"),  # Blip2 模型的条件生成版本
        ("git", "GitForCausalLM"),  # Git 模型的因果语言建模版本
        ("instructblip", "InstructBlipForConditionalGeneration"),  # InstructBlip 模型的条件生成版本
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),  # Kosmos2 模型的条件生成版本
        ("llava", "LlavaForConditionalGeneration"),  # Llava 模型的条件生成版本
        ("pix2struct", "Pix2StructForConditionalGeneration"),  # Pix2Struct 模型的条件生成版本
        ("vipllava", "VipLlavaForConditionalGeneration"),  # VipLlava 模型的条件生成版本
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),  # 视觉编码解码模型
    ]
)

# 定义一个有序字典，用于将模型名称映射到遮蔽语言模型任务的模型类名
MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # 创建一个列表，其中每个元素是一个元组，包含了模型名称和对应的模型类名称
        # 模型名称对应的是一种预训练语言模型
        # 模型类名称对应的是该语言模型的掩码语言建模（Masked LM）模型类
        # 这个列表用于将模型名称映射到其相应的掩码语言建模模型类
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("big_bird", "BigBirdForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("ernie", "ErnieForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("flaubert", "FlaubertWithLMHeadModel"),
        ("fnet", "FNetForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("mega", "MegaForMaskedLM"),
        ("megatron-bert", "MegatronBertForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mra", "MraForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nezha", "NezhaForMaskedLM"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("perceiver", "PerceiverForMaskedLM"),
        ("qdqbert", "QDQBertForMaskedLM"),
        ("reformer", "ReformerForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xmod", "XmodForMaskedLM"),
        ("yoso", "YosoForMaskedLM"),
    ]
# 定义用于对象检测的模型名称映射字典
MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # 对象检测模型映射
        ("conditional_detr", "ConditionalDetrForObjectDetection"),
        ("deformable_detr", "DeformableDetrForObjectDetection"),
        ("deta", "DetaForObjectDetection"),
        ("detr", "DetrForObjectDetection"),
        ("table-transformer", "TableTransformerForObjectDetection"),
        ("yolos", "YolosForObjectDetection"),
    ]
)

# 定义用于零样本对象检测的模型名称映射字典
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # 零样本对象检测模型映射
        ("owlv2", "Owlv2ForObjectDetection"),
        ("owlvit", "OwlViTForObjectDetection"),
    ]
)

# 定义用于深度估计的模型名称映射字典
MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # 深度估计模型映射
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
    ]
)

# 定义用于序列到序列因果语言建模的模型名称映射字典
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # 序列到序列因果语言建模模型映射
        ("bart", "BartForConditionalGeneration"),
        ("bigbird_pegasus", "BigBirdPegasusForConditionalGeneration"),
        ("blenderbot", "BlenderbotForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("gptsan-japanese", "GPTSanJapaneseForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("mbart", "MBartForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("prophetnet", "ProphetNetForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToText"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToText"),
        ("switch_transformers", "SwitchTransformersForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("umt5", "UMT5ForConditionalGeneration"),
        ("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
    ]
)

# 定义用于语音序列到序列的模型名称映射字典
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForSpeechToText"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForSpeechToText"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderModel"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("speecht5", "SpeechT5ForSpeechToText"),
        ("whisper", "WhisperForConditionalGeneration"),
    ]
)

# 定义用于序列分类的模型名称映射字典
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    # 空的序列分类模型映射字典
    []
)
# 定义用于问题回答的模型名称映射字典
MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    ]
)

# 定义用于表格问题回答的模型名称映射字典
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

# 定义用于视觉问题回答的模型名称映射字典
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("vilt", "ViltForQuestionAnswering"),
    ]
)

# 定义用于文档问题回答的模型名称映射字典
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
    ]
)

# 定义用于标记分类的模型名称映射字典
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    ]
)

# 定义用于多项选择的模型名称映射字典
MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "AlbertForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("big_bird", "BigBirdForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("data2vec-text", "Data2VecTextForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("ernie", "ErnieForMultipleChoice"),
        ("ernie_m", "ErnieMForMultipleChoice"),
        ("flaubert", "FlaubertForMultipleChoice"),
        ("fnet", "FNetForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("luke", "LukeForMultipleChoice"),
        ("mega", "MegaForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("mra", "MraForMultipleChoice"),
        ("nezha", "NezhaForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("qdqbert", "QDQBertForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMultipleChoice"),
        ("roc_bert", "RoCBertForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
        ("xlm", "XLMForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
        ("xlnet", "XLNetForMultipleChoice"),
        ("xmod", "XmodForMultipleChoice"),
        ("yoso", "YosoForMultipleChoice"),
    ]
)

# 定义用于下一个句子预测的模型名称映射字典
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    # 创建一个包含元组的列表，每个元组包含两个字符串，代表模型的名称和类名
    [
        ("bert", "BertForNextSentencePrediction"),  # BERT 模型的名称和类名
        ("ernie", "ErnieForNextSentencePrediction"),  # ERNIE 模型的名称和类名
        ("fnet", "FNetForNextSentencePrediction"),  # FNet 模型的名称和类名
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),  # Megatron-BERT 模型的名称和类名
        ("mobilebert", "MobileBertForNextSentencePrediction"),  # MobileBERT 模型的名称和类名
        ("nezha", "NezhaForNextSentencePrediction"),  # Nezha 模型的名称和类名
        ("qdqbert", "QDQBertForNextSentencePrediction"),  # QDQBert 模型的名称和类名
    ]
# 模型用于音频分类的映射名称
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 音频分类模型映射
        ("audio-spectrogram-transformer", "ASTForAudioClassification"),  # AST 音频分类
        ("data2vec-audio", "Data2VecAudioForSequenceClassification"),    # Data2Vec 音频分类
        ("hubert", "HubertForSequenceClassification"),                   # Hubert 音频分类
        ("sew", "SEWForSequenceClassification"),                         # SEW 音频分类
        ("sew-d", "SEWDForSequenceClassification"),                      # SEW-D 音频分类
        ("unispeech", "UniSpeechForSequenceClassification"),             # UniSpeech 音频分类
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),     # UniSpeech-SAT 音频分类
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),              # Wav2Vec2 音频分类
        ("wav2vec2-bert", "Wav2Vec2BertForSequenceClassification"),      # Wav2Vec2-BERT 音频分类
        ("wav2vec2-conformer", "Wav2Vec2ConformerForSequenceClassification"),  # Wav2Vec2-Conformer 音频分类
        ("wavlm", "WavLMForSequenceClassification"),                     # WavLM 音频分类
        ("whisper", "WhisperForAudioClassification"),                    # Whisper 音频分类
    ]
)

# 模型用于 CTC（连接主义时序分类）的映射名称
MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # CTC（连接主义时序分类）模型映射
        ("data2vec-audio", "Data2VecAudioForCTC"),              # Data2Vec CTC
        ("hubert", "HubertForCTC"),                             # Hubert CTC
        ("mctct", "MCTCTForCTC"),                               # MCTCT CTC
        ("sew", "SEWForCTC"),                                   # SEW CTC
        ("sew-d", "SEWDForCTC"),                                # SEW-D CTC
        ("unispeech", "UniSpeechForCTC"),                       # UniSpeech CTC
        ("unispeech-sat", "UniSpeechSatForCTC"),               # UniSpeech-SAT CTC
        ("wav2vec2", "Wav2Vec2ForCTC"),                         # Wav2Vec2 CTC
        ("wav2vec2-bert", "Wav2Vec2BertForCTC"),                # Wav2Vec2-BERT CTC
        ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),      # Wav2Vec2-Conformer CTC
        ("wavlm", "WavLMForCTC"),                               # WavLM CTC
    ]
)

# 模型用于音频帧分类的映射名称
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 音频帧分类模型映射
        ("data2vec-audio", "Data2VecAudioForAudioFrameClassification"),              # Data2Vec 音频帧分类
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),               # UniSpeech-SAT 音频帧分类
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),                        # Wav2Vec2 音频帧分类
        ("wav2vec2-bert", "Wav2Vec2BertForAudioFrameClassification"),                # Wav2Vec2-BERT 音频帧分类
        ("wav2vec2-conformer", "Wav2Vec2ConformerForAudioFrameClassification"),      # Wav2Vec2-Conformer 音频帧分类
        ("wavlm", "WavLMForAudioFrameClassification"),                               # WavLM 音频帧分类
    ]
)

# 模型用于音频 X 矢量的映射名称
MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        # 音频 X 矢量模型映射
        ("data2vec-audio", "Data2VecAudioForXVector"),                  # Data2Vec 音频 X 矢量
        ("unispeech-sat", "UniSpeechSatForXVector"),                    # UniSpeech-SAT 音频 X 矢量
        ("wav2vec2", "Wav2Vec2ForXVector"),                             # Wav2Vec2 音频 X 矢量
        ("wav2vec2-bert", "Wav2Vec2BertForXVector"),                    # Wav2Vec2-BERT 音频 X 矢量
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),          # Wav2Vec2-Conformer 音频 X 矢量
        ("wavlm", "WavLMForXVector"),                                   # WavLM 音频 X 矢量
    ]
)

# 模型用于文本到频谱图的映射名称
MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        # 文本到频谱图模型映射
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),     # FastSpeech2-Conformer 文本到频谱图
        ("speecht5", "SpeechT5ForTextToSpeech"),                    # SpeechT5 文本到频谱图
    ]
)

# 模型用于文本到波形的映射名称
MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict(
    # 定义一个列表，其中包含了模型名称和对应的模型类名
    [
        # 模型名称 "bark" 对应的模型类名 "BarkModel"
        ("bark", "BarkModel"),
        # 模型名称 "fastspeech2_conformer" 对应的模型类名 "FastSpeech2ConformerWithHifiGan"
        ("fastspeech2_conformer", "FastSpeech2ConformerWithHifiGan"),
        # 模型名称 "musicgen" 对应的模型类名 "MusicgenForConditionalGeneration"
        ("musicgen", "MusicgenForConditionalGeneration"),
        # 模型名称 "seamless_m4t" 对应的模型类名 "SeamlessM4TForTextToSpeech"
        ("seamless_m4t", "SeamlessM4TForTextToSpeech"),
        # 模型名称 "seamless_m4t_v2" 对应的模型类名 "SeamlessM4Tv2ForTextToSpeech"
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToSpeech"),
        # 模型名称 "vits" 对应的模型类名 "VitsModel"
        ("vits", "VitsModel"),
    ]
# 导入OrderedDict模块，用于创建有序字典
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Zero Shot Image Classification模型映射
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("blip", "BlipModel"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("clip", "CLIPModel"),
        ("clipseg", "CLIPSegModel"),
        ("siglip", "SiglipModel"),
    ]
)

# 创建Backbone模型映射
MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # Backbone映射
        ("beit", "BeitBackbone"),
        ("bit", "BitBackbone"),
        ("convnext", "ConvNextBackbone"),
        ("convnextv2", "ConvNextV2Backbone"),
        ("dinat", "DinatBackbone"),
        ("dinov2", "Dinov2Backbone"),
        ("focalnet", "FocalNetBackbone"),
        ("maskformer-swin", "MaskFormerSwinBackbone"),
        ("nat", "NatBackbone"),
        ("resnet", "ResNetBackbone"),
        ("swin", "SwinBackbone"),
        ("swinv2", "Swinv2Backbone"),
        ("timm_backbone", "TimmBackbone"),
        ("vitdet", "VitDetBackbone"),
    ]
)

# 创建Mask Generation模型映射
MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)

# 创建Text Encoding模型映射
MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "AlbertModel"),
        ("bert", "BertModel"),
        ("big_bird", "BigBirdModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("distilbert", "DistilBertModel"),
        ("electra", "ElectraModel"),
        ("flaubert", "FlaubertModel"),
        ("ibert", "IBertModel"),
        ("longformer", "LongformerModel"),
        ("mobilebert", "MobileBertModel"),
        ("mt5", "MT5EncoderModel"),
        ("nystromformer", "NystromformerModel"),
        ("reformer", "ReformerModel"),
        ("rembert", "RemBertModel"),
        ("roberta", "RobertaModel"),
        ("roberta-prelayernorm", "RobertaPreLayerNormModel"),
        ("roc_bert", "RoCBertModel"),
        ("roformer", "RoFormerModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("t5", "T5EncoderModel"),
        ("umt5", "UMT5EncoderModel"),
        ("xlm", "XLMModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
    ]
)

# 创建Time Series Classification模型映射
MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("patchtsmixer", "PatchTSMixerForTimeSeriesClassification"),
        ("patchtst", "PatchTSTForClassification"),
    ]
)

# 创建Time Series Regression模型映射
MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES = OrderedDict(
    [
        ("patchtsmixer", "PatchTSMixerForRegression"),
        ("patchtst", "PatchTSTForRegression"),
    ]
)

# 创建Image to Image模型映射
MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        ("swin2sr", "Swin2SRForImageSuperResolution"),
    ]
)

# 创建模型映射的_LazyAutoMapping实例
MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
# 创建预训练模型映射的_LazyAutoMapping实例
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
# 定义模型与语言模型头的映射关系
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
# 定义因果语言模型的模型映射关系
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
# 定义因果图像建模的模型映射关系
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES)
# 定义图像分类模型的映射关系
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES)
# 定义零样本图像分类模型的映射关系
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES)
# 定义图像分割模型的映射关系
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES)
# 定义语义分割模型的映射关系
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES)
# 定义实例分割模型的映射关系
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES)
# 定义通用分割模型的映射关系
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)
# 定义视频分类模型的映射关系
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)
# 定义视觉到序列模型的映射关系
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
# 定义视觉问答模型的映射关系
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES)
# 定义文档问答模型的映射关系
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES)
# 定义遮蔽语言模型的模型映射关系
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)
# 定义遮蔽图像建模的模型映射关系
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES)
# 定义目标检测模型的映射关系
MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES)
# 定义零样本目标检测模型的映射关系
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES)
# 定义深度估计模型的映射关系
MODEL_FOR_DEPTH_ESTIMATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)
# 定义序列到序列因果语言模型的映射关系
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)
# 定义序列分类模型的映射关系
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)
# 定义问题回答模型的映射关系
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)
# 定义表格问题回答模型的映射关系
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES)
# 定义标记分类模型的映射关系
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES)
# 定义模型类型与模型类之间的映射，用于自动加载模型
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES)

# 定义文本转频谱模型类型与模型类之间的映射
MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
)

# 定义文本转波形模型类型与模型类之间的映射
MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES)

# 定义骨干网络模型类型与模型类之间的映射
MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES)

# 定义生成掩码模型类型与模型类之间的映射
MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)

# 定义文本编码模型类型与模型类之间的映射
MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)

# 定义时间序列分类模型类型与模型类之间的映射
MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES
)

# 定义时间序列回归模型类型与模型类之间的映射
MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES
)

# 定义图像到图像模型类型与模型类之间的映射
MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)

# 自动加载生成掩码模型的类
class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING

# 自动加载文本编码模型的类
class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING

# 自动加载图像到图像模型的类
class AutoModelForImageToImage(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING

# 自动加载模型的基类
class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING

# 自动加载预训练模型的类
AutoModel = auto_class_update(AutoModel)

# 自动加载用于预训练的模型的类
class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING

# 自动加载具有语言模型头的模型的类
_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")

# 自动加载具有因果语言模型头的模型的类
class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

# 自动加载具有因果语言模型头的模型的类，并添加头部文档
AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")
# 自动创建用于掩码语言建模的模型类
class AutoModelForMaskedLM(_BaseAutoModelClass):
    # 模型映射为掩码语言建模模型
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


# 更新自动模型类以适应掩码语言建模任务，添加头部文档说明
AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")


# 自动创建用于序列到序列语言建模的模型类
class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    # 模型映射为序列到序列语言建模模型
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


# 更新自动模型类以适应序列到序列语言建模任务，添加头部文档说明，示例检查点为"t5-base"
AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="t5-base",
)


# 自动创建用于序列分类任务的模型类
class AutoModelForSequenceClassification(_BaseAutoModelClass):
    # 模型映射为序列分类模型
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


# 更新自动模型类以适应序列分类任务，添加头部文档说明
AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification"
)


# 自动创建用于问答任务的模型类
class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    # 模型映射为问答模型
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


# 更新自动模型类以适应问答任务，添加头部文档说明
AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


# 自动创建用于表格问答任务的模型类
class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    # 模型映射为表格问答模型
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


# 更新自动模型类以适应表格问答任务，添加头部文档说明，示例检查点为"google/tapas-base-finetuned-wtq"
AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


# 自动创建用于视觉问答任务的模型类
class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    # 模型映射为视觉问答模型
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


# 更新自动模型类以适应视觉问答任务，添加头部文档说明，示例检查点为"dandelin/vilt-b32-finetuned-vqa"
AutoModelForVisualQuestionAnswering = auto_class_update(
    AutoModelForVisualQuestionAnswering,
    head_doc="visual question answering",
    checkpoint_for_example="dandelin/vilt-b32-finetuned-vqa",
)


# 自动创建用于文档问答任务的模型类
class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    # 模型映射为文档问答模型
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


# 更新自动模型类以适应文档问答任务，添加头部文档说明，示例检查点为'impira/layoutlm-document-qa"，修订版本为"52e01b3'
AutoModelForDocumentQuestionAnswering = auto_class_update(
    AutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)


# 自动创建用于标记分类任务的模型类
class AutoModelForTokenClassification(_BaseAutoModelClass):
    # 模型映射为标记分类模型
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


# 更新自动模型类以适应标记分类任务，添加头部文档说明
AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


# 自动创建用于多选题任务的模型类
class AutoModelForMultipleChoice(_BaseAutoModelClass):
    # 模型映射为多选题模型
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


# 更新自动模型类以适应多选题任务，添加头部文档说明
AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")


# 自动创建用于下一句预测任务的模型类
class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    # 模型映射为下一句预测模型
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


# 更新自动模型类以适应下一句预测任务，添加头部文档说明
AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


# 自动创建用于图像分类任务的模型类
class AutoModelForImageClassification(_BaseAutoModelClass):
    # 模型映射为图像分类模型
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


# 更新自动模型类以适应图像分类任务，添加头部文档说明
AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")
# 定义一个自动化模型类，用于零样本图像分类
class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    # 将模型映射表设置为零样本图像分类的模型映射表
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"zero-shot image classification"
AutoModelForZeroShotImageClassification = auto_class_update(
    AutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
)


# 定义一个自动化模型类，用于图像分割
class AutoModelForImageSegmentation(_BaseAutoModelClass):
    # 将模型映射表设置为图像分割的模型映射表
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"image segmentation"
AutoModelForImageSegmentation = auto_class_update(AutoModelForImageSegmentation, head_doc="image segmentation")


# 定义一个自动化模型类，用于语义分割
class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    # 将模型映射表设置为语义分割的模型映射表
    _model_mapping = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"semantic segmentation"
AutoModelForSemanticSegmentation = auto_class_update(
    AutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)


# 定义一个自动化模型类，用于通用图像分割
class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    # 将模型映射表设置为通用图像分割的模型映射表
    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"universal image segmentation"
AutoModelForUniversalSegmentation = auto_class_update(
    AutoModelForUniversalSegmentation, head_doc="universal image segmentation"
)


# 定义一个自动化模型类，用于实例分割
class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    # 将模型映射表设置为实例分割的模型映射表
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"instance segmentation"
AutoModelForInstanceSegmentation = auto_class_update(
    AutoModelForInstanceSegmentation, head_doc="instance segmentation"
)


# 定义一个自动化模型类，用于目标检测
class AutoModelForObjectDetection(_BaseAutoModelClass):
    # 将模型映射表设置为目标检测的模型映射表
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


# 更新自动化模型类，添加头部文档说明为"object detection"
AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")


# 定义一个自动化模型类，用于零样本目标检测
class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    # 将模型映射表设置为零样本目标检测的模型映射表
    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING


# 更新自动化模型类，添加头部文档说明为"zero-shot object detection"
AutoModelForZeroShotObjectDetection = auto_class_update(
    AutoModelForZeroShotObjectDetection, head_doc="zero-shot object detection"
)


# 定义一个自动化模型类，用于深度估计
class AutoModelForDepthEstimation(_BaseAutoModelClass):
    # 将模型映射表设置为深度估计的模型映射表
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"depth estimation"
AutoModelForDepthEstimation = auto_class_update(AutoModelForDepthEstimation, head_doc="depth estimation")


# 定义一个自动化模型类，用于视频分类
class AutoModelForVideoClassification(_BaseAutoModelClass):
    # 将模型映射表设置为视频分类的模型映射表
    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"video classification"
AutoModelForVideoClassification = auto_class_update(AutoModelForVideoClassification, head_doc="video classification")


# 定义一个自动化模型类，用于图像到文本的建模
class AutoModelForVision2Seq(_BaseAutoModelClass):
    # 将模型映射表设置为图像到文本的建模的模型映射表
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


# 更新自动化模型类，添加头部文档说明为"vision-to-text modeling"
AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")


# 定义一个自动化模型类，用于音频分类
class AutoModelForAudioClassification(_BaseAutoModelClass):
    # 将模型映射表设置为音频分类的模型映射表
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


# 更新自动化模型类，添加头部文档说明为"audio classification"
AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")


# 定义一个自动化模型类，用于连接主义时间分类
class AutoModelForCTC(_BaseAutoModelClass):
    # 将模型映射表设置为连接主义时间分类的模型映射表
    _model_mapping = MODEL_FOR_CTC_MAPPING


# 更新自动化模型类，添加头部文档说明为"connectionist temporal classification"
AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")


# 定义一个自动化模型类，用于语音序列到序列
class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    # 将一个全局变量指向用于语音序列到序列任务的模型映射字典
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
# 更新 AutoModelForSpeechSeq2Seq 类，并添加头部文档说明为“sequence-to-sequence speech-to-text modeling”
AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)


# 定义 AutoModelForAudioFrameClassification 类，其模型映射为 MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING
class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING


# 更新 AutoModelForAudioFrameClassification 类，并添加头部文档说明为“audio frame (token) classification”
AutoModelForAudioFrameClassification = auto_class_update(
    AutoModelForAudioFrameClassification, head_doc="audio frame (token) classification"
)


# 定义 AutoModelForAudioXVector 类，其模型映射为 MODEL_FOR_AUDIO_XVECTOR_MAPPING
class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING


# 定义 AutoModelForTextToSpectrogram 类，其模型映射为 MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING
class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING


# 定义 AutoModelForTextToWaveform 类，其模型映射为 MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING
class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING


# 定义 AutoBackbone 类，其模型映射为 MODEL_FOR_BACKBONE_MAPPING
class AutoBackbone(_BaseAutoBackboneClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING


# 更新 AutoModelForAudioXVector 类，并添加头部文档说明为“audio retrieval via x-vector”
AutoModelForAudioXVector = auto_class_update(AutoModelForAudioXVector, head_doc="audio retrieval via x-vector")


# 定义 AutoModelForMaskedImageModeling 类，其模型映射为 MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING
class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


# 更新 AutoModelForMaskedImageModeling 类，并添加头部文档说明为“masked image modeling”
AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")


# 定义 AutoModelWithLMHead 类，继承自 _AutoModelWithLMHead 类
class AutoModelWithLMHead(_AutoModelWithLMHead):
    # 根据配置创建对象
    @classmethod
    def from_config(cls, config):
        # 发出警告，提示该类已弃用
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        # 调用父类方法创建对象
        return super().from_config(config)

    # 从预训练模型创建对象
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 发出警告，提示该类已弃用
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        # 调用父类方法创建对象
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
```