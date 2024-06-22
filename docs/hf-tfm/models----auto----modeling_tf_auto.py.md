# `.\transformers\models\auto\modeling_tf_auto.py`

```py
# 导入警告模块
import warnings
# 导入有序字典模块
from collections import OrderedDict
# 导入日志记录工具
from ...utils import logging
# 从自动生成的模型工厂中导入基类和惰性自动生成的映射
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
# 从自动生成的配置映射中导入配置名称
from .configuration_auto import CONFIG_MAPPING_NAMES
# 获取日志记录器
logger = logging.get_logger(__name__)
# 定义 TensorFlow 模型的映射名称为有序字典
TF_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # 定义模型名称与对应的 TensorFlow 模型类的映射关系
        ("albert", "TFAlbertModel"),
        ("bart", "TFBartModel"),
        ("bert", "TFBertModel"),
        ("blenderbot", "TFBlenderbotModel"),
        ("blenderbot-small", "TFBlenderbotSmallModel"),
        ("blip", "TFBlipModel"),
        ("camembert", "TFCamembertModel"),
        ("clip", "TFCLIPModel"),
        ("convbert", "TFConvBertModel"),
        ("convnext", "TFConvNextModel"),
        ("convnextv2", "TFConvNextV2Model"),
        ("ctrl", "TFCTRLModel"),
        ("cvt", "TFCvtModel"),
        ("data2vec-vision", "TFData2VecVisionModel"),
        ("deberta", "TFDebertaModel"),
        ("deberta-v2", "TFDebertaV2Model"),
        ("deit", "TFDeiTModel"),
        ("distilbert", "TFDistilBertModel"),
        ("dpr", "TFDPRQuestionEncoder"),
        ("efficientformer", "TFEfficientFormerModel"),
        ("electra", "TFElectraModel"),
        ("esm", "TFEsmModel"),
        ("flaubert", "TFFlaubertModel"),
        ("funnel", ("TFFunnelModel", "TFFunnelBaseModel")),
        ("gpt-sw3", "TFGPT2Model"),
        ("gpt2", "TFGPT2Model"),
        ("gptj", "TFGPTJModel"),
        ("groupvit", "TFGroupViTModel"),
        ("hubert", "TFHubertModel"),
        ("layoutlm", "TFLayoutLMModel"),
        ("layoutlmv3", "TFLayoutLMv3Model"),
        ("led", "TFLEDModel"),
        ("longformer", "TFLongformerModel"),
        ("lxmert", "TFLxmertModel"),
        ("marian", "TFMarianModel"),
        ("mbart", "TFMBartModel"),
        ("mobilebert", "TFMobileBertModel"),
        ("mobilevit", "TFMobileViTModel"),
        ("mpnet", "TFMPNetModel"),
        ("mt5", "TFMT5Model"),
        ("openai-gpt", "TFOpenAIGPTModel"),
        ("opt", "TFOPTModel"),
        ("pegasus", "TFPegasusModel"),
        ("regnet", "TFRegNetModel"),
        ("rembert", "TFRemBertModel"),
        ("resnet", "TFResNetModel"),
        ("roberta", "TFRobertaModel"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormModel"),
        ("roformer", "TFRoFormerModel"),
        ("sam", "TFSamModel"),
        ("segformer", "TFSegformerModel"),
        ("speech_to_text", "TFSpeech2TextModel"),
        ("swin", "TFSwinModel"),
        ("t5", "TFT5Model"),
        ("tapas", "TFTapasModel"),
        ("transfo-xl", "TFTransfoXLModel"),
        ("vision-text-dual-encoder", "TFVisionTextDualEncoderModel"),
        ("vit", "TFViTModel"),
        ("vit_mae", "TFViTMAEModel"),
        ("wav2vec2", "TFWav2Vec2Model"),
        ("whisper", "TFWhisperModel"),
        ("xglm", "TFXGLMModel"),
        ("xlm", "TFXLMModel"),
        ("xlm-roberta", "TFXLMRobertaModel"),
        ("xlnet", "TFXLNetModel"),
    ]
# 导入 OrderedDict 类，用于创建有序字典
from collections import OrderedDict

# 包含了各种模型和对应的 TensorFlow 类名，用于预训练映射
TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # 模型名 "albert" 对应的 TensorFlow 类名 "TFAlbertForPreTraining"
        ("albert", "TFAlbertForPreTraining"),
        # 模型名 "bart" 对应的 TensorFlow 类名 "TFBartForConditionalGeneration"
        ("bart", "TFBartForConditionalGeneration"),
        # 模型名 "bert" 对应的 TensorFlow 类名 "TFBertForPreTraining"
        ("bert", "TFBertForPreTraining"),
        # 模型名 "camembert" 对应的 TensorFlow 类名 "TFCamembertForMaskedLM"
        ("camembert", "TFCamembertForMaskedLM"),
        # 模型名 "ctrl" 对应的 TensorFlow 类名 "TFCTRLLMHeadModel"
        ("ctrl", "TFCTRLLMHeadModel"),
        # 模型名 "distilbert" 对应的 TensorFlow 类名 "TFDistilBertForMaskedLM"
        ("distilbert", "TFDistilBertForMaskedLM"),
        # 模型名 "electra" 对应的 TensorFlow 类名 "TFElectraForPreTraining"
        ("electra", "TFElectraForPreTraining"),
        # 模型名 "flaubert" 对应的 TensorFlow 类名 "TFFlaubertWithLMHeadModel"
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        # 模型名 "funnel" 对应的 TensorFlow 类名 "TFFunnelForPreTraining"
        ("funnel", "TFFunnelForPreTraining"),
        # 模型名 "gpt-sw3" 对应的 TensorFlow 类名 "TFGPT2LMHeadModel"
        ("gpt-sw3", "TFGPT2LMHeadModel"),
        # 模型名 "gpt2" 对应的 TensorFlow 类名 "TFGPT2LMHeadModel"
        ("gpt2", "TFGPT2LMHeadModel"),
        # 模型名 "layoutlm" 对应的 TensorFlow 类名 "TFLayoutLMForMaskedLM"
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        # 模型名 "lxmert" 对应的 TensorFlow 类名 "TFLxmertForPreTraining"
        ("lxmert", "TFLxmertForPreTraining"),
        # 模型名 "mobilebert" 对应的 TensorFlow 类名 "TFMobileBertForPreTraining"
        ("mobilebert", "TFMobileBertForPreTraining"),
        # 模型名 "mpnet" 对应的 TensorFlow 类名 "TFMPNetForMaskedLM"
        ("mpnet", "TFMPNetForMaskedLM"),
        # 模型名 "openai-gpt" 对应的 TensorFlow 类名 "TFOpenAIGPTLMHeadModel"
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        # 模型名 "roberta" 对应的 TensorFlow 类名 "TFRobertaForMaskedLM"
        ("roberta", "TFRobertaForMaskedLM"),
        # 模型名 "roberta-prelayernorm" 对应的 TensorFlow 类名 "TFRobertaPreLayerNormForMaskedLM"
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),
        # 模型名 "t5" 对应的 TensorFlow 类名 "TFT5ForConditionalGeneration"
        ("t5", "TFT5ForConditionalGeneration"),
        # 模型名 "tapas" 对应的 TensorFlow 类名 "TFTapasForMaskedLM"
        ("tapas", "TFTapasForMaskedLM"),
        # 模型名 "transfo-xl" 对应的 TensorFlow 类名 "TFTransfoXLLMHeadModel"
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        # 模型名 "vit_mae" 对应的 TensorFlow 类名 "TFViTMAEForPreTraining"
        ("vit_mae", "TFViTMAEForPreTraining"),
        # 模型名 "xlm" 对应的 TensorFlow 类名 "TFXLMWithLMHeadModel"
        ("xlm", "TFXLMWithLMHeadModel"),
        # 模型名 "xlm-roberta" 对应的 TensorFlow 类名 "TFXLMRobertaForMaskedLM"
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        # 模型名 "xlnet" 对应的 TensorFlow 类名 "TFXLNetLMHeadModel"
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
)

# 包含了各种模型和对应的 TensorFlow 类名，用于带有 LM heads 的模型映射
TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # 模型名 "albert" 对应的 TensorFlow 类名 "TFAlbertForMaskedLM"
        ("albert", "TFAlbertForMaskedLM"),
        # 模型名 "bart" 对应的 TensorFlow 类名 "TFBartForConditionalGeneration"
        ("bart", "TFBartForConditionalGeneration"),
        # 模型名 "bert" 对应的 TensorFlow 类名 "TFBertForMaskedLM"
        ("bert", "TFBertForMaskedLM"),
        # 模型名 "camembert" 对应的 TensorFlow 类名 "TFCamembertForMaskedLM"
        ("camembert", "TFCamembertForMaskedLM"),
        # 模型名 "convbert" 对应的 TensorFlow 类名 "TFConvBertForMaskedLM"
        ("convbert", "TFConvBertForMaskedLM"),
        # 模型名 "ctrl" 对应的 TensorFlow 类名 "TFCTRLLMHeadModel"
        ("ctrl", "TFCTRLLMHeadModel"),
        # 模型名 "distilbert" 对应的 TensorFlow 类名 "TFDistilBertForMaskedLM"
        ("distilbert", "TFDistilBertForMaskedLM"),
        # 模型名 "electra" 对应的 TensorFlow 类名 "TFElectraForMaskedLM"
        ("electra", "TFElectraForMaskedLM"),
        # 模型名 "esm" 对应的 TensorFlow 类名 "TFEsmForMaskedLM"
        ("esm", "TFEsmForMaskedLM"),
        # 模型名 "flaubert" 对应的 TensorFlow 类名 "TFFlaubertWithLMHeadModel"
        ("flaubert", "TFFlaubert
    [
        # 定义了不同模型和对应的 TensorFlow 模型类名
        ("bert", "TFBertLMHeadModel"),  # BERT 模型对应的 TensorFlow 模型类名
        ("camembert", "TFCamembertForCausalLM"),  # CamemBERT 模型对应的 TensorFlow 模型类名
        ("ctrl", "TFCTRLLMHeadModel"),  # CTRL 模型对应的 TensorFlow 模型类名
        ("gpt-sw3", "TFGPT2LMHeadModel"),  # GPT-SW3 模型对应的 TensorFlow 模型类名
        ("gpt2", "TFGPT2LMHeadModel"),  # GPT-2 模型对应的 TensorFlow 模型类名
        ("gptj", "TFGPTJForCausalLM"),  # GPT-J 模型对应的 TensorFlow 模型类名
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),  # OpenAI GPT 模型对应的 TensorFlow 模型类名
        ("opt", "TFOPTForCausalLM"),  # OPT 模型对应的 TensorFlow 模型类名
        ("rembert", "TFRemBertForCausalLM"),  # RemBERT 模型对应的 TensorFlow 模型类名
        ("roberta", "TFRobertaForCausalLM"),  # RoBERTa 模型对应的 TensorFlow 模型类名
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForCausalLM"),  # RoBERTa with pre-layer normalization 模型对应的 TensorFlow 模型类名
        ("roformer", "TFRoFormerForCausalLM"),  # RoFormer 模型对应的 TensorFlow 模型类名
        ("transfo-xl", "TFTransfoXLLMHeadModel"),  # Transformer-XL 模型对应的 TensorFlow 模型类名
        ("xglm", "TFXGLMForCausalLM"),  # XGLM 模型对应的 TensorFlow 模型类名
        ("xlm", "TFXLMWithLMHeadModel"),  # XLM 模型对应的 TensorFlow 模型类名
        ("xlm-roberta", "TFXLMRobertaForCausalLM"),  # XLM-RoBERTa 模型对应的 TensorFlow 模型类名
        ("xlnet", "TFXLNetLMHeadModel"),  # XLNet 模型对应的 TensorFlow 模型类名
    ]
# 导入OrderedDict模块，用于创建有序字典
from collections import OrderedDict

# 定义TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES，包含模型名称和对应类名的有序字典
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        # deit模型对应TFDeiTForMaskedImageModeling类
        ("deit", "TFDeiTForMaskedImageModeling"),
        # swin模型对应TFSwinForMaskedImageModeling类
        ("swin", "TFSwinForMaskedImageModeling"),
    ]
)

# 定义TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES，包含图像分类模型名称和对应类名的有序字典
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image-classsification
        # convnext模型对应TFConvNextForImageClassification类
        ("convnext", "TFConvNextForImageClassification"),
        # convnextv2模型对应TFConvNextV2ForImageClassification类
        ("convnextv2", "TFConvNextV2ForImageClassification"),
        # cvt模型对应TFCvtForImageClassification类
        ("cvt", "TFCvtForImageClassification"),
        # data2vec-vision模型对应TFData2VecVisionForImageClassification类
        ("data2vec-vision", "TFData2VecVisionForImageClassification"),
        # deit模型对应TFDeiTForImageClassification和TFDeiTForImageClassificationWithTeacher类
        ("deit", ("TFDeiTForImageClassification", "TFDeiTForImageClassificationWithTeacher")),
        # efficientformer模型对应TFEfficientFormerForImageClassification和TFEfficientFormerForImageClassificationWithTeacher类
        (
            "efficientformer",
            ("TFEfficientFormerForImageClassification", "TFEfficientFormerForImageClassificationWithTeacher"),
        ),
        # mobilevit模型对应TFMobileViTForImageClassification类
        ("mobilevit", "TFMobileViTForImageClassification"),
        # regnet模型对应TFRegNetForImageClassification类
        ("regnet", "TFRegNetForImageClassification"),
        # resnet模型对应TFResNetForImageClassification类
        ("resnet", "TFResNetForImageClassification"),
        # segformer模型对应TFSegformerForImageClassification类
        ("segformer", "TFSegformerForImageClassification"),
        # swin模型对应TFSwinForImageClassification类
        ("swin", "TFSwinForImageClassification"),
        # vit模型对应TFViTForImageClassification类
        ("vit", "TFViTForImageClassification"),
    ]
)

# 定义TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES，包含零样本图像分类模型名称和对应类名的有序字典
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        # blip模型对应TFBlipModel类
        ("blip", "TFBlipModel"),
        # clip模型对应TFCLIPModel类
        ("clip", "TFCLIPModel"),
    ]
)

# 定义TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES，包含语义分割模型名称和对应类名的有序字典
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Semantic Segmentation mapping
        # data2vec-vision模型对应TFData2VecVisionForSemanticSegmentation类
        ("data2vec-vision", "TFData2VecVisionForSemanticSegmentation"),
        # mobilevit模型对应TFMobileViTForSemanticSegmentation类
        ("mobilevit", "TFMobileViTForSemanticSegmentation"),
        # segformer模型对应TFSegformerForSemanticSegmentation类
        ("segformer", "TFSegformerForSemanticSegmentation"),
    ]
)

# 定义TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES，包含图像到序列模型名称和对应类名的有序字典
TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        # blip模型对应TFBlipForConditionalGeneration类
        ("blip", "TFBlipForConditionalGeneration"),
        # vision-encoder-decoder模型对应TFVisionEncoderDecoderModel类
        ("vision-encoder-decoder", "TFVisionEncoderDecoderModel"),
    ]
)

# 定义TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES，包含Masked LM模型名称和对应类名的有序字典
TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        # albert模型对应TFAlbertForMaskedLM类
        ("albert", "TFAlbertForMaskedLM"),
        # bert模型对应TFBertForMaskedLM类
        ("bert", "TFBertForMaskedLM"),
        # camembert模型对应TFCamembertForMaskedLM类
        ("camembert", "TFCamembertForMaskedLM"),
        # convbert模型对应TFConvBertForMaskedLM类
        ("convbert", "TFConvBertForMaskedLM"),
        # deberta模型对应TFDebertaForMaskedLM类
        ("deberta", "TFDebertaForMaskedLM"),
        # deberta-v2模型对应TFDebertaV2ForMaskedLM类
        ("deberta-v2", "TFDebertaV2ForMaskedLM"),
        # distilbert模型对应TFDistilBertForMaskedLM类
        ("distilbert", "TFDistilBertForMaskedLM"),
        # electra模型对应TFElectraForMaskedLM类
        ("electra", "TFElectraForMaskedLM"),
        # esm模型对应TFEsmForMaskedLM类
        ("esm", "TFEsmForMaskedLM"),
        # flaubert模型对应TFFlaubertWithLMHeadModel类
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        # funnel模型对应TFFunnelForMaskedLM类
        ("funnel", "TFFunnelForMaskedLM"),
        # layoutlm模型对应TFLayoutLMForMaskedLM类
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        # longformer模型对应TFLongformerForMaskedLM类
        ("longformer", "TFLongformerForMaskedLM"),
        # mobilebert模型对应TFMobileBertForMaskedLM类
        ("mobile
# 定义一个有序字典，用于将模型名称映射到相应的序列到序列因果语言模型类
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # 用于序列到序列因果语言模型的模型映射
        ("bart", "TFBartForConditionalGeneration"),  # BART模型
        ("blenderbot", "TFBlenderbotForConditionalGeneration"),  # Blenderbot模型
        ("blenderbot-small", "TFBlenderbotSmallForConditionalGeneration"),  # 小型Blenderbot模型
        ("encoder-decoder", "TFEncoderDecoderModel"),  # 编码器-解码器模型
        ("led", "TFLEDForConditionalGeneration"),  # LED模型
        ("marian", "TFMarianMTModel"),  # Marian模型
        ("mbart", "TFMBartForConditionalGeneration"),  # MBART模型
        ("mt5", "TFMT5ForConditionalGeneration"),  # MT5模型
        ("pegasus", "TFPegasusForConditionalGeneration"),  # Pegasus模型
        ("t5", "TFT5ForConditionalGeneration"),  # T5模型
    ]
)

# 定义一个有序字典，用于将语音序列到序列模型的名称映射到相应的类
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),  # 语音到文本模型
        ("whisper", "TFWhisperForConditionalGeneration"),  # Whisper模型
    ]
)

# 定义一个有序字典，用于将序列分类模型的名称映射到相应的类
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 用于序列分类的模型映射
        ("albert", "TFAlbertForSequenceClassification"),  # ALBERT模型
        ("bart", "TFBartForSequenceClassification"),  # BART模型
        ("bert", "TFBertForSequenceClassification"),  # BERT模型
        ("camembert", "TFCamembertForSequenceClassification"),  # CamemBERT模型
        ("convbert", "TFConvBertForSequenceClassification"),  # ConvBERT模型
        ("ctrl", "TFCTRLForSequenceClassification"),  # CTRL模型
        ("deberta", "TFDebertaForSequenceClassification"),  # DeBERTa模型
        ("deberta-v2", "TFDebertaV2ForSequenceClassification"),  # DeBERTa-v2模型
        ("distilbert", "TFDistilBertForSequenceClassification"),  # DistilBERT模型
        ("electra", "TFElectraForSequenceClassification"),  # ELECTRA模型
        ("esm", "TFEsmForSequenceClassification"),  # ESM模型
        ("flaubert", "TFFlaubertForSequenceClassification"),  # FlauBERT模型
        ("funnel", "TFFunnelForSequenceClassification"),  # Funnel模型
        ("gpt-sw3", "TFGPT2ForSequenceClassification"),  # GPT-SW3模型
        ("gpt2", "TFGPT2ForSequenceClassification"),  # GPT-2模型
        ("gptj", "TFGPTJForSequenceClassification"),  # GPT-J模型
        ("layoutlm", "TFLayoutLMForSequenceClassification"),  # LayoutLM模型
        ("layoutlmv3", "TFLayoutLMv3ForSequenceClassification"),  # LayoutLMv3模型
        ("longformer", "TFLongformerForSequenceClassification"),  # Longformer模型
        ("mobilebert", "TFMobileBertForSequenceClassification"),  # MobileBERT模型
        ("mpnet", "TFMPNetForSequenceClassification"),  # MPNet模型
        ("openai-gpt", "TFOpenAIGPTForSequenceClassification"),  # OpenAI GPT模型
        ("rembert", "TFRemBertForSequenceClassification"),  # RemBERT模型
        ("roberta", "TFRobertaForSequenceClassification"),  # RoBERTa模型
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForSequenceClassification"),  # RoBERTa PreLayerNorm模型
        ("roformer", "TFRoFormerForSequenceClassification"),  # RoFormer模型
        ("tapas", "TFTapasForSequenceClassification"),  # Tapas模型
        ("transfo-xl", "TFTransfoXLForSequenceClassification"),  # TransfoXL模型
        ("xlm", "TFXLMForSequenceClassification"),  # XLM模型
        ("xlm-roberta", "TFXLMRobertaForSequenceClassification"),  # XLM-RoBERTa模型
        ("xlnet", "TFXLNetForSequenceClassification"),  # XLNet模型
    ]
)

# 定义一个有序字典，用于将问答模型的名称映射到相应的类
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    # 这里省略了映射关系的声明，将在下面补充
)
    # 定义了一个列表，包含了各种预训练模型和相应的问题回答模型映射关系
    [
        # 使用 ALBERT 模型的问题回答模型
        ("albert", "TFAlbertForQuestionAnswering"),
        # 使用 BERT 模型的问题回答模型
        ("bert", "TFBertForQuestionAnswering"),
        # 使用 CamemBERT 模型的问题回答模型
        ("camembert", "TFCamembertForQuestionAnswering"),
        # 使用 ConvBERT 模型的问题回答模型
        ("convbert", "TFConvBertForQuestionAnswering"),
        # 使用 DeBERTa 模型的问题回答模型
        ("deberta", "TFDebertaForQuestionAnswering"),
        # 使用 DeBERTa-v2 模型的问题回答模型
        ("deberta-v2", "TFDebertaV2ForQuestionAnswering"),
        # 使用 DistilBERT 模型的问题回答模型
        ("distilbert", "TFDistilBertForQuestionAnswering"),
        # 使用 ELECTRA 模型的问题回答模型
        ("electra", "TFElectraForQuestionAnswering"),
        # 使用 FlauBERT 模型的问题回答模型
        ("flaubert", "TFFlaubertForQuestionAnsweringSimple"),
        # 使用 Funnel 模型的问题回答模型
        ("funnel", "TFFunnelForQuestionAnswering"),
        # 使用 GPT-J 模型的问题回答模型
        ("gptj", "TFGPTJForQuestionAnswering"),
        # 使用 LayoutLMv3 模型的问题回答模型
        ("layoutlmv3", "TFLayoutLMv3ForQuestionAnswering"),
        # 使用 Longformer 模型的问题回答模型
        ("longformer", "TFLongformerForQuestionAnswering"),
        # 使用 MobileBERT 模型的问题回答模型
        ("mobilebert", "TFMobileBertForQuestionAnswering"),
        # 使用 MPNet 模型的问题回答模型
        ("mpnet", "TFMPNetForQuestionAnswering"),
        # 使用 RemBERT 模型的问题回答模型
        ("rembert", "TFRemBertForQuestionAnswering"),
        # 使用 RoBERTa 模型的问题回答模型
        ("roberta", "TFRobertaForQuestionAnswering"),
        # 使用 RoBERTa-prelayernorm 模型的问题回答模型
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForQuestionAnswering"),
        # 使用 RoFormer 模型的问题回答模型
        ("roformer", "TFRoFormerForQuestionAnswering"),
        # 使用 XLM 模型的问题回答模型
        ("xlm", "TFXLMForQuestionAnsweringSimple"),
        # 使用 XLM-RoBERTa 模型的问题回答模型
        ("xlm-roberta", "TFXLMRobertaForQuestionAnswering"),
        # 使用 XLNet 模型的问题回答模型
        ("xlnet", "TFXLNetForQuestionAnsweringSimple"),
    ]
# 定义 TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES 字典，映射模型名称到对应的类名
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict([("wav2vec2", "TFWav2Vec2ForSequenceClassification")])

# 定义 TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES 字典，映射模型名称到对应的类名
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "TFLayoutLMForQuestionAnswering"),
        ("layoutlmv3", "TFLayoutLMv3ForQuestionAnswering"),
    ]
)

# 定义 TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES 字典，映射模型名称到对应的类名
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TFTapasForQuestionAnswering"),
    ]
)

# 定义 TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES 字典，映射模型名称到对应的类名
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "TFAlbertForTokenClassification"),
        ("bert", "TFBertForTokenClassification"),
        ("camembert", "TFCamembertForTokenClassification"),
        ("convbert", "TFConvBertForTokenClassification"),
        ("deberta", "TFDebertaForTokenClassification"),
        ("deberta-v2", "TFDebertaV2ForTokenClassification"),
        ("distilbert", "TFDistilBertForTokenClassification"),
        ("electra", "TFElectraForTokenClassification"),
        ("esm", "TFEsmForTokenClassification"),
        ("flaubert", "TFFlaubertForTokenClassification"),
        ("funnel", "TFFunnelForTokenClassification"),
        ("layoutlm", "TFLayoutLMForTokenClassification"),
        ("layoutlmv3", "TFLayoutLMv3ForTokenClassification"),
        ("longformer", "TFLongformerForTokenClassification"),
        ("mobilebert", "TFMobileBertForTokenClassification"),
        ("mpnet", "TFMPNetForTokenClassification"),
        ("rembert", "TFRemBertForTokenClassification"),
        ("roberta", "TFRobertaForTokenClassification"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForTokenClassification"),
        ("roformer", "TFRoFormerForTokenClassification"),
        ("xlm", "TFXLMForTokenClassification"),
        ("xlm-roberta", "TFXLMRobertaForTokenClassification"),
        ("xlnet", "TFXLNetForTokenClassification"),
    ]
)

# 定义 TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES 字典
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # 定义了不同模型和对应的多选题模型类名的映射关系
        ("albert", "TFAlbertForMultipleChoice"),
        ("bert", "TFBertForMultipleChoice"),
        ("camembert", "TFCamembertForMultipleChoice"),
        ("convbert", "TFConvBertForMultipleChoice"),
        ("deberta-v2", "TFDebertaV2ForMultipleChoice"),
        ("distilbert", "TFDistilBertForMultipleChoice"),
        ("electra", "TFElectraForMultipleChoice"),
        ("flaubert", "TFFlaubertForMultipleChoice"),
        ("funnel", "TFFunnelForMultipleChoice"),
        ("longformer", "TFLongformerForMultipleChoice"),
        ("mobilebert", "TFMobileBertForMultipleChoice"),
        ("mpnet", "TFMPNetForMultipleChoice"),
        ("rembert", "TFRemBertForMultipleChoice"),
        ("roberta", "TFRobertaForMultipleChoice"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMultipleChoice"),
        ("roformer", "TFRoFormerForMultipleChoice"),
        ("xlm", "TFXLMForMultipleChoice"),
        ("xlm-roberta", "TFXLMRobertaForMultipleChoice"),
        ("xlnet", "TFXLNetForMultipleChoice"),
    ]
# 导入 OrderedDict 模块，用于创建有序字典
from collections import OrderedDict

# 定义下一个句子预测模型到 TensorFlow 模型类名的映射关系字典
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        # BERT 模型对应的 TensorFlow 下一个句子预测模型类名
        ("bert", "TFBertForNextSentencePrediction"),
        # MobileBERT 模型对应的 TensorFlow 下一个句子预测模型类名
        ("mobilebert", "TFMobileBertForNextSentencePrediction"),
    ]
)

# 定义掩码生成模型到 TensorFlow 模型类名的映射关系字典
TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        # SAM 模型对应的 TensorFlow 模型类名
        ("sam", "TFSamModel"),
    ]
)

# 定义文本编码模型到 TensorFlow 模型类名的映射关系字典
TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        # ALBERT 模型对应的 TensorFlow 模型类名
        ("albert", "TFAlbertModel"),
        # BERT 模型对应的 TensorFlow 模型类名
        ("bert", "TFBertModel"),
        # ConvBERT 模型对应的 TensorFlow 模型类名
        ("convbert", "TFConvBertModel"),
        # DeBERTa 模型对应的 TensorFlow 模型类名
        ("deberta", "TFDebertaModel"),
        # DeBERTa-v2 模型对应的 TensorFlow 模型类名
        ("deberta-v2", "TFDebertaV2Model"),
        # DistilBERT 模型对应的 TensorFlow 模型类名
        ("distilbert", "TFDistilBertModel"),
        # Electra 模型对应的 TensorFlow 模型类名
        ("electra", "TFElectraModel"),
        # FlauBERT 模型对应的 TensorFlow 模型类名
        ("flaubert", "TFFlaubertModel"),
        # Longformer 模型对应的 TensorFlow 模型类名
        ("longformer", "TFLongformerModel"),
        # MobileBERT 模型对应的 TensorFlow 模型类名
        ("mobilebert", "TFMobileBertModel"),
        # MT5 模型对应的 TensorFlow 模型类名
        ("mt5", "TFMT5EncoderModel"),
        # RemBERT 模型对应的 TensorFlow 模型类名
        ("rembert", "TFRemBertModel"),
        # RoBERTa 模型对应的 TensorFlow 模型类名
        ("roberta", "TFRobertaModel"),
        # RoBERTa-prelayernorm 模型对应的 TensorFlow 模型类名
        ("roberta-prelayernorm", "TFRobertaPreLayerNormModel"),
        # RoFormer 模型对应的 TensorFlow 模型类名
        ("roformer", "TFRoFormerModel"),
        # T5 模型对应的 TensorFlow 模型类名
        ("t5", "TFT5EncoderModel"),
        # XLM 模型对应的 TensorFlow 模型类名
        ("xlm", "TFXLMModel"),
        # XLM-RoBERTa 模型对应的 TensorFlow 模型类名
        ("xlm-roberta", "TFXLMRobertaModel"),
    ]
)

# 定义 TensorFlow 模型的通用映射字典，用于自动映射配置名称到 TensorFlow 模型类名
TF_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_MAPPING_NAMES)
TF_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES)
TF_MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES)
TF_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
)
TF_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
TF_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES)
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    # 导入 CONFIG_MAPPING_NAMES 和 TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES 这两个变量
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 表格问答模型
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 标记分类模型
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 多选模型
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)
# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 下一句预测模型
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 音频分类模型
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)

# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 掩码生成模型
TF_MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES
)

# 创建一个自动映射，将配置名称映射到相应的 TensorFlow 文本编码模型
TF_MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)


# 定义 TFAutoModelForMaskGeneration 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_MASK_GENERATION_MAPPING 自动映射
class TFAutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_MASK_GENERATION_MAPPING


# 定义 TFAutoModelForTextEncoding 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_TEXT_ENCODING_MAPPING 自动映射
class TFAutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_TEXT_ENCODING_MAPPING


# 定义 TFAutoModel 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_MAPPING 自动映射
class TFAutoModel(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_MAPPING

# 对 TFAutoModel 类进行自动更新
TFAutoModel = auto_class_update(TFAutoModel)


# 定义 TFAutoModelForAudioClassification 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING 自动映射
class TFAutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


# 对 TFAutoModelForAudioClassification 类进行自动更新，添加头部文档为 "audio classification"
TFAutoModelForAudioClassification = auto_class_update(
    TFAutoModelForAudioClassification, head_doc="audio classification"
)


# 定义 TFAutoModelForPreTraining 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_PRETRAINING_MAPPING 自动映射
class TFAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_PRETRAINING_MAPPING


# 对 TFAutoModelForPreTraining 类进行自动更新，添加头部文档为 "pretraining"
TFAutoModelForPreTraining = auto_class_update(TFAutoModelForPreTraining, head_doc="pretraining")


# 私有类，故意私有，公共类将添加弃用警告。
class _TFAutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_WITH_LM_HEAD_MAPPING


# 对 _TFAutoModelWithLMHead 类进行自动更新，添加头部文档为 "language modeling"
_TFAutoModelWithLMHead = auto_class_update(_TFAutoModelWithLMHead, head_doc="language modeling")


# 定义 TFAutoModelForCausalLM 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_CAUSAL_LM_MAPPING 自动映射
class TFAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_CAUSAL_LM_MAPPING


# 对 TFAutoModelForCausalLM 类进行自动更新，添加头部文档为 "causal language modeling"
TFAutoModelForCausalLM = auto_class_update(TFAutoModelForCausalLM, head_doc="causal language modeling")


# 定义 TFAutoModelForMaskedImageModeling 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING 自动映射
class TFAutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


# 对 TFAutoModelForMaskedImageModeling 类进行自动更新，添加头部文档为 "masked image modeling"
TFAutoModelForMaskedImageModeling = auto_class_update(
    TFAutoModelForMaskedImageModeling, head_doc="masked image modeling"
)


# 定义 TFAutoModelForImageClassification 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING 自动映射
class TFAutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


# 对 TFAutoModelForImageClassification 类进行自动更新，添加头部文档为 "image classification"
TFAutoModelForImageClassification = auto_class_update(
    TFAutoModelForImageClassification, head_doc="image classification"
)


# 定义 TFAutoModelForZeroShotImageClassification 类，继承自 _BaseAutoModelClass，使用 TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING 自动映射
class TFAutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


# 对 TFAutoModelForZeroShotImageClassification 类进行自动更新
TFAutoModelForZeroShotImageClassification = auto_class_update(
    # 导入 TFAutoModelForZeroShotImageClassification 类，用于零样本图像分类
    TFAutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
class TFAutoModelForSemanticSegmentation(_BaseAutoModelClass):
    # 语义分割模型自动化类
    _model_mapping = TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


# 更新语义分割模型自动化类的文档头
TFAutoModelForSemanticSegmentation = auto_class_update(
    TFAutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)


class TFAutoModelForVision2Seq(_BaseAutoModelClass):
    # 视觉到文本建模模型自动化类
    _model_mapping = TF_MODEL_FOR_VISION_2_SEQ_MAPPING


# 更新视觉到文本建模模型自动化类的文档头
TFAutoModelForVision2Seq = auto_class_update(TFAutoModelForVision2Seq, head_doc="vision-to-text modeling")


class TFAutoModelForMaskedLM(_BaseAutoModelClass):
    # 掩码语言建模模型自动化类
    _model_mapping = TF_MODEL_FOR_MASKED_LM_MAPPING


# 更新掩码语言建模模型自动化类的文档头
TFAutoModelForMaskedLM = auto_class_update(TFAutoModelForMaskedLM, head_doc="masked language modeling")


class TFAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    # 序列到序列因果语言建模模型自动化类
    _model_mapping = TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


# 更新序列到序列因果语言建模模型自动化类的文档头和示例检查点
TFAutoModelForSeq2SeqLM = auto_class_update(
    TFAutoModelForSeq2SeqLM, head_doc="sequence-to-sequence language modeling", checkpoint_for_example="t5-base"
)


class TFAutoModelForSequenceClassification(_BaseAutoModelClass):
    # 序列分类模型自动化类
    _model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


# 更新序列分类模型自动化类的文档头
TFAutoModelForSequenceClassification = auto_class_update(
    TFAutoModelForSequenceClassification, head_doc="sequence classification"
)


class TFAutoModelForQuestionAnswering(_BaseAutoModelClass):
    # 问题回答模型自动化类
    _model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING


# 更新问题回答模型自动化类的文档头
TFAutoModelForQuestionAnswering = auto_class_update(TFAutoModelForQuestionAnswering, head_doc="question answering")


class TFAutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    # 文档问题回答模型自动化类
    _model_mapping = TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


# 更新文档问题回答模型自动化类的文档头和示例检查点
TFAutoModelForDocumentQuestionAnswering = auto_class_update(
    TFAutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)


class TFAutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    # 表格问题回答模型自动化类
    _model_mapping = TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


# 更新表格问题回答模型自动化类的文档头和示例检查点
TFAutoModelForTableQuestionAnswering = auto_class_update(
    TFAutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class TFAutoModelForTokenClassification(_BaseAutoModelClass):
    # 标记分类模型自动化类
    _model_mapping = TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


# 更新标记分类模型自动化类的文档头
TFAutoModelForTokenClassification = auto_class_update(
    TFAutoModelForTokenClassification, head_doc="token classification"
)


class TFAutoModelForMultipleChoice(_BaseAutoModelClass):
    # 多项选择模型自动化类
    _model_mapping = TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


# 更新多项选择模型自动化类的文档头
TFAutoModelForMultipleChoice = auto_class_update(TFAutoModelForMultipleChoice, head_doc="multiple choice")


class TFAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    # 下一句预测模型自动化类
    _model_mapping = TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


# 更新下一句预测模型自动化类的文档头
TFAutoModelForNextSentencePrediction = auto_class_update(
    TFAutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)
# 定义了一个名为 TFAutoModelForSpeechSeq2Seq 的类，继承自 _BaseAutoModelClass 类
class TFAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    # 设置模型映射表，用于自动加载预训练模型
    _model_mapping = TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING

# 使用 auto_class_update 函数更新 TFAutoModelForSpeechSeq2Seq 类的文档字符串
TFAutoModelForSpeechSeq2Seq = auto_class_update(
    TFAutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)

# 定义了一个名为 TFAutoModelWithLMHead 的类，继承自 _TFAutoModelWithLMHead 类
class TFAutoModelWithLMHead(_TFAutoModelWithLMHead):
    # 定义了一个从配置创建模型的类方法
    @classmethod
    def from_config(cls, config):
        # 发出警告，提醒用户该类将来会被移除，并建议使用其他类
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use"
            " `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models"
            " and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        # 调用父类的 from_config 方法
        return super().from_config(config)

    # 定义了一个从预训练模型创建模型的类方法
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 发出警告，提醒用户该类将来会被移除，并建议使用其他类
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use"
            " `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models"
            " and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        # 调用父类的 from_pretrained 方法
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
```  
```