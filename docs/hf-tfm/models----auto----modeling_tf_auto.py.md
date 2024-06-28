# `.\models\auto\modeling_tf_auto.py`

```py
# 指定编码格式为UTF-8

# 版权声明和许可证信息，告知代码的版权和许可使用条款
# 版权归The HuggingFace Inc.团队所有，许可类型为Apache License, Version 2.0
# 除非符合许可证规定，否则不得使用此文件

# 引入警告模块，用于处理警告信息
import warnings

# 引入有序字典模块，用于保存模型映射关系的有序字典
from collections import OrderedDict

# 从当前包中的utils模块中导入logging工具
from ...utils import logging

# 从当前包中的.auto_factory模块导入自动模型工厂的基类和映射类相关函数
from .auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update

# 从当前包中的.configuration_auto模块导入配置映射名称
from .configuration_auto import CONFIG_MAPPING_NAMES

# 获取或创建当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义TensorFlow模型映射名称的有序字典，用于存储模型名称和类别的映射关系
TF_MODEL_MAPPING_NAMES = OrderedDict(
    # 定义一个列表，包含模型名称到对应 TensorFlow 模型类的映射关系
    
    [
        # "albert" 模型对应的 TensorFlow 模型类为 "TFAlbertModel"
        ("albert", "TFAlbertModel"),
        # "bart" 模型对应的 TensorFlow 模型类为 "TFBartModel"
        ("bart", "TFBartModel"),
        # "bert" 模型对应的 TensorFlow 模型类为 "TFBertModel"
        ("bert", "TFBertModel"),
        # "blenderbot" 模型对应的 TensorFlow 模型类为 "TFBlenderbotModel"
        ("blenderbot", "TFBlenderbotModel"),
        # "blenderbot-small" 模型对应的 TensorFlow 模型类为 "TFBlenderbotSmallModel"
        ("blenderbot-small", "TFBlenderbotSmallModel"),
        # "blip" 模型对应的 TensorFlow 模型类为 "TFBlipModel"
        ("blip", "TFBlipModel"),
        # "camembert" 模型对应的 TensorFlow 模型类为 "TFCamembertModel"
        ("camembert", "TFCamembertModel"),
        # "clip" 模型对应的 TensorFlow 模型类为 "TFCLIPModel"
        ("clip", "TFCLIPModel"),
        # "convbert" 模型对应的 TensorFlow 模型类为 "TFConvBertModel"
        ("convbert", "TFConvBertModel"),
        # "convnext" 模型对应的 TensorFlow 模型类为 "TFConvNextModel"
        ("convnext", "TFConvNextModel"),
        # "convnextv2" 模型对应的 TensorFlow 模型类为 "TFConvNextV2Model"
        ("convnextv2", "TFConvNextV2Model"),
        # "ctrl" 模型对应的 TensorFlow 模型类为 "TFCTRLModel"
        ("ctrl", "TFCTRLModel"),
        # "cvt" 模型对应的 TensorFlow 模型类为 "TFCvtModel"
        ("cvt", "TFCvtModel"),
        # "data2vec-vision" 模型对应的 TensorFlow 模型类为 "TFData2VecVisionModel"
        ("data2vec-vision", "TFData2VecVisionModel"),
        # "deberta" 模型对应的 TensorFlow 模型类为 "TFDebertaModel"
        ("deberta", "TFDebertaModel"),
        # "deberta-v2" 模型对应的 TensorFlow 模型类为 "TFDebertaV2Model"
        ("deberta-v2", "TFDebertaV2Model"),
        # "deit" 模型对应的 TensorFlow 模型类为 "TFDeiTModel"
        ("deit", "TFDeiTModel"),
        # "distilbert" 模型对应的 TensorFlow 模型类为 "TFDistilBertModel"
        ("distilbert", "TFDistilBertModel"),
        # "dpr" 模型对应的 TensorFlow 模型类为 "TFDPRQuestionEncoder"
        ("dpr", "TFDPRQuestionEncoder"),
        # "efficientformer" 模型对应的 TensorFlow 模型类为 "TFEfficientFormerModel"
        ("efficientformer", "TFEfficientFormerModel"),
        # "electra" 模型对应的 TensorFlow 模型类为 "TFElectraModel"
        ("electra", "TFElectraModel"),
        # "esm" 模型对应的 TensorFlow 模型类为 "TFEsmModel"
        ("esm", "TFEsmModel"),
        # "flaubert" 模型对应的 TensorFlow 模型类为 "TFFlaubertModel"
        ("flaubert", "TFFlaubertModel"),
        # "funnel" 模型对应的 TensorFlow 模型类为 ("TFFunnelModel", "TFFunnelBaseModel")
        ("funnel", ("TFFunnelModel", "TFFunnelBaseModel")),
        # "gpt-sw3" 模型对应的 TensorFlow 模型类为 "TFGPT2Model"
        ("gpt-sw3", "TFGPT2Model"),
        # "gpt2" 模型对应的 TensorFlow 模型类为 "TFGPT2Model"
        ("gpt2", "TFGPT2Model"),
        # "gptj" 模型对应的 TensorFlow 模型类为 "TFGPTJModel"
        ("gptj", "TFGPTJModel"),
        # "groupvit" 模型对应的 TensorFlow 模型类为 "TFGroupViTModel"
        ("groupvit", "TFGroupViTModel"),
        # "hubert" 模型对应的 TensorFlow 模型类为 "TFHubertModel"
        ("hubert", "TFHubertModel"),
        # "layoutlm" 模型对应的 TensorFlow 模型类为 "TFLayoutLMModel"
        ("layoutlm", "TFLayoutLMModel"),
        # "layoutlmv3" 模型对应的 TensorFlow 模型类为 "TFLayoutLMv3Model"
        ("layoutlmv3", "TFLayoutLMv3Model"),
        # "led" 模型对应的 TensorFlow 模型类为 "TFLEDModel"
        ("led", "TFLEDModel"),
        # "longformer" 模型对应的 TensorFlow 模型类为 "TFLongformerModel"
        ("longformer", "TFLongformerModel"),
        # "lxmert" 模型对应的 TensorFlow 模型类为 "TFLxmertModel"
        ("lxmert", "TFLxmertModel"),
        # "marian" 模型对应的 TensorFlow 模型类为 "TFMarianModel"
        ("marian", "TFMarianModel"),
        # "mbart" 模型对应的 TensorFlow 模型类为 "TFMBartModel"
        ("mbart", "TFMBartModel"),
        # "mobilebert" 模型对应的 TensorFlow 模型类为 "TFMobileBertModel"
        ("mobilebert", "TFMobileBertModel"),
        # "mobilevit" 模型对应的 TensorFlow 模型类为 "TFMobileViTModel"
        ("mobilevit", "TFMobileViTModel"),
        # "mpnet" 模型对应的 TensorFlow 模型类为 "TFMPNetModel"
        ("mpnet", "TFMPNetModel"),
        # "mt5" 模型对应的 TensorFlow 模型类为 "TFMT5Model"
        ("mt5", "TFMT5Model"),
        # "openai-gpt" 模型对应的 TensorFlow 模型类为 "TFOpenAIGPTModel"
        ("openai-gpt", "TFOpenAIGPTModel"),
        # "opt" 模型对应的 TensorFlow 模型类为 "TFOPTModel"
        ("opt", "TFOPTModel"),
        # "pegasus" 模型对应的 TensorFlow 模型类为 "TFPegasusModel"
        ("pegasus", "TFPegasusModel"),
        # "regnet" 模型对应的 TensorFlow 模型类为 "TFRegNetModel"
        ("regnet", "TFRegNetModel"),
        # "rembert" 模型对应的 TensorFlow 模型类为 "TFRemBertModel"
        ("rembert", "TFRemBertModel"),
        # "resnet" 模型对应的 TensorFlow 模型类为 "TFResNetModel"
        ("resnet", "TFResNetModel"),
        # "roberta" 模型对应的 TensorFlow 模型类为 "TFRobertaModel"
        ("roberta", "TFRobertaModel"),
        # "roberta-prelayernorm" 模型对应的 TensorFlow 模型类为 "TFRobertaPreLayerNormModel"
        ("roberta-prelayernorm", "TFRobertaPreLayerNormModel"),
        # "roformer" 模型对应的 TensorFlow 模型类为 "TFRoFormerModel"
        ("roformer", "TFRoFormerModel"),
        # "sam" 模型对应的 TensorFlow 模型类为 "TFSamModel"
        ("sam", "TFSamModel"),
        # "segformer" 模型对应的 TensorFlow 模型类为 "TFSegformerModel"
        ("segformer", "TFSegformerModel"),
        # "speech_to_text" 模型对应的 TensorFlow 模型类为 "TFSpeech2TextModel"
        ("speech_to_text", "TFSpeech2TextModel"),
        # "swin" 模型对应的 TensorFlow 模型类为 "TFSwinModel"
        ("swin", "TFSwinModel"),
        # "t5" 模型对应的 TensorFlow 模型类
# 定义一个有序字典，映射模型名称到TensorFlow模型类名，用于预训练模型的映射
TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # 各种预训练模型的映射关系
        ("albert", "TFAlbertForPreTraining"),
        ("bart", "TFBartForConditionalGeneration"),
        ("bert", "TFBertForPreTraining"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("electra", "TFElectraForPreTraining"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("funnel", "TFFunnelForPreTraining"),
        ("gpt-sw3", "TFGPT2LMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("lxmert", "TFLxmertForPreTraining"),
        ("mobilebert", "TFMobileBertForPreTraining"),
        ("mpnet", "TFMPNetForMaskedLM"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("tapas", "TFTapasForMaskedLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("vit_mae", "TFViTMAEForPreTraining"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
)

# 定义另一个有序字典，映射模型名称到TensorFlow模型类名，用于带有语言模型头部的模型的映射
TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # 各种带有语言模型头部的模型的映射关系
        ("albert", "TFAlbertForMaskedLM"),
        ("bart", "TFBartForConditionalGeneration"),
        ("bert", "TFBertForMaskedLM"),
        ("camembert", "TFCamembertForMaskedLM"),
        ("convbert", "TFConvBertForMaskedLM"),
        ("ctrl", "TFCTRLLMHeadModel"),
        ("distilbert", "TFDistilBertForMaskedLM"),
        ("electra", "TFElectraForMaskedLM"),
        ("esm", "TFEsmForMaskedLM"),
        ("flaubert", "TFFlaubertWithLMHeadModel"),
        ("funnel", "TFFunnelForMaskedLM"),
        ("gpt-sw3", "TFGPT2LMHeadModel"),
        ("gpt2", "TFGPT2LMHeadModel"),
        ("gptj", "TFGPTJForCausalLM"),
        ("layoutlm", "TFLayoutLMForMaskedLM"),
        ("led", "TFLEDForConditionalGeneration"),
        ("longformer", "TFLongformerForMaskedLM"),
        ("marian", "TFMarianMTModel"),
        ("mobilebert", "TFMobileBertForMaskedLM"),
        ("mpnet", "TFMPNetForMaskedLM"),
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
        ("rembert", "TFRemBertForMaskedLM"),
        ("roberta", "TFRobertaForMaskedLM"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),
        ("roformer", "TFRoFormerForMaskedLM"),
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),
        ("t5", "TFT5ForConditionalGeneration"),
        ("tapas", "TFTapasForMaskedLM"),
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
        ("whisper", "TFWhisperForConditionalGeneration"),
        ("xlm", "TFXLMWithLMHeadModel"),
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
)

# 定义另一个有序字典，映射模型名称到TensorFlow模型类名，用于因果语言模型的映射
TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # 定义一个列表，包含了多个元组，每个元组代表了一个模型及其对应的类名
        # 第一个元素是模型的缩写或名称，第二个元素是该模型对应的 TensorFlow 类名
    
        # 模型 "bert" 对应的类是 "TFBertLMHeadModel"
        ("bert", "TFBertLMHeadModel"),
    
        # 模型 "camembert" 对应的类是 "TFCamembertForCausalLM"
        ("camembert", "TFCamembertForCausalLM"),
    
        # 模型 "ctrl" 对应的类是 "TFCTRLLMHeadModel"
        ("ctrl", "TFCTRLLMHeadModel"),
    
        # 模型 "gpt-sw3" 对应的类是 "TFGPT2LMHeadModel"
        ("gpt-sw3", "TFGPT2LMHeadModel"),
    
        # 模型 "gpt2" 对应的类是 "TFGPT2LMHeadModel"
        ("gpt2", "TFGPT2LMHeadModel"),
    
        # 模型 "gptj" 对应的类是 "TFGPTJForCausalLM"
        ("gptj", "TFGPTJForCausalLM"),
    
        # 模型 "openai-gpt" 对应的类是 "TFOpenAIGPTLMHeadModel"
        ("openai-gpt", "TFOpenAIGPTLMHeadModel"),
    
        # 模型 "opt" 对应的类是 "TFOPTForCausalLM"
        ("opt", "TFOPTForCausalLM"),
    
        # 模型 "rembert" 对应的类是 "TFRemBertForCausalLM"
        ("rembert", "TFRemBertForCausalLM"),
    
        # 模型 "roberta" 对应的类是 "TFRobertaForCausalLM"
        ("roberta", "TFRobertaForCausalLM"),
    
        # 模型 "roberta-prelayernorm" 对应的类是 "TFRobertaPreLayerNormForCausalLM"
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForCausalLM"),
    
        # 模型 "roformer" 对应的类是 "TFRoFormerForCausalLM"
        ("roformer", "TFRoFormerForCausalLM"),
    
        # 模型 "transfo-xl" 对应的类是 "TFTransfoXLLMHeadModel"
        ("transfo-xl", "TFTransfoXLLMHeadModel"),
    
        # 模型 "xglm" 对应的类是 "TFXGLMForCausalLM"
        ("xglm", "TFXGLMForCausalLM"),
    
        # 模型 "xlm" 对应的类是 "TFXLMWithLMHeadModel"
        ("xlm", "TFXLMWithLMHeadModel"),
    
        # 模型 "xlm-roberta" 对应的类是 "TFXLMRobertaForCausalLM"
        ("xlm-roberta", "TFXLMRobertaForCausalLM"),
    
        # 模型 "xlnet" 对应的类是 "TFXLNetLMHeadModel"
        ("xlnet", "TFXLNetLMHeadModel"),
    ]
# 模型到类的映射，用于模型在 TensorFlow 中的命名
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "TFDeiTForMaskedImageModeling"),  # DEIT模型对应的命名为TFDeiTForMaskedImageModeling
        ("swin", "TFSwinForMaskedImageModeling"),  # Swin模型对应的命名为TFSwinForMaskedImageModeling
    ]
)

# 图像分类模型到类的映射
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 图像分类模型
        ("convnext", "TFConvNextForImageClassification"),  # ConvNext模型对应的命名为TFConvNextForImageClassification
        ("convnextv2", "TFConvNextV2ForImageClassification"),  # ConvNextV2模型对应的命名为TFConvNextV2ForImageClassification
        ("cvt", "TFCvtForImageClassification"),  # CVT模型对应的命名为TFCvtForImageClassification
        ("data2vec-vision", "TFData2VecVisionForImageClassification"),  # Data2Vec-Vision模型对应的命名为TFData2VecVisionForImageClassification
        ("deit", ("TFDeiTForImageClassification", "TFDeiTForImageClassificationWithTeacher")),  # DEIT模型对应的命名为TFDeiTForImageClassification和TFDeiTForImageClassificationWithTeacher
        (
            "efficientformer",
            ("TFEfficientFormerForImageClassification", "TFEfficientFormerForImageClassificationWithTeacher"),  # EfficientFormer模型对应的命名为TFEfficientFormerForImageClassification和TFEfficientFormerForImageClassificationWithTeacher
        ),
        ("mobilevit", "TFMobileViTForImageClassification"),  # MobileViT模型对应的命名为TFMobileViTForImageClassification
        ("regnet", "TFRegNetForImageClassification"),  # RegNet模型对应的命名为TFRegNetForImageClassification
        ("resnet", "TFResNetForImageClassification"),  # ResNet模型对应的命名为TFResNetForImageClassification
        ("segformer", "TFSegformerForImageClassification"),  # Segformer模型对应的命名为TFSegformerForImageClassification
        ("swin", "TFSwinForImageClassification"),  # Swin模型对应的命名为TFSwinForImageClassification
        ("vit", "TFViTForImageClassification"),  # ViT模型对应的命名为TFViTForImageClassification
    ]
)

# 零样本图像分类模型到类的映射
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 零样本图像分类模型映射
        ("blip", "TFBlipModel"),  # BLIP模型对应的命名为TFBlipModel
        ("clip", "TFCLIPModel"),  # CLIP模型对应的命名为TFCLIPModel
    ]
)

# 语义分割模型到类的映射
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 语义分割模型映射
        ("data2vec-vision", "TFData2VecVisionForSemanticSegmentation"),  # Data2Vec-Vision模型对应的命名为TFData2VecVisionForSemanticSegmentation
        ("mobilevit", "TFMobileViTForSemanticSegmentation"),  # MobileViT模型对应的命名为TFMobileViTForSemanticSegmentation
        ("segformer", "TFSegformerForSemanticSegmentation"),  # Segformer模型对应的命名为TFSegformerForSemanticSegmentation
    ]
)

# 视觉到序列模型到类的映射
TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "TFBlipForConditionalGeneration"),  # BLIP模型对应的命名为TFBlipForConditionalGeneration
        ("vision-encoder-decoder", "TFVisionEncoderDecoderModel"),  # Vision-Encoder-Decoder模型对应的命名为TFVisionEncoderDecoderModel
    ]
)

# Masked LM模型到类的映射
TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Masked LM模型映射
        ("albert", "TFAlbertForMaskedLM"),  # ALBERT模型对应的命名为TFAlbertForMaskedLM
        ("bert", "TFBertForMaskedLM"),  # BERT模型对应的命名为TFBertForMaskedLM
        ("camembert", "TFCamembertForMaskedLM"),  # Camembert模型对应的命名为TFCamembertForMaskedLM
        ("convbert", "TFConvBertForMaskedLM"),  # ConvBERT模型对应的命名为TFConvBertForMaskedLM
        ("deberta", "TFDebertaForMaskedLM"),  # DeBERTa模型对应的命名为TFDebertaForMaskedLM
        ("deberta-v2", "TFDebertaV2ForMaskedLM"),  # DeBERTa-v2模型对应的命名为TFDebertaV2ForMaskedLM
        ("distilbert", "TFDistilBertForMaskedLM"),  # DistilBERT模型对应的命名为TFDistilBertForMaskedLM
        ("electra", "TFElectraForMaskedLM"),  # Electra模型对应的命名为TFElectraForMaskedLM
        ("esm", "TFEsmForMaskedLM"),  # ESM模型对应的命名为TFEsmForMaskedLM
        ("flaubert", "TFFlaubertWithLMHeadModel"),  # FlauBERT模型对应的命名为TFFlaubertWithLMHeadModel
        ("funnel", "TFFunnelForMaskedLM"),  # Funnel模型对应的命名为TFFunnelForMaskedLM
        ("layoutlm", "TFLayoutLMForMaskedLM"),  # LayoutLM模型对应的命名为TFLayoutLMForMaskedLM
        ("longformer", "TFLongformerForMaskedLM"),  # Longformer模型对应的命名为TFLongformerForMaskedLM
        ("mobilebert", "TFMobileBertForMaskedLM"),  # MobileBERT模型对应的命名为TFMobileBertForMaskedLM
        ("mpnet", "TFMPNetForMaskedLM"),  # MPNet模型对应的命名为TFMPNetForMaskedLM
        ("rembert", "TFRemBertForMaskedLM"),  # RemBERT模型对应的命名为TFRemBertForMaskedLM
        ("roberta", "TFRobertaForMaskedLM"),  # RoBERTa模型对应的命名为TFRobertaForMaskedLM
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForMaskedLM"),  # RoBERTa-prelayernorm模型对应的命名为TFRobertaPreLayerNormForMaskedLM
        ("roformer", "TFRoFormerForMaskedLM"),  # RoFormer模型对应的命名为TFRoFormerForMaskedLM
        ("tapas", "TFTapasForMaskedLM"),  # TAPAS模型对应的命名为TFTapasForMaskedLM
        ("xlm", "TFXLMWithLMHeadModel"),  # XLM模型对应的命名为TFXLMWithLMHeadModel
        ("xlm-roberta", "TFXLMRobertaForMaskedLM"),  # XLM-RoBERTa模型对应的命名为TFXLMRobertaForMaskedLM
    ]
)
# 创建一个有序字典，用于将模型名称映射到对应的 TensorFlow 序列到序列因果语言建模模型类名
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "TFBartForConditionalGeneration"),  # BART模型的条件生成器
        ("blenderbot", "TFBlenderbotForConditionalGeneration"),  # Blenderbot模型的条件生成器
        ("blenderbot-small", "TFBlenderbotSmallForConditionalGeneration"),  # 小型Blenderbot模型的条件生成器
        ("encoder-decoder", "TFEncoderDecoderModel"),  # 编码-解码模型
        ("led", "TFLEDForConditionalGeneration"),  # LED模型的条件生成器
        ("marian", "TFMarianMTModel"),  # Marian机器翻译模型
        ("mbart", "TFMBartForConditionalGeneration"),  # mBART模型的条件生成器
        ("mt5", "TFMT5ForConditionalGeneration"),  # MT5模型的条件生成器
        ("pegasus", "TFPegasusForConditionalGeneration"),  # Pegasus模型的条件生成器
        ("t5", "TFT5ForConditionalGeneration"),  # T5模型的条件生成器
    ]
)

# 创建一个有序字典，用于将模型名称映射到对应的 TensorFlow 语音序列到序列模型类名
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speech_to_text", "TFSpeech2TextForConditionalGeneration"),  # 语音转文本模型的条件生成器
        ("whisper", "TFWhisperForConditionalGeneration"),  # Whisper模型的条件生成器
    ]
)

# 创建一个有序字典，用于将模型名称映射到对应的 TensorFlow 序列分类模型类名
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "TFAlbertForSequenceClassification"),  # Albert模型的序列分类器
        ("bart", "TFBartForSequenceClassification"),  # BART模型的序列分类器
        ("bert", "TFBertForSequenceClassification"),  # BERT模型的序列分类器
        ("camembert", "TFCamembertForSequenceClassification"),  # CamemBERT模型的序列分类器
        ("convbert", "TFConvBertForSequenceClassification"),  # ConvBERT模型的序列分类器
        ("ctrl", "TFCTRLForSequenceClassification"),  # CTRL模型的序列分类器
        ("deberta", "TFDebertaForSequenceClassification"),  # DeBERTa模型的序列分类器
        ("deberta-v2", "TFDebertaV2ForSequenceClassification"),  # DeBERTa-v2模型的序列分类器
        ("distilbert", "TFDistilBertForSequenceClassification"),  # DistilBERT模型的序列分类器
        ("electra", "TFElectraForSequenceClassification"),  # Electra模型的序列分类器
        ("esm", "TFEsmForSequenceClassification"),  # ESM模型的序列分类器
        ("flaubert", "TFFlaubertForSequenceClassification"),  # FlauBERT模型的序列分类器
        ("funnel", "TFFunnelForSequenceClassification"),  # Funnel模型的序列分类器
        ("gpt-sw3", "TFGPT2ForSequenceClassification"),  # GPT-SW3模型的序列分类器
        ("gpt2", "TFGPT2ForSequenceClassification"),  # GPT-2模型的序列分类器
        ("gptj", "TFGPTJForSequenceClassification"),  # GPT-J模型的序列分类器
        ("layoutlm", "TFLayoutLMForSequenceClassification"),  # LayoutLM模型的序列分类器
        ("layoutlmv3", "TFLayoutLMv3ForSequenceClassification"),  # LayoutLMv3模型的序列分类器
        ("longformer", "TFLongformerForSequenceClassification"),  # Longformer模型的序列分类器
        ("mobilebert", "TFMobileBertForSequenceClassification"),  # MobileBERT模型的序列分类器
        ("mpnet", "TFMPNetForSequenceClassification"),  # MPNet模型的序列分类器
        ("openai-gpt", "TFOpenAIGPTForSequenceClassification"),  # OpenAI-GPT模型的序列分类器
        ("rembert", "TFRemBertForSequenceClassification"),  # RemBERT模型的序列分类器
        ("roberta", "TFRobertaForSequenceClassification"),  # RoBERTa模型的序列分类器
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForSequenceClassification"),  # RoBERTa-prelayernorm模型的序列分类器
        ("roformer", "TFRoFormerForSequenceClassification"),  # RoFormer模型的序列分类器
        ("tapas", "TFTapasForSequenceClassification"),  # TAPAS模型的序列分类器
        ("transfo-xl", "TFTransfoXLForSequenceClassification"),  # TransfoXL模型的序列分类器
        ("xlm", "TFXLMForSequenceClassification"),  # XLM模型的序列分类器
        ("xlm-roberta", "TFXLMRobertaForSequenceClassification"),  # XLM-RoBERTa模型的序列分类器
        ("xlnet", "TFXLNetForSequenceClassification"),  # XLNet模型的序列分类器
    ]
)

# 创建一个有序字典，用于将模型名称映射到对应的 TensorFlow 问答模型类名
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    # 定义了一个模型到类的映射关系列表，用于问答任务
    [
        # 使用 ALBERT 模型进行问答的类
        ("albert", "TFAlbertForQuestionAnswering"),
        # 使用 BERT 模型进行问答的类
        ("bert", "TFBertForQuestionAnswering"),
        # 使用 CamemBERT 模型进行问答的类
        ("camembert", "TFCamembertForQuestionAnswering"),
        # 使用 ConvBERT 模型进行问答的类
        ("convbert", "TFConvBertForQuestionAnswering"),
        # 使用 DeBERTa 模型进行问答的类
        ("deberta", "TFDebertaForQuestionAnswering"),
        # 使用 DeBERTa-v2 模型进行问答的类
        ("deberta-v2", "TFDebertaV2ForQuestionAnswering"),
        # 使用 DistilBERT 模型进行问答的类
        ("distilbert", "TFDistilBertForQuestionAnswering"),
        # 使用 Electra 模型进行问答的类
        ("electra", "TFElectraForQuestionAnswering"),
        # 使用 FlauBERT 模型进行问答的类
        ("flaubert", "TFFlaubertForQuestionAnsweringSimple"),
        # 使用 Funnel 模型进行问答的类
        ("funnel", "TFFunnelForQuestionAnswering"),
        # 使用 GPT-J 模型进行问答的类
        ("gptj", "TFGPTJForQuestionAnswering"),
        # 使用 LayoutLMv3 模型进行问答的类
        ("layoutlmv3", "TFLayoutLMv3ForQuestionAnswering"),
        # 使用 Longformer 模型进行问答的类
        ("longformer", "TFLongformerForQuestionAnswering"),
        # 使用 MobileBERT 模型进行问答的类
        ("mobilebert", "TFMobileBertForQuestionAnswering"),
        # 使用 MPNet 模型进行问答的类
        ("mpnet", "TFMPNetForQuestionAnswering"),
        # 使用 RemBERT 模型进行问答的类
        ("rembert", "TFRemBertForQuestionAnswering"),
        # 使用 RoBERTa 模型进行问答的类
        ("roberta", "TFRobertaForQuestionAnswering"),
        # 使用 RoBERTa-prelayernorm 模型进行问答的类
        ("roberta-prelayernorm", "TFRobertaPreLayerNormForQuestionAnswering"),
        # 使用 RoFormer 模型进行问答的类
        ("roformer", "TFRoFormerForQuestionAnswering"),
        # 使用 XLM 模型进行问答的类
        ("xlm", "TFXLMForQuestionAnsweringSimple"),
        # 使用 XLM-RoBERTa 模型进行问答的类
        ("xlm-roberta", "TFXLMRobertaForQuestionAnswering"),
        # 使用 XLNet 模型进行问答的类
        ("xlnet", "TFXLNetForQuestionAnsweringSimple"),
    ]
# 导入 OrderedDict 类型，用于创建有序字典，记录模型名称到 TensorFlow 类的映射关系
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict([("wav2vec2", "TFWav2Vec2ForSequenceClassification")])

# 导入 OrderedDict 类型，用于创建有序字典，记录模型名称到 TensorFlow 类的映射关系
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "TFLayoutLMForQuestionAnswering"),
        ("layoutlmv3", "TFLayoutLMv3ForQuestionAnswering"),
    ]
)

# 导入 OrderedDict 类型，用于创建有序字典，记录模型名称到 TensorFlow 类的映射关系
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # 用于表格问答的模型映射
        ("tapas", "TFTapasForQuestionAnswering"),
    ]
)

# 导入 OrderedDict 类型，用于创建有序字典，记录模型名称到 TensorFlow 类的映射关系
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 用于标记分类的模型映射
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

# 导入 OrderedDict 类型，用于创建有序字典，记录模型名称到 TensorFlow 类的映射关系
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    # 此处是多选题的模型映射
    [
        # 模型名称和对应的TensorFlow模型类名，用于多选题任务
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
# 创建一个有序字典，用于将模型名称映射到相应的 TensorFlow 下一句预测模型类名
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "TFBertForNextSentencePrediction"),
        ("mobilebert", "TFMobileBertForNextSentencePrediction"),
    ]
)

# 创建一个有序字典，用于将模型名称映射到相应的 TensorFlow 掩码生成模型类名
TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "TFSamModel"),
    ]
)

# 创建一个有序字典，用于将模型名称映射到相应的 TensorFlow 文本编码模型类名
TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "TFAlbertModel"),
        ("bert", "TFBertModel"),
        ("convbert", "TFConvBertModel"),
        ("deberta", "TFDebertaModel"),
        ("deberta-v2", "TFDebertaV2Model"),
        ("distilbert", "TFDistilBertModel"),
        ("electra", "TFElectraModel"),
        ("flaubert", "TFFlaubertModel"),
        ("longformer", "TFLongformerModel"),
        ("mobilebert", "TFMobileBertModel"),
        ("mt5", "TFMT5EncoderModel"),
        ("rembert", "TFRemBertModel"),
        ("roberta", "TFRobertaModel"),
        ("roberta-prelayernorm", "TFRobertaPreLayerNormModel"),
        ("roformer", "TFRoFormerModel"),
        ("t5", "TFT5EncoderModel"),
        ("xlm", "TFXLMModel"),
        ("xlm-roberta", "TFXLMRobertaModel"),
    ]
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_MAPPING_NAMES
TF_MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_MAPPING_NAMES)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES
TF_MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_PRETRAINING_MAPPING_NAMES)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES
TF_MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_WITH_LM_HEAD_MAPPING_NAMES)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
TF_MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
TF_MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES
TF_MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASKED_LM_MAPPING_NAMES)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)

# 创建 LazyAutoMapping 对象，将 CONFIG_MAPPING_NAMES 映射到 TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)
    # 导入模块中的特定变量，CONFIG_MAPPING_NAMES 和 TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_MASK_GENERATION_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES
TF_MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_MASK_GENERATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建 TF_MODEL_FOR_TEXT_ENCODING_MAPPING 对象，映射配置名称到 TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES
TF_MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES
)


class TFAutoModelForMaskGeneration(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_MASK_GENERATION_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_MASK_GENERATION_MAPPING


class TFAutoModelForTextEncoding(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_TEXT_ENCODING_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_TEXT_ENCODING_MAPPING


class TFAutoModel(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_MAPPING


TFAutoModel = auto_class_update(TFAutoModel)


class TFAutoModelForAudioClassification(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


TFAutoModelForAudioClassification = auto_class_update(
    TFAutoModelForAudioClassification, head_doc="audio classification"
)


class TFAutoModelForPreTraining(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_PRETRAINING_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_PRETRAINING_MAPPING


TFAutoModelForPreTraining = auto_class_update(TFAutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _TFAutoModelWithLMHead(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_WITH_LM_HEAD_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_WITH_LM_HEAD_MAPPING


_TFAutoModelWithLMHead = auto_class_update(_TFAutoModelWithLMHead, head_doc="language modeling")


class TFAutoModelForCausalLM(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_CAUSAL_LM_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_CAUSAL_LM_MAPPING


TFAutoModelForCausalLM = auto_class_update(TFAutoModelForCausalLM, head_doc="causal language modeling")


class TFAutoModelForMaskedImageModeling(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


TFAutoModelForMaskedImageModeling = auto_class_update(
    TFAutoModelForMaskedImageModeling, head_doc="masked image modeling"
)


class TFAutoModelForImageClassification(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


TFAutoModelForImageClassification = auto_class_update(
    TFAutoModelForImageClassification, head_doc="image classification"
)


class TFAutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    # 设置类属性 _model_mapping 为 TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING，用于自动模型选择
    _model_mapping = TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


TFAutoModelForZeroShotImageClassification = auto_class_update(
    TFAutoModelForZeroShotImageClassification,
    head_doc="zero-shot image classification"
)
    TFAutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"


# 导入 TensorFlow 自动模型用于零样本图像分类，指定头部文档为“zero-shot image classification”
class TFAutoModelForSemanticSegmentation(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于语义分割任务
    _model_mapping = TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


# 更新 TFAutoModelForSemanticSegmentation 类，添加头部文档描述为“semantic segmentation”
TFAutoModelForSemanticSegmentation = auto_class_update(
    TFAutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)


class TFAutoModelForVision2Seq(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于视觉到文本任务
    _model_mapping = TF_MODEL_FOR_VISION_2_SEQ_MAPPING


# 更新 TFAutoModelForVision2Seq 类，添加头部文档描述为“vision-to-text modeling”
TFAutoModelForVision2Seq = auto_class_update(
    TFAutoModelForVision2Seq, head_doc="vision-to-text modeling"
)


class TFAutoModelForMaskedLM(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于掩码语言建模任务
    _model_mapping = TF_MODEL_FOR_MASKED_LM_MAPPING


# 更新 TFAutoModelForMaskedLM 类，添加头部文档描述为“masked language modeling”
TFAutoModelForMaskedLM = auto_class_update(
    TFAutoModelForMaskedLM, head_doc="masked language modeling"
)


class TFAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于序列到序列因果语言建模任务
    _model_mapping = TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


# 更新 TFAutoModelForSeq2SeqLM 类，添加头部文档描述为“sequence-to-sequence language modeling”，
# 并指定一个示例的检查点名称为“google-t5/t5-base”
TFAutoModelForSeq2SeqLM = auto_class_update(
    TFAutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google-t5/t5-base",
)


class TFAutoModelForSequenceClassification(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于序列分类任务
    _model_mapping = TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


# 更新 TFAutoModelForSequenceClassification 类，添加头部文档描述为“sequence classification”
TFAutoModelForSequenceClassification = auto_class_update(
    TFAutoModelForSequenceClassification, head_doc="sequence classification"
)


class TFAutoModelForQuestionAnswering(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于问答任务
    _model_mapping = TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING


# 更新 TFAutoModelForQuestionAnswering 类，添加头部文档描述为“question answering”
TFAutoModelForQuestionAnswering = auto_class_update(
    TFAutoModelForQuestionAnswering, head_doc="question answering"
)


class TFAutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于文档问答任务
    _model_mapping = TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


# 更新 TFAutoModelForDocumentQuestionAnswering 类，添加头部文档描述为“document question answering”，
# 并指定一个示例的检查点名称和修订版本号
TFAutoModelForDocumentQuestionAnswering = auto_class_update(
    TFAutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)


class TFAutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于表格问答任务
    _model_mapping = TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


# 更新 TFAutoModelForTableQuestionAnswering 类，添加头部文档描述为“table question answering”，
# 并指定一个示例的检查点名称为“google/tapas-base-finetuned-wtq”
TFAutoModelForTableQuestionAnswering = auto_class_update(
    TFAutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class TFAutoModelForTokenClassification(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于标记分类任务
    _model_mapping = TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


# 更新 TFAutoModelForTokenClassification 类，添加头部文档描述为“token classification”
TFAutoModelForTokenClassification = auto_class_update(
    TFAutoModelForTokenClassification, head_doc="token classification"
)


class TFAutoModelForMultipleChoice(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于多项选择任务
    _model_mapping = TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING


# 更新 TFAutoModelForMultipleChoice 类，添加头部文档描述为“multiple choice”
TFAutoModelForMultipleChoice = auto_class_update(
    TFAutoModelForMultipleChoice, head_doc="multiple choice"
)


class TFAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    # 定义自动化创建的 TensorFlow 模型类，用于下一句预测任务
    _model_mapping = TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


# 更新 TFAutoModelForNextSentencePrediction 类，添加头部文档描述为“next sentence prediction”
TFAutoModelForNextSentencePrediction = auto_class_update(
    TFAutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)
class TFAutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    # 定义了一个名为 TFAutoModelForSpeechSeq2Seq 的类，继承自 _BaseAutoModelClass
    _model_mapping = TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
    # 设置了一个类变量 _model_mapping，用于映射语音序列到序列模型

# 对 TFAutoModelForSpeechSeq2Seq 类进行更新，添加了头部文档信息，说明其为序列到序列语音转文本建模
TFAutoModelForSpeechSeq2Seq = auto_class_update(
    TFAutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)


class TFAutoModelWithLMHead(_TFAutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        # 发出警告，提醒该类即将被弃用，建议使用特定的子类代替
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use"
            " `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models"
            " and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        # 调用父类方法，从给定的配置中创建对象
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 发出警告，提醒该类即将被弃用，建议使用特定的子类代替
        warnings.warn(
            "The class `TFAutoModelWithLMHead` is deprecated and will be removed in a future version. Please use"
            " `TFAutoModelForCausalLM` for causal language models, `TFAutoModelForMaskedLM` for masked language models"
            " and `TFAutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        # 调用父类方法，从预训练模型名或路径创建对象
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
```