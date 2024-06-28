# `.\models\auto\modeling_auto.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息
#
# 根据 Apache 许可证版本 2.0 授权使用此文件
# 除非符合许可证的条件，否则不得使用此文件
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于"原样"分发，无任何担保或条件
# 请查阅许可证了解具体的法律条文和允许条件
""" Auto Model class."""

# 导入警告模块
import warnings
# 导入有序字典模块
from collections import OrderedDict

# 导入日志记录工具
from ...utils import logging
# 从 auto_factory 模块导入相关类和函数
from .auto_factory import (
    _BaseAutoBackboneClass,
    _BaseAutoModelClass,
    _LazyAutoMapping,
    auto_class_update,
)
# 导入自动生成的配置映射
from .configuration_auto import CONFIG_MAPPING_NAMES

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义模型映射名称的有序字典
MODEL_MAPPING_NAMES = OrderedDict(
    # 这里是一个空的有序字典，用于存储模型映射名称
)

# 定义用于预训练的模型映射名称的有序字典
MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    # 这里是一个空的有序字典，用于存储预训练模型映射名称
)

# 定义带语言模型头部的模型映射名称的有序字典
MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    # 这里是一个空的有序字典，用于存储带语言模型头部的模型映射名称
)

# 定义用于因果语言模型的模型映射名称的有序字典
MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    # 这里是一个空的有序字典，用于存储因果语言模型的模型映射名称
)

# 定义用于图像任务的模型映射名称的有序字典
MODEL_FOR_IMAGE_MAPPING_NAMES = OrderedDict(
    # 这里是一个空的有序字典，用于存储图像任务的模型映射名称
)
    # 创建一个元组列表，每个元组包含模型的名称和相应的模型类名
    [
        # 模型名 "beit" 对应的模型类名 "BeitModel"
        ("beit", "BeitModel"),
        # 模型名 "bit" 对应的模型类名 "BitModel"
        ("bit", "BitModel"),
        # 模型名 "conditional_detr" 对应的模型类名 "ConditionalDetrModel"
        ("conditional_detr", "ConditionalDetrModel"),
        # 模型名 "convnext" 对应的模型类名 "ConvNextModel"
        ("convnext", "ConvNextModel"),
        # 模型名 "convnextv2" 对应的模型类名 "ConvNextV2Model"
        ("convnextv2", "ConvNextV2Model"),
        # 模型名 "data2vec-vision" 对应的模型类名 "Data2VecVisionModel"
        ("data2vec-vision", "Data2VecVisionModel"),
        # 模型名 "deformable_detr" 对应的模型类名 "DeformableDetrModel"
        ("deformable_detr", "DeformableDetrModel"),
        # 模型名 "deit" 对应的模型类名 "DeiTModel"
        ("deit", "DeiTModel"),
        # 模型名 "deta" 对应的模型类名 "DetaModel"
        ("deta", "DetaModel"),
        # 模型名 "detr" 对应的模型类名 "DetrModel"
        ("detr", "DetrModel"),
        # 模型名 "dinat" 对应的模型类名 "DinatModel"
        ("dinat", "DinatModel"),
        # 模型名 "dinov2" 对应的模型类名 "Dinov2Model"
        ("dinov2", "Dinov2Model"),
        # 模型名 "dpt" 对应的模型类名 "DPTModel"
        ("dpt", "DPTModel"),
        # 模型名 "efficientformer" 对应的模型类名 "EfficientFormerModel"
        ("efficientformer", "EfficientFormerModel"),
        # 模型名 "efficientnet" 对应的模型类名 "EfficientNetModel"
        ("efficientnet", "EfficientNetModel"),
        # 模型名 "focalnet" 对应的模型类名 "FocalNetModel"
        ("focalnet", "FocalNetModel"),
        # 模型名 "glpn" 对应的模型类名 "GLPNModel"
        ("glpn", "GLPNModel"),
        # 模型名 "imagegpt" 对应的模型类名 "ImageGPTModel"
        ("imagegpt", "ImageGPTModel"),
        # 模型名 "levit" 对应的模型类名 "LevitModel"
        ("levit", "LevitModel"),
        # 模型名 "mobilenet_v1" 对应的模型类名 "MobileNetV1Model"
        ("mobilenet_v1", "MobileNetV1Model"),
        # 模型名 "mobilenet_v2" 对应的模型类名 "MobileNetV2Model"
        ("mobilenet_v2", "MobileNetV2Model"),
        # 模型名 "mobilevit" 对应的模型类名 "MobileViTModel"
        ("mobilevit", "MobileViTModel"),
        # 模型名 "mobilevitv2" 对应的模型类名 "MobileViTV2Model"
        ("mobilevitv2", "MobileViTV2Model"),
        # 模型名 "nat" 对应的模型类名 "NatModel"
        ("nat", "NatModel"),
        # 模型名 "poolformer" 对应的模型类名 "PoolFormerModel"
        ("poolformer", "PoolFormerModel"),
        # 模型名 "pvt" 对应的模型类名 "PvtModel"
        ("pvt", "PvtModel"),
        # 模型名 "regnet" 对应的模型类名 "RegNetModel"
        ("regnet", "RegNetModel"),
        # 模型名 "resnet" 对应的模型类名 "ResNetModel"
        ("resnet", "ResNetModel"),
        # 模型名 "segformer" 对应的模型类名 "SegformerModel"
        ("segformer", "SegformerModel"),
        # 模型名 "siglip_vision_model" 对应的模型类名 "SiglipVisionModel"
        ("siglip_vision_model", "SiglipVisionModel"),
        # 模型名 "swiftformer" 对应的模型类名 "SwiftFormerModel"
        ("swiftformer", "SwiftFormerModel"),
        # 模型名 "swin" 对应的模型类名 "SwinModel"
        ("swin", "SwinModel"),
        # 模型名 "swin2sr" 对应的模型类名 "Swin2SRModel"
        ("swin2sr", "Swin2SRModel"),
        # 模型名 "swinv2" 对应的模型类名 "Swinv2Model"
        ("swinv2", "Swinv2Model"),
        # 模型名 "table-transformer" 对应的模型类名 "TableTransformerModel"
        ("table-transformer", "TableTransformerModel"),
        # 模型名 "timesformer" 对应的模型类名 "TimesformerModel"
        ("timesformer", "TimesformerModel"),
        # 模型名 "timm_backbone" 对应的模型类名 "TimmBackbone"
        ("timm_backbone", "TimmBackbone"),
        # 模型名 "van" 对应的模型类名 "VanModel"
        ("van", "VanModel"),
        # 模型名 "videomae" 对应的模型类名 "VideoMAEModel"
        ("videomae", "VideoMAEModel"),
        # 模型名 "vit" 对应的模型类名 "ViTModel"
        ("vit", "ViTModel"),
        # 模型名 "vit_hybrid" 对应的模型类名 "ViTHybridModel"
        ("vit_hybrid", "ViTHybridModel"),
        # 模型名 "vit_mae" 对应的模型类名 "ViTMAEModel"
        ("vit_mae", "ViTMAEModel"),
        # 模型名 "vit_msn" 对应的模型类名 "ViTMSNModel"
        ("vit_msn", "ViTMSNModel"),
        # 模型名 "vitdet" 对应的模型类名 "VitDetModel"
        ("vitdet", "VitDetModel"),
        # 模型名 "vivit" 对应的模型类名 "VivitModel"
        ("vivit", "VivitModel"),
        # 模型名 "yolos" 对应的模型类名 "YolosModel"
        ("yolos", "YolosModel"),
    ]
# 定义一个有序字典，映射不同模型到相应的类名称，用于掩模图像建模模型
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "DeiTForMaskedImageModeling"),   # 将 "deit" 映射到 "DeiTForMaskedImageModeling"
        ("focalnet", "FocalNetForMaskedImageModeling"),  # 将 "focalnet" 映射到 "FocalNetForMaskedImageModeling"
        ("swin", "SwinForMaskedImageModeling"),   # 将 "swin" 映射到 "SwinForMaskedImageModeling"
        ("swinv2", "Swinv2ForMaskedImageModeling"),  # 将 "swinv2" 映射到 "Swinv2ForMaskedImageModeling"
        ("vit", "ViTForMaskedImageModeling"),   # 将 "vit" 映射到 "ViTForMaskedImageModeling"
    ]
)

# 定义一个有序字典，映射不同模型到因果图像建模模型的类名称
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("imagegpt", "ImageGPTForCausalImageModeling"),  # 将 "imagegpt" 映射到 "ImageGPTForCausalImageModeling"
    ]
)

# 定义一个有序字典，映射不同模型到图像分类模型的类名称
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 定义了多个模型名称与对应的类名的映射关系
        ("beit", "BeitForImageClassification"),  # BEiT 模型的图像分类器
        ("bit", "BitForImageClassification"),  # BiT 模型的图像分类器
        ("clip", "CLIPForImageClassification"),  # CLIP 模型的图像分类器
        ("convnext", "ConvNextForImageClassification"),  # ConvNext 模型的图像分类器
        ("convnextv2", "ConvNextV2ForImageClassification"),  # ConvNextV2 模型的图像分类器
        ("cvt", "CvtForImageClassification"),  # CvT 模型的图像分类器
        ("data2vec-vision", "Data2VecVisionForImageClassification"),  # Data2VecVision 模型的图像分类器
        (
            "deit",
            ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher"),  # DeiT 模型的图像分类器及其带教师的版本
        ),
        ("dinat", "DinatForImageClassification"),  # DINO 模型的图像分类器
        ("dinov2", "Dinov2ForImageClassification"),  # DINOv2 模型的图像分类器
        (
            "efficientformer",
            (
                "EfficientFormerForImageClassification",  # EfficientFormer 模型的图像分类器
                "EfficientFormerForImageClassificationWithTeacher",  # EfficientFormer 模型的图像分类器带教师版本
            ),
        ),
        ("efficientnet", "EfficientNetForImageClassification"),  # EfficientNet 模型的图像分类器
        ("focalnet", "FocalNetForImageClassification"),  # FocalNet 模型的图像分类器
        ("imagegpt", "ImageGPTForImageClassification"),  # ImageGPT 模型的图像分类器
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),  # LeViT 模型的图像分类器及其带教师的版本
        ),
        ("mobilenet_v1", "MobileNetV1ForImageClassification"),  # MobileNetV1 模型的图像分类器
        ("mobilenet_v2", "MobileNetV2ForImageClassification"),  # MobileNetV2 模型的图像分类器
        ("mobilevit", "MobileViTForImageClassification"),  # MobileViT 模型的图像分类器
        ("mobilevitv2", "MobileViTV2ForImageClassification"),  # MobileViTV2 模型的图像分类器
        ("nat", "NatForImageClassification"),  # NAT 模型的图像分类器
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",  # Perceiver 模型的图像分类器（学习）
                "PerceiverForImageClassificationFourier",  # Perceiver 模型的图像分类器（Fourier变换）
                "PerceiverForImageClassificationConvProcessing",  # Perceiver 模型的图像分类器（卷积处理）
            ),
        ),
        ("poolformer", "PoolFormerForImageClassification"),  # PoolFormer 模型的图像分类器
        ("pvt", "PvtForImageClassification"),  # PVT 模型的图像分类器
        ("pvt_v2", "PvtV2ForImageClassification"),  # PvtV2 模型的图像分类器
        ("regnet", "RegNetForImageClassification"),  # RegNet 模型的图像分类器
        ("resnet", "ResNetForImageClassification"),  # ResNet 模型的图像分类器
        ("segformer", "SegformerForImageClassification"),  # Segformer 模型的图像分类器
        ("siglip", "SiglipForImageClassification"),  # Siglip 模型的图像分类器
        ("swiftformer", "SwiftFormerForImageClassification"),  # SwiftFormer 模型的图像分类器
        ("swin", "SwinForImageClassification"),  # Swin 模型的图像分类器
        ("swinv2", "Swinv2ForImageClassification"),  # SwinV2 模型的图像分类器
        ("van", "VanForImageClassification"),  # ViT 模型的图像分类器
        ("vit", "ViTForImageClassification"),  # ViT 模型的图像分类器
        ("vit_hybrid", "ViTHybridForImageClassification"),  # ViT 混合模型的图像分类器
        ("vit_msn", "ViTMSNForImageClassification"),  # ViT-MSN 模型的图像分类器
    ]
# 定义一个有序字典，映射不同的模型名称到对应的类名，用于图像分割模型
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 不要在这里添加新的模型，此类将来会被弃用。
        # 图像分割模型的映射
        ("detr", "DetrForSegmentation"),
    ]
)

# 定义一个有序字典，映射不同的模型名称到对应的类名，用于语义分割模型
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 语义分割模型的映射
        ("beit", "BeitForSemanticSegmentation"),
        ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),
        ("dpt", "DPTForSemanticSegmentation"),
        ("mobilenet_v2", "MobileNetV2ForSemanticSegmentation"),
        ("mobilevit", "MobileViTForSemanticSegmentation"),
        ("mobilevitv2", "MobileViTV2ForSemanticSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
        ("upernet", "UperNetForSemanticSegmentation"),
    ]
)

# 定义一个有序字典，映射不同的模型名称到对应的类名，用于实例分割模型
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 实例分割模型的映射
        # MaskFormerForInstanceSegmentation 在 v5 中可以从这个映射中移除
        ("maskformer", "MaskFormerForInstanceSegmentation"),
    ]
)

# 定义一个有序字典，映射不同的模型名称到对应的类名，用于通用分割模型
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # 通用分割模型的映射
        ("detr", "DetrForSegmentation"),
        ("mask2former", "Mask2FormerForUniversalSegmentation"),
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)

# 定义一个有序字典，映射不同的模型名称到对应的类名，用于视频分类模型
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("timesformer", "TimesformerForVideoClassification"),
        ("videomae", "VideoMAEForVideoClassification"),
        ("vivit", "VivitForVideoClassification"),
    ]
)

# 定义一个有序字典，映射不同的模型名称到对应的类名，用于视觉到序列模型
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("git", "GitForCausalLM"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)

# 定义一个有序字典，映射不同的模型名称到对应的类名，用于掩码语言建模模型
MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # 模型名称与对应的 PyTorch 模型类名的映射关系列表
        ("albert", "AlbertForMaskedLM"),                # Albert 模型用于 Masked LM
        ("bart", "BartForConditionalGeneration"),       # Bart 模型用于条件生成
        ("bert", "BertForMaskedLM"),                    # Bert 模型用于 Masked LM
        ("big_bird", "BigBirdForMaskedLM"),             # BigBird 模型用于 Masked LM
        ("camembert", "CamembertForMaskedLM"),          # Camembert 模型用于 Masked LM
        ("convbert", "ConvBertForMaskedLM"),            # ConvBert 模型用于 Masked LM
        ("data2vec-text", "Data2VecTextForMaskedLM"),   # Data2Vec-Text 模型用于 Masked LM
        ("deberta", "DebertaForMaskedLM"),              # Deberta 模型用于 Masked LM
        ("deberta-v2", "DebertaV2ForMaskedLM"),         # Deberta-v2 模型用于 Masked LM
        ("distilbert", "DistilBertForMaskedLM"),        # DistilBert 模型用于 Masked LM
        ("electra", "ElectraForMaskedLM"),              # Electra 模型用于 Masked LM
        ("ernie", "ErnieForMaskedLM"),                  # Ernie 模型用于 Masked LM
        ("esm", "EsmForMaskedLM"),                      # ESM 模型用于 Masked LM
        ("flaubert", "FlaubertWithLMHeadModel"),        # Flaubert 模型用于 Masked LM
        ("fnet", "FNetForMaskedLM"),                    # FNet 模型用于 Masked LM
        ("funnel", "FunnelForMaskedLM"),                # Funnel 模型用于 Masked LM
        ("ibert", "IBertForMaskedLM"),                  # IBert 模型用于 Masked LM
        ("layoutlm", "LayoutLMForMaskedLM"),            # LayoutLM 模型用于 Masked LM
        ("longformer", "LongformerForMaskedLM"),        # Longformer 模型用于 Masked LM
        ("luke", "LukeForMaskedLM"),                    # Luke 模型用于 Masked LM
        ("mbart", "MBartForConditionalGeneration"),     # MBart 模型用于条件生成
        ("mega", "MegaForMaskedLM"),                    # Mega 模型用于 Masked LM
        ("megatron-bert", "MegatronBertForMaskedLM"),   # Megatron-Bert 模型用于 Masked LM
        ("mobilebert", "MobileBertForMaskedLM"),        # MobileBert 模型用于 Masked LM
        ("mpnet", "MPNetForMaskedLM"),                  # MPNet 模型用于 Masked LM
        ("mra", "MraForMaskedLM"),                      # Mra 模型用于 Masked LM
        ("mvp", "MvpForConditionalGeneration"),         # Mvp 模型用于条件生成
        ("nezha", "NezhaForMaskedLM"),                  # Nezha 模型用于 Masked LM
        ("nystromformer", "NystromformerForMaskedLM"),  # Nystromformer 模型用于 Masked LM
        ("perceiver", "PerceiverForMaskedLM"),          # Perceiver 模型用于 Masked LM
        ("qdqbert", "QDQBertForMaskedLM"),              # QDQBert 模型用于 Masked LM
        ("reformer", "ReformerForMaskedLM"),            # Reformer 模型用于 Masked LM
        ("rembert", "RemBertForMaskedLM"),              # RemBert 模型用于 Masked LM
        ("roberta", "RobertaForMaskedLM"),              # Roberta 模型用于 Masked LM
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),  # Roberta with PreLayerNorm 模型用于 Masked LM
        ("roc_bert", "RoCBertForMaskedLM"),             # RoCBert 模型用于 Masked LM
        ("roformer", "RoFormerForMaskedLM"),            # RoFormer 模型用于 Masked LM
        ("squeezebert", "SqueezeBertForMaskedLM"),      # SqueezeBert 模型用于 Masked LM
        ("tapas", "TapasForMaskedLM"),                  # Tapas 模型用于 Masked LM
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),            # Wav2Vec2 模型用于 Masked LM
        ("xlm", "XLMWithLMHeadModel"),                  # XLM 模型用于 Masked LM
        ("xlm-roberta", "XLMRobertaForMaskedLM"),       # XLM-RoBERTa 模型用于 Masked LM
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),  # XLM-RoBERTa-XL 模型用于 Masked LM
        ("xmod", "XmodForMaskedLM"),                    # Xmod 模型用于 Masked LM
        ("yoso", "YosoForMaskedLM"),                    # Yoso 模型用于 Masked LM
    ]
# 定义用于对象检测模型的名称映射字典，使用有序字典确保顺序性
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

# 定义用于零样本对象检测模型的名称映射字典，使用有序字典确保顺序性
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # 零样本对象检测模型映射
        ("owlv2", "Owlv2ForObjectDetection"),
        ("owlvit", "OwlViTForObjectDetection"),
    ]
)

# 定义深度估计模型的名称映射字典，使用有序字典确保顺序性
MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # 深度估计模型映射
        ("depth_anything", "DepthAnythingForDepthEstimation"),
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
    ]
)

# 定义序列到序列因果语言模型的名称映射字典，使用有序字典确保顺序性
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # 序列到序列因果语言模型映射
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

# 定义语音序列到序列模型的名称映射字典，使用有序字典确保顺序性
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        # 语音序列到序列模型映射
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForSpeechToText"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForSpeechToText"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderModel"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("speecht5", "SpeechT5ForSpeechToText"),
        ("whisper", "WhisperForConditionalGeneration"),
    ]
)
# 定义用于序列分类模型的名称映射字典
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    ]
)

# 定义用于问答模型的名称映射字典
MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    ]
)

# 定义用于表格问答模型的名称映射字典
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # 表格问答模型映射
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

# 定义用于视觉问答模型的名称映射字典
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForQuestionAnswering"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("vilt", "ViltForQuestionAnswering"),
    ]
)

# 定义用于文档问答模型的名称映射字典
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
    ]
)

# 定义用于标记分类模型的名称映射字典
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    ]
)

# 定义用于多项选择模型的名称映射字典
MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # 多项选择模型映射
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

# 定义用于下一句预测模型的名称映射字典
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    # 留空，等待后续添加
)
    # 定义一个包含模型名称和类名的元组列表，每个元组包含模型的简称和完整类名
    [
        ("bert", "BertForNextSentencePrediction"),  # Bert 模型的简称及其完整类名
        ("ernie", "ErnieForNextSentencePrediction"),  # Ernie 模型的简称及其完整类名
        ("fnet", "FNetForNextSentencePrediction"),  # FNet 模型的简称及其完整类名
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),  # Megatron-Bert 模型的简称及其完整类名
        ("mobilebert", "MobileBertForNextSentencePrediction"),  # MobileBERT 模型的简称及其完整类名
        ("nezha", "NezhaForNextSentencePrediction"),  # Nezha 模型的简称及其完整类名
        ("qdqbert", "QDQBertForNextSentencePrediction"),  # QDQBert 模型的简称及其完整类名
    ]
# 定义一个有序字典，用于映射音频分类模型名称到对应的类名
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 音频分类模型映射
        ("audio-spectrogram-transformer", "ASTForAudioClassification"),
        ("data2vec-audio", "Data2VecAudioForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("wav2vec2-bert", "Wav2Vec2BertForSequenceClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForSequenceClassification"),
        ("wavlm", "WavLMForSequenceClassification"),
        ("whisper", "WhisperForAudioClassification"),
    ]
)

# 定义一个有序字典，用于映射连接主义时间分类（CTC）模型名称到对应的类名
MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        # 连接主义时间分类（CTC）模型映射
        ("data2vec-audio", "Data2VecAudioForCTC"),
        ("hubert", "HubertForCTC"),
        ("mctct", "MCTCTForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("unispeech", "UniSpeechForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("wav2vec2-bert", "Wav2Vec2BertForCTC"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),
        ("wavlm", "WavLMForCTC"),
    ]
)

# 定义一个有序字典，用于映射音频帧分类模型名称到对应的类名
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 音频帧分类模型映射
        ("data2vec-audio", "Data2VecAudioForAudioFrameClassification"),
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
        ("wav2vec2-bert", "Wav2Vec2BertForAudioFrameClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForAudioFrameClassification"),
        ("wavlm", "WavLMForAudioFrameClassification"),
    ]
)

# 定义一个有序字典，用于映射音频 X-向量模型名称到对应的类名
MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        # 音频 X-向量模型映射
        ("data2vec-audio", "Data2VecAudioForXVector"),
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wav2vec2", "Wav2Vec2ForXVector"),
        ("wav2vec2-bert", "Wav2Vec2BertForXVector"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),
        ("wavlm", "WavLMForXVector"),
    ]
)

# 定义一个有序字典，用于映射文本到频谱图模型名称到对应的类名
MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        # 文本到频谱图模型映射
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("speecht5", "SpeechT5ForTextToSpeech"),
    ]
)

# 定义一个有序字典，用于映射文本到波形图模型名称到对应的类名
MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict(
    [
        # 定义了多个元组，每个元组表示一个模型名称和对应的类名
        ("bark", "BarkModel"),                            # 模型名 "bark" 对应的类名 "BarkModel"
        ("fastspeech2_conformer", "FastSpeech2ConformerWithHifiGan"),   # 模型名 "fastspeech2_conformer" 对应的类名 "FastSpeech2ConformerWithHifiGan"
        ("musicgen", "MusicgenForConditionalGeneration"),   # 模型名 "musicgen" 对应的类名 "MusicgenForConditionalGeneration"
        ("musicgen_melody", "MusicgenMelodyForConditionalGeneration"),   # 模型名 "musicgen_melody" 对应的类名 "MusicgenMelodyForConditionalGeneration"
        ("seamless_m4t", "SeamlessM4TForTextToSpeech"),     # 模型名 "seamless_m4t" 对应的类名 "SeamlessM4TForTextToSpeech"
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToSpeech"),    # 模型名 "seamless_m4t_v2" 对应的类名 "SeamlessM4Tv2ForTextToSpeech"
        ("vits", "VitsModel"),                             # 模型名 "vits" 对应的类名 "VitsModel"
    ]
# 用于零样本图像分类模型映射的有序字典
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # 零样本图像分类模型映射
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("blip", "BlipModel"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("clip", "CLIPModel"),
        ("clipseg", "CLIPSegModel"),
        ("siglip", "SiglipModel"),
    ]
)

# 用于骨干网络映射的有序字典
MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # 骨干网络映射
        ("beit", "BeitBackbone"),
        ("bit", "BitBackbone"),
        ("convnext", "ConvNextBackbone"),
        ("convnextv2", "ConvNextV2Backbone"),
        ("dinat", "DinatBackbone"),
        ("dinov2", "Dinov2Backbone"),
        ("focalnet", "FocalNetBackbone"),
        ("maskformer-swin", "MaskFormerSwinBackbone"),
        ("nat", "NatBackbone"),
        ("pvt_v2", "PvtV2Backbone"),
        ("resnet", "ResNetBackbone"),
        ("swin", "SwinBackbone"),
        ("swinv2", "Swinv2Backbone"),
        ("timm_backbone", "TimmBackbone"),
        ("vitdet", "VitDetBackbone"),
    ]
)

# 用于遮罩生成模型映射的有序字典
MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)

# 用于关键点检测模型映射的有序字典
MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        ("superpoint", "SuperPointForKeypointDetection"),
    ]
)

# 用于文本编码模型映射的有序字典
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

# 用于时间序列分类模型映射的有序字典
MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("patchtsmixer", "PatchTSMixerForTimeSeriesClassification"),
        ("patchtst", "PatchTSTForClassification"),
    ]
)

# 用于时间序列回归模型映射的有序字典
MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES = OrderedDict(
    [
        ("patchtsmixer", "PatchTSMixerForRegression"),
        ("patchtst", "PatchTSTForRegression"),
    ]
)

# 用于图像到图像映射的有序字典
MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        ("swin2sr", "Swin2SRForImageSuperResolution"),
    ]
)

# 使用懒加载自动映射生成的模型映射
MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
# 创建用于预训练模型映射的惰性自动映射对象
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)

# 创建带有语言模型头的模型映射的惰性自动映射对象
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)

# 创建用于因果语言模型的模型映射的惰性自动映射对象
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)

# 创建用于因果图像建模的模型映射的惰性自动映射对象
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES)

# 创建用于图像分类的模型映射的惰性自动映射对象
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES)

# 创建用于零样本图像分类的模型映射的惰性自动映射对象
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES)

# 创建用于图像分割的模型映射的惰性自动映射对象
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES)

# 创建用于语义分割的模型映射的惰性自动映射对象
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES)

# 创建用于实例分割的模型映射的惰性自动映射对象
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES)

# 创建用于通用分割的模型映射的惰性自动映射对象
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)

# 创建用于视频分类的模型映射的惰性自动映射对象
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES)

# 创建用于视觉到序列的模型映射的惰性自动映射对象
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)

# 创建用于视觉问答的模型映射的惰性自动映射对象
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES)

# 创建用于文档问答的模型映射的惰性自动映射对象
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES)

# 创建用于掩蔽语言模型的模型映射的惰性自动映射对象
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)

# 创建用于图像处理的模型映射的惰性自动映射对象
MODEL_FOR_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_MAPPING_NAMES)

# 创建用于掩蔽图像建模的模型映射的惰性自动映射对象
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES)

# 创建用于目标检测的模型映射的惰性自动映射对象
MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES)

# 创建用于零样本目标检测的模型映射的惰性自动映射对象
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES)

# 创建用于深度估计的模型映射的惰性自动映射对象
MODEL_FOR_DEPTH_ESTIMATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)

# 创建用于序列到序列因果语言模型的模型映射的惰性自动映射对象
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)

# 创建用于序列分类的模型映射的惰性自动映射对象
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

# 创建用于问答的模型映射的惰性自动映射对象
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES)

# 创建用于表格问答的模型映射的惰性自动映射对象
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
    # 导入变量 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
    CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_CTC_MAPPING_NAMES
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES
MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_BACKBONE_MAPPING_NAMES
MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_MASK_GENERATION_MAPPING_NAMES
MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES
MODEL_FOR_KEYPOINT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES
MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES
MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES
MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES
)

# 使用 _LazyAutoMapping 类创建模型到配置映射，基于 CONFIG_MAPPING_NAMES 和 MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES
MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)
    # 将 MODEL_WITH_LM_HEAD_MAPPING 赋值给 _model_mapping 变量
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING
# 更新 _AutoModelWithLMHead 类，自动设置头部文档为 "language modeling"
_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")

# 定义 AutoModelForCausalLM 类，使用 MODEL_FOR_CAUSAL_LM_MAPPING 映射
class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

# 更新 AutoModelForCausalLM 类，自动设置头部文档为 "causal language modeling"
AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")

# 定义 AutoModelForMaskedLM 类，使用 MODEL_FOR_MASKED_LM_MAPPING 映射
class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING

# 更新 AutoModelForMaskedLM 类，自动设置头部文档为 "masked language modeling"
AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")

# 定义 AutoModelForSeq2SeqLM 类，使用 MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING 映射
class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING

# 更新 AutoModelForSeq2SeqLM 类，自动设置头部文档为 "sequence-to-sequence language modeling"
# 同时设置示例的检查点为 "google-t5/t5-base"
AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google-t5/t5-base",
)

# 定义 AutoModelForSequenceClassification 类，使用 MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING 映射
class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

# 更新 AutoModelForSequenceClassification 类，自动设置头部文档为 "sequence classification"
AutoModelForSequenceClassification = auto_class_update(AutoModelForSequenceClassification, head_doc="sequence classification")

# 定义 AutoModelForQuestionAnswering 类，使用 MODEL_FOR_QUESTION_ANSWERING_MAPPING 映射
class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING

# 更新 AutoModelForQuestionAnswering 类，自动设置头部文档为 "question answering"
AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")

# 定义 AutoModelForTableQuestionAnswering 类，使用 MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING 映射
class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING

# 更新 AutoModelForTableQuestionAnswering 类，自动设置头部文档为 "table question answering"
# 同时设置示例的检查点为 "google/tapas-base-finetuned-wtq"
AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)

# 定义 AutoModelForVisualQuestionAnswering 类，使用 MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING 映射
class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING

# 更新 AutoModelForVisualQuestionAnswering 类，自动设置头部文档为 "visual question answering"
# 同时设置示例的检查点为 "dandelin/vilt-b32-finetuned-vqa"
AutoModelForVisualQuestionAnswering = auto_class_update(
    AutoModelForVisualQuestionAnswering,
    head_doc="visual question answering",
    checkpoint_for_example="dandelin/vilt-b32-finetuned-vqa",
)

# 定义 AutoModelForDocumentQuestionAnswering 类，使用 MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING 映射
class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING

# 更新 AutoModelForDocumentQuestionAnswering 类，自动设置头部文档为 "document question answering"
# 同时设置示例的检查点为 'impira/layoutlm-document-qa", revision="52e01b3'
AutoModelForDocumentQuestionAnswering = auto_class_update(
    AutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)

# 定义 AutoModelForTokenClassification 类，使用 MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING 映射
class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING

# 更新 AutoModelForTokenClassification 类，自动设置头部文档为 "token classification"
AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")

# 定义 AutoModelForMultipleChoice 类，使用 MODEL_FOR_MULTIPLE_CHOICE_MAPPING 映射
class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING

# 更新 AutoModelForMultipleChoice 类，自动设置头部文档为 "multiple choice"
AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")

# 定义 AutoModelForNextSentencePrediction 类，使用 MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING 映射
class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING

# 更新 AutoModelForNextSentencePrediction 类，未完成的部分，可能有其它设置或定义。
    # 导入 AutoModelForNextSentencePrediction 类，并为其指定 head_doc 参数为 "next sentence prediction"
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
class AutoModelForImageClassification(_BaseAutoModelClass):
    # 自动化生成的图像分类模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    # 自动化生成的零样本图像分类模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


AutoModelForZeroShotImageClassification = auto_class_update(
    AutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
)


class AutoModelForImageSegmentation(_BaseAutoModelClass):
    # 自动化生成的图像分割模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING


AutoModelForImageSegmentation = auto_class_update(AutoModelForImageSegmentation, head_doc="image segmentation")


class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    # 自动化生成的语义分割模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


AutoModelForSemanticSegmentation = auto_class_update(
    AutoModelForSemanticSegmentation, head_doc="semantic segmentation"
)


class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    # 自动化生成的通用图像分割模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING


AutoModelForUniversalSegmentation = auto_class_update(
    AutoModelForUniversalSegmentation, head_doc="universal image segmentation"
)


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    # 自动化生成的实例分割模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


AutoModelForInstanceSegmentation = auto_class_update(
    AutoModelForInstanceSegmentation, head_doc="instance segmentation"
)


class AutoModelForObjectDetection(_BaseAutoModelClass):
    # 自动化生成的物体检测模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")


class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    # 自动化生成的零样本物体检测模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING


AutoModelForZeroShotObjectDetection = auto_class_update(
    AutoModelForZeroShotObjectDetection, head_doc="zero-shot object detection"
)


class AutoModelForDepthEstimation(_BaseAutoModelClass):
    # 自动化生成的深度估计模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING


AutoModelForDepthEstimation = auto_class_update(AutoModelForDepthEstimation, head_doc="depth estimation")


class AutoModelForVideoClassification(_BaseAutoModelClass):
    # 自动化生成的视频分类模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING


AutoModelForVideoClassification = auto_class_update(AutoModelForVideoClassification, head_doc="video classification")


class AutoModelForVision2Seq(_BaseAutoModelClass):
    # 自动化生成的视觉到文本模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")


class AutoModelForAudioClassification(_BaseAutoModelClass):
    # 自动化生成的音频分类模型类，使用预定义的模型映射
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")
    # 将 MODEL_FOR_CTC_MAPPING 赋值给 _model_mapping
    _model_mapping = MODEL_FOR_CTC_MAPPING
# 使用 auto_class_update 函数更新 AutoModelForCTC 类，添加头部文档说明
AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")

# 定义 AutoModelForSpeechSeq2Seq 类，映射到 MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING

# 使用 auto_class_update 函数更新 AutoModelForSpeechSeq2Seq 类，添加头部文档说明
AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)

# 定义 AutoModelForAudioFrameClassification 类，映射到 MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING
class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING

# 使用 auto_class_update 函数更新 AutoModelForAudioFrameClassification 类，添加头部文档说明
AutoModelForAudioFrameClassification = auto_class_update(
    AutoModelForAudioFrameClassification, head_doc="audio frame (token) classification"
)

# 定义 AutoModelForAudioXVector 类，映射到 MODEL_FOR_AUDIO_XVECTOR_MAPPING
class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING

# 定义 AutoModelForTextToSpectrogram 类，映射到 MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING
class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING

# 定义 AutoModelForTextToWaveform 类，映射到 MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING
class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING

# 定义 AutoBackbone 类，映射到 MODEL_FOR_BACKBONE_MAPPING
class AutoBackbone(_BaseAutoBackboneClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING

# 使用 auto_class_update 函数更新 AutoModelForAudioXVector 类，添加头部文档说明
AutoModelForAudioXVector = auto_class_update(AutoModelForAudioXVector, head_doc="audio retrieval via x-vector")

# 定义 AutoModelForMaskedImageModeling 类，映射到 MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING
class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING

# 使用 auto_class_update 函数更新 AutoModelForMaskedImageModeling 类，添加头部文档说明
AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")

# 定义 AutoModelWithLMHead 类，继承自 _AutoModelWithLMHead
class AutoModelWithLMHead(_AutoModelWithLMHead):
    # 从给定配置创建对象的类方法，发出未来版本移除警告
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    # 从预训练模型创建对象的类方法，发出未来版本移除警告
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
```