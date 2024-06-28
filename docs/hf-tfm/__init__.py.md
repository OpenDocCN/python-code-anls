# `.\__init__.py`

```py
`
# 版权声明和许可证信息，指明代码的使用和分发条件
# 版本号定义
__version__ = "4.39.0"

# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入依赖版本检查模块
from . import dependency_versions_check
# 导入工具函数和异常类
from .utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_g2p_en_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchaudio_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)

# 获取日志记录器对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义导入结构，用于延迟实际导入
# 这些对象将在被请求时导入，而不是立即导入
_import_structure = {
    "audio_utils": [],
    "benchmark": [],
    "commands": [],
    "configuration_utils": ["PretrainedConfig"],
    "convert_graph_to_onnx": [],
    "convert_slow_tokenizers_checkpoints_to_fast": [],
    "convert_tf_hub_seq_to_seq_bert_to_pytorch": [],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    # 定义了数据处理相关的类名称列表
    "data.data_collator": [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    # 空列表，未包含任何数据指标
    "data.metrics": [],
    # 空列表，未包含任何数据处理器
    "data.processors": [],
    # 空列表，未包含任何调试工具相关的内容
    "debug_utils": [],
    # 空列表，未包含任何 DeepSpeed 相关的内容
    "deepspeed": [],
    # 空列表，未包含任何依赖版本检查相关的内容
    "dependency_versions_check": [],
    # 空列表，未包含任何依赖版本表格相关的内容
    "dependency_versions_table": [],
    # 空列表，未包含任何动态模块工具相关的内容
    "dynamic_module_utils": [],
    # 定义了特征提取序列工具类名称列表
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    # 定义了特征提取工具类名称列表
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    # 空列表，未包含任何文件工具相关的内容
    "file_utils": [],
    # 定义了生成任务相关的类名称列表
    "generation": ["GenerationConfig", "TextIteratorStreamer", "TextStreamer"],
    # 定义了 Hugging Face 参数解析器类名称列表
    "hf_argparser": ["HfArgumentParser"],
    # 空列表，未包含任何超参数搜索相关的内容
    "hyperparameter_search": [],
    # 空列表，未包含任何图像变换相关的内容
    "image_transforms": [],
    # 定义了集成工具是否可用的函数名称列表
    "integrations": [
        "is_clearml_available",
        "is_comet_available",
        "is_dvclive_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    # 定义了模型卡片相关的类名称列表
    "modelcard": ["ModelCard"],
    # 定义了 TensorFlow 和 PyTorch 模型工具类名称列表
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",
        "load_pytorch_checkpoint_in_tf2_model",
        "load_pytorch_model_in_tf2_model",
        "load_pytorch_weights_in_tf2_model",
        "load_tf2_checkpoint_in_pytorch_model",
        "load_tf2_model_in_pytorch_model",
        "load_tf2_weights_in_pytorch_model",
    ],
    # 空列表，未包含任何模型相关的内容
    "models": [],
    # 定义了 Albert 模型相关的类名称列表和预训练配置映射
    "models.albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig"],
    # 定义了 Align 模型相关的类名称列表和预训练配置映射
    "models.align": [
        "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AlignConfig",
        "AlignProcessor",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    # 定义了 AltCLIP 模型相关的类名称列表和预训练配置映射
    "models.altclip": [
        "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AltCLIPConfig",
        "AltCLIPProcessor",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    # 定义了 Audio Spectrogram Transformer 模型相关的类名称列表和预训练配置映射
    "models.audio_spectrogram_transformer": [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ASTConfig",
        "ASTFeatureExtractor",
    ],
    # 定义了 Auto 模型相关的类名称列表和预训练配置映射
    "models.auto": [
        "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CONFIG_MAPPING",
        "FEATURE_EXTRACTOR_MAPPING",
        "IMAGE_PROCESSOR_MAPPING",
        "MODEL_NAMES_MAPPING",
        "PROCESSOR_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoFeatureExtractor",
        "AutoImageProcessor",
        "AutoProcessor",
        "AutoTokenizer",
    ],
    # 定义了 Autoformer 模型相关的类名称列表和预训练配置映射
    "models.autoformer": [
        "AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AutoformerConfig",
    ],
    "models.bark": [
        "BarkCoarseConfig",  # 定义了模块 'models.bark' 下的 'BarkCoarseConfig' 类
        "BarkConfig",  # 定义了模块 'models.bark' 下的 'BarkConfig' 类
        "BarkFineConfig",  # 定义了模块 'models.bark' 下的 'BarkFineConfig' 类
        "BarkProcessor",  # 定义了模块 'models.bark' 下的 'BarkProcessor' 类
        "BarkSemanticConfig",  # 定义了模块 'models.bark' 下的 'BarkSemanticConfig' 类
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],  # 定义了模块 'models.bart' 下的 'BartConfig' 类和 'BartTokenizer' 类
    "models.barthez": [],  # 'models.barthez' 模块为空，没有类或对象定义
    "models.bartpho": [],  # 'models.bartpho' 模块为空，没有类或对象定义
    "models.beit": ["BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BeitConfig"],  # 定义了模块 'models.beit' 下的 'BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象和 'BeitConfig' 类
    "models.bert": [
        "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.bert' 下的 'BERT_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BasicTokenizer",  # 定义了模块 'models.bert' 下的 'BasicTokenizer' 类
        "BertConfig",  # 定义了模块 'models.bert' 下的 'BertConfig' 类
        "BertTokenizer",  # 定义了模块 'models.bert' 下的 'BertTokenizer' 类
        "WordpieceTokenizer",  # 定义了模块 'models.bert' 下的 'WordpieceTokenizer' 类
    ],
    "models.bert_generation": ["BertGenerationConfig"],  # 定义了模块 'models.bert_generation' 下的 'BertGenerationConfig' 类
    "models.bert_japanese": [
        "BertJapaneseTokenizer",  # 定义了模块 'models.bert_japanese' 下的 'BertJapaneseTokenizer' 类
        "CharacterTokenizer",  # 定义了模块 'models.bert_japanese' 下的 'CharacterTokenizer' 类
        "MecabTokenizer",  # 定义了模块 'models.bert_japanese' 下的 'MecabTokenizer' 类
    ],
    "models.bertweet": ["BertweetTokenizer"],  # 定义了模块 'models.bertweet' 下的 'BertweetTokenizer' 类
    "models.big_bird": ["BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigBirdConfig"],  # 定义了模块 'models.big_bird' 下的 'BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象和 'BigBirdConfig' 类
    "models.bigbird_pegasus": [
        "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.bigbird_pegasus' 下的 'BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BigBirdPegasusConfig",  # 定义了模块 'models.bigbird_pegasus' 下的 'BigBirdPegasusConfig' 类
    ],
    "models.biogpt": [
        "BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.biogpt' 下的 'BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BioGptConfig",  # 定义了模块 'models.biogpt' 下的 'BioGptConfig' 类
        "BioGptTokenizer",  # 定义了模块 'models.biogpt' 下的 'BioGptTokenizer' 类
    ],
    "models.bit": ["BIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BitConfig"],  # 定义了模块 'models.bit' 下的 'BIT_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象和 'BitConfig' 类
    "models.blenderbot": [
        "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.blenderbot' 下的 'BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BlenderbotConfig",  # 定义了模块 'models.blenderbot' 下的 'BlenderbotConfig' 类
        "BlenderbotTokenizer",  # 定义了模块 'models.blenderbot' 下的 'BlenderbotTokenizer' 类
    ],
    "models.blenderbot_small": [
        "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.blenderbot_small' 下的 'BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BlenderbotSmallConfig",  # 定义了模块 'models.blenderbot_small' 下的 'BlenderbotSmallConfig' 类
        "BlenderbotSmallTokenizer",  # 定义了模块 'models.blenderbot_small' 下的 'BlenderbotSmallTokenizer' 类
    ],
    "models.blip": [
        "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.blip' 下的 'BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BlipConfig",  # 定义了模块 'models.blip' 下的 'BlipConfig' 类
        "BlipProcessor",  # 定义了模块 'models.blip' 下的 'BlipProcessor' 类
        "BlipTextConfig",  # 定义了模块 'models.blip' 下的 'BlipTextConfig' 类
        "BlipVisionConfig",  # 定义了模块 'models.blip' 下的 'BlipVisionConfig' 类
    ],
    "models.blip_2": [
        "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.blip_2' 下的 'BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "Blip2Config",  # 定义了模块 'models.blip_2' 下的 'Blip2Config' 类
        "Blip2Processor",  # 定义了模块 'models.blip_2' 下的 'Blip2Processor' 类
        "Blip2QFormerConfig",  # 定义了模块 'models.blip_2' 下的 'Blip2QFormerConfig' 类
        "Blip2VisionConfig",  # 定义了模块 'models.blip_2' 下的 'Blip2VisionConfig' 类
    ],
    "models.bloom": ["BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP", "BloomConfig"],  # 定义了模块 'models.bloom' 下的 'BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象和 'BloomConfig' 类
    "models.bridgetower": [
        "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.bridgetower' 下的 'BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BridgeTowerConfig",  # 定义了模块 'models.bridgetower' 下的 'BridgeTowerConfig' 类
        "BridgeTowerProcessor",  # 定义了模块 'models.bridgetower' 下的 'BridgeTowerProcessor' 类
        "BridgeTowerTextConfig",  # 定义了模块 'models.bridgetower' 下的 'BridgeTowerTextConfig' 类
        "BridgeTowerVisionConfig",  # 定义了模块 'models.bridgetower' 下的 'BridgeTowerVisionConfig' 类
    ],
    "models.bros": [
        "BROS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了模块 'models.bros' 下的 'BROS_PRETRAINED_CONFIG_ARCHIVE_MAP' 对象
        "BrosConfig",  # 定义了模块 'models.bros' 下的 'BrosConfig' 类
        "BrosProcessor",  # 定义了模块 'models.bros' 下的 'BrosProcessor' 类
    ],
    "models.byt5": ["ByT5Tokenizer"],  # 定义了模块 'models.byt5' 下的 'ByT5Tokenizer' 类
    "models.camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig"],  # 定义了模块 'models.camembert' 下的 'CAMEMBERT_PRETRAINED
    "models.clip": [
        "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLIPConfig",
        "CLIPProcessor",
        "CLIPTextConfig",
        "CLIPTokenizer",
        "CLIPVisionConfig",
    ],
    "models.clipseg": [
        "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CLIPSegConfig",
        "CLIPSegProcessor",
        "CLIPSegTextConfig",
        "CLIPSegVisionConfig",
    ],
    "models.clvp": [
        "CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ClvpConfig",
        "ClvpDecoderConfig",
        "ClvpEncoderConfig",
        "ClvpFeatureExtractor",
        "ClvpProcessor",
        "ClvpTokenizer",
    ],
    "models.code_llama": [],
    "models.codegen": [
        "CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CodeGenConfig",
        "CodeGenTokenizer",
    ],
    "models.cohere": ["COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP", "CohereConfig"],
    "models.conditional_detr": [
        "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConditionalDetrConfig",
    ],
    "models.convbert": [
        "CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConvBertConfig",
        "ConvBertTokenizer",
    ],
    "models.convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig"],
    "models.convnextv2": [
        "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConvNextV2Config",
    ],
    "models.cpm": [],
    "models.cpmant": [
        "CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CpmAntConfig",
        "CpmAntTokenizer",
    ],
    "models.ctrl": [
        "CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "CTRLConfig",
        "CTRLTokenizer",
    ],
    "models.cvt": ["CVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CvtConfig"],
    "models.data2vec": [
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Data2VecAudioConfig",
        "Data2VecTextConfig",
        "Data2VecVisionConfig",
    ],
    "models.deberta": [
        "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DebertaConfig",
        "DebertaTokenizer",
    ],
    "models.deberta_v2": [
        "DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DebertaV2Config",
    ],
    "models.decision_transformer": [
        "DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DecisionTransformerConfig",
    ],
    "models.deformable_detr": [
        "DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DeformableDetrConfig",
    ],
    "models.deit": ["DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeiTConfig"],
    "models.deprecated": [],
    "models.deprecated.bort": [],
    "models.deprecated.mctct": [
        "MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MCTCTConfig",
        "MCTCTFeatureExtractor",
        "MCTCTProcessor",
    ],
    "models.deprecated.mmbt": ["MMBTConfig"],
    "models.deprecated.open_llama": [
        "OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "OpenLlamaConfig",
    ],


注释：

    "models.clip": [
        "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CLIP 模型预训练配置文件映射
        "CLIPConfig",                         # CLIP 模型配置
        "CLIPProcessor",                      # CLIP 模型处理器
        "CLIPTextConfig",                     # CLIP 文本配置
        "CLIPTokenizer",                      # CLIP 分词器
        "CLIPVisionConfig",                   # CLIP 视觉配置
    ],
    "models.clipseg": [
        "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CLIPSeg 模型预训练配置文件映射
        "CLIPSegConfig",                         # CLIPSeg 模型配置
        "CLIPSegProcessor",                      # CLIPSeg 模型处理器
        "CLIPSegTextConfig",                     # CLIPSeg 文本配置
        "CLIPSegVisionConfig",                   # CLIPSeg 视觉配置
    ],
    "models.clvp": [
        "CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP",   # CLVP 模型预训练配置文件映射
        "ClvpConfig",                           # CLVP 模型配置
        "ClvpDecoderConfig",                    # CLVP 解码器配置
        "ClvpEncoderConfig",                    # CLVP 编码器配置
        "ClvpFeatureExtractor",                 # CLVP 特征提取器
        "ClvpProcessor",                        # CLVP 模型处理器
        "ClvpTokenizer",                        # CLVP 分词器
    ],
    "models.code_llama": [],                    # Code LLAMA 模型为空列表
    "models.codegen": [
        "CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CodeGen 模型预训练配置文件映射
        "CodeGenConfig",                         # CodeGen 模型配置
        "CodeGenTokenizer",                      # CodeGen 分词器
    ],
    "models.cohere": ["COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP", "CohereConfig"],  # Cohere 模型预训练配置文件映射和配置
    "models.conditional_detr": [
        "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Conditional DETR 模型预训练配置文件映射
        "ConditionalDetrConfig",                          # Conditional DETR 模型配置
    ],
    "models.convbert": [
        "CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # ConvBERT 模型预训练配置文件映射
        "ConvBertConfig",                          # ConvBERT 模型配置
        "ConvBertTokenizer",                       # ConvBERT 分词器
    ],
    "models.convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig"],  # ConvNext 模型预训练配置文件映射和配置
    "models.convnextv2": [
        "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # ConvNextV2 模型预训练配置文件映射
        "ConvNextV2Config",                          # ConvNextV2 模型配置
    ],
    "models.cpm": [],  # CPM 模型为空列表
    "models.cpmant": [
        "CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CpmAnt 模型预训练配置文件映射
        "CpmAntConfig",                          # CpmAnt 模型配置
        "CpmAntTokenizer",                       # CpmAnt 分词器
    ],
    "models.ctrl": [
        "CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CTRL 模型预训练配置文件映射
        "CTRLConfig",                          # CTRL 模型配置
        "CTRLTokenizer",                       # CTRL 分词器
    ],
    "models.cvt": ["CVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CvtConfig"],  # CVT 模型预训练配置文件映射和配置
    "models.data2vec": [
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",    # Data2Vec 文本预训练配置文件映射
        "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Data2Vec 视觉预训练配置文件映射
        "Data2VecAudioConfig",                           # Data2Vec 音频配置
        "Data2VecTextConfig",                            # Data2Vec 文本配置
        "Data2VecVisionConfig",                          # Data2Vec 视觉配置
    ],
    "models.deberta": [
        "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Deberta 模型预训练配置文件映射
        "DebertaConfig",                          # Deberta 模型配置
        "DebertaTokenizer",                       # Deberta 分词器
    ],
    "models.deberta_v2": [
        "DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DebertaV2 模型预训练配置文件映射
        "DebertaV2Config",                          # DebertaV2 模型配置
    ],
    "models.decision_transformer": [
        "DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Decision Transformer 模型预训练配置文件映射
        "DecisionTransformerConfig",                          # Decision Transformer 模型配置
    ],
    "models.deformable_d
    {
        "models.deprecated.retribert": [
            "RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "RetriBertConfig",
            "RetriBertTokenizer",
        ],
        "models.deprecated.tapex": ["TapexTokenizer"],
        "models.deprecated.trajectory_transformer": [
            "TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "TrajectoryTransformerConfig",
        ],
        "models.deprecated.transfo_xl": [
            "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "TransfoXLConfig",
            "TransfoXLCorpus",
            "TransfoXLTokenizer",
        ],
        "models.deprecated.van": ["VAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "VanConfig"],
        "models.depth_anything": ["DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP", "DepthAnythingConfig"],
        "models.deta": ["DETA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetaConfig"],
        "models.detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig"],
        "models.dialogpt": [],
        "models.dinat": ["DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DinatConfig"],
        "models.dinov2": ["DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Dinov2Config"],
        "models.distilbert": [
            "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "DistilBertConfig",
            "DistilBertTokenizer",
        ],
        "models.dit": [],
        "models.donut": [
            "DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "DonutProcessor",
            "DonutSwinConfig",
        ],
        "models.dpr": [
            "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "DPRConfig",
            "DPRContextEncoderTokenizer",
            "DPRQuestionEncoderTokenizer",
            "DPRReaderOutput",
            "DPRReaderTokenizer",
        ],
        "models.dpt": ["DPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPTConfig"],
        "models.efficientformer": [
            "EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "EfficientFormerConfig",
        ],
        "models.efficientnet": [
            "EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "EfficientNetConfig",
        ],
        "models.electra": [
            "ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "ElectraConfig",
            "ElectraTokenizer",
        ],
        "models.encodec": [
            "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "EncodecConfig",
            "EncodecFeatureExtractor",
        ],
        "models.encoder_decoder": ["EncoderDecoderConfig"],
        "models.ernie": [
            "ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "ErnieConfig",
        ],
        "models.ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
        "models.esm": ["ESM_PRETRAINED_CONFIG_ARCHIVE_MAP", "EsmConfig", "EsmTokenizer"],
        "models.falcon": ["FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP", "FalconConfig"],
        "models.fastspeech2_conformer": [
            "FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
            "FastSpeech2ConformerConfig",
            "FastSpeech2ConformerHifiGanConfig",
            "FastSpeech2ConformerTokenizer",
            "FastSpeech2ConformerWithHifiGanConfig",
        ],
    }
    
    
    注释：
    
    
    # models.deprecated.retribert 模块中的预训练配置映射、RetriBertConfig和RetriBertTokenizer类
    {
        "models.deprecated.retribert": [
            "RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
            "RetriBertConfig",  # RetriBertConfig类
            "RetriBertTokenizer",  # RetriBertTokenizer类
        ],
        # models.deprecated.tapex 模块中的TapexTokenizer类
        "models.deprecated.tapex": ["TapexTokenizer"],  # TapexTokenizer类
        # models.deprecated.trajectory_transformer 模块中的预训练配置映射和TrajectoryTransformerConfig类
        "models.deprecated.trajectory_transformer": [
            "TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
            "TrajectoryTransformerConfig",  # TrajectoryTransformerConfig类
        ],
        # models.deprecated.transfo_xl 模块中的预训练配置映射、TransfoXLConfig类、TransfoXLCorpus类和TransfoXLTokenizer类
        "models.deprecated.transfo_xl": [
            "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
            "TransfoXLConfig",  # TransfoXLConfig类
            "TransfoXLCorpus",  # TransfoXLCorpus类
            "TransfoXLTokenizer",  # TransfoXLTokenizer类
        ],
        # models.deprecated.van 模块中的预训练配置映射和VanConfig类
        "models.deprecated.van": ["VAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "VanConfig"],  # 预训练配置映射和VanConfig类
        # models.depth_anything 模块中的预训练配置映射和DepthAnythingConfig类
        "models.depth_anything": ["DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP", "DepthAnythingConfig"],  # 预训练配置映射和DepthAnythingConfig类
        # models.deta 模块中的预训练配置映射和DetaConfig类
        "models.deta": ["DETA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetaConfig"],  # 预训练配置映射和DetaConfig类
        # models.detr 模块中的预训练配置映射和DetrConfig类
        "models.detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig"],  # 预训练配置映射和DetrConfig类
        # models.dialogpt 模块为空，没有内容
        "models.dialogpt": [],  # 空列表，没有内容
        # models.dinat 模块中的预训练配置映射和DinatConfig类
        "models.dinat": ["DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DinatConfig"],  # 预训练配置映射和DinatConfig类
        # models.dinov2 模块中的预训练配置映射和Dinov2Config类
        "models.dinov2": ["DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Dinov2Config"],  # 预训练配置映射和Dinov2Config类
        # models.distilbert 模块中的预训练配置映射、DistilBertConfig类和DistilBertTokenizer类
        "models.distilbert": [
            "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
            "DistilBertConfig",  # DistilBertConfig类
            "DistilBertTokenizer",  # DistilBertTokenizer类
        ],
        # models.dit 模块为空，没有内容
        "models.dit": [],  # 空列表，没有内容
        # models.donut 模块中的DonutProcessor类、DonutSwinConfig类和预训练配置映射
        "models.donut": [
            "DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
            "DonutProcessor",  # DonutProcessor类
            "DonutSwinConfig",  # DonutSwinConfig类
        ],
        # models.dpr 模块中的多个类和预训练配置映射
        "models.dpr": [
            "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
            "DPRConfig",  # DPRConfig类
            "DPRContextEncoderTokenizer",  # DPRContextEncoderTokenizer类
            "DPRQuestionEncoderTokenizer",  # DPRQuestionEncoderTokenizer类
            "DPRReaderOutput",  # DPRReaderOutput类
            "DPRReaderTokenizer",  # DPRReaderTokenizer类
        ],
        # models.dpt 模块中的预训练配置映射和DPTConfig
    ],
    # models.flaubert 模块下的常量和类名列表
    "models.flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertTokenizer"],
    # models.flava 模块下的常量和类名列表
    "models.flava": [
        "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
    # models.fnet 模块下的常量和类名列表
    "models.fnet": ["FNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FNetConfig"],
    # models.focalnet 模块下的常量和类名列表
    "models.focalnet": ["FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FocalNetConfig"],
    # models.fsmt 模块下的常量和类名列表
    "models.fsmt": [
        "FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FSMTConfig",
        "FSMTTokenizer",
    ],
    # models.funnel 模块下的常量和类名列表
    "models.funnel": [
        "FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FunnelConfig",
        "FunnelTokenizer",
    ],
    # models.fuyu 模块下的常量和类名列表
    "models.fuyu": ["FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP", "FuyuConfig"],
    # models.gemma 模块下的常量和类名列表
    "models.gemma": ["GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "GemmaConfig"],
    # models.git 模块下的常量和类名列表
    "models.git": [
        "GIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GitConfig",
        "GitProcessor",
        "GitVisionConfig",
    ],
    # models.glpn 模块下的常量和类名列表
    "models.glpn": ["GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP", "GLPNConfig"],
    # models.gpt2 模块下的常量和类名列表
    "models.gpt2": [
        "GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GPT2Config",
        "GPT2Tokenizer",
    ],
    # models.gpt_bigcode 模块下的常量和类名列表
    "models.gpt_bigcode": [
        "GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GPTBigCodeConfig",
    ],
    # models.gpt_neo 模块下的常量和类名列表
    "models.gpt_neo": ["GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoConfig"],
    # models.gpt_neox 模块下的常量和类名列表
    "models.gpt_neox": ["GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXConfig"],
    # models.gpt_neox_japanese 模块下的常量和类名列表
    "models.gpt_neox_japanese": [
        "GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GPTNeoXJapaneseConfig",
    ],
    # models.gpt_sw3 模块下的空列表，无常量和类名
    "models.gpt_sw3": [],
    # models.gptj 模块下的常量和类名列表
    "models.gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig"],
    # models.gptsan_japanese 模块下的常量和类名列表
    "models.gptsan_japanese": [
        "GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GPTSanJapaneseConfig",
        "GPTSanJapaneseTokenizer",
    ],
    # models.graphormer 模块下的常量和类名列表
    "models.graphormer": [
        "GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GraphormerConfig",
    ],
    # models.groupvit 模块下的常量和类名列表
    "models.groupvit": [
        "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "GroupViTConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
    # models.herbert 模块下的常量和类名列表
    "models.herbert": ["HerbertTokenizer"],
    # models.hubert 模块下的常量和类名列表
    "models.hubert": ["HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "HubertConfig"],
    # models.ibert 模块下的常量和类名列表
    "models.ibert": ["IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "IBertConfig"],
    # models.idefics 模块下的常量和类名列表
    "models.idefics": [
        "IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "IdeficsConfig",
    ],
    # models.imagegpt 模块下的常量和类名列表
    "models.imagegpt": ["IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ImageGPTConfig"],
    # models.informer 模块下的常量和类名列表
    "models.informer": ["INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "InformerConfig"],
    # models.instructblip 模块下的常量和类名列表
    "models.instructblip": [
        "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "InstructBlipConfig",
        "InstructBlipProcessor",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    {
        "models.jukebox": [
            "JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 音乐盒预训练配置文件映射
            "JukeboxConfig",  # 音乐盒配置
            "JukeboxPriorConfig",  # 音乐盒先验配置
            "JukeboxTokenizer",  # 音乐盒分词器
            "JukeboxVQVAEConfig",  # 音乐盒VQ-VAE配置
        ],
        "models.kosmos2": [
            "KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Kosmos2预训练配置文件映射
            "Kosmos2Config",  # Kosmos2配置
            "Kosmos2Processor",  # Kosmos2处理器
        ],
        "models.layoutlm": [
            "LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LayoutLM预训练配置文件映射
            "LayoutLMConfig",  # LayoutLM配置
            "LayoutLMTokenizer",  # LayoutLM分词器
        ],
        "models.layoutlmv2": [
            "LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LayoutLMv2预训练配置文件映射
            "LayoutLMv2Config",  # LayoutLMv2配置
            "LayoutLMv2FeatureExtractor",  # LayoutLMv2特征提取器
            "LayoutLMv2ImageProcessor",  # LayoutLMv2图像处理器
            "LayoutLMv2Processor",  # LayoutLMv2处理器
            "LayoutLMv2Tokenizer",  # LayoutLMv2分词器
        ],
        "models.layoutlmv3": [
            "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LayoutLMv3预训练配置文件映射
            "LayoutLMv3Config",  # LayoutLMv3配置
            "LayoutLMv3FeatureExtractor",  # LayoutLMv3特征提取器
            "LayoutLMv3ImageProcessor",  # LayoutLMv3图像处理器
            "LayoutLMv3Processor",  # LayoutLMv3处理器
            "LayoutLMv3Tokenizer",  # LayoutLMv3分词器
        ],
        "models.layoutxlm": ["LayoutXLMProcessor"],  # LayoutXLM处理器
        "models.led": [
            "LED_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LED预训练配置文件映射
            "LEDConfig",  # LED配置
            "LEDTokenizer",  # LED分词器
        ],
        "models.levit": ["LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Levit预训练配置文件映射
                        "LevitConfig"],  # Levit配置
        "models.lilt": ["LILT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Lilt预训练配置文件映射
                        "LiltConfig"],  # Lilt配置
        "models.llama": ["LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Llama预训练配置文件映射
                         "LlamaConfig"],  # Llama配置
        "models.llava": [
            "LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Llava预训练配置文件映射
            "LlavaConfig",  # Llava配置
            "LlavaProcessor",  # Llava处理器
        ],
        "models.llava_next": [
            "LLAVA_NEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Llava Next预训练配置文件映射
            "LlavaNextConfig",  # Llava Next配置
            "LlavaNextProcessor",  # Llava Next处理器
        ],
        "models.longformer": [
            "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Longformer预训练配置文件映射
            "LongformerConfig",  # Longformer配置
            "LongformerTokenizer",  # Longformer分词器
        ],
        "models.longt5": ["LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LongT5预训练配置文件映射
                          "LongT5Config"],  # LongT5配置
        "models.luke": [
            "LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LUKE预训练配置文件映射
            "LukeConfig",  # LUKE配置
            "LukeTokenizer",  # LUKE分词器
        ],
        "models.lxmert": [
            "LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # LXMERT预训练配置文件映射
            "LxmertConfig",  # LXMERT配置
            "LxmertTokenizer",  # LXMERT分词器
        ],
        "models.m2m_100": ["M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP",  # M2M-100预训练配置文件映射
                           "M2M100Config"],  # M2M-100配置
        "models.mamba": ["MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Mamba预训练配置文件映射
                         "MambaConfig"],  # Mamba配置
        "models.marian": ["MarianConfig"],  # Marian配置
        "models.markuplm": [
            "MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # MarkupLM预训练配置文件映射
            "MarkupLMConfig",  # MarkupLM配置
            "MarkupLMFeatureExtractor",  # MarkupLM特征提取器
            "MarkupLMProcessor",  # MarkupLM处理器
            "MarkupLMTokenizer",  # MarkupLM分词器
        ],
        "models.mask2former": [
            "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Mask2Former预训练配置文件映射
            "Mask2FormerConfig",  # Mask2Former配置
        ],
        "models.maskformer": [
            "MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # MaskFormer预训练配置文件映射
            "MaskFormerConfig",  # MaskFormer配置
            "MaskFormerSwinConfig",  # MaskFormerSwin配置
        ],
        "models.mbart": ["MBartConfig"],  # MBart配置
        "models.mbart50": [],  # MBart50
        "models.mega": ["MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Mega预训练配置文件映射
                        "MegaConfig"],  # Mega配置
        "models.megatron_bert": [
            "MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # Megatron BERT预训练配置文件映射
            "MegatronBertConfig",  # Megatron BERT配置
        ],
    }
    # 定义一个字典，包含了多个模型名称和它们对应的空列表
    {
        "models.megatron_gpt2": [],
        "models.mgp_str": [
            "MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MgpstrConfig",  # MGPSTR 配置类
            "MgpstrProcessor",  # MGPSTR 处理器类
            "MgpstrTokenizer",  # MGPSTR 分词器类
        ],
        "models.mistral": [
            "MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MistralConfig",  # Mistral 配置类
        ],
        "models.mixtral": [
            "MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MixtralConfig",  # Mixtral 配置类
        ],
        "models.mluke": [],  # mluke 模型，空列表
        "models.mobilebert": [
            "MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MobileBertConfig",  # MobileBERT 配置类
            "MobileBertTokenizer",  # MobileBERT 分词器类
        ],
        "models.mobilenet_v1": [
            "MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MobileNetV1Config",  # MobileNetV1 配置类
        ],
        "models.mobilenet_v2": [
            "MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MobileNetV2Config",  # MobileNetV2 配置类
        ],
        "models.mobilevit": [
            "MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MobileViTConfig",  # MobileViT 配置类
        ],
        "models.mobilevitv2": [
            "MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MobileViTV2Config",  # MobileViTV2 配置类
        ],
        "models.mpnet": [
            "MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MPNetConfig",  # MPNet 配置类
            "MPNetTokenizer",  # MPNet 分词器类
        ],
        "models.mpt": [
            "MPT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MptConfig",  # MPT 配置类
        ],
        "models.mra": [
            "MRA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MraConfig",  # MRA 配置类
        ],
        "models.mt5": [
            "MT5Config",  # MT5 配置类
        ],
        "models.musicgen": [
            "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "MusicgenConfig",  # Musicgen 配置类
            "MusicgenDecoderConfig",  # Musicgen 解码器配置类
        ],
        "models.musicgen_melody": [
            "MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "MusicgenMelodyConfig",  # Musicgen Melody 配置类
            "MusicgenMelodyDecoderConfig",  # Musicgen Melody 解码器配置类
        ],
        "models.mvp": [
            "MvpConfig",  # MVP 配置类
            "MvpTokenizer",  # MVP 分词器类
        ],
        "models.nat": [
            "NAT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "NatConfig",  # Nat 配置类
        ],
        "models.nezha": [
            "NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "NezhaConfig",  # Nezha 配置类
        ],
        "models.nllb": [],  # nllb 模型，空列表
        "models.nllb_moe": [
            "NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "NllbMoeConfig",  # NLLB MOE 配置类
        ],
        "models.nougat": [
            "NougatProcessor",  # Nougat 处理器类
        ],
        "models.nystromformer": [
            "NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "NystromformerConfig",  # Nystromformer 配置类
        ],
        "models.oneformer": [
            "ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "OneFormerConfig",  # Oneformer 配置类
            "OneFormerProcessor",  # Oneformer 处理器类
        ],
        "models.openai": [
            "OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "OpenAIGPTConfig",  # OpenAI GPT 配置类
            "OpenAIGPTTokenizer",  # OpenAI GPT 分词器类
        ],
        "models.opt": [
            "OPTConfig",  # OPT 配置类
        ],
        "models.owlv2": [
            "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "Owlv2Config",  # Owlv2 配置类
            "Owlv2Processor",  # Owlv2 处理器类
            "Owlv2TextConfig",  # Owlv2 文本配置类
            "Owlv2VisionConfig",  # Owlv2 视觉配置类
        ],
        "models.owlvit": [
            "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "OwlViTConfig",  # Owlvit 配置类
            "OwlViTProcessor",  # Owlvit 处理器类
            "OwlViTTextConfig",  # Owlvit 文本配置类
            "OwlViTVisionConfig",  # Owlvit 视觉配置类
        ],
        "models.patchtsmixer": [
            "PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "PatchTSMixerConfig",  # PatchTSMixer 配置类
        ],
        "models.patchtst": [
            "PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
            "PatchTSTConfig",  # PatchTST 配置类
        ],
    }
    "models.pegasus": [
        "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PegasusConfig",
        "PegasusTokenizer",
    ],
    "models.pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],
    "models.perceiver": [
        "PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PerceiverConfig",
        "PerceiverTokenizer",
    ],
    "models.persimmon": ["PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP", "PersimmonConfig"],
    "models.phi": ["PHI_PRETRAINED_CONFIG_ARCHIVE_MAP", "PhiConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.pix2struct": [
        "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "models.plbart": ["PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "PLBartConfig"],
    "models.poolformer": [
        "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PoolFormerConfig",
    ],
    "models.pop2piano": [
        "POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pop2PianoConfig",
    ],
    "models.prophetnet": [
        "PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ProphetNetConfig",
        "ProphetNetTokenizer",
    ],
    "models.pvt": ["PVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtConfig"],
    "models.pvt_v2": ["PVT_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtV2Config"],
    "models.qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"],
    "models.qwen2": [
        "QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Qwen2Config",
        "Qwen2Tokenizer",
    ],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.realm": [
        "REALM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RealmConfig",
        "RealmTokenizer",
    ],
    "models.reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"],
    "models.regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"],
    "models.rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig"],
    "models.resnet": ["RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetConfig"],
    "models.roberta": [
        "ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaConfig",
        "RobertaTokenizer",
    ],
    "models.roberta_prelayernorm": [
        "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaPreLayerNormConfig",
    ],
    "models.roc_bert": [
        "ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RoCBertConfig",
        "RoCBertTokenizer",
    ],
    "models.roformer": [
        "ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RoFormerConfig",
        "RoFormerTokenizer",
    ],
    "models.rwkv": ["RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP", "RwkvConfig"],
    "models.sam": [
        "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],



    # 以下是一系列模型和相关的配置映射、配置类或者分词器的定义，每个条目对应于一个模型或者组件
    "models.pegasus": [
        "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PegasusConfig",
        "PegasusTokenizer",
    ],
    # Pegasus 模型及其相关配置和分词器
    "models.pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],
    # PegasusX 模型及其相关配置
    "models.perceiver": [
        "PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PerceiverConfig",
        "PerceiverTokenizer",
    ],
    # Perceiver 模型及其相关配置和分词器
    "models.persimmon": ["PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP", "PersimmonConfig"],
    # Persimmon 模型及其相关配置
    "models.phi": ["PHI_PRETRAINED_CONFIG_ARCHIVE_MAP", "PhiConfig"],
    # Phi 模型及其相关配置
    "models.phobert": ["PhobertTokenizer"],
    # Phobert 分词器
    "models.pix2struct": [
        "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    # Pix2Struct 模型及其相关配置和处理器
    "models.plbart": ["PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "PLBartConfig"],
    # PLBart 模型及其相关配置
    "models.poolformer": [
        "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PoolFormerConfig",
    ],
    # PoolFormer 模型及其相关配置
    "models.pop2piano": [
        "POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Pop2PianoConfig",
    ],
    # Pop2Piano 模型及其相关配置
    "models.prophetnet": [
        "PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ProphetNetConfig",
        "ProphetNetTokenizer",
    ],
    # ProphetNet 模型及其相关配置和分词器
    "models.pvt": ["PVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtConfig"],
    # Pvt 模型及其相关配置
    "models.pvt_v2": ["PVT_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtV2Config"],
    # PvtV2 模型及其相关配置
    "models.qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"],
    # QDQBert 模型及其相关配置
    "models.qwen2": [
        "QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Qwen2Config",
        "Qwen2Tokenizer",
    ],
    # Qwen2 模型及其相关配置和分词器
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    # Rag 模型及其相关配置和检索器
    "models.realm": [
        "REALM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RealmConfig",
        "RealmTokenizer",
    ],
    # Realm 模型及其相关配置和分词器
    "models.reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"],
    # Reformer 模型及其相关配置
    "models.regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"],
    # RegNet 模型及其相关配置
    "models.rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig"],
    # RemBert 模型及其相关配置
    "models.resnet": ["RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetConfig"],
    # ResNet 模型及其相关配置
    "models.roberta": [
        "ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaConfig",
        "RobertaTokenizer",
    ],
    # Roberta 模型及其相关配置和分词器
    "models.roberta_prelayernorm": [
        "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaPreLayerNormConfig",
    ],
    # Roberta with PreLayerNorm 模型及其相关配置
    "models.roc_bert": [
        "ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RoCBertConfig",
        "RoCBertTokenizer",
    ],
    # RoCBert 模型及其相关配置和分词器
    "models.roformer": [
        "ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RoFormerConfig",
        "RoFormerTokenizer",
    ],
    # RoFormer 模型及其相关配置和分词器
    "models.rwkv": ["RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP", "RwkvConfig"],
    # Rwkv 模型及其相关配置
    "models.sam": [
        "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    # Sam 模型及其相关配置和处理器、编码器、视觉配置
    # 定义了多个模型及其相关配置和工具的映射关系，每个键对应一个模型的相关信息列表

    "models.seamless_m4t": [
        "SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SEAMLESS_M4T 的预训练配置文件映射
        "SeamlessM4TConfig",  # SEAMLESS_M4T 的配置类
        "SeamlessM4TFeatureExtractor",  # SEAMLESS_M4T 的特征提取器类
        "SeamlessM4TProcessor",  # SEAMLESS_M4T 的处理器类
    ],
    "models.seamless_m4t_v2": [
        "SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SEAMLESS_M4T_V2 的预训练配置文件映射
        "SeamlessM4Tv2Config",  # SEAMLESS_M4T_V2 的配置类
    ],
    "models.segformer": [
        "SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SEGFORMER 的预训练配置文件映射
        "SegformerConfig",  # SEGFORMER 的配置类
    ],
    "models.seggpt": [
        "SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SEGGPT 的预训练配置文件映射
        "SegGptConfig",  # SEGGPT 的配置类
    ],
    "models.sew": [
        "SEW_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SEW 的预训练配置文件映射
        "SEWConfig",  # SEW 的配置类
    ],
    "models.sew_d": [
        "SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SEW_D 的预训练配置文件映射
        "SEWDConfig",  # SEW_D 的配置类
    ],
    "models.siglip": [
        "SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SIGLIP 的预训练配置文件映射
        "SiglipConfig",  # SIGLIP 的配置类
        "SiglipProcessor",  # SIGLIP 的处理器类
        "SiglipTextConfig",  # SIGLIP 文本处理的配置类
        "SiglipVisionConfig",  # SIGLIP 视觉处理的配置类
    ],
    "models.speech_encoder_decoder": [
        "SpeechEncoderDecoderConfig",  # 语音编解码器的配置类
    ],
    "models.speech_to_text": [
        "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 语音到文本的预训练配置文件映射
        "Speech2TextConfig",  # 语音到文本的配置类
        "Speech2TextFeatureExtractor",  # 语音到文本的特征提取器类
        "Speech2TextProcessor",  # 语音到文本的处理器类
    ],
    "models.speech_to_text_2": [
        "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 语音到文本2的预训练配置文件映射
        "Speech2Text2Config",  # 语音到文本2的配置类
        "Speech2Text2Processor",  # 语音到文本2的处理器类
        "Speech2Text2Tokenizer",  # 语音到文本2的分词器类
    ],
    "models.speecht5": [
        "SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SPEECHT5 的预训练配置文件映射
        "SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP",  # SPEECHT5 HiFiGAN 的预训练配置文件映射
        "SpeechT5Config",  # SPEECHT5 的配置类
        "SpeechT5FeatureExtractor",  # SPEECHT5 的特征提取器类
        "SpeechT5HifiGanConfig",  # SPEECHT5 HiFiGAN 的配置类
        "SpeechT5Processor",  # SPEECHT5 的处理器类
    ],
    "models.splinter": [
        "SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SPLINTER 的预训练配置文件映射
        "SplinterConfig",  # SPLINTER 的配置类
        "SplinterTokenizer",  # SPLINTER 的分词器类
    ],
    "models.squeezebert": [
        "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SQUEEZEBERT 的预训练配置文件映射
        "SqueezeBertConfig",  # SQUEEZEBERT 的配置类
        "SqueezeBertTokenizer",  # SQUEEZEBERT 的分词器类
    ],
    "models.stablelm": [
        "STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # STABLELM 的预训练配置文件映射
        "StableLmConfig",  # STABLELM 的配置类
    ],
    "models.starcoder2": [
        "STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # STARCODER2 的预训练配置文件映射
        "Starcoder2Config",  # STARCODER2 的配置类
    ],
    "models.superpoint": [
        "SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SUPERPOINT 的预训练配置文件映射
        "SuperPointConfig",  # SUPERPOINT 的配置类
    ],
    "models.swiftformer": [
        "SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SWIFTFORMER 的预训练配置文件映射
        "SwiftFormerConfig",  # SWIFTFORMER 的配置类
    ],
    "models.swin": [
        "SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SWIN 的预训练配置文件映射
        "SwinConfig",  # SWIN 的配置类
    ],
    "models.swin2sr": [
        "SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SWIN2SR 的预训练配置文件映射
        "Swin2SRConfig",  # SWIN2SR 的配置类
    ],
    "models.swinv2": [
        "SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SWINV2 的预训练配置文件映射
        "Swinv2Config",  # SWINV2 的配置类
    ],
    "models.switch_transformers": [
        "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # SWITCH_TRANSFORMERS 的预训练配置文件映射
        "SwitchTransformersConfig",  # SWITCH_TRANSFORMERS 的配置类
    ],
    "models.t5": [
        "T5_PRETRAINED_CONFIG_ARCHIVE_MAP",  # T5 的预训练配置文件映射
        "T5Config",  # T5 的配置类
    ],
    "models.table_transformer": [
        "TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # TABLE_TRANSFORMER 的预训练配置文件映射
        "TableTransformerConfig",  # TABLE_TRANSFORMER 的配置类
    ],
    "models.tapas": [
        "TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # TAPAS 的预训练配置文件映射
        "TapasConfig",  # TAPAS 的配置类
        "TapasTokenizer",  # TAPAS 的分词器类
    ],
    # 导入模型 "models.time_series_transformer" 的相关配置和类名
    "models.time_series_transformer": [
        "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TimeSeriesTransformerConfig",
    ],
    # 导入模型 "models.timesformer" 的相关配置和类名
    "models.timesformer": [
        "TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TimesformerConfig",
    ],
    # 导入模型 "models.timm_backbone" 的相关配置类名
    "models.timm_backbone": ["TimmBackboneConfig"],
    # 导入模型 "models.trocr" 的相关配置、类名和处理器名
    "models.trocr": [
        "TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TrOCRConfig",
        "TrOCRProcessor",
    ],
    # 导入模型 "models.tvlt" 的相关配置、类名和特征提取器、处理器名
    "models.tvlt": [
        "TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TvltConfig",
        "TvltFeatureExtractor",
        "TvltProcessor",
    ],
    # 导入模型 "models.tvp" 的相关配置、类名和处理器名
    "models.tvp": [
        "TVP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TvpConfig",
        "TvpProcessor",
    ],
    # 导入模型 "models.udop" 的相关配置、类名和处理器名
    "models.udop": [
        "UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UdopConfig",
        "UdopProcessor",
    ],
    # 导入模型 "models.umt5" 的相关配置类名
    "models.umt5": ["UMT5Config"],
    # 导入模型 "models.unispeech" 的相关配置和类名
    "models.unispeech": [
        "UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UniSpeechConfig",
    ],
    # 导入模型 "models.unispeech_sat" 的相关配置和类名
    "models.unispeech_sat": [
        "UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UniSpeechSatConfig",
    ],
    # 导入模型 "models.univnet" 的相关配置、类名和特征提取器名
    "models.univnet": [
        "UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UnivNetConfig",
        "UnivNetFeatureExtractor",
    ],
    # 导入模型 "models.upernet" 的相关配置类名
    "models.upernet": ["UperNetConfig"],
    # 导入模型 "models.videomae" 的相关配置类名
    "models.videomae": ["VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VideoMAEConfig"],
    # 导入模型 "models.vilt" 的相关配置、类名和特征提取器、图像处理器、处理器名
    "models.vilt": [
        "VILT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ViltConfig",
        "ViltFeatureExtractor",
        "ViltImageProcessor",
        "ViltProcessor",
    ],
    # 导入模型 "models.vipllava" 的相关配置和类名
    "models.vipllava": [
        "VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VipLlavaConfig",
    ],
    # 导入模型 "models.vision_encoder_decoder" 的相关配置类名
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],
    # 导入模型 "models.vision_text_dual_encoder" 的相关配置和类名
    "models.vision_text_dual_encoder": [
        "VisionTextDualEncoderConfig",
        "VisionTextDualEncoderProcessor",
    ],
    # 导入模型 "models.visual_bert" 的相关配置和类名
    "models.visual_bert": [
        "VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VisualBertConfig",
    ],
    # 导入模型 "models.vit" 的相关配置类名
    "models.vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],
    # 导入模型 "models.vit_hybrid" 的相关配置类名
    "models.vit_hybrid": [
        "VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ViTHybridConfig",
    ],
    # 导入模型 "models.vit_mae" 的相关配置类名
    "models.vit_mae": ["VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMAEConfig"],
    # 导入模型 "models.vit_msn" 的相关配置类名
    "models.vit_msn": ["VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMSNConfig"],
    # 导入模型 "models.vitdet" 的相关配置类名
    "models.vitdet": ["VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitDetConfig"],
    # 导入模型 "models.vitmatte" 的相关配置类名
    "models.vitmatte": ["VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitMatteConfig"],
    # 导入模型 "models.vits" 的相关配置、类名和分词器、处理器名
    "models.vits": [
        "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VitsConfig",
        "VitsTokenizer",
    ],
    # 导入模型 "models.vivit" 的相关配置和类名
    "models.vivit": [
        "VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VivitConfig",
    ],
    # 导入模型 "models.wav2vec2" 的相关配置、类名和CTC标记器、特征提取器、处理器、分词器名
    "models.wav2vec2": [
        "WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Wav2Vec2Config",
        "Wav2Vec2CTCTokenizer",
        "Wav2Vec2FeatureExtractor",
        "Wav2Vec2Processor",
        "Wav2Vec2Tokenizer",
    ],
    "models.wav2vec2_bert": [
        "WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.wav2vec2_bert 的预训练配置映射
        "Wav2Vec2BertConfig",  # Wav2Vec2BertConfig 是用于 models.wav2vec2_bert 的配置类
        "Wav2Vec2BertProcessor",  # Wav2Vec2BertProcessor 是处理 models.wav2vec2_bert 的处理器类
    ],
    "models.wav2vec2_conformer": [
        "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.wav2vec2_conformer 的预训练配置映射
        "Wav2Vec2ConformerConfig",  # Wav2Vec2ConformerConfig 是用于 models.wav2vec2_conformer 的配置类
    ],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],  # Wav2Vec2PhonemeCTCTokenizer 是 models.wav2vec2_phoneme 的音素 CTC 分词器类
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],  # Wav2Vec2ProcessorWithLM 是带有语言模型的 models.wav2vec2_with_lm 的处理器类
    "models.wavlm": [
        "WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.wavlm 的预训练配置映射
        "WavLMConfig",  # WavLMConfig 是用于 models.wavlm 的配置类
    ],
    "models.whisper": [
        "WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.whisper 的预训练配置映射
        "WhisperConfig",  # WhisperConfig 是用于 models.whisper 的配置类
        "WhisperFeatureExtractor",  # WhisperFeatureExtractor 是用于 models.whisper 的特征提取器类
        "WhisperProcessor",  # WhisperProcessor 是处理 models.whisper 的处理器类
        "WhisperTokenizer",  # WhisperTokenizer 是 models.whisper 的分词器类
    ],
    "models.x_clip": [
        "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.x_clip 的预训练配置映射
        "XCLIPConfig",  # XCLIPConfig 是用于 models.x_clip 的配置类
        "XCLIPProcessor",  # XCLIPProcessor 是处理 models.x_clip 的处理器类
        "XCLIPTextConfig",  # XCLIPTextConfig 是 models.x_clip 的文本配置类
        "XCLIPVisionConfig",  # XCLIPVisionConfig 是 models.x_clip 的视觉配置类
    ],
    "models.xglm": ["XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XGLMConfig"],  # XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xglm 的预训练配置映射，XGLMConfig 是用于 models.xglm 的配置类
    "models.xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMTokenizer"],  # XLM_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xlm 的预训练配置映射，XLMConfig 是用于 models.xlm 的配置类，XLMTokenizer 是 models.xlm 的分词器类
    "models.xlm_prophetnet": [
        "XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP",  # XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xlm_prophetnet 的预训练配置映射
        "XLMProphetNetConfig",  # XLMProphetNetConfig 是用于 models.xlm_prophetnet 的配置类
    ],
    "models.xlm_roberta": [
        "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xlm_roberta 的预训练配置映射
        "XLMRobertaConfig",  # XLMRobertaConfig 是用于 models.xlm_roberta 的配置类
    ],
    "models.xlm_roberta_xl": [
        "XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xlm_roberta_xl 的预训练配置映射
        "XLMRobertaXLConfig",  # XLMRobertaXLConfig 是用于 models.xlm_roberta_xl 的配置类
    ],
    "models.xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"],  # XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xlnet 的预训练配置映射，XLNetConfig 是用于 models.xlnet 的配置类
    "models.xmod": ["XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP", "XmodConfig"],  # XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.xmod 的预训练配置映射，XmodConfig 是用于 models.xmod 的配置类
    "models.yolos": ["YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP", "YolosConfig"],  # YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.yolos 的预训练配置映射，YolosConfig 是用于 models.yolos 的配置类
    "models.yoso": ["YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP", "YosoConfig"],  # YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP 是 models.yoso 的预训练配置映射，YosoConfig 是用于 models.yoso 的配置类
    "onnx": [],  # 空列表，表示没有与 onnx 相关的配置
    "pipelines": [
        "AudioClassificationPipeline",  # 音频分类管道
        "AutomaticSpeechRecognitionPipeline",  # 自动语音识别管道
        "Conversation",  # 对话管道
        "ConversationalPipeline",  # 会话式管道
        "CsvPipelineDataFormat",  # CSV 数据格式管道
        "DepthEstimationPipeline",  # 深度估计管道
        "DocumentQuestionAnsweringPipeline",  # 文档问答管道
        "FeatureExtractionPipeline",  # 特征提取管道
        "FillMaskPipeline",  # 填充掩码管道
        "ImageClassificationPipeline",  # 图像分类管道
        "ImageFeatureExtractionPipeline",  # 图像特征提取管道
        "ImageSegmentationPipeline",  # 图像分割管道
        "ImageToImagePipeline",  # 图像到图像管道
        "ImageToTextPipeline",  # 图像到文本管道
        "JsonPipelineDataFormat",  # JSON 数据格式管道
        "MaskGenerationPipeline",  # 掩码生成管道
        "NerPipeline",  # 命名实体识别管道
        "ObjectDetectionPipeline",  # 目标检测管道
        "PipedPipelineDataFormat",  # 管道数据格式管道
        "Pipeline",  # 管道
        "PipelineDataFormat",  # 管道数据格式
        "QuestionAnsweringPipeline",  # 问答管道
        "SummarizationPipeline",  # 摘要生成管道
        "TableQuestionAnsweringPipeline",  # 表格问答管道
        "Text2TextGenerationPipeline",  # 文本到文本生成管道
        "TextClassificationPipeline",  # 文本分类管道
        "TextGenerationPipeline",  # 文本生成管道
        "TextToAudioPipeline",  # 文本到音频管道
        "TokenClassificationPipeline",  # 标记分类管道
        "TranslationPipeline",  # 翻译管道
        "VideoClassificationPipeline",  # 视频分类管道
        "VisualQuestionAnsweringPipeline",  # 视觉问答管道
        "ZeroShotAudioClassificationPipeline",  # 零样本音频分类管道
        "ZeroShotClassificationPipeline",  # 零样本分类管道
        "ZeroShotImageClassificationPipeline",  # 零样本图像分类管道
        "ZeroShotObjectDetectionPipeline",  # 零样本目标检测管道
        "pipeline",  # 管道（通用概念）
    ],
    "processing_utils": ["ProcessorMixin"],  # 处理工具：混合处理器
    "quantizers": [],  # 量化器（空列表，暂无量化器）
    "testing_utils": [],  # 测试工具（空列表，暂无测试工具）
    "tokenization_utils": ["PreTrainedTokenizer"],  # 标记化工具：预训练标记器
    "tokenization_utils_base": [
        "AddedToken",  # 增加的标记
        "BatchEncoding",  # 批次编码
        "CharSpan",  # 字符跨度
        "PreTrainedTokenizerBase",  # 预训练标记器基类
        "SpecialTokensMixin",  # 特殊标记混合类
        "TokenSpan",  # 标记跨度
    ],
    "tools": [
        "Agent",  # 代理
        "AzureOpenAiAgent",  # Azure OpenAI 代理
        "HfAgent",  # Hugging Face 代理
        "LocalAgent",  # 本地代理
        "OpenAiAgent",  # OpenAI 代理
        "PipelineTool",  # 管道工具
        "RemoteTool",  # 远程工具
        "Tool",  # 工具
        "launch_gradio_demo",  # 启动 Gradio 演示
        "load_tool",  # 加载工具
    ],
    "trainer_callback": [
        "DefaultFlowCallback",  # 默认流程回调
        "EarlyStoppingCallback",  # 提前停止回调
        "PrinterCallback",  # 打印回调
        "ProgressCallback",  # 进度回调
        "TrainerCallback",  # 训练器回调
        "TrainerControl",  # 训练控制
        "TrainerState",  # 训练状态
    ],
    "trainer_utils": [
        "EvalPrediction",  # 评估预测
        "IntervalStrategy",  # 间隔策略
        "SchedulerType",  # 调度器类型
        "enable_full_determinism",  # 启用完全确定性
        "set_seed",  # 设置种子
    ],
    "training_args": ["TrainingArguments"],  # 训练参数：训练参数
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],  # 序列到序列训练参数：序列到序列训练参数
    "training_args_tf": ["TFTrainingArguments"],  # TensorFlow 训练参数：TensorFlow 训练参数
    "utils": [
        "CONFIG_NAME",  # 配置名称常量
        "MODEL_CARD_NAME",  # 模型卡片名称常量
        "PYTORCH_PRETRAINED_BERT_CACHE",  # PyTorch预训练BERT模型缓存常量
        "PYTORCH_TRANSFORMERS_CACHE",  # PyTorch Transformers模型缓存常量
        "SPIECE_UNDERLINE",  # 分词符号下划线常量
        "TF2_WEIGHTS_NAME",  # TensorFlow 2.x模型权重名称常量
        "TF_WEIGHTS_NAME",  # TensorFlow模型权重名称常量
        "TRANSFORMERS_CACHE",  # Transformers模型缓存常量
        "WEIGHTS_NAME",  # 模型权重名称常量
        "TensorType",  # 张量类型
        "add_end_docstrings",  # 添加结尾文档字符串函数
        "add_start_docstrings",  # 添加开头文档字符串函数
        "is_apex_available",  # 是否可用Apex库
        "is_bitsandbytes_available",  # 是否可用BitsAndBytes库
        "is_datasets_available",  # 是否可用datasets库
        "is_decord_available",  # 是否可用Decord库
        "is_faiss_available",  # 是否可用Faiss库
        "is_flax_available",  # 是否可用Flax库
        "is_keras_nlp_available",  # 是否可用Keras NLP库
        "is_phonemizer_available",  # 是否可用Phonemizer库
        "is_psutil_available",  # 是否可用psutil库
        "is_py3nvml_available",  # 是否可用py3nvml库
        "is_pyctcdecode_available",  # 是否可用PyCTCDecode库
        "is_sacremoses_available",  # 是否可用sacremoses库
        "is_safetensors_available",  # 是否可用SafeTensors库
        "is_scipy_available",  # 是否可用SciPy库
        "is_sentencepiece_available",  # 是否可用SentencePiece库
        "is_sklearn_available",  # 是否可用scikit-learn库
        "is_speech_available",  # 是否可用speech库
        "is_tensorflow_text_available",  # 是否可用TensorFlow Text库
        "is_tf_available",  # 是否可用TensorFlow库
        "is_timm_available",  # 是否可用Timm库
        "is_tokenizers_available",  # 是否可用tokenizers库
        "is_torch_available",  # 是否可用PyTorch库
        "is_torch_neuroncore_available",  # 是否可用Torch NeuronCore库
        "is_torch_npu_available",  # 是否可用Torch NPU库
        "is_torch_tpu_available",  # 是否可用Torch TPU库
        "is_torchvision_available",  # 是否可用TorchVision库
        "is_torch_xla_available",  # 是否可用Torch XLA库
        "is_torch_xpu_available",  # 是否可用Torch XPU库
        "is_vision_available",  # 是否可用Vision库
        "logging",  # 日志记录模块
    ],
    "utils.quantization_config": [
        "AqlmConfig",  # Aqlm配置类
        "AwqConfig",  # Awq配置类
        "BitsAndBytesConfig",  # BitsAndBytes配置类
        "GPTQConfig",  # GPTQ配置类
        "QuantoConfig",  # Quanto配置类
    ],
# sentencepiece-backed objects

# 尝试检查是否存在 sentencepiece 库的可用性
try:
    # 如果 sentencepiece 库不可用，引发 OptionalDependencyNotAvailable 异常
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果引发了 OptionalDependencyNotAvailable 异常，则从 utils 中导入 dummy_sentencepiece_objects
    from .utils import dummy_sentencepiece_objects

    # 将 dummy_sentencepiece_objects 中所有非私有成员的名称添加到 _import_structure["utils.dummy_sentencepiece_objects"] 列表中
    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    # 如果 sentencepiece 库可用，分别添加以下模块的 Tokenizer 到对应的 _import_structure 中的列表中
    _import_structure["models.albert"].append("AlbertTokenizer")
    _import_structure["models.barthez"].append("BarthezTokenizer")
    _import_structure["models.bartpho"].append("BartphoTokenizer")
    _import_structure["models.bert_generation"].append("BertGenerationTokenizer")
    _import_structure["models.big_bird"].append("BigBirdTokenizer")
    _import_structure["models.camembert"].append("CamembertTokenizer")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizer")
    _import_structure["models.cpm"].append("CpmTokenizer")
    _import_structure["models.deberta_v2"].append("DebertaV2Tokenizer")
    _import_structure["models.ernie_m"].append("ErnieMTokenizer")
    _import_structure["models.fnet"].append("FNetTokenizer")
    _import_structure["models.gemma"].append("GemmaTokenizer")
    _import_structure["models.gpt_sw3"].append("GPTSw3Tokenizer")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizer")
    _import_structure["models.llama"].append("LlamaTokenizer")
    _import_structure["models.m2m_100"].append("M2M100Tokenizer")
    _import_structure["models.marian"].append("MarianTokenizer")
    _import_structure["models.mbart"].append("MBartTokenizer")
    _import_structure["models.mbart50"].append("MBart50Tokenizer")
    _import_structure["models.mluke"].append("MLukeTokenizer")
    _import_structure["models.mt5"].append("MT5Tokenizer")
    _import_structure["models.nllb"].append("NllbTokenizer")
    _import_structure["models.pegasus"].append("PegasusTokenizer")
    _import_structure["models.plbart"].append("PLBartTokenizer")
    _import_structure["models.reformer"].append("ReformerTokenizer")
    _import_structure["models.rembert"].append("RemBertTokenizer")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizer")
    _import_structure["models.siglip"].append("SiglipTokenizer")
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.udop"].append("UdopTokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
    _import_structure["models.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")
    # 将 "utils.dummy_tokenizers_objects" 添加到 _import_structure 字典中，其对应的值是一个列表
    _import_structure["utils.dummy_tokenizers_objects"] = [
        # 遍历 dummy_tokenizers_objects 模块中的所有名称，但排除以 "_" 开头的名称
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # 如果不是慢速分词器，则使用快速分词器结构

    # 将 AlbertTokenizerFast 添加到 _import_structure["models.albert"] 中
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    # 将 BartTokenizerFast 添加到 _import_structure["models.bart"] 中
    _import_structure["models.bart"].append("BartTokenizerFast")
    # 将 BarthezTokenizerFast 添加到 _import_structure["models.barthez"] 中
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    # 将 BertTokenizerFast 添加到 _import_structure["models.bert"] 中
    _import_structure["models.bert"].append("BertTokenizerFast")
    # 将 BigBirdTokenizerFast 添加到 _import_structure["models.big_bird"] 中
    _import_structure["models.big_bird"].append("BigBirdTokenizerFast")
    # 将 BlenderbotTokenizerFast 添加到 _import_structure["models.blenderbot"] 中
    _import_structure["models.blenderbot"].append("BlenderbotTokenizerFast")
    # 将 BlenderbotSmallTokenizerFast 添加到 _import_structure["models.blenderbot_small"] 中
    _import_structure["models.blenderbot_small"].append("BlenderbotSmallTokenizerFast")
    # 将 BloomTokenizerFast 添加到 _import_structure["models.bloom"] 中
    _import_structure["models.bloom"].append("BloomTokenizerFast")
    # 将 CamembertTokenizerFast 添加到 _import_structure["models.camembert"] 中
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    # 将 CLIPTokenizerFast 添加到 _import_structure["models.clip"] 中
    _import_structure["models.clip"].append("CLIPTokenizerFast")
    # 将 CodeLlamaTokenizerFast 添加到 _import_structure["models.code_llama"] 中
    _import_structure["models.code_llama"].append("CodeLlamaTokenizerFast")
    # 将 CodeGenTokenizerFast 添加到 _import_structure["models.codegen"] 中
    _import_structure["models.codegen"].append("CodeGenTokenizerFast")
    # 将 CohereTokenizerFast 添加到 _import_structure["models.cohere"] 中
    _import_structure["models.cohere"].append("CohereTokenizerFast")
    # 将 ConvBertTokenizerFast 添加到 _import_structure["models.convbert"] 中
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    # 将 CpmTokenizerFast 添加到 _import_structure["models.cpm"] 中
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    # 将 DebertaTokenizerFast 添加到 _import_structure["models.deberta"] 中
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    # 将 DebertaV2TokenizerFast 添加到 _import_structure["models.deberta_v2"] 中
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    # 将 RetriBertTokenizerFast 添加到 _import_structure["models.deprecated.retribert"] 中
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    # 将 DistilBertTokenizerFast 添加到 _import_structure["models.distilbert"] 中
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    # 将 DPRContextEncoderTokenizerFast、DPRQuestionEncoderTokenizerFast、DPRReaderTokenizerFast 添加到 _import_structure["models.dpr"] 中
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoderTokenizerFast",
            "DPRQuestionEncoderTokenizerFast",
            "DPRReaderTokenizerFast",
        ]
    )
    # 将 ElectraTokenizerFast 添加到 _import_structure["models.electra"] 中
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    # 将 FNetTokenizerFast 添加到 _import_structure["models.fnet"] 中
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    # 将 FunnelTokenizerFast 添加到 _import_structure["models.funnel"] 中
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    # 将 GemmaTokenizerFast 添加到 _import_structure["models.gemma"] 中
    _import_structure["models.gemma"].append("GemmaTokenizerFast")
    # 将 GPT2TokenizerFast 添加到 _import_structure["models.gpt2"] 中
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    # 将 GPTNeoXTokenizerFast 添加到 _import_structure["models.gpt_neox"] 中
    _import_structure["models.gpt_neox"].append("GPTNeoXTokenizerFast")
    # 将 GPTNeoXJapaneseTokenizer 添加到 _import_structure["models.gpt_neox_japanese"] 中
    _import_structure["models.gpt_neox_japanese"].append("GPTNeoXJapaneseTokenizer")
    # 将 HerbertTokenizerFast 添加到 _import_structure["models.herbert"] 中
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    # 将 LayoutLMTokenizerFast 添加到 _import_structure["models.layoutlm"] 中
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    # 将 LayoutLMv2TokenizerFast 添加到 _import_structure["models.layoutlmv2"] 中
    _import_structure["models.layoutlmv2"].append("LayoutLMv2TokenizerFast")
    # 将 LayoutLMv3TokenizerFast 添加到 _import_structure["models.layoutlmv3"] 中
    _import_structure["models.layoutlmv3"].append("LayoutLMv3TokenizerFast")
    # 将 LayoutXLMTokenizerFast 添加到 _import_structure["models.layoutxlm"] 中
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizerFast")
    # 将 LEDTokenizerFast 添加到 _import_structure["models.led"] 中
    _import_structure["models.led"].append("LEDTokenizerFast")
    # 将 LlamaTokenizerFast 添加到 _import_structure["models.llama"] 中
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    # 将 LongformerTokenizerFast 添加到 _import_structure["models.longformer"] 中
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    # 将 LxmertTokenizerFast 添加到 _import_structure["models.lxmert"] 中
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    # 将 MarkupLMTokenizerFast 添加到 _import_structure["models.markuplm"] 中
    _import_structure["models.markuplm"].append("MarkupLMTokenizerFast")
    # 将 MBartTokenizerFast 添加到 _import_structure["models.mbart"] 中
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    # 将 MBart50TokenizerFast 添加到 _import_structure["models.mbart50"] 中
    _import_structure["models.mbart50"].append("MBart50TokenizerFast")
    # 将 "MobileBertTokenizerFast" 添加到 _import_structure["models.mobilebert"] 列表中
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    # 将 "MPNetTokenizerFast" 添加到 _import_structure["models.mpnet"] 列表中
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    # 将 "MT5TokenizerFast" 添加到 _import_structure["models.mt5"] 列表中
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    # 将 "MvpTokenizerFast" 添加到 _import_structure["models.mvp"] 列表中
    _import_structure["models.mvp"].append("MvpTokenizerFast")
    # 将 "NllbTokenizerFast" 添加到 _import_structure["models.nllb"] 列表中
    _import_structure["models.nllb"].append("NllbTokenizerFast")
    # 将 "NougatTokenizerFast" 添加到 _import_structure["models.nougat"] 列表中
    _import_structure["models.nougat"].append("NougatTokenizerFast")
    # 将 "OpenAIGPTTokenizerFast" 添加到 _import_structure["models.openai"] 列表中
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    # 将 "PegasusTokenizerFast" 添加到 _import_structure["models.pegasus"] 列表中
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    # 将 "Qwen2TokenizerFast" 添加到 _import_structure["models.qwen2"] 列表中
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    # 将 "RealmTokenizerFast" 添加到 _import_structure["models.realm"] 列表中
    _import_structure["models.realm"].append("RealmTokenizerFast")
    # 将 "ReformerTokenizerFast" 添加到 _import_structure["models.reformer"] 列表中
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    # 将 "RemBertTokenizerFast" 添加到 _import_structure["models.rembert"] 列表中
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    # 将 "RobertaTokenizerFast" 添加到 _import_structure["models.roberta"] 列表中
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    # 将 "RoFormerTokenizerFast" 添加到 _import_structure["models.roformer"] 列表中
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    # 将 "SeamlessM4TTokenizerFast" 添加到 _import_structure["models.seamless_m4t"] 列表中
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    # 将 "SplinterTokenizerFast" 添加到 _import_structure["models.splinter"] 列表中
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    # 将 "SqueezeBertTokenizerFast" 添加到 _import_structure["models.squeezebert"] 列表中
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    # 将 "T5TokenizerFast" 添加到 _import_structure["models.t5"] 列表中
    _import_structure["models.t5"].append("T5TokenizerFast")
    # 将 "UdopTokenizerFast" 添加到 _import_structure["models.udop"] 列表中
    _import_structure["models.udop"].append("UdopTokenizerFast")
    # 将 "WhisperTokenizerFast" 添加到 _import_structure["models.whisper"] 列表中
    _import_structure["models.whisper"].append("WhisperTokenizerFast")
    # 将 "XGLMTokenizerFast" 添加到 _import_structure["models.xglm"] 列表中
    _import_structure["models.xglm"].append("XGLMTokenizerFast")
    # 将 "XLMRobertaTokenizerFast" 添加到 _import_structure["models.xlm_roberta"] 列表中
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    # 将 "XLNetTokenizerFast" 添加到 _import_structure["models.xlnet"] 列表中
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    # 将 "PreTrainedTokenizerFast" 添加到 _import_structure["tokenization_utils_fast"] 列表中
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]
# 检查是否安装了句子分词器和分词器模块，如果未安装则引发自定义的OptionalDependencyNotAvailable异常
try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入替代的虚拟模块，用于句子分词器和分词器
    from .utils import dummy_sentencepiece_and_tokenizers_objects

    # 将虚拟模块中的非私有名称添加到_import_structure字典中
    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # 如果依赖可用，则将相关名称添加到_import_structure字典中的"convert_slow_tokenizer"键下
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

# Tensorflow-text-specific objects
# 检查是否安装了TensorFlow Text模块，如果未安装则引发自定义的OptionalDependencyNotAvailable异常
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入替代的虚拟模块，用于TensorFlow Text
    from .utils import dummy_tensorflow_text_objects

    # 将虚拟模块中的非私有名称添加到_import_structure字典中
    _import_structure["utils.dummy_tensorflow_text_objects"] = [
        name for name in dir(dummy_tensorflow_text_objects) if not name.startswith("_")
    ]
else:
    # 如果依赖可用，则将"TFBertTokenizer"添加到"models.bert"键对应的列表中
    _import_structure["models.bert"].append("TFBertTokenizer")

# keras-nlp-specific objects
# 检查是否安装了Keras NLP模块，如果未安装则引发自定义的OptionalDependencyNotAvailable异常
try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入替代的虚拟模块，用于Keras NLP
    from .utils import dummy_keras_nlp_objects

    # 将虚拟模块中的非私有名称添加到_import_structure字典中
    _import_structure["utils.dummy_keras_nlp_objects"] = [
        name for name in dir(dummy_keras_nlp_objects) if not name.startswith("_")
    ]
else:
    # 如果依赖可用，则将"TFGPT2Tokenizer"添加到"models.gpt2"键对应的列表中
    _import_structure["models.gpt2"].append("TFGPT2Tokenizer")

# Vision-specific objects
# 检查是否安装了Vision模块，如果未安装则引发自定义的OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入替代的虚拟模块，用于Vision
    from .utils import dummy_vision_objects

    # 将虚拟模块中的非私有名称添加到_import_structure字典中
    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    # 如果依赖可用，则按需添加各种Vision模型处理器到对应的_import_structure字典中的不同键下
    _import_structure["image_processing_utils"] = ["ImageProcessingMixin"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.beit"].extend(["BeitFeatureExtractor", "BeitImageProcessor"])
    _import_structure["models.bit"].extend(["BitImageProcessor"])
    _import_structure["models.blip"].extend(["BlipImageProcessor"])
    _import_structure["models.bridgetower"].append("BridgeTowerImageProcessor")
    _import_structure["models.chinese_clip"].extend(["ChineseCLIPFeatureExtractor", "ChineseCLIPImageProcessor"])
    _import_structure["models.clip"].extend(["CLIPFeatureExtractor", "CLIPImageProcessor"])
    _import_structure["models.conditional_detr"].extend(
        ["ConditionalDetrFeatureExtractor", "ConditionalDetrImageProcessor"]
    )
    _import_structure["models.convnext"].extend(["ConvNextFeatureExtractor", "ConvNextImageProcessor"])
    _import_structure["models.deformable_detr"].extend(
        ["DeformableDetrFeatureExtractor", "DeformableDetrImageProcessor"]
    )
    _import_structure["models.deit"].extend(["DeiTFeatureExtractor", "DeiTImageProcessor"])
    _import_structure["models.deta"].append("DetaImageProcessor")
    # 将 "models.detr" 模块下的 "DetrFeatureExtractor" 和 "DetrImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor"])
    
    # 将 "models.donut" 模块下的 "DonutFeatureExtractor" 和 "DonutImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    
    # 将 "models.dpt" 模块下的 "DPTFeatureExtractor" 和 "DPTImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    
    # 将 "models.efficientformer" 模块下的 "EfficientFormerImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.efficientformer"].append("EfficientFormerImageProcessor")
    
    # 将 "models.efficientnet" 模块下的 "EfficientNetImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    
    # 将 "models.flava" 模块下的 "FlavaFeatureExtractor", "FlavaImageProcessor" 和 "FlavaProcessor" 添加到 _import_structure 字典中
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    
    # 将 "models.fuyu" 模块下的 "FuyuImageProcessor" 和 "FuyuProcessor" 添加到 _import_structure 字典中
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    
    # 将 "models.glpn" 模块下的 "GLPNFeatureExtractor" 和 "GLPNImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    
    # 将 "models.idefics" 模块下的 "IdeficsImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    
    # 将 "models.imagegpt" 模块下的 "ImageGPTFeatureExtractor" 和 "ImageGPTImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    
    # 将 "models.layoutlmv2" 模块下的 "LayoutLMv2FeatureExtractor" 和 "LayoutLMv2ImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    
    # 将 "models.layoutlmv3" 模块下的 "LayoutLMv3FeatureExtractor" 和 "LayoutLMv3ImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    
    # 将 "models.levit" 模块下的 "LevitFeatureExtractor" 和 "LevitImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    
    # 将 "models.llava_next" 模块下的 "LlavaNextImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.llava_next"].append("LlavaNextImageProcessor")
    
    # 将 "models.mask2former" 模块下的 "Mask2FormerImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    
    # 将 "models.maskformer" 模块下的 "MaskFormerFeatureExtractor" 和 "MaskFormerImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    
    # 将 "models.mobilenet_v1" 模块下的 "MobileNetV1FeatureExtractor" 和 "MobileNetV1ImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    
    # 将 "models.mobilenet_v2" 模块下的 "MobileNetV2FeatureExtractor" 和 "MobileNetV2ImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    
    # 将 "models.mobilevit" 模块下的 "MobileViTFeatureExtractor" 和 "MobileViTImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    
    # 将 "models.nougat" 模块下的 "NougatImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.nougat"].append("NougatImageProcessor")
    
    # 将 "models.oneformer" 模块下的 "OneFormerImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    
    # 将 "models.owlv2" 模块下的 "Owlv2ImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    
    # 将 "models.owlvit" 模块下的 "OwlViTFeatureExtractor" 和 "OwlViTImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    
    # 将 "models.perceiver" 模块下的 "PerceiverFeatureExtractor" 和 "PerceiverImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    
    # 将 "models.pix2struct" 模块下的 "Pix2StructImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    
    # 将 "models.poolformer" 模块下的 "PoolFormerFeatureExtractor" 和 "PoolFormerImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    
    # 将 "models.pvt" 模块下的 "PvtImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    
    # 将 "models.sam" 模块下的 "SamImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    
    # 将 "models.segformer" 模块下的 "SegformerFeatureExtractor" 和 "SegformerImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    
    # 将 "models.seggpt" 模块下的 "SegGptImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.seggpt"].extend(["SegGptImageProcessor"])
    
    # 将 "models.siglip" 模块下的 "SiglipImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.siglip"].append("SiglipImageProcessor")
    
    # 将 "models.superpoint" 模块下的 "SuperPointImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.superpoint"].extend(["SuperPointImageProcessor"])
    
    # 将 "models.swin2sr" 模块下的 "Swin2SRImageProcessor" 添加到 _import_structure 字典中
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    # 将 "TvltImageProcessor" 添加到 _import_structure 字典中的 "models.tvlt" 键对应的列表中
    _import_structure["models.tvlt"].append("TvltImageProcessor")
    
    # 将 "TvpImageProcessor" 添加到 _import_structure 字典中的 "models.tvp" 键对应的列表中
    _import_structure["models.tvp"].append("TvpImageProcessor")
    
    # 将 ["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"] 扩展添加到 _import_structure 字典中的 "models.videomae" 键对应的列表中
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    
    # 将 ["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"] 扩展添加到 _import_structure 字典中的 "models.vilt" 键对应的列表中
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    
    # 将 ["ViTFeatureExtractor", "ViTImageProcessor"] 扩展添加到 _import_structure 字典中的 "models.vit" 键对应的列表中
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    
    # 将 ["ViTHybridImageProcessor"] 扩展添加到 _import_structure 字典中的 "models.vit_hybrid" 键对应的列表中
    _import_structure["models.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    
    # 将 "VitMatteImageProcessor" 添加到 _import_structure 字典中的 "models.vitmatte" 键对应的列表中
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    
    # 将 "VivitImageProcessor" 添加到 _import_structure 字典中的 "models.vivit" 键对应的列表中
    _import_structure["models.vivit"].append("VivitImageProcessor")
    
    # 将 ["YolosFeatureExtractor", "YolosImageProcessor"] 扩展添加到 _import_structure 字典中的 "models.yolos" 键对应的列表中
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])
# 尝试检测是否存在 PyTorch 库
try:
    # 如果 PyTorch 不可用，则抛出 OptionalDependencyNotAvailable 异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入 dummy_pt_objects 模块替代
    from .utils import dummy_pt_objects

    # 将 dummy_pt_objects 模块内的非下划线开头的所有名称加入 _import_structure
    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    # 如果 PyTorch 可用，则分别设置以下模块的导入结构
    _import_structure["activations"] = []
    _import_structure["benchmark.benchmark"] = ["PyTorchBenchmark"]
    _import_structure["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
    _import_structure["cache_utils"] = ["Cache", "DynamicCache", "SinkCache", "StaticCache"]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "ForceTokensLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "LogitsWarper",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )
    _import_structure["generation_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_utils"] = ["PreTrainedModel"]

    # PyTorch 模型结构的导入部分
    # 将 "models.albert" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.albert"].extend(
        [
            "ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # ALBERT 模型的预训练模型存档列表
            "AlbertForMaskedLM",  # 用于Masked Language Modeling的ALBERT模型
            "AlbertForMultipleChoice",  # 用于多项选择任务的ALBERT模型
            "AlbertForPreTraining",  # ALBERT的预训练模型
            "AlbertForQuestionAnswering",  # 用于问答任务的ALBERT模型
            "AlbertForSequenceClassification",  # 用于序列分类任务的ALBERT模型
            "AlbertForTokenClassification",  # 用于标记分类任务的ALBERT模型
            "AlbertModel",  # ALBERT模型
            "AlbertPreTrainedModel",  # ALBERT预训练模型基类
            "load_tf_weights_in_albert",  # 加载在ALBERT中的TensorFlow权重
        ]
    )

    # 将 "models.align" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.align"].extend(
        [
            "ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST",  # ALIGN模型的预训练模型存档列表
            "AlignModel",  # ALIGN模型
            "AlignPreTrainedModel",  # ALIGN预训练模型基类
            "AlignTextModel",  # ALIGN的文本模型
            "AlignVisionModel",  # ALIGN的视觉模型
        ]
    )

    # 将 "models.altclip" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.altclip"].extend(
        [
            "ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # ALTCLIP模型的预训练模型存档列表
            "AltCLIPModel",  # ALTCLIP模型
            "AltCLIPPreTrainedModel",  # ALTCLIP预训练模型基类
            "AltCLIPTextModel",  # ALTCLIP的文本模型
            "AltCLIPVisionModel",  # ALTCLIP的视觉模型
        ]
    )

    # 将 "models.audio_spectrogram_transformer" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
            "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 音频频谱变换器模型的预训练模型存档列表
            "ASTForAudioClassification",  # 用于音频分类任务的AST模型
            "ASTModel",  # AST模型
            "ASTPreTrainedModel",  # AST预训练模型基类
        ]
    )

    # 将 "models.autoformer" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.autoformer"].extend(
        [
            "AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # AUTOFORMER模型的预训练模型存档列表
            "AutoformerForPrediction",  # 用于预测任务的Autoformer模型
            "AutoformerModel",  # Autoformer模型
            "AutoformerPreTrainedModel",  # Autoformer预训练模型基类
        ]
    )

    # 将 "models.bark" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.bark"].extend(
        [
            "BARK_PRETRAINED_MODEL_ARCHIVE_LIST",  # BARK模型的预训练模型存档列表
            "BarkCausalModel",  # 用于因果建模的BARK模型
            "BarkCoarseModel",  # 粗粒度任务的BARK模型
            "BarkFineModel",  # 细粒度任务的BARK模型
            "BarkModel",  # BARK模型
            "BarkPreTrainedModel",  # BARK预训练模型基类
            "BarkSemanticModel",  # 语义任务的BARK模型
        ]
    )

    # 将 "models.bart" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.bart"].extend(
        [
            "BART_PRETRAINED_MODEL_ARCHIVE_LIST",  # BART模型的预训练模型存档列表
            "BartForCausalLM",  # 用于因果语言建模的BART模型
            "BartForConditionalGeneration",  # 用于条件生成任务的BART模型
            "BartForQuestionAnswering",  # 用于问答任务的BART模型
            "BartForSequenceClassification",  # 用于序列分类任务的BART模型
            "BartModel",  # BART模型
            "BartPretrainedModel",  # BART预训练模型
            "BartPreTrainedModel",  # BART预训练模型基类
            "PretrainedBartModel",  # 预训练的BART模型
        ]
    )

    # 将 "models.beit" 模块的导入结构扩展，包括以下内容：
    _import_structure["models.beit"].extend(
        [
            "BEIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # BEIT模型的预训练模型存档列表
            "BeitBackbone",  # BEIT的骨干模型
            "BeitForImageClassification",  # 用于图像分类任务的BEIT模型
            "BeitForMaskedImageModeling",  # 用于图像建模任务的BEIT模型
            "BeitForSemanticSegmentation",  # 用于语义分割任务的BEIT模型
            "BeitModel",  # BEIT模型
            "BeitPreTrainedModel",  # BEIT预训练模型基类
        ]
    )
    # 将 "models.bert" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.bert"].extend(
        [
            "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # BERT 预训练模型归档列表
            "BertForMaskedLM",                    # 用于遮蔽语言建模的 BERT 模型
            "BertForMultipleChoice",               # 用于多项选择任务的 BERT 模型
            "BertForNextSentencePrediction",       # 用于下一个句子预测任务的 BERT 模型
            "BertForPreTraining",                  # 用于预训练的 BERT 模型
            "BertForQuestionAnswering",            # 用于问答任务的 BERT 模型
            "BertForSequenceClassification",       # 用于序列分类任务的 BERT 模型
            "BertForTokenClassification",          # 用于标记分类任务的 BERT 模型
            "BertLayer",                           # BERT 的层类
            "BertLMHeadModel",                     # BERT 语言建模头模型
            "BertModel",                           # BERT 模型
            "BertPreTrainedModel",                 # BERT 预训练模型基类
            "load_tf_weights_in_bert",             # 加载 TF 格式权重到 BERT 模型中的函数
        ]
    )
    # 将 "models.bert_generation" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",               # Bert Generation 解码器
            "BertGenerationEncoder",               # Bert Generation 编码器
            "BertGenerationPreTrainedModel",       # Bert Generation 预训练模型基类
            "load_tf_weights_in_bert_generation",  # 加载 TF 格式权重到 Bert Generation 模型中的函数
        ]
    )
    # 将 "models.big_bird" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.big_bird"].extend(
        [
            "BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST",   # BigBird 预训练模型归档列表
            "BigBirdForCausalLM",                      # 用于因果语言建模的 BigBird 模型
            "BigBirdForMaskedLM",                      # 用于遮蔽语言建模的 BigBird 模型
            "BigBirdForMultipleChoice",                # 用于多项选择任务的 BigBird 模型
            "BigBirdForPreTraining",                   # 用于预训练的 BigBird 模型
            "BigBirdForQuestionAnswering",             # 用于问答任务的 BigBird 模型
            "BigBirdForSequenceClassification",        # 用于序列分类任务的 BigBird 模型
            "BigBirdForTokenClassification",           # 用于标记分类任务的 BigBird 模型
            "BigBirdLayer",                            # BigBird 的层类
            "BigBirdModel",                            # BigBird 模型
            "BigBirdPreTrainedModel",                  # BigBird 预训练模型基类
            "load_tf_weights_in_big_bird",             # 加载 TF 格式权重到 BigBird 模型中的函数
        ]
    )
    # 将 "models.bigbird_pegasus" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.bigbird_pegasus"].extend(
        [
            "BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST",   # BigBirdPegasus 预训练模型归档列表
            "BigBirdPegasusForCausalLM",                      # 用于因果语言建模的 BigBirdPegasus 模型
            "BigBirdPegasusForConditionalGeneration",         # 用于条件生成任务的 BigBirdPegasus 模型
            "BigBirdPegasusForQuestionAnswering",             # 用于问答任务的 BigBirdPegasus 模型
            "BigBirdPegasusForSequenceClassification",        # 用于序列分类任务的 BigBirdPegasus 模型
            "BigBirdPegasusModel",                           # BigBirdPegasus 模型
            "BigBirdPegasusPreTrainedModel",                 # BigBirdPegasus 预训练模型基类
        ]
    )
    # 将 "models.biogpt" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.biogpt"].extend(
        [
            "BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST",   # BioGpt 预训练模型归档列表
            "BioGptForCausalLM",                      # 用于因果语言建模的 BioGpt 模型
            "BioGptForSequenceClassification",        # 用于序列分类任务的 BioGpt 模型
            "BioGptForTokenClassification",           # 用于标记分类任务的 BioGpt 模型
            "BioGptModel",                            # BioGpt 模型
            "BioGptPreTrainedModel",                  # BioGpt 预训练模型基类
        ]
    )
    # 将 "models.bit" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.bit"].extend(
        [
            "BIT_PRETRAINED_MODEL_ARCHIVE_LIST",    # Bit 预训练模型归档列表
            "BitBackbone",                          # Bit 的骨干网络
            "BitForImageClassification",            # 用于图像分类任务的 Bit 模型
            "BitModel",                             # Bit 模型
            "BitPreTrainedModel",                   # Bit 预训练模型基类
        ]
    )
    # 将 "models.blenderbot" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.blenderbot"].extend(
        [
            "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",   # Blenderbot 预训练模型归档列表
            "BlenderbotForCausalLM",                      # 用于因果语言建模的 Blenderbot 模型
            "BlenderbotForConditionalGeneration",         # 用于条件生成任务的 Blenderbot 模型
            "BlenderbotModel",                            # Blenderbot 模型
            "BlenderbotPreTrainedModel",                  # Blenderbot 预训练模型基类
        ]
    )
    # 将 "models.blenderbot_small" 模块中的特定对象列表扩展到 _import_structure 字典中
    _import_structure["models.blenderbot_small"].extend(
        [
            "BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST",   # BlenderbotSmall 预训练模型归档列表
            "BlenderbotSmallForCausalLM",                      # 用于因果语言建模的 BlenderbotSmall 模型
            "BlenderbotSmallForConditionalGeneration",         # 用于条件生成任务的 BlenderbotSmall 模型
            "BlenderbotSmallModel",                            # BlenderbotSmall 模型
            "BlenderbotSmallPreTrainedModel",                  # BlenderbotSmall 预训练模型基类
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.blip条目中
    _import_structure["models.blip"].extend(
        [
            "BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # BLIP预训练模型存档列表
            "BlipForConditionalGeneration",  # 用于条件生成的Blip模型
            "BlipForImageTextRetrieval",  # 用于图像文本检索的Blip模型
            "BlipForQuestionAnswering",  # 用于问答的Blip模型
            "BlipModel",  # Blip模型基类
            "BlipPreTrainedModel",  # Blip预训练模型基类
            "BlipTextModel",  # 用于文本任务的Blip模型
            "BlipVisionModel",  # 用于视觉任务的Blip模型
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.blip_2条目中
    _import_structure["models.blip_2"].extend(
        [
            "BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST",  # BLIP_2预训练模型存档列表
            "Blip2ForConditionalGeneration",  # 用于条件生成的Blip2模型
            "Blip2Model",  # Blip2模型基类
            "Blip2PreTrainedModel",  # Blip2预训练模型基类
            "Blip2QFormerModel",  # Blip2的Q-Former模型
            "Blip2VisionModel",  # 用于视觉任务的Blip2模型
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.bloom条目中
    _import_structure["models.bloom"].extend(
        [
            "BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST",  # BLOOM预训练模型存档列表
            "BloomForCausalLM",  # 用于因果语言建模的Bloom模型
            "BloomForQuestionAnswering",  # 用于问答的Bloom模型
            "BloomForSequenceClassification",  # 用于序列分类的Bloom模型
            "BloomForTokenClassification",  # 用于标记分类的Bloom模型
            "BloomModel",  # Bloom模型基类
            "BloomPreTrainedModel",  # Bloom预训练模型基类
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.bridgetower条目中
    _import_structure["models.bridgetower"].extend(
        [
            "BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST",  # BRIDGETOWER预训练模型存档列表
            "BridgeTowerForContrastiveLearning",  # 用于对比学习的BridgeTower模型
            "BridgeTowerForImageAndTextRetrieval",  # 用于图像文本检索的BridgeTower模型
            "BridgeTowerForMaskedLM",  # 用于掩蔽语言建模的BridgeTower模型
            "BridgeTowerModel",  # BridgeTower模型基类
            "BridgeTowerPreTrainedModel",  # BridgeTower预训练模型基类
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.bros条目中
    _import_structure["models.bros"].extend(
        [
            "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",  # BROS预训练模型存档列表
            "BrosForTokenClassification",  # 用于标记分类的Bros模型
            "BrosModel",  # Bros模型基类
            "BrosPreTrainedModel",  # Bros预训练模型基类
            "BrosProcessor",  # Bros处理器
            "BrosSpadeEEForTokenClassification",  # 用于标记分类的Bros Spade EE模型
            "BrosSpadeELForTokenClassification",  # 用于标记分类的Bros Spade EL模型
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.camembert条目中
    _import_structure["models.camembert"].extend(
        [
            "CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # CAMEMBERT预训练模型存档列表
            "CamembertForCausalLM",  # 用于因果语言建模的Camembert模型
            "CamembertForMaskedLM",  # 用于掩蔽语言建模的Camembert模型
            "CamembertForMultipleChoice",  # 用于多项选择任务的Camembert模型
            "CamembertForQuestionAnswering",  # 用于问答的Camembert模型
            "CamembertForSequenceClassification",  # 用于序列分类的Camembert模型
            "CamembertForTokenClassification",  # 用于标记分类的Camembert模型
            "CamembertModel",  # Camembert模型基类
            "CamembertPreTrainedModel",  # Camembert预训练模型基类
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.canine条目中
    _import_structure["models.canine"].extend(
        [
            "CANINE_PRETRAINED_MODEL_ARCHIVE_LIST",  # CANINE预训练模型存档列表
            "CanineForMultipleChoice",  # 用于多项选择任务的Canine模型
            "CanineForQuestionAnswering",  # 用于问答的Canine模型
            "CanineForSequenceClassification",  # 用于序列分类的Canine模型
            "CanineForTokenClassification",  # 用于标记分类的Canine模型
            "CanineLayer",  # Canine模型的层
            "CanineModel",  # Canine模型基类
            "CaninePreTrainedModel",  # Canine预训练模型基类
            "load_tf_weights_in_canine",  # 加载TensorFlow权重到Canine模型中的函数
        ]
    )
    # 将指定模块中的一组预定义字符串和类名添加到_import_structure字典中的models.chinese_clip条目中
    _import_structure["models.chinese_clip"].extend(
        [
            "CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # CHINESE_CLIP预训练模型存档列表
            "ChineseCLIPModel",  # 中文CLIP模型
            "ChineseCLIPPreTrainedModel",  # 中文CLIP预训练模型基类
            "ChineseCLIPTextModel",  # 用于文本任务的中文CLIP模型
            "ChineseCLIPVisionModel",  # 用于视觉任务的中文CLIP模型
        ]
    )
    # 将 "models.clap" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.clap"].extend(
        [
            "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ClapAudioModel",
            "ClapAudioModelWithProjection",
            "ClapFeatureExtractor",
            "ClapModel",
            "ClapPreTrainedModel",
            "ClapTextModel",
            "ClapTextModelWithProjection",
        ]
    )
    
    # 将 "models.clip" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.clip"].extend(
        [
            "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CLIPForImageClassification",
            "CLIPModel",
            "CLIPPreTrainedModel",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
        ]
    )
    
    # 将 "models.clipseg" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.clipseg"].extend(
        [
            "CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CLIPSegForImageSegmentation",
            "CLIPSegModel",
            "CLIPSegPreTrainedModel",
            "CLIPSegTextModel",
            "CLIPSegVisionModel",
        ]
    )
    
    # 将 "models.clvp" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.clvp"].extend(
        [
            "CLVP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ClvpDecoder",
            "ClvpEncoder",
            "ClvpForCausalLM",
            "ClvpModel",
            "ClvpModelForConditionalGeneration",
            "ClvpPreTrainedModel",
        ]
    )
    
    # 将 "models.codegen" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.codegen"].extend(
        [
            "CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    
    # 将 "models.cohere" 中的模块列表扩展，包括各个特定模型类
    _import_structure["models.cohere"].extend(
        [
            "CohereForCausalLM",
            "CohereModel",
            "CoherePreTrainedModel",
        ]
    )
    
    # 将 "models.conditional_detr" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.conditional_detr"].extend(
        [
            "CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    
    # 将 "models.convbert" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.convbert"].extend(
        [
            "CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertLayer",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
            "load_tf_weights_in_convbert",
        ]
    )
    
    # 将 "models.convnext" 中的模块列表扩展，包括预训练模型存档列表和各个特定模型类
    _import_structure["models.convnext"].extend(
        [
            "CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.convnextv2"].extend(
        [
            "CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # ConvNextV2模型的预训练模型存档列表
            "ConvNextV2Backbone",  # ConvNextV2的主干模型
            "ConvNextV2ForImageClassification",  # 用于图像分类的ConvNextV2模型
            "ConvNextV2Model",  # ConvNextV2模型
            "ConvNextV2PreTrainedModel",  # ConvNextV2预训练模型
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.cpmant"].extend(
        [
            "CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST",  # CpmAnt模型的预训练模型存档列表
            "CpmAntForCausalLM",  # 用于因果语言建模的CpmAnt模型
            "CpmAntModel",  # CpmAnt模型
            "CpmAntPreTrainedModel",  # CpmAnt预训练模型
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.ctrl"].extend(
        [
            "CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",  # CTRL模型的预训练模型存档列表
            "CTRLForSequenceClassification",  # 用于序列分类的CTRL模型
            "CTRLLMHeadModel",  # CTRL模型的LM头部模型
            "CTRLModel",  # CTRL模型
            "CTRLPreTrainedModel",  # CTRL预训练模型
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.cvt"].extend(
        [
            "CVT_PRETRAINED_MODEL_ARCHIVE_LIST",  # CVT模型的预训练模型存档列表
            "CvtForImageClassification",  # 用于图像分类的CVT模型
            "CvtModel",  # CVT模型
            "CvtPreTrainedModel",  # CVT预训练模型
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.data2vec"].extend(
        [
            "DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST",  # Data2Vec音频模型的预训练模型存档列表
            "DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Data2Vec文本模型的预训练模型存档列表
            "DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST",  # Data2Vec视觉模型的预训练模型存档列表
            "Data2VecAudioForAudioFrameClassification",  # 用于音频帧分类的Data2Vec音频模型
            "Data2VecAudioForCTC",  # 用于CTC的Data2Vec音频模型
            "Data2VecAudioForSequenceClassification",  # 用于序列分类的Data2Vec音频模型
            "Data2VecAudioForXVector",  # 用于X向量的Data2Vec音频模型
            "Data2VecAudioModel",  # Data2Vec音频模型
            "Data2VecAudioPreTrainedModel",  # Data2Vec音频预训练模型
            "Data2VecTextForCausalLM",  # 用于因果语言建模的Data2Vec文本模型
            "Data2VecTextForMaskedLM",  # 用于掩码语言建模的Data2Vec文本模型
            "Data2VecTextForMultipleChoice",  # 用于多选题的Data2Vec文本模型
            "Data2VecTextForQuestionAnswering",  # 用于问答的Data2Vec文本模型
            "Data2VecTextForSequenceClassification",  # 用于序列分类的Data2Vec文本模型
            "Data2VecTextForTokenClassification",  # 用于标记分类的Data2Vec文本模型
            "Data2VecTextModel",  # Data2Vec文本模型
            "Data2VecTextPreTrainedModel",  # Data2Vec文本预训练模型
            "Data2VecVisionForImageClassification",  # 用于图像分类的Data2Vec视觉模型
            "Data2VecVisionForSemanticSegmentation",  # 用于语义分割的Data2Vec视觉模型
            "Data2VecVisionModel",  # Data2Vec视觉模型
            "Data2VecVisionPreTrainedModel",  # Data2Vec视觉预训练模型
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.deberta"].extend(
        [
            "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",  # Deberta模型的预训练模型存档列表
            "DebertaForMaskedLM",  # 用于掩码语言建模的Deberta模型
            "DebertaForQuestionAnswering",  # 用于问答的Deberta模型
            "DebertaForSequenceClassification",  # 用于序列分类的Deberta模型
            "DebertaForTokenClassification",  # 用于标记分类的Deberta模型
            "DebertaModel",  # Deberta模型
            "DebertaPreTrainedModel",  # Deberta预训练模型
        ]
    )
    # 将以下模块的名称列表扩展到_import_structure字典中的对应键
    _import_structure["models.deberta_v2"].extend(
        [
            "DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",  # DebertaV2模型的预训练模型存档列表
            "DebertaV2ForMaskedLM",  # 用于掩码语言建模的DebertaV2模型
            "DebertaV2ForMultipleChoice",  # 用于多选题的DebertaV2模型
            "DebertaV2ForQuestionAnswering",  # 用于问答的DebertaV2模型
            "DebertaV2ForSequenceClassification",  # 用于序列分类的DebertaV2模型
            "DebertaV2ForTokenClassification",  # 用于标记分类的DebertaV2模型
            "DebertaV2Model",  # DebertaV2模型
            "DebertaV2PreTrainedModel",  # DebertaV2预训练模型
        ]
    )
    _import_structure["models.decision_transformer"].extend(
        [
            "DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deformable_detr"].extend(
        [
            "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DeiTForImageClassification",
            "DeiTForImageClassificationWithTeacher",
            "DeiTForMaskedImageModeling",
            "DeiTModel",
            "DeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mmbt"].extend(
        [
            "MMBTForClassification",
            "MMBTModel",
            "ModalEmbeddings"
        ]
    )
    _import_structure["models.deprecated.open_llama"].extend(
        [
            "OpenLlamaForCausalLM",
            "OpenLlamaForSequenceClassification",
            "OpenLlamaModel",
            "OpenLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.retribert"].extend(
        [
            "RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RetriBertModel",
            "RetriBertPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
            "load_tf_weights_in_transfo_xl",
        ]
    )
    _import_structure["models.deprecated.van"].extend(
        [
            "VAN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    _import_structure["models.depth_anything"].extend(
        [
            "DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DepthAnythingForDepthEstimation",
            "DepthAnythingPreTrainedModel",
        ]
    )


注释：


# 扩展导入结构中的 models.decision_transformer，包括预训练模型存档列表和相关模型类
_import_structure["models.decision_transformer"].extend(
    [
        "DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DecisionTransformerGPT2Model",
        "DecisionTransformerGPT2PreTrainedModel",
        "DecisionTransformerModel",
        "DecisionTransformerPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deformable_detr，包括预训练模型存档列表和相关模型类
_import_structure["models.deformable_detr"].extend(
    [
        "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeformableDetrForObjectDetection",
        "DeformableDetrModel",
        "DeformableDetrPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deit，包括预训练模型存档列表和相关模型类
_import_structure["models.deit"].extend(
    [
        "DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeiTForImageClassification",
        "DeiTForImageClassificationWithTeacher",
        "DeiTForMaskedImageModeling",
        "DeiTModel",
        "DeiTPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deprecated.mctct，包括预训练模型存档列表和相关模型类
_import_structure["models.deprecated.mctct"].extend(
    [
        "MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MCTCTForCTC",
        "MCTCTModel",
        "MCTCTPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deprecated.mmbt，包括相关分类和嵌入模型
_import_structure["models.deprecated.mmbt"].extend(
    [
        "MMBTForClassification",
        "MMBTModel",
        "ModalEmbeddings"
    ]
)

# 扩展导入结构中的 models.deprecated.open_llama，包括用于因果语言建模和序列分类的模型
_import_structure["models.deprecated.open_llama"].extend(
    [
        "OpenLlamaForCausalLM",
        "OpenLlamaForSequenceClassification",
        "OpenLlamaModel",
        "OpenLlamaPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deprecated.retribert，包括预训练模型存档列表和相关模型类
_import_structure["models.deprecated.retribert"].extend(
    [
        "RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RetriBertModel",
        "RetriBertPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deprecated.trajectory_transformer，包括轨迹转换模型类和预训练模型
_import_structure["models.deprecated.trajectory_transformer"].extend(
    [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TrajectoryTransformerModel",
        "TrajectoryTransformerPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.deprecated.transfo_xl，包括用于序列分类和语言建模的 TransfoXL 相关模型和工具
_import_structure["models.deprecated.transfo_xl"].extend(
    [
        "TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AdaptiveEmbedding",
        "TransfoXLForSequenceClassification",
        "TransfoXLLMHeadModel",
        "TransfoXLModel",
        "TransfoXLPreTrainedModel",
        "load_tf_weights_in_transfo_xl",
    ]
)

# 扩展导入结构中的 models.deprecated.van，包括用于图像分类的 Van 模型和相关预训练模型
_import_structure["models.deprecated.van"].extend(
    [
        "VAN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VanForImageClassification",
        "VanModel",
        "VanPreTrainedModel",
    ]
)

# 扩展导入结构中的 models.depth_anything，包括深度估计相关模型和预训练模型
_import_structure["models.depth_anything"].extend(
    [
        "DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DepthAnythingForDepthEstimation",
        "DepthAnythingPreTrainedModel",
    ]
)
    # 将 "models.deta" 模块的相关对象添加到导入结构中
    _import_structure["models.deta"].extend(
        [
            "DETA_PRETRAINED_MODEL_ARCHIVE_LIST",  # DETA 预训练模型的存档列表
            "DetaForObjectDetection",  # 用于目标检测的 Deta 模型
            "DetaModel",  # Deta 模型
            "DetaPreTrainedModel",  # Deta 预训练模型
        ]
    )
    
    # 将 "models.detr" 模块的相关对象添加到导入结构中
    _import_structure["models.detr"].extend(
        [
            "DETR_PRETRAINED_MODEL_ARCHIVE_LIST",  # DETR 预训练模型的存档列表
            "DetrForObjectDetection",  # 用于目标检测的 Detr 模型
            "DetrForSegmentation",  # 用于分割任务的 Detr 模型
            "DetrModel",  # Detr 模型
            "DetrPreTrainedModel",  # Detr 预训练模型
        ]
    )
    
    # 将 "models.dinat" 模块的相关对象添加到导入结构中
    _import_structure["models.dinat"].extend(
        [
            "DINAT_PRETRAINED_MODEL_ARCHIVE_LIST",  # DINAT 预训练模型的存档列表
            "DinatBackbone",  # Dinat 的骨干网络
            "DinatForImageClassification",  # 用于图像分类的 Dinat 模型
            "DinatModel",  # Dinat 模型
            "DinatPreTrainedModel",  # Dinat 预训练模型
        ]
    )
    
    # 将 "models.dinov2" 模块的相关对象添加到导入结构中
    _import_structure["models.dinov2"].extend(
        [
            "DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # DINOV2 预训练模型的存档列表
            "Dinov2Backbone",  # Dinov2 的骨干网络
            "Dinov2ForImageClassification",  # 用于图像分类的 Dinov2 模型
            "Dinov2Model",  # Dinov2 模型
            "Dinov2PreTrainedModel",  # Dinov2 预训练模型
        ]
    )
    
    # 将 "models.distilbert" 模块的相关对象添加到导入结构中
    _import_structure["models.distilbert"].extend(
        [
            "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # DISTILBERT 预训练模型的存档列表
            "DistilBertForMaskedLM",  # 用于遮蔽语言建模任务的 DistilBERT 模型
            "DistilBertForMultipleChoice",  # 用于多项选择任务的 DistilBERT 模型
            "DistilBertForQuestionAnswering",  # 用于问答任务的 DistilBERT 模型
            "DistilBertForSequenceClassification",  # 用于序列分类任务的 DistilBERT 模型
            "DistilBertForTokenClassification",  # 用于标记分类任务的 DistilBERT 模型
            "DistilBertModel",  # DistilBERT 模型
            "DistilBertPreTrainedModel",  # DistilBERT 预训练模型
        ]
    )
    
    # 将 "models.donut" 模块的相关对象添加到导入结构中
    _import_structure["models.donut"].extend(
        [
            "DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",  # DONUT SWIN 预训练模型的存档列表
            "DonutSwinModel",  # DonutSwin 模型
            "DonutSwinPreTrainedModel",  # DonutSwin 预训练模型
        ]
    )
    
    # 将 "models.dpr" 模块的相关对象添加到导入结构中
    _import_structure["models.dpr"].extend(
        [
            "DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPR 上下文编码器预训练模型的存档列表
            "DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPR 问题编码器预训练模型的存档列表
            "DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPR 阅读器预训练模型的存档列表
            "DPRContextEncoder",  # DPR 上下文编码器
            "DPRPretrainedContextEncoder",  # DPR 预训练上下文编码器
            "DPRPreTrainedModel",  # DPR 预训练模型
            "DPRPretrainedQuestionEncoder",  # DPR 预训练问题编码器
            "DPRPretrainedReader",  # DPR 预训练阅读器
            "DPRQuestionEncoder",  # DPR 问题编码器
            "DPRReader",  # DPR 阅读器
        ]
    )
    
    # 将 "models.dpt" 模块的相关对象添加到导入结构中
    _import_structure["models.dpt"].extend(
        [
            "DPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPT 预训练模型的存档列表
            "DPTForDepthEstimation",  # 用于深度估计任务的 DPT 模型
            "DPTForSemanticSegmentation",  # 用于语义分割任务的 DPT 模型
            "DPTModel",  # DPT 模型
            "DPTPreTrainedModel",  # DPT 预训练模型
        ]
    )
    
    # 将 "models.efficientformer" 模块的相关对象添加到导入结构中
    _import_structure["models.efficientformer"].extend(
        [
            "EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # EFFICIENTFORMER 预训练模型的存档列表
            "EfficientFormerForImageClassification",  # 用于图像分类的 EfficientFormer 模型
            "EfficientFormerForImageClassificationWithTeacher",  # 带有教师模型的图像分类 EfficientFormer 模型
            "EfficientFormerModel",  # EfficientFormer 模型
            "EfficientFormerPreTrainedModel",  # EfficientFormer 预训练模型
        ]
    )
    
    # 将 "models.efficientnet" 模块的相关对象添加到导入结构中
    _import_structure["models.efficientnet"].extend(
        [
            "EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # EFFICIENTNET 预训练模型的存档列表
            "EfficientNetForImageClassification",  # 用于图像分类的 EfficientNet 模型
            "EfficientNetModel",  # EfficientNet 模型
            "EfficientNetPreTrainedModel",  # EfficientNet 预训练模型
        ]
    )
    # 扩展 _import_structure 字典中 "models.electra" 键的值，添加多个 Electra 相关的模型和常量
    _import_structure["models.electra"].extend(
        [
            "ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ElectraForCausalLM",
            "ElectraForMaskedLM",
            "ElectraForMultipleChoice",
            "ElectraForPreTraining",
            "ElectraForQuestionAnswering",
            "ElectraForSequenceClassification",
            "ElectraForTokenClassification",
            "ElectraModel",
            "ElectraPreTrainedModel",
            "load_tf_weights_in_electra",
        ]
    )
    # 扩展 _import_structure 字典中 "models.encodec" 键的值，添加 Encodec 相关的模型和常量
    _import_structure["models.encodec"].extend(
        [
            "ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    # 向 _import_structure 字典中 "models.encoder_decoder" 键的值添加 EncoderDecoderModel
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    # 扩展 _import_structure 字典中 "models.ernie" 键的值，添加多个 Ernie 相关的模型和常量
    _import_structure["models.ernie"].extend(
        [
            "ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ErnieForCausalLM",
            "ErnieForMaskedLM",
            "ErnieForMultipleChoice",
            "ErnieForNextSentencePrediction",
            "ErnieForPreTraining",
            "ErnieForQuestionAnswering",
            "ErnieForSequenceClassification",
            "ErnieForTokenClassification",
            "ErnieModel",
            "ErniePreTrainedModel",
        ]
    )
    # 扩展 _import_structure 字典中 "models.ernie_m" 键的值，添加多个 ErnieM 相关的模型和常量
    _import_structure["models.ernie_m"].extend(
        [
            "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ErnieMForInformationExtraction",
            "ErnieMForMultipleChoice",
            "ErnieMForQuestionAnswering",
            "ErnieMForSequenceClassification",
            "ErnieMForTokenClassification",
            "ErnieMModel",
            "ErnieMPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 字典中 "models.esm" 键的值，添加多个 Esm 相关的模型和常量
    _import_structure["models.esm"].extend(
        [
            "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EsmFoldPreTrainedModel",
            "EsmForMaskedLM",
            "EsmForProteinFolding",
            "EsmForSequenceClassification",
            "EsmForTokenClassification",
            "EsmModel",
            "EsmPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 字典中 "models.falcon" 键的值，添加多个 Falcon 相关的模型和常量
    _import_structure["models.falcon"].extend(
        [
            "FALCON_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FalconForCausalLM",
            "FalconForQuestionAnswering",
            "FalconForSequenceClassification",
            "FalconForTokenClassification",
            "FalconModel",
            "FalconPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 字典中 "models.fastspeech2_conformer" 键的值，添加多个 FastSpeech2 Conformer 相关的模型和常量
    _import_structure["models.fastspeech2_conformer"].extend(
        [
            "FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FastSpeech2ConformerHifiGan",
            "FastSpeech2ConformerModel",
            "FastSpeech2ConformerPreTrainedModel",
            "FastSpeech2ConformerWithHifiGan",
        ]
    )
    # 将 "models.flaubert" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.flaubert"].extend(
        [
            "FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "FlaubertForMultipleChoice",  # 用于多项选择任务的 Flaubert 模型
            "FlaubertForQuestionAnswering",  # 用于问答任务的 Flaubert 模型
            "FlaubertForQuestionAnsweringSimple",  # 简化版本的问答任务 Flaubert 模型
            "FlaubertForSequenceClassification",  # 用于序列分类任务的 Flaubert 模型
            "FlaubertForTokenClassification",  # 用于标记分类任务的 Flaubert 模型
            "FlaubertModel",  # Flaubert 模型基类
            "FlaubertPreTrainedModel",  # Flaubert 预训练模型基类
            "FlaubertWithLMHeadModel",  # 带有语言模型头的 Flaubert 模型
        ]
    )
    # 将 "models.flava" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.flava"].extend(
        [
            "FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",  # FLAVA 模型预训练模型存档列表
            "FlavaForPreTraining",  # 用于预训练任务的 FLAVA 模型
            "FlavaImageCodebook",  # 图像编码簿模块
            "FlavaImageModel",  # 图像模型
            "FlavaModel",  # FLAVA 模型基类
            "FlavaMultimodalModel",  # 多模态 FLAVA 模型
            "FlavaPreTrainedModel",  # FLAVA 预训练模型基类
            "FlavaTextModel",  # 文本模型
        ]
    )
    # 将 "models.fnet" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.fnet"].extend(
        [
            "FNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # FNET 模型预训练模型存档列表
            "FNetForMaskedLM",  # 用于掩码语言建模任务的 FNet 模型
            "FNetForMultipleChoice",  # 用于多项选择任务的 FNet 模型
            "FNetForNextSentencePrediction",  # 用于下一个句子预测任务的 FNet 模型
            "FNetForPreTraining",  # 用于预训练任务的 FNet 模型
            "FNetForQuestionAnswering",  # 用于问答任务的 FNet 模型
            "FNetForSequenceClassification",  # 用于序列分类任务的 FNet 模型
            "FNetForTokenClassification",  # 用于标记分类任务的 FNet 模型
            "FNetLayer",  # FNet 模型的层
            "FNetModel",  # FNet 模型基类
            "FNetPreTrainedModel",  # FNet 预训练模型基类
        ]
    )
    # 将 "models.focalnet" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.focalnet"].extend(
        [
            "FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # FOCALNET 模型预训练模型存档列表
            "FocalNetBackbone",  # FocalNet 模型的骨干网络
            "FocalNetForImageClassification",  # 用于图像分类任务的 FocalNet 模型
            "FocalNetForMaskedImageModeling",  # 用于遮罩图像建模任务的 FocalNet 模型
            "FocalNetModel",  # FocalNet 模型基类
            "FocalNetPreTrainedModel",  # FocalNet 预训练模型基类
        ]
    )
    # 将 "models.fsmt" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    # 将 "models.funnel" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.funnel"].extend(
        [
            "FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",  # FUNNEL 模型预训练模型存档列表
            "FunnelBaseModel",  # Funnel 模型基类
            "FunnelForMaskedLM",  # 用于掩码语言建模任务的 Funnel 模型
            "FunnelForMultipleChoice",  # 用于多项选择任务的 Funnel 模型
            "FunnelForPreTraining",  # 用于预训练任务的 Funnel 模型
            "FunnelForQuestionAnswering",  # 用于问答任务的 Funnel 模型
            "FunnelForSequenceClassification",  # 用于序列分类任务的 Funnel 模型
            "FunnelForTokenClassification",  # 用于标记分类任务的 Funnel 模型
            "FunnelModel",  # Funnel 模型基类
            "FunnelPreTrainedModel",  # Funnel 预训练模型基类
            "load_tf_weights_in_funnel",  # 加载 TensorFlow 权重到 Funnel 模型中的函数
        ]
    )
    # 将 "models.fuyu" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.fuyu"].extend(["FuyuForCausalLM", "FuyuPreTrainedModel"])
    # 将 "models.gemma" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.gemma"].extend(
        [
            "GemmaForCausalLM",  # 用于因果语言建模任务的 Gemma 模型
            "GemmaForSequenceClassification",  # 用于序列分类任务的 Gemma 模型
            "GemmaModel",  # Gemma 模型基类
            "GemmaPreTrainedModel",  # Gemma 预训练模型基类
        ]
    )
    # 将 "models.git" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.git"].extend(
        [
            "GIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # GIT 模型预训练模型存档列表
            "GitForCausalLM",  # 用于因果语言建模任务的 Git 模型
            "GitModel",  # Git 模型基类
            "GitPreTrainedModel",  # Git 预训练模型基类
            "GitVisionModel",  # Git 视觉模型
        ]
    )
    # 将 "models.glpn" 中指定的模块名列表扩展到 _import_structure 字典的值中
    _import_structure["models.glpn"].extend(
        [
            "GLPN_PRETRAINED_MODEL_ARCHIVE_LIST",  # GLPN 模型预训练模型存档列表
            "GLPNForDepthEstimation",  # 用于深度估计任务的 GLPN 模型
            "GLPNModel",  # GLPN 模型基类
            "GLPNPreTrainedModel",  # GLPN 预训练模型基类
        ]
    )
    # 将指定的模型模块导入结构中，并扩展模块列表
    _import_structure["models.gpt2"].extend(
        [
            "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",   # GPT-2 预训练模型存档列表
            "GPT2DoubleHeadsModel",                 # GPT-2 双头模型
            "GPT2ForQuestionAnswering",             # GPT-2 问答模型
            "GPT2ForSequenceClassification",       # GPT-2 序列分类模型
            "GPT2ForTokenClassification",          # GPT-2 标记分类模型
            "GPT2LMHeadModel",                     # GPT-2 语言模型头
            "GPT2Model",                           # GPT-2 模型
            "GPT2PreTrainedModel",                 # GPT-2 预训练模型基类
            "load_tf_weights_in_gpt2",             # 加载 TensorFlow 权重到 GPT-2 模型
        ]
    )
    # 将大代码模块的相关类名添加到导入结构中
    _import_structure["models.gpt_bigcode"].extend(
        [
            "GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST",    # 大代码预训练模型存档列表
            "GPTBigCodeForCausalLM",                        # 大代码因果语言模型
            "GPTBigCodeForSequenceClassification",          # 大代码序列分类模型
            "GPTBigCodeForTokenClassification",             # 大代码标记分类模型
            "GPTBigCodeModel",                              # 大代码模型
            "GPTBigCodePreTrainedModel",                    # 大代码预训练模型基类
        ]
    )
    # 将 GPT-Neo 模块的相关类名添加到导入结构中
    _import_structure["models.gpt_neo"].extend(
        [
            "GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST",    # GPT-Neo 预训练模型存档列表
            "GPTNeoForCausalLM",                        # GPT-Neo 因果语言模型
            "GPTNeoForQuestionAnswering",               # GPT-Neo 问答模型
            "GPTNeoForSequenceClassification",          # GPT-Neo 序列分类模型
            "GPTNeoForTokenClassification",             # GPT-Neo 标记分类模型
            "GPTNeoModel",                              # GPT-Neo 模型
            "GPTNeoPreTrainedModel",                    # GPT-Neo 预训练模型基类
            "load_tf_weights_in_gpt_neo",               # 加载 TensorFlow 权重到 GPT-Neo 模型
        ]
    )
    # 将 GPT-NeoX 模块的相关类名添加到导入结构中
    _import_structure["models.gpt_neox"].extend(
        [
            "GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST",   # GPT-NeoX 预训练模型存档列表
            "GPTNeoXForCausalLM",                       # GPT-NeoX 因果语言模型
            "GPTNeoXForQuestionAnswering",              # GPT-NeoX 问答模型
            "GPTNeoXForSequenceClassification",         # GPT-NeoX 序列分类模型
            "GPTNeoXForTokenClassification",            # GPT-NeoX 标记分类模型
            "GPTNeoXLayer",                             # GPT-NeoX 模型层
            "GPTNeoXModel",                             # GPT-NeoX 模型
            "GPTNeoXPreTrainedModel",                   # GPT-NeoX 预训练模型基类
        ]
    )
    # 将 GPT-NeoX 日本语模块的相关类名添加到导入结构中
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",   # GPT-NeoX 日本语预训练模型存档列表
            "GPTNeoXJapaneseForCausalLM",                        # GPT-NeoX 日本语因果语言模型
            "GPTNeoXJapaneseLayer",                              # GPT-NeoX 日本语模型层
            "GPTNeoXJapaneseModel",                              # GPT-NeoX 日本语模型
            "GPTNeoXJapanesePreTrainedModel",                    # GPT-NeoX 日本语预训练模型基类
        ]
    )
    # 将 GPT-J 模块的相关类名添加到导入结构中
    _import_structure["models.gptj"].extend(
        [
            "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",    # GPT-J 预训练模型存档列表
            "GPTJForCausalLM",                        # GPT-J 因果语言模型
            "GPTJForQuestionAnswering",               # GPT-J 问答模型
            "GPTJForSequenceClassification",          # GPT-J 序列分类模型
            "GPTJModel",                              # GPT-J 模型
            "GPTJPreTrainedModel",                    # GPT-J 预训练模型基类
        ]
    )
    # 将 GPT-SAN 日本语模块的相关类名添加到导入结构中
    _import_structure["models.gptsan_japanese"].extend(
        [
            "GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",    # GPT-SAN 日本语预训练模型存档列表
            "GPTSanJapaneseForConditionalGeneration",           # GPT-SAN 日本语条件生成模型
            "GPTSanJapaneseModel",                              # GPT-SAN 日本语模型
            "GPTSanJapanesePreTrainedModel",                    # GPT-SAN 日本语预训练模型基类
        ]
    )
    # 将 Graphormer 模块的相关类名添加到导入结构中
    _import_structure["models.graphormer"].extend(
        [
            "GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST",    # Graphormer 预训练模型存档列表
            "GraphormerForGraphClassification",            # Graphormer 图分类模型
            "GraphormerModel",                             # Graphormer 模型
            "GraphormerPreTrainedModel",                   # Graphormer 预训练模型基类
        ]
    )
    # 将 GroupViT 模块的相关类名添加到导入结构中
    _import_structure["models.groupvit"].extend(
        [
            "GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",    # GroupViT 预训练模型存档列表
            "GroupViTModel",                             # GroupViT 模型
            "GroupViTPreTrainedModel",                   # GroupViT 预训练模型基类
            "GroupViTTextModel",                         # GroupViT 文本模型
            "GroupViTVisionModel",                       # GroupViT 视觉模型
        ]
    )
    # 将 "models.hubert" 中的模块名称列表扩展，包括预训练模型存档列表和各种 Hubert 模型类
    _import_structure["models.hubert"].extend(
        [
            "HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "HubertForCTC",
            "HubertForSequenceClassification",
            "HubertModel",
            "HubertPreTrainedModel",
        ]
    )
    # 将 "models.ibert" 中的模块名称列表扩展，包括预训练模型存档列表和各种 IBert 模型类
    _import_structure["models.ibert"].extend(
        [
            "IBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "IBertForMaskedLM",
            "IBertForMultipleChoice",
            "IBertForQuestionAnswering",
            "IBertForSequenceClassification",
            "IBertForTokenClassification",
            "IBertModel",
            "IBertPreTrainedModel",
        ]
    )
    # 将 "models.idefics" 中的模块名称列表扩展，包括预训练模型存档列表和各种 IDEFICS 模型类
    _import_structure["models.idefics"].extend(
        [
            "IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "IdeficsForVisionText2Text",
            "IdeficsModel",
            "IdeficsPreTrainedModel",
            "IdeficsProcessor",
        ]
    )
    # 将 "models.imagegpt" 中的模块名称列表扩展，包括预训练模型存档列表和各种 ImageGPT 模型类以及相关函数
    _import_structure["models.imagegpt"].extend(
        [
            "IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ImageGPTForCausalImageModeling",
            "ImageGPTForImageClassification",
            "ImageGPTModel",
            "ImageGPTPreTrainedModel",
            "load_tf_weights_in_imagegpt",
        ]
    )
    # 将 "models.informer" 中的模块名称列表扩展，包括预训练模型存档列表和各种 Informer 模型类
    _import_structure["models.informer"].extend(
        [
            "INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "InformerForPrediction",
            "InformerModel",
            "InformerPreTrainedModel",
        ]
    )
    # 将 "models.instructblip" 中的模块名称列表扩展，包括预训练模型存档列表和各种 InstructBlip 模型类
    _import_structure["models.instructblip"].extend(
        [
            "INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "InstructBlipForConditionalGeneration",
            "InstructBlipPreTrainedModel",
            "InstructBlipQFormerModel",
            "InstructBlipVisionModel",
        ]
    )
    # 将 "models.jukebox" 中的模块名称列表扩展，包括预训练模型存档列表和各种 Jukebox 模型类
    _import_structure["models.jukebox"].extend(
        [
            "JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST",
            "JukeboxModel",
            "JukeboxPreTrainedModel",
            "JukeboxPrior",
            "JukeboxVQVAE",
        ]
    )
    # 将 "models.kosmos2" 中的模块名称列表扩展，包括预训练模型存档列表和各种 Kosmos2 模型类
    _import_structure["models.kosmos2"].extend(
        [
            "KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Kosmos2ForConditionalGeneration",
            "Kosmos2Model",
            "Kosmos2PreTrainedModel",
        ]
    )
    # 将 "models.layoutlm" 中的模块名称列表扩展，包括预训练模型存档列表和各种 LayoutLM 模型类
    _import_structure["models.layoutlm"].extend(
        [
            "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMForMaskedLM",
            "LayoutLMForQuestionAnswering",
            "LayoutLMForSequenceClassification",
            "LayoutLMForTokenClassification",
            "LayoutLMModel",
            "LayoutLMPreTrainedModel",
        ]
    )
    # 将 "models.layoutlmv2" 中的模块名称列表扩展，包括预训练模型存档列表和各种 LayoutLMv2 模型类
    _import_structure["models.layoutlmv2"].extend(
        [
            "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMv2ForQuestionAnswering",
            "LayoutLMv2ForSequenceClassification",
            "LayoutLMv2ForTokenClassification",
            "LayoutLMv2Model",
            "LayoutLMv2PreTrainedModel",
        ]
    )
    )
    # 扩展 _import_structure 中 "models.layoutlmv3" 的列表，添加以下条目
    _import_structure["models.layoutlmv3"].extend(
        [
            "LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LayoutLMv3ForQuestionAnswering",
            "LayoutLMv3ForSequenceClassification",
            "LayoutLMv3ForTokenClassification",
            "LayoutLMv3Model",
            "LayoutLMv3PreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.led" 的列表，添加以下条目
    _import_structure["models.led"].extend(
        [
            "LED_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
            "LEDPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.levit" 的列表，添加以下条目
    _import_structure["models.levit"].extend(
        [
            "LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LevitForImageClassification",
            "LevitForImageClassificationWithTeacher",
            "LevitModel",
            "LevitPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.lilt" 的列表，添加以下条目
    _import_structure["models.lilt"].extend(
        [
            "LILT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LiltForQuestionAnswering",
            "LiltForSequenceClassification",
            "LiltForTokenClassification",
            "LiltModel",
            "LiltPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.llama" 的列表，添加以下条目
    _import_structure["models.llama"].extend(
        [
            "LlamaForCausalLM",
            "LlamaForQuestionAnswering",
            "LlamaForSequenceClassification",
            "LlamaModel",
            "LlamaPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.llava" 的列表，添加以下条目
    _import_structure["models.llava"].extend(
        [
            "LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LlavaForConditionalGeneration",
            "LlavaPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.llava_next" 的列表，添加以下条目
    _import_structure["models.llava_next"].extend(
        [
            "LLAVA_NEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LlavaNextForConditionalGeneration",
            "LlavaNextPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.longformer" 的列表，添加以下条目
    _import_structure["models.longformer"].extend(
        [
            "LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerPreTrainedModel",
            "LongformerSelfAttention",
        ]
    )
    # 扩展 _import_structure 中 "models.longt5" 的列表，添加以下条目
    _import_structure["models.longt5"].extend(
        [
            "LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LongT5EncoderModel",
            "LongT5ForConditionalGeneration",
            "LongT5Model",
            "LongT5PreTrainedModel",
        ]
    )
    # 将 "models.luke" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.luke"].extend(
        [
            "LUKE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "LukeForEntityClassification",
            "LukeForEntityPairClassification",
            "LukeForEntitySpanClassification",
            "LukeForMaskedLM",
            "LukeForMultipleChoice",
            "LukeForQuestionAnswering",
            "LukeForSequenceClassification",
            "LukeForTokenClassification",
            "LukeModel",
            "LukePreTrainedModel",
        ]
    )
    
    # 将 "models.lxmert" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",
            "LxmertForPreTraining",
            "LxmertForQuestionAnswering",
            "LxmertModel",
            "LxmertPreTrainedModel",
            "LxmertVisualFeatureEncoder",
            "LxmertXLayer",
        ]
    )
    
    # 将 "models.m2m_100" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.m2m_100"].extend(
        [
            "M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST",
            "M2M100ForConditionalGeneration",
            "M2M100Model",
            "M2M100PreTrainedModel",
        ]
    )
    
    # 将 "models.mamba" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.mamba"].extend(
        [
            "MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MambaForCausalLM",
            "MambaModel",
            "MambaPreTrainedModel",
        ]
    )
    
    # 将 "models.marian" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.marian"].extend(
        ["MarianForCausalLM", "MarianModel", "MarianMTModel"]
    )
    
    # 将 "models.markuplm" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.markuplm"].extend(
        [
            "MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MarkupLMForQuestionAnswering",
            "MarkupLMForSequenceClassification",
            "MarkupLMForTokenClassification",
            "MarkupLMModel",
            "MarkupLMPreTrainedModel",
        ]
    )
    
    # 将 "models.mask2former" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.mask2former"].extend(
        [
            "MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Mask2FormerForUniversalSegmentation",
            "Mask2FormerModel",
            "Mask2FormerPreTrainedModel",
        ]
    )
    
    # 将 "models.maskformer" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.maskformer"].extend(
        [
            "MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MaskFormerForInstanceSegmentation",
            "MaskFormerModel",
            "MaskFormerPreTrainedModel",
            "MaskFormerSwinBackbone",
        ]
    )
    
    # 将 "models.mbart" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",
            "MBartForConditionalGeneration",
            "MBartForQuestionAnswering",
            "MBartForSequenceClassification",
            "MBartModel",
            "MBartPreTrainedModel",
        ]
    )
    
    # 将 "models.mega" 模块的若干标识符添加到 _import_structure 字典中
    _import_structure["models.mega"].extend(
        [
            "MEGA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MegaForCausalLM",
            "MegaForMaskedLM",
            "MegaForMultipleChoice",
            "MegaForQuestionAnswering",
            "MegaForSequenceClassification",
            "MegaForTokenClassification",
            "MegaModel",
            "MegaPreTrainedModel",
        ]
    )
    # 将 "models.megatron_bert" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.megatron_bert"].extend(
        [
            "MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MegatronBertForCausalLM",
            "MegatronBertForMaskedLM",
            "MegatronBertForMultipleChoice",
            "MegatronBertForNextSentencePrediction",
            "MegatronBertForPreTraining",
            "MegatronBertForQuestionAnswering",
            "MegatronBertForSequenceClassification",
            "MegatronBertForTokenClassification",
            "MegatronBertModel",
            "MegatronBertPreTrainedModel",
        ]
    )
    # 将 "models.mgp_str" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mgp_str"].extend(
        [
            "MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MgpstrForSceneTextRecognition",
            "MgpstrModel",
            "MgpstrPreTrainedModel",
        ]
    )
    # 将 "models.mistral" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mistral"].extend(
        [
            "MistralForCausalLM",
            "MistralForSequenceClassification",
            "MistralModel",
            "MistralPreTrainedModel",
        ]
    )
    # 将 "models.mixtral" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mixtral"].extend(
        ["MixtralForCausalLM", "MixtralForSequenceClassification", "MixtralModel", "MixtralPreTrainedModel"]
    )
    # 将 "models.mobilebert" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mobilebert"].extend(
        [
            "MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertLayer",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
            "load_tf_weights_in_mobilebert",
        ]
    )
    # 将 "models.mobilenet_v1" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileNetV1ForImageClassification",
            "MobileNetV1Model",
            "MobileNetV1PreTrainedModel",
            "load_tf_weights_in_mobilenet_v1",
        ]
    )
    # 将 "models.mobilenet_v2" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileNetV2ForImageClassification",
            "MobileNetV2ForSemanticSegmentation",
            "MobileNetV2Model",
            "MobileNetV2PreTrainedModel",
            "load_tf_weights_in_mobilenet_v2",
        ]
    )
    # 将 "models.mobilevit" 中指定的模块列表添加到 _import_structure 字典中
    _import_structure["models.mobilevit"].extend(
        [
            "MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MobileViTForImageClassification",
            "MobileViTForSemanticSegmentation",
            "MobileViTModel",
            "MobileViTPreTrainedModel",
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.mobilevitv2"
    _import_structure["models.mobilevitv2"].extend(
        [
            "MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "MobileViTV2ForImageClassification",  # 图像分类的MobileViTV2模型
            "MobileViTV2ForSemanticSegmentation",  # 语义分割的MobileViTV2模型
            "MobileViTV2Model",  # MobileViTV2模型
            "MobileViTV2PreTrainedModel",  # MobileViTV2预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.mpnet"
    _import_structure["models.mpnet"].extend(
        [
            "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # MPNet预训练模型存档列表
            "MPNetForMaskedLM",  # 掩码语言建模的MPNet模型
            "MPNetForMultipleChoice",  # 多项选择的MPNet模型
            "MPNetForQuestionAnswering",  # 问答的MPNet模型
            "MPNetForSequenceClassification",  # 序列分类的MPNet模型
            "MPNetForTokenClassification",  # 标记分类的MPNet模型
            "MPNetLayer",  # MPNet层
            "MPNetModel",  # MPNet模型
            "MPNetPreTrainedModel",  # MPNet预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.mpt"
    _import_structure["models.mpt"].extend(
        [
            "MPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # MPT预训练模型存档列表
            "MptForCausalLM",  # 因果语言建模的Mpt模型
            "MptForQuestionAnswering",  # 问答的Mpt模型
            "MptForSequenceClassification",  # 序列分类的Mpt模型
            "MptForTokenClassification",  # 标记分类的Mpt模型
            "MptModel",  # Mpt模型
            "MptPreTrainedModel",  # Mpt预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.mra"
    _import_structure["models.mra"].extend(
        [
            "MRA_PRETRAINED_MODEL_ARCHIVE_LIST",  # MRA预训练模型存档列表
            "MraForMaskedLM",  # 掩码语言建模的Mra模型
            "MraForMultipleChoice",  # 多项选择的Mra模型
            "MraForQuestionAnswering",  # 问答的Mra模型
            "MraForSequenceClassification",  # 序列分类的Mra模型
            "MraForTokenClassification",  # 标记分类的Mra模型
            "MraModel",  # Mra模型
            "MraPreTrainedModel",  # Mra预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.mt5"
    _import_structure["models.mt5"].extend(
        [
            "MT5EncoderModel",  # MT5编码器模型
            "MT5ForConditionalGeneration",  # 条件生成的MT5模型
            "MT5ForQuestionAnswering",  # 问答的MT5模型
            "MT5ForSequenceClassification",  # 序列分类的MT5模型
            "MT5ForTokenClassification",  # 标记分类的MT5模型
            "MT5Model",  # MT5模型
            "MT5PreTrainedModel",  # MT5预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.musicgen"
    _import_structure["models.musicgen"].extend(
        [
            "MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST",  # MUSICGEN预训练模型存档列表
            "MusicgenForCausalLM",  # 因果语言建模的Musicgen模型
            "MusicgenForConditionalGeneration",  # 条件生成的Musicgen模型
            "MusicgenModel",  # Musicgen模型
            "MusicgenPreTrainedModel",  # Musicgen预训练模型
            "MusicgenProcessor",  # Musicgen处理器
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.musicgen_melody"
    _import_structure["models.musicgen_melody"].extend(
        [
            "MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST",  # MUSICGEN_MELODY预训练模型存档列表
            "MusicgenMelodyForCausalLM",  # 因果语言建模的MusicgenMelody模型
            "MusicgenMelodyForConditionalGeneration",  # 条件生成的MusicgenMelody模型
            "MusicgenMelodyModel",  # MusicgenMelody模型
            "MusicgenMelodyPreTrainedModel",  # MusicgenMelody预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.mvp"
    _import_structure["models.mvp"].extend(
        [
            "MVP_PRETRAINED_MODEL_ARCHIVE_LIST",  # MVP预训练模型存档列表
            "MvpForCausalLM",  # 因果语言建模的Mvp模型
            "MvpForConditionalGeneration",  # 条件生成的Mvp模型
            "MvpForQuestionAnswering",  # 问答的Mvp模型
            "MvpForSequenceClassification",  # 序列分类的Mvp模型
            "MvpModel",  # Mvp模型
            "MvpPreTrainedModel",  # Mvp预训练模型
        ]
    )
    # 将以下模型名称和相关属性扩展到_import_structure字典中的"models.nat"
    _import_structure["models.nat"].extend(
        [
            "NAT_PRETRAINED_MODEL_ARCHIVE_LIST",  # NAT预训练模型存档列表
            "NatBackbone",  # NatBackbone
            "NatForImageClassification",  # 图像分类的Nat模型
            "NatModel",  # Nat模型
            "NatPreTrainedModel",  # Nat预训练模型
        ]
    )
    # 将 "models.nezha" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.nezha"].extend(
        [
            "NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST",  # NEZHA 预训练模型存档列表常量
            "NezhaForMaskedLM",  # NEZHA Masked LM 模型类
            "NezhaForMultipleChoice",  # NEZHA 多选题模型类
            "NezhaForNextSentencePrediction",  # NEZHA 下一句预测模型类
            "NezhaForPreTraining",  # NEZHA 预训练模型类
            "NezhaForQuestionAnswering",  # NEZHA 问答模型类
            "NezhaForSequenceClassification",  # NEZHA 序列分类模型类
            "NezhaForTokenClassification",  # NEZHA 标记分类模型类
            "NezhaModel",  # NEZHA 模型基类
            "NezhaPreTrainedModel",  # NEZHA 预训练模型基类
        ]
    )
    
    # 将 "models.nllb_moe" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.nllb_moe"].extend(
        [
            "NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST",  # NLLB_MOE 预训练模型存档列表常量
            "NllbMoeForConditionalGeneration",  # NLLB_MOE 生成条件模型类
            "NllbMoeModel",  # NLLB_MOE 模型类
            "NllbMoePreTrainedModel",  # NLLB_MOE 预训练模型基类
            "NllbMoeSparseMLP",  # NLLB_MOE 稀疏MLP模型类
            "NllbMoeTop2Router",  # NLLB_MOE Top2路由模型类
        ]
    )
    
    # 将 "models.nystromformer" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.nystromformer"].extend(
        [
            "NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # NYSTROMFORMER 预训练模型存档列表常量
            "NystromformerForMaskedLM",  # NYSTROMFORMER Masked LM 模型类
            "NystromformerForMultipleChoice",  # NYSTROMFORMER 多选题模型类
            "NystromformerForQuestionAnswering",  # NYSTROMFORMER 问答模型类
            "NystromformerForSequenceClassification",  # NYSTROMFORMER 序列分类模型类
            "NystromformerForTokenClassification",  # NYSTROMFORMER 标记分类模型类
            "NystromformerLayer",  # NYSTROMFORMER 模型层类
            "NystromformerModel",  # NYSTROMFORMER 模型类
            "NystromformerPreTrainedModel",  # NYSTROMFORMER 预训练模型基类
        ]
    )
    
    # 将 "models.oneformer" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.oneformer"].extend(
        [
            "ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # ONEFORMER 预训练模型存档列表常量
            "OneFormerForUniversalSegmentation",  # ONEFORMER 通用分割模型类
            "OneFormerModel",  # ONEFORMER 模型类
            "OneFormerPreTrainedModel",  # ONEFORMER 预训练模型基类
        ]
    )
    
    # 将 "models.openai" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.openai"].extend(
        [
            "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # OPENAI_GPT 预训练模型存档列表常量
            "OpenAIGPTDoubleHeadsModel",  # OPENAI_GPT 双头模型类
            "OpenAIGPTForSequenceClassification",  # OPENAI_GPT 序列分类模型类
            "OpenAIGPTLMHeadModel",  # OPENAI_GPT LM头模型类
            "OpenAIGPTModel",  # OPENAI_GPT 模型类
            "OpenAIGPTPreTrainedModel",  # OPENAI_GPT 预训练模型基类
            "load_tf_weights_in_openai_gpt",  # 在 OPENAI_GPT 中加载 TensorFlow 权重函数
        ]
    )
    
    # 将 "models.opt" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.opt"].extend(
        [
            "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # OPT 预训练模型存档列表常量
            "OPTForCausalLM",  # OPT 因果LM模型类
            "OPTForQuestionAnswering",  # OPT 问答模型类
            "OPTForSequenceClassification",  # OPT 序列分类模型类
            "OPTModel",  # OPT 模型类
            "OPTPreTrainedModel",  # OPT 预训练模型基类
        ]
    )
    
    # 将 "models.owlv2" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.owlv2"].extend(
        [
            "OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # OWLV2 预训练模型存档列表常量
            "Owlv2ForObjectDetection",  # OWLV2 目标检测模型类
            "Owlv2Model",  # OWLV2 模型类
            "Owlv2PreTrainedModel",  # OWLV2 预训练模型基类
            "Owlv2TextModel",  # OWLV2 文本模型类
            "Owlv2VisionModel",  # OWLV2 视觉模型类
        ]
    )
    
    # 将 "models.owlvit" 模块下的特定类名和常量列表扩展到 _import_structure 字典中
    _import_structure["models.owlvit"].extend(
        [
            "OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # OWLVIT 预训练模型存档列表常量
            "OwlViTForObjectDetection",  # OWLVIT 目标检测模型类
            "OwlViTModel",  # OWLVIT 模型类
            "OwlViTPreTrainedModel",  # OWLVIT 预训练模型基类
            "OwlViTTextModel",  # OWLVIT 文本模型类
            "OwlViTVisionModel",  # OWLVIT 视觉模型类
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.patchtsmixer" 键对应的列表中
    _import_structure["models.patchtsmixer"].extend(
        [
            "PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PatchTSMixerForPrediction",
            "PatchTSMixerForPretraining",
            "PatchTSMixerForRegression",
            "PatchTSMixerForTimeSeriesClassification",
            "PatchTSMixerModel",
            "PatchTSMixerPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.patchtst" 键对应的列表中
    _import_structure["models.patchtst"].extend(
        [
            "PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PatchTSTForClassification",
            "PatchTSTForPrediction",
            "PatchTSTForPretraining",
            "PatchTSTForRegression",
            "PatchTSTModel",
            "PatchTSTPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.pegasus" 键对应的列表中
    _import_structure["models.pegasus"].extend(
        [
            "PegasusForCausalLM",
            "PegasusForConditionalGeneration",
            "PegasusModel",
            "PegasusPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.pegasus_x" 键对应的列表中
    _import_structure["models.pegasus_x"].extend(
        [
            "PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PegasusXForConditionalGeneration",
            "PegasusXModel",
            "PegasusXPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.perceiver" 键对应的列表中
    _import_structure["models.perceiver"].extend(
        [
            "PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PerceiverForImageClassificationConvProcessing",
            "PerceiverForImageClassificationFourier",
            "PerceiverForImageClassificationLearned",
            "PerceiverForMaskedLM",
            "PerceiverForMultimodalAutoencoding",
            "PerceiverForOpticalFlow",
            "PerceiverForSequenceClassification",
            "PerceiverLayer",
            "PerceiverModel",
            "PerceiverPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.persimmon" 键对应的列表中
    _import_structure["models.persimmon"].extend(
        [
            "PersimmonForCausalLM",
            "PersimmonForSequenceClassification",
            "PersimmonModel",
            "PersimmonPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.phi" 键对应的列表中
    _import_structure["models.phi"].extend(
        [
            "PHI_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PhiForCausalLM",
            "PhiForSequenceClassification",
            "PhiForTokenClassification",
            "PhiModel",
            "PhiPreTrainedModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.pix2struct" 键对应的列表中
    _import_structure["models.pix2struct"].extend(
        [
            "PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    # 将以下模块添加到 `_import_structure` 字典中的 "models.plbart" 键对应的列表中
    _import_structure["models.plbart"].extend(
        [
            "PLBART_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PLBartForCausalLM",
            "PLBartForConditionalGeneration",
            "PLBartForSequenceClassification",
            "PLBartModel",
            "PLBartPreTrainedModel",
        ]
    )
    # 将 "models.poolformer" 模块的预训练模型列表、图像分类的 PoolFormer 类、PoolFormer 模型、PoolFormer 的预训练模型添加到导入结构中
    _import_structure["models.poolformer"].extend(
        [
            "POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    # 将 "models.pop2piano" 模块的预训练模型列表、条件生成的 Pop2Piano 类、Pop2Piano 模型、Pop2Piano 的预训练模型添加到导入结构中
    _import_structure["models.pop2piano"].extend(
        [
            "POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    # 将 "models.prophetnet" 模块的预训练模型列表、ProphetNet 解码器、ProphetNet 编码器等添加到导入结构中
    _import_structure["models.prophetnet"].extend(
        [
            "PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ProphetNetDecoder",
            "ProphetNetEncoder",
            "ProphetNetForCausalLM",
            "ProphetNetForConditionalGeneration",
            "ProphetNetModel",
            "ProphetNetPreTrainedModel",
        ]
    )
    # 将 "models.pvt" 模块的预训练模型列表、图像分类的 Pvt 类、Pvt 模型、Pvt 的预训练模型添加到导入结构中
    _import_structure["models.pvt"].extend(
        [
            "PVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    # 将 "models.pvt_v2" 模块的预训练模型列表、PvtV2 的骨干网络、图像分类的 PvtV2 类、PvtV2 模型、PvtV2 的预训练模型添加到导入结构中
    _import_structure["models.pvt_v2"].extend(
        [
            "PVT_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PvtV2Backbone",
            "PvtV2ForImageClassification",
            "PvtV2Model",
            "PvtV2PreTrainedModel",
        ]
    )
    # 将 "models.qdqbert" 模块的预训练模型列表、QDQBert 的各种任务专用类、QDQBert 层、QDQBert 语言模型头部等添加到导入结构中
    _import_structure["models.qdqbert"].extend(
        [
            "QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "QDQBertForMaskedLM",
            "QDQBertForMultipleChoice",
            "QDQBertForNextSentencePrediction",
            "QDQBertForQuestionAnswering",
            "QDQBertForSequenceClassification",
            "QDQBertForTokenClassification",
            "QDQBertLayer",
            "QDQBertLMHeadModel",
            "QDQBertModel",
            "QDQBertPreTrainedModel",
            "load_tf_weights_in_qdqbert",
        ]
    )
    # 将 "models.qwen2" 模块的条件生成 Qwen2 类、Qwen2 模型、Qwen2 的预训练模型添加到导入结构中
    _import_structure["models.qwen2"].extend(
        [
            "Qwen2ForCausalLM",
            "Qwen2ForSequenceClassification",
            "Qwen2Model",
            "Qwen2PreTrainedModel",
        ]
    )
    # 将 "models.rag" 模块的 RAG 模型、RAG 的预训练模型添加到导入结构中
    _import_structure["models.rag"].extend(
        [
            "RagModel",
            "RagPreTrainedModel",
            "RagSequenceForGeneration",
            "RagTokenForGeneration",
        ]
    )
    # 将 "models.realm" 模块的预训练模型列表、RealmEmbedder、RealmForOpenQA 等添加到导入结构中
    _import_structure["models.realm"].extend(
        [
            "REALM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RealmEmbedder",
            "RealmForOpenQA",
            "RealmKnowledgeAugEncoder",
            "RealmPreTrainedModel",
            "RealmReader",
            "RealmRetriever",
            "RealmScorer",
            "load_tf_weights_in_realm",
        ]
    )
    # 将以下模型和相关内容导入到_import_structure中的"models.reformer"模块中
    _import_structure["models.reformer"].extend(
        [
            "REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",    # 引入预训练模型存档列表
            "ReformerAttention",                        # 引入ReformerAttention类
            "ReformerForMaskedLM",                      # 引入ReformerForMaskedLM类
            "ReformerForQuestionAnswering",              # 引入ReformerForQuestionAnswering类
            "ReformerForSequenceClassification",        # 引入ReformerForSequenceClassification类
            "ReformerLayer",                            # 引入ReformerLayer类
            "ReformerModel",                            # 引入ReformerModel类
            "ReformerModelWithLMHead",                  # 引入ReformerModelWithLMHead类
            "ReformerPreTrainedModel",                  # 引入ReformerPreTrainedModel类
        ]
    )
    
    # 将以下模型和相关内容导入到_import_structure中的"models.regnet"模块中
    _import_structure["models.regnet"].extend(
        [
            "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",      # 引入预训练模型存档列表
            "RegNetForImageClassification",             # 引入RegNetForImageClassification类
            "RegNetModel",                              # 引入RegNetModel类
            "RegNetPreTrainedModel",                    # 引入RegNetPreTrainedModel类
        ]
    )
    
    # 将以下模型和相关内容导入到_import_structure中的"models.rembert"模块中
    _import_structure["models.rembert"].extend(
        [
            "REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",     # 引入预训练模型存档列表
            "RemBertForCausalLM",                       # 引入RemBertForCausalLM类
            "RemBertForMaskedLM",                       # 引入RemBertForMaskedLM类
            "RemBertForMultipleChoice",                 # 引入RemBertForMultipleChoice类
            "RemBertForQuestionAnswering",              # 引入RemBertForQuestionAnswering类
            "RemBertForSequenceClassification",         # 引入RemBertForSequenceClassification类
            "RemBertForTokenClassification",            # 引入RemBertForTokenClassification类
            "RemBertLayer",                             # 引入RemBertLayer类
            "RemBertModel",                             # 引入RemBertModel类
            "RemBertPreTrainedModel",                   # 引入RemBertPreTrainedModel类
            "load_tf_weights_in_rembert",               # 引入load_tf_weights_in_rembert函数
        ]
    )
    
    # 将以下模型和相关内容导入到_import_structure中的"models.resnet"模块中
    _import_structure["models.resnet"].extend(
        [
            "RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",      # 引入预训练模型存档列表
            "ResNetBackbone",                           # 引入ResNetBackbone类
            "ResNetForImageClassification",             # 引入ResNetForImageClassification类
            "ResNetModel",                              # 引入ResNetModel类
            "ResNetPreTrainedModel",                    # 引入ResNetPreTrainedModel类
        ]
    )
    
    # 将以下模型和相关内容导入到_import_structure中的"models.roberta"模块中
    _import_structure["models.roberta"].extend(
        [
            "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",     # 引入预训练模型存档列表
            "RobertaForCausalLM",                       # 引入RobertaForCausalLM类
            "RobertaForMaskedLM",                       # 引入RobertaForMaskedLM类
            "RobertaForMultipleChoice",                 # 引入RobertaForMultipleChoice类
            "RobertaForQuestionAnswering",              # 引入RobertaForQuestionAnswering类
            "RobertaForSequenceClassification",         # 引入RobertaForSequenceClassification类
            "RobertaForTokenClassification",            # 引入RobertaForTokenClassification类
            "RobertaModel",                             # 引入RobertaModel类
            "RobertaPreTrainedModel",                   # 引入RobertaPreTrainedModel类
        ]
    )
    
    # 将以下模型和相关内容导入到_import_structure中的"models.roberta_prelayernorm"模块中
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",     # 引入预训练模型存档列表
            "RobertaPreLayerNormForCausalLM",                         # 引入RobertaPreLayerNormForCausalLM类
            "RobertaPreLayerNormForMaskedLM",                         # 引入RobertaPreLayerNormForMaskedLM类
            "RobertaPreLayerNormForMultipleChoice",                   # 引入RobertaPreLayerNormForMultipleChoice类
            "RobertaPreLayerNormForQuestionAnswering",                # 引入RobertaPreLayerNormForQuestionAnswering类
            "RobertaPreLayerNormForSequenceClassification",           # 引入RobertaPreLayerNormForSequenceClassification类
            "RobertaPreLayerNormForTokenClassification",              # 引入RobertaPreLayerNormForTokenClassification类
            "RobertaPreLayerNormModel",                               # 引入RobertaPreLayerNormModel类
            "RobertaPreLayerNormPreTrainedModel",                     # 引入RobertaPreLayerNormPreTrainedModel类
        ]
    )
    
    # 将以下模型和相关内容导入到_import_structure中的"models.roc_bert"模块中
    _import_structure["models.roc_bert"].extend(
        [
            "ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",      # 引入预训练模型存档列表
            "RoCBertForCausalLM",                         # 引入RoCBertForCausalLM类
            "RoCBertForMaskedLM",                         # 引入RoCBertForMaskedLM类
            "RoCBertForMultipleChoice",                   # 引入RoCBertForMultipleChoice类
            "RoCBertForPreTraining",                      # 引入RoCBertForPreTraining类
            "RoCBertForQuestionAnswering",                # 引入RoCBertForQuestionAnswering类
            "RoCBertForSequenceClassification",           # 引入RoCBertForSequenceClassification类
            "RoCBertForTokenClassification",              # 引入RoCBertForTokenClassification类
            "RoCBertLayer",                               # 引入RoCBertLayer类
            "RoCBertModel",                               # 引入RoCBertModel类
            "RoCBertPreTrainedModel",                     # 引入RoCBertPreTrainedModel类
            "load_tf_weights_in_roc_bert",                # 引入load_tf_weights_in_roc_bert函数
        ]
    )
    # 将 "models.roformer" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.roformer"].extend(
        [
            "ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # RoFormer 预训练模型的存档列表
            "RoFormerForCausalLM",  # 用于因果语言建模的 RoFormer 模型
            "RoFormerForMaskedLM",  # 用于遮蔽语言建模的 RoFormer 模型
            "RoFormerForMultipleChoice",  # 用于多选题任务的 RoFormer 模型
            "RoFormerForQuestionAnswering",  # 用于问答任务的 RoFormer 模型
            "RoFormerForSequenceClassification",  # 用于序列分类任务的 RoFormer 模型
            "RoFormerForTokenClassification",  # 用于标记分类任务的 RoFormer 模型
            "RoFormerLayer",  # RoFormer 模型的层定义
            "RoFormerModel",  # RoFormer 模型的主体定义
            "RoFormerPreTrainedModel",  # RoFormer 模型的预训练模型基类
            "load_tf_weights_in_roformer",  # 加载 TensorFlow 权重到 RoFormer 模型中的函数
        ]
    )
    # 将 "models.rwkv" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.rwkv"].extend(
        [
            "RWKV_PRETRAINED_MODEL_ARCHIVE_LIST",  # Rwkv 预训练模型的存档列表
            "RwkvForCausalLM",  # 用于因果语言建模的 Rwkv 模型
            "RwkvModel",  # Rwkv 模型的主体定义
            "RwkvPreTrainedModel",  # Rwkv 模型的预训练模型基类
        ]
    )
    # 将 "models.sam" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.sam"].extend(
        [
            "SAM_PRETRAINED_MODEL_ARCHIVE_LIST",  # SAM 预训练模型的存档列表
            "SamModel",  # SAM 模型的主体定义
            "SamPreTrainedModel",  # SAM 模型的预训练模型基类
        ]
    )
    # 将 "models.seamless_m4t" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.seamless_m4t"].extend(
        [
            "SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST",  # Seamless M4T 预训练模型的存档列表
            "SeamlessM4TCodeHifiGan",  # 用于代码 Hifi Gan 的 Seamless M4T 模型
            "SeamlessM4TForSpeechToSpeech",  # 用于语音转语音任务的 Seamless M4T 模型
            "SeamlessM4TForSpeechToText",  # 用于语音转文本任务的 Seamless M4T 模型
            "SeamlessM4TForTextToSpeech",  # 用于文本转语音任务的 Seamless M4T 模型
            "SeamlessM4TForTextToText",  # 用于文本转文本任务的 Seamless M4T 模型
            "SeamlessM4THifiGan",  # 用于 Hifi Gan 的 Seamless M4T 模型
            "SeamlessM4TModel",  # Seamless M4T 模型的主体定义
            "SeamlessM4TPreTrainedModel",  # Seamless M4T 模型的预训练模型基类
            "SeamlessM4TTextToUnitForConditionalGeneration",  # 用于条件生成的 Seamless M4T 模型
            "SeamlessM4TTextToUnitModel",  # 用于文本生成的 Seamless M4T 模型
        ]
    )
    # 将 "models.seamless_m4t_v2" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.seamless_m4t_v2"].extend(
        [
            "SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST",  # Seamless M4T V2 预训练模型的存档列表
            "SeamlessM4Tv2ForSpeechToSpeech",  # 用于语音转语音任务的 Seamless M4T V2 模型
            "SeamlessM4Tv2ForSpeechToText",  # 用于语音转文本任务的 Seamless M4T V2 模型
            "SeamlessM4Tv2ForTextToSpeech",  # 用于文本转语音任务的 Seamless M4T V2 模型
            "SeamlessM4Tv2ForTextToText",  # 用于文本转文本任务的 Seamless M4T V2 模型
            "SeamlessM4Tv2Model",  # Seamless M4T V2 模型的主体定义
            "SeamlessM4Tv2PreTrainedModel",  # Seamless M4T V2 模型的预训练模型基类
        ]
    )
    # 将 "models.segformer" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.segformer"].extend(
        [
            "SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # Segformer 预训练模型的存档列表
            "SegformerDecodeHead",  # Segformer 解码头部定义
            "SegformerForImageClassification",  # 用于图像分类任务的 Segformer 模型
            "SegformerForSemanticSegmentation",  # 用于语义分割任务的 Segformer 模型
            "SegformerLayer",  # Segformer 模型的层定义
            "SegformerModel",  # Segformer 模型的主体定义
            "SegformerPreTrainedModel",  # Segformer 模型的预训练模型基类
        ]
    )
    # 将 "models.seggpt" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.seggpt"].extend(
        [
            "SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # SegGpt 预训练模型的存档列表
            "SegGptForImageSegmentation",  # 用于图像分割任务的 SegGpt 模型
            "SegGptModel",  # SegGpt 模型的主体定义
            "SegGptPreTrainedModel",  # SegGpt 模型的预训练模型基类
        ]
    )
    # 将 "models.sew" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.sew"].extend(
        [
            "SEW_PRETRAINED_MODEL_ARCHIVE_LIST",  # SEW 预训练模型的存档列表
            "SEWForCTC",  # 用于 CTC 模型的 SEW 模型
            "SEWForSequenceClassification",  # 用于序列分类任务的 SEW 模型
            "SEWModel",  # SEW 模型的主体定义
            "SEWPreTrainedModel",  # SEW 模型的预训练模型基类
        ]
    )
    # 将 "models.sew_d" 模块中的一组成员添加到 _import_structure 字典的列表中
    _import_structure["models.sew_d"].extend(
        [
            "SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST",  # SEW_D 预训练模型的存档列表
            "SEWDForCTC",  # 用于 CTC 模型的 SEW_D 模型
            "SEWDForSequenceClassification",  # 用于序列分类任务的 SEW_D 模型
            "SEWDModel",  # SEW_D 模型的主体定义
            "SEWDPreTrainedModel",  # SEW_D 模型的预训练模型基类
        ]
    )
    # 扩展导入结构中 "models.siglip" 的内容列表，包括预训练模型列表、图像分类模型、基础模型、预训练模型和文本模型、视觉模型
    _import_structure["models.siglip"].extend(
        [
            "SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SiglipForImageClassification",
            "SiglipModel",
            "SiglipPreTrainedModel",
            "SiglipTextModel",
            "SiglipVisionModel",
        ]
    )
    # 扩展导入结构中 "models.speech_encoder_decoder" 的内容列表，包括语音编码解码模型
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    # 扩展导入结构中 "models.speech_to_text" 的内容列表，包括语音转文本预训练模型列表、语音转文本生成模型、语音转文本基础模型、语音转文本预训练模型
    _import_structure["models.speech_to_text"].extend(
        [
            "SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Speech2TextForConditionalGeneration",
            "Speech2TextModel",
            "Speech2TextPreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.speech_to_text_2" 的内容列表，包括第二个语音转文本条件LM模型和第二个语音转文本预训练模型
    _import_structure["models.speech_to_text_2"].extend(["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"])
    # 扩展导入结构中 "models.speecht5" 的内容列表，包括T5风格语音到语音预训练模型列表、T5风格语音到文本模型、T5风格文本到语音模型、T5风格HifiGan模型、T5风格基础模型、T5风格预训练模型
    _import_structure["models.speecht5"].extend(
        [
            "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SpeechT5ForSpeechToSpeech",
            "SpeechT5ForSpeechToText",
            "SpeechT5ForTextToSpeech",
            "SpeechT5HifiGan",
            "SpeechT5Model",
            "SpeechT5PreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.splinter" 的内容列表，包括SPLINTER预训练模型列表、SPLINTER预训练模型、SPLINTER问答模型、SPLINTER层、SPLINTER基础模型、SPLINTER预训练模型
    _import_structure["models.splinter"].extend(
        [
            "SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SplinterForPreTraining",
            "SplinterForQuestionAnswering",
            "SplinterLayer",
            "SplinterModel",
            "SplinterPreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.squeezebert" 的内容列表，包括SQUEEZEBERT预训练模型列表、SQUEEZEBERT遮蔽LM模型、SQUEEZEBERT多项选择模型、SQUEEZEBERT问答模型、SQUEEZEBERT序列分类模型、SQUEEZEBERT标记分类模型、SQUEEZEBERT基础模型、SQUEEZEBERT模块、SQUEEZEBERT预训练模型
    _import_structure["models.squeezebert"].extend(
        [
            "SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertModule",
            "SqueezeBertPreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.stablelm" 的内容列表，包括稳定LM条件LM模型、稳定LM序列分类模型、稳定LM基础模型、稳定LM预训练模型
    _import_structure["models.stablelm"].extend(
        [
            "StableLmForCausalLM",
            "StableLmForSequenceClassification",
            "StableLmModel",
            "StableLmPreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.starcoder2" 的内容列表，包括Starcoder2条件LM模型、Starcoder2序列分类模型、Starcoder2基础模型、Starcoder2预训练模型
    _import_structure["models.starcoder2"].extend(
        [
            "Starcoder2ForCausalLM",
            "Starcoder2ForSequenceClassification",
            "Starcoder2Model",
            "Starcoder2PreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.superpoint" 的内容列表，包括SUPERPOINT预训练模型列表、SUPERPOINT关键点检测模型、SUPERPOINT基础模型
    _import_structure["models.superpoint"].extend(
        [
            "SUPERPOINT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SuperPointForKeypointDetection",
            "SuperPointPreTrainedModel",
        ]
    )
    # 扩展导入结构中 "models.swiftformer" 的内容列表，包括SWIFTFORMER预训练模型列表、SWIFTFORMER图像分类模型、SWIFTFORMER基础模型、SWIFTFORMER预训练模型
    _import_structure["models.swiftformer"].extend(
        [
            "SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SwiftFormerForImageClassification",
            "SwiftFormerModel",
            "SwiftFormerPreTrainedModel",
        ]
    )
    # 将指定模块的一组模型名称添加到_import_structure字典中
    _import_structure["models.swin"].extend(
        [
            "SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型列表
            "SwinBackbone",  # Swin模型的骨干网络
            "SwinForImageClassification",  # 用于图像分类的Swin模型
            "SwinForMaskedImageModeling",  # 用于带遮罩图像建模的Swin模型
            "SwinModel",  # Swin模型
            "SwinPreTrainedModel",  # Swin预训练模型
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理swin2sr模块
    _import_structure["models.swin2sr"].extend(
        [
            "SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST",  # SWIN2SR的预训练模型列表
            "Swin2SRForImageSuperResolution",  # 用于图像超分辨率的Swin2SR模型
            "Swin2SRModel",  # Swin2SR模型
            "Swin2SRPreTrainedModel",  # Swin2SR预训练模型
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理swinv2模块
    _import_structure["models.swinv2"].extend(
        [
            "SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # SWINV2的预训练模型列表
            "Swinv2Backbone",  # Swinv2模型的骨干网络
            "Swinv2ForImageClassification",  # 用于图像分类的Swinv2模型
            "Swinv2ForMaskedImageModeling",  # 用于带遮罩图像建模的Swinv2模型
            "Swinv2Model",  # Swinv2模型
            "Swinv2PreTrainedModel",  # Swinv2预训练模型
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理switch_transformers模块
    _import_structure["models.switch_transformers"].extend(
        [
            "SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",  # SWITCH_TRANSFORMERS的预训练模型列表
            "SwitchTransformersEncoderModel",  # SwitchTransformers编码器模型
            "SwitchTransformersForConditionalGeneration",  # SwitchTransformers条件生成模型
            "SwitchTransformersModel",  # SwitchTransformers通用模型
            "SwitchTransformersPreTrainedModel",  # SwitchTransformers预训练模型
            "SwitchTransformersSparseMLP",  # SwitchTransformers稀疏MLP模型
            "SwitchTransformersTop1Router",  # SwitchTransformers顶级1路由器模型
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理t5模块
    _import_structure["models.t5"].extend(
        [
            "T5_PRETRAINED_MODEL_ARCHIVE_LIST",  # T5的预训练模型列表
            "T5EncoderModel",  # T5编码器模型
            "T5ForConditionalGeneration",  # T5条件生成模型
            "T5ForQuestionAnswering",  # T5问答模型
            "T5ForSequenceClassification",  # T5序列分类模型
            "T5ForTokenClassification",  # T5标记分类模型
            "T5Model",  # T5通用模型
            "T5PreTrainedModel",  # T5预训练模型
            "load_tf_weights_in_t5",  # 在T5中加载TensorFlow权重
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理table_transformer模块
    _import_structure["models.table_transformer"].extend(
        [
            "TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # TABLE_TRANSFORMER的预训练模型列表
            "TableTransformerForObjectDetection",  # 用于对象检测的TableTransformer模型
            "TableTransformerModel",  # TableTransformer通用模型
            "TableTransformerPreTrainedModel",  # TableTransformer预训练模型
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理tapas模块
    _import_structure["models.tapas"].extend(
        [
            "TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",  # TAPAS的预训练模型列表
            "TapasForMaskedLM",  # 用于掩码语言模型的Tapas模型
            "TapasForQuestionAnswering",  # Tapas问答模型
            "TapasForSequenceClassification",  # Tapas序列分类模型
            "TapasModel",  # Tapas通用模型
            "TapasPreTrainedModel",  # Tapas预训练模型
            "load_tf_weights_in_tapas",  # 在Tapas中加载TensorFlow权重
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理time_series_transformer模块
    _import_structure["models.time_series_transformer"].extend(
        [
            "TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # TIME_SERIES_TRANSFORMER的预训练模型列表
            "TimeSeriesTransformerForPrediction",  # 用于预测的TimeSeriesTransformer模型
            "TimeSeriesTransformerModel",  # TimeSeriesTransformer通用模型
            "TimeSeriesTransformerPreTrainedModel",  # TimeSeriesTransformer预训练模型
        ]
    )
    # 将另一组模型名称添加到_import_structure字典中，此处处理timesformer模块
    _import_structure["models.timesformer"].extend(
        [
            "TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # TIMESFORMER的预训练模型列表
            "TimesformerForVideoClassification",  # 用于视频分类的Timesformer模型
            "TimesformerModel",  # Timesformer通用模型
            "TimesformerPreTrainedModel",  # Timesformer预训练模型
        ]
    )
    # 将一个模型名称添加到_import_structure字典中，此处处理timm_backbone模块
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])  # TimmBackbone模型
    # 将 "models.trocr" 模块的列表扩展，包括预训练模型存档列表、特定模型类等
    _import_structure["models.trocr"].extend(
        [
            "TROCR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TrOCRForCausalLM",
            "TrOCRPreTrainedModel",
        ]
    )
    # 将 "models.tvlt" 模块的列表扩展，包括预训练模型存档列表、特定模型类等
    _import_structure["models.tvlt"].extend(
        [
            "TVLT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TvltForAudioVisualClassification",
            "TvltForPreTraining",
            "TvltModel",
            "TvltPreTrainedModel",
        ]
    )
    # 将 "models.tvp" 模块的列表扩展，包括预训练模型存档列表、特定模型类等
    _import_structure["models.tvp"].extend(
        [
            "TVP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TvpForVideoGrounding",
            "TvpModel",
            "TvpPreTrainedModel",
        ]
    )
    # 将 "models.udop" 模块的列表扩展，包括预训练模型存档列表、特定模型类等
    _import_structure["models.udop"].extend(
        [
            "UDOP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "UdopEncoderModel",
            "UdopForConditionalGeneration",
            "UdopModel",
            "UdopPreTrainedModel",
        ],
    )
    # 将 "models.umt5" 模块的列表扩展，包括特定模型类，如编码器、生成条件模型等
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",
            "UMT5ForConditionalGeneration",
            "UMT5ForQuestionAnswering",
            "UMT5ForSequenceClassification",
            "UMT5ForTokenClassification",
            "UMT5Model",
            "UMT5PreTrainedModel",
        ]
    )
    # 将 "models.unispeech" 模块的列表扩展，包括预训练模型存档列表、特定语音处理模型类等
    _import_structure["models.unispeech"].extend(
        [
            "UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST",
            "UniSpeechForCTC",
            "UniSpeechForPreTraining",
            "UniSpeechForSequenceClassification",
            "UniSpeechModel",
            "UniSpeechPreTrainedModel",
        ]
    )
    # 将 "models.unispeech_sat" 模块的列表扩展，包括预训练模型存档列表、特定语音处理模型类等
    _import_structure["models.unispeech_sat"].extend(
        [
            "UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "UniSpeechSatForAudioFrameClassification",
            "UniSpeechSatForCTC",
            "UniSpeechSatForPreTraining",
            "UniSpeechSatForSequenceClassification",
            "UniSpeechSatForXVector",
            "UniSpeechSatModel",
            "UniSpeechSatPreTrainedModel",
        ]
    )
    # 将 "models.univnet" 模块的列表扩展，包括预训练模型存档列表、特定模型类等
    _import_structure["models.univnet"].extend(
        [
            "UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "UnivNetModel",
        ]
    )
    # 将 "models.upernet" 模块的列表扩展，包括语义分割模型类、预训练模型类等
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",
            "UperNetPreTrainedModel",
        ]
    )
    # 将 "models.videomae" 模块的列表扩展，包括预训练模型存档列表、视频分类模型类等
    _import_structure["models.videomae"].extend(
        [
            "VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VideoMAEForPreTraining",
            "VideoMAEForVideoClassification",
            "VideoMAEModel",
            "VideoMAEPreTrainedModel",
        ]
    )
    # 将指定模块内的预定义符号（变量和类）扩展到_import_structure字典中的models.vilt模块
    _import_structure["models.vilt"].extend(
        [
            "VILT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展预训练模型存档列表
            "ViltForImageAndTextRetrieval",  # 扩展图像和文本检索模型
            "ViltForImagesAndTextClassification",  # 扩展图像和文本分类模型
            "ViltForMaskedLM",  # 扩展掩码语言建模模型
            "ViltForQuestionAnswering",  # 扩展问答模型
            "ViltForTokenClassification",  # 扩展标记分类模型
            "ViltLayer",  # 扩展VILT层
            "ViltModel",  # 扩展VILT模型
            "ViltPreTrainedModel",  # 扩展VILT预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vipllava模块
    _import_structure["models.vipllava"].extend(
        [
            "VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VIPLLAVA预训练模型存档列表
            "VipLlavaForConditionalGeneration",  # 扩展条件生成模型
            "VipLlavaPreTrainedModel",  # 扩展VIP LLAVA预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vision_encoder_decoder模块
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vision_text_dual_encoder模块
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.visual_bert模块
    _import_structure["models.visual_bert"].extend(
        [
            "VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VISUAL BERT预训练模型存档列表
            "VisualBertForMultipleChoice",  # 扩展多选题模型
            "VisualBertForPreTraining",  # 扩展预训练模型
            "VisualBertForQuestionAnswering",  # 扩展视觉BERT问答模型
            "VisualBertForRegionToPhraseAlignment",  # 扩展区域到短语对齐模型
            "VisualBertForVisualReasoning",  # 扩展视觉推理模型
            "VisualBertLayer",  # 扩展Visual BERT层
            "VisualBertModel",  # 扩展Visual BERT模型
            "VisualBertPreTrainedModel",  # 扩展Visual BERT预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vit模块
    _import_structure["models.vit"].extend(
        [
            "VIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VIT预训练模型存档列表
            "ViTForImageClassification",  # 扩展图像分类模型
            "ViTForMaskedImageModeling",  # 扩展掩码图像建模模型
            "ViTModel",  # 扩展ViT模型
            "ViTPreTrainedModel",  # 扩展ViT预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vit_hybrid模块
    _import_structure["models.vit_hybrid"].extend(
        [
            "VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VIT HYBRID预训练模型存档列表
            "ViTHybridForImageClassification",  # 扩展图像分类模型
            "ViTHybridModel",  # 扩展ViT Hybrid模型
            "ViTHybridPreTrainedModel",  # 扩展ViT Hybrid预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vit_mae模块
    _import_structure["models.vit_mae"].extend(
        [
            "VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VIT MAE预训练模型存档列表
            "ViTMAEForPreTraining",  # 扩展预训练模型
            "ViTMAELayer",  # 扩展ViT MAE层
            "ViTMAEModel",  # 扩展ViT MAE模型
            "ViTMAEPreTrainedModel",  # 扩展ViT MAE预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vit_msn模块
    _import_structure["models.vit_msn"].extend(
        [
            "VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VIT MSN预训练模型存档列表
            "ViTMSNForImageClassification",  # 扩展图像分类模型
            "ViTMSNModel",  # 扩展ViT MSN模型
            "ViTMSNPreTrainedModel",  # 扩展ViT MSN预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vitdet模块
    _import_structure["models.vitdet"].extend(
        [
            "VITDET_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VITDET预训练模型存档列表
            "VitDetBackbone",  # 扩展VitDet后端模型
            "VitDetModel",  # 扩展VitDet模型
            "VitDetPreTrainedModel",  # 扩展VitDet预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vitmatte模块
    _import_structure["models.vitmatte"].extend(
        [
            "VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VITMATTE预训练模型存档列表
            "VitMatteForImageMatting",  # 扩展图像抠图模型
            "VitMattePreTrainedModel",  # 扩展VitMatte预训练模型
        ]
    )
    # 将指定模块内的预定义符号扩展到_import_structure字典中的models.vits模块
    _import_structure["models.vits"].extend(
        [
            "VITS_PRETRAINED_MODEL_ARCHIVE_LIST",  # 扩展VITS预训练模型存档列表
            "VitsModel",  # 扩展Vits模型
            "VitsPreTrainedModel",  # 扩展Vits预训练模型
        ]
    )
    # 将以下模型类和预训练模型列表添加到_import_structure字典中的各自模块中
    
    _import_structure["models.vivit"].extend(
        [
            "VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Vivit模块的预训练模型归档列表
            "VivitForVideoClassification",  # Vivit模型用于视频分类
            "VivitModel",  # Vivit模型
            "VivitPreTrainedModel",  # Vivit预训练模型
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",  # Wav2Vec2模块的预训练模型归档列表
            "Wav2Vec2ForAudioFrameClassification",  # Wav2Vec2模型用于音频帧分类
            "Wav2Vec2ForCTC",  # Wav2Vec2模型用于CTC任务
            "Wav2Vec2ForMaskedLM",  # Wav2Vec2模型用于遮蔽语言建模
            "Wav2Vec2ForPreTraining",  # Wav2Vec2模型用于预训练
            "Wav2Vec2ForSequenceClassification",  # Wav2Vec2模型用于序列分类
            "Wav2Vec2ForXVector",  # Wav2Vec2模型用于X向量生成
            "Wav2Vec2Model",  # Wav2Vec2模型
            "Wav2Vec2PreTrainedModel",  # Wav2Vec2预训练模型
        ]
    )
    _import_structure["models.wav2vec2_bert"].extend(
        [
            "WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Wav2Vec2 BERT模块的预训练模型归档列表
            "Wav2Vec2BertForAudioFrameClassification",  # Wav2Vec2 BERT模型用于音频帧分类
            "Wav2Vec2BertForCTC",  # Wav2Vec2 BERT模型用于CTC任务
            "Wav2Vec2BertForSequenceClassification",  # Wav2Vec2 BERT模型用于序列分类
            "Wav2Vec2BertForXVector",  # Wav2Vec2 BERT模型用于X向量生成
            "Wav2Vec2BertModel",  # Wav2Vec2 BERT模型
            "Wav2Vec2BertPreTrainedModel",  # Wav2Vec2 BERT预训练模型
        ]
    )
    _import_structure["models.wav2vec2_conformer"].extend(
        [
            "WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # Wav2Vec2 Conformer模块的预训练模型归档列表
            "Wav2Vec2ConformerForAudioFrameClassification",  # Wav2Vec2 Conformer模型用于音频帧分类
            "Wav2Vec2ConformerForCTC",  # Wav2Vec2 Conformer模型用于CTC任务
            "Wav2Vec2ConformerForPreTraining",  # Wav2Vec2 Conformer模型用于预训练
            "Wav2Vec2ConformerForSequenceClassification",  # Wav2Vec2 Conformer模型用于序列分类
            "Wav2Vec2ConformerForXVector",  # Wav2Vec2 Conformer模型用于X向量生成
            "Wav2Vec2ConformerModel",  # Wav2Vec2 Conformer模型
            "Wav2Vec2ConformerPreTrainedModel",  # Wav2Vec2 Conformer预训练模型
        ]
    )
    _import_structure["models.wavlm"].extend(
        [
            "WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST",  # WavLM模块的预训练模型归档列表
            "WavLMForAudioFrameClassification",  # WavLM模型用于音频帧分类
            "WavLMForCTC",  # WavLM模型用于CTC任务
            "WavLMForSequenceClassification",  # WavLM模型用于序列分类
            "WavLMForXVector",  # WavLM模型用于X向量生成
            "WavLMModel",  # WavLM模型
            "WavLMPreTrainedModel",  # WavLM预训练模型
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",  # Whisper模块的预训练模型归档列表
            "WhisperForAudioClassification",  # Whisper模型用于音频分类
            "WhisperForCausalLM",  # Whisper模型用于因果语言建模
            "WhisperForConditionalGeneration",  # Whisper模型用于条件生成
            "WhisperModel",  # Whisper模型
            "WhisperPreTrainedModel",  # Whisper预训练模型
        ]
    )
    _import_structure["models.x_clip"].extend(
        [
            "XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # XCLIP模块的预训练模型归档列表
            "XCLIPModel",  # XCLIP模型
            "XCLIPPreTrainedModel",  # XCLIP预训练模型
            "XCLIPTextModel",  # XCLIP文本模型
            "XCLIPVisionModel",  # XCLIP视觉模型
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",  # XGLM模块的预训练模型归档列表
            "XGLMForCausalLM",  # XGLM模型用于因果语言建模
            "XGLMModel",  # XGLM模型
            "XGLMPreTrainedModel",  # XGLM预训练模型
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.xlm" 键对应的值列表中
    _import_structure["models.xlm"].extend(
        [
            "XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMForMultipleChoice",
            "XLMForQuestionAnswering",
            "XLMForQuestionAnsweringSimple",
            "XLMForSequenceClassification",
            "XLMForTokenClassification",
            "XLMModel",
            "XLMPreTrainedModel",
            "XLMWithLMHeadModel",
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.xlm_prophetnet" 键对应的值列表中
    _import_structure["models.xlm_prophetnet"].extend(
        [
            "XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
            "XLMProphetNetPreTrainedModel",
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.xlm_roberta" 键对应的值列表中
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMRobertaForCausalLM",
            "XLMRobertaForMaskedLM",
            "XLMRobertaForMultipleChoice",
            "XLMRobertaForQuestionAnswering",
            "XLMRobertaForSequenceClassification",
            "XLMRobertaForTokenClassification",
            "XLMRobertaModel",
            "XLMRobertaPreTrainedModel",
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.xlm_roberta_xl" 键对应的值列表中
    _import_structure["models.xlm_roberta_xl"].extend(
        [
            "XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLMRobertaXLForCausalLM",
            "XLMRobertaXLForMaskedLM",
            "XLMRobertaXLForMultipleChoice",
            "XLMRobertaXLForQuestionAnswering",
            "XLMRobertaXLForSequenceClassification",
            "XLMRobertaXLForTokenClassification",
            "XLMRobertaXLModel",
            "XLMRobertaXLPreTrainedModel",
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.xlnet" 键对应的值列表中
    _import_structure["models.xlnet"].extend(
        [
            "XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XLNetForMultipleChoice",
            "XLNetForQuestionAnswering",
            "XLNetForQuestionAnsweringSimple",
            "XLNetForSequenceClassification",
            "XLNetForTokenClassification",
            "XLNetLMHeadModel",
            "XLNetModel",
            "XLNetPreTrainedModel",
            "load_tf_weights_in_xlnet",
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.xmod" 键对应的值列表中
    _import_structure["models.xmod"].extend(
        [
            "XMOD_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XmodForCausalLM",
            "XmodForMaskedLM",
            "XmodForMultipleChoice",
            "XmodForQuestionAnswering",
            "XmodForSequenceClassification",
            "XmodForTokenClassification",
            "XmodModel",
            "XmodPreTrainedModel",
        ]
    )
    # 将以下模型名称添加到 _import_structure 字典的 "models.yolos" 键对应的值列表中
    _import_structure["models.yolos"].extend(
        [
            "YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YOSO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "YosoForMaskedLM",
            "YosoForMultipleChoice",
            "YosoForQuestionAnswering",
            "YosoForSequenceClassification",
            "YosoForTokenClassification",
            "YosoLayer",
            "YosoModel",
            "YosoPreTrainedModel",
        ]
    )
    # 将列表中的模块名称扩展到 "models.yoso" 的导入结构中
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ]
    # 设置 "optimization" 的导入结构为包含的优化模块列表
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
    ]
    # 设置 "pytorch_utils" 的导入结构为包含的 PyTorch 实用工具列表
    _import_structure["sagemaker"] = []
    # 设置 "sagemaker" 的导入结构为空列表
    _import_structure["time_series_utils"] = []
    # 设置 "time_series_utils" 的导入结构为空列表
    _import_structure["trainer"] = ["Trainer"]
    # 设置 "trainer" 的导入结构为包含单个元素 "Trainer" 的列表
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    # 设置 "trainer_pt_utils" 的导入结构为包含单个元素 "torch_distributed_zero_first" 的列表
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]
    # 设置 "trainer_seq2seq" 的导入结构为包含单个元素 "Seq2SeqTrainer" 的列表
# TensorFlow-backed objects
# 尝试检查是否可用 TensorFlow
try:
    if not is_tf_available():
        # 如果 TensorFlow 不可用，抛出自定义的依赖未满足异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果依赖未满足，从本地导入虚拟的 TensorFlow 对象
    from .utils import dummy_tf_objects

    # 更新导入结构，将 dummy_tf_objects 中非私有的对象名称添加到 _import_structure 中
    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    # 如果 TensorFlow 可用，更新导入结构来包含以下模块和类
    _import_structure["activations_tf"] = []
    _import_structure["benchmark.benchmark_args_tf"] = ["TensorFlowBenchmarkArguments"]
    _import_structure["benchmark.benchmark_tf"] = ["TensorFlowBenchmark"]
    _import_structure["generation"].extend(
        [
            "TFForcedBOSTokenLogitsProcessor",
            "TFForcedEOSTokenLogitsProcessor",
            "TFForceTokensLogitsProcessor",
            "TFGenerationMixin",
            "TFLogitsProcessor",
            "TFLogitsProcessorList",
            "TFLogitsWarper",
            "TFMinLengthLogitsProcessor",
            "TFNoBadWordsLogitsProcessor",
            "TFNoRepeatNGramLogitsProcessor",
            "TFRepetitionPenaltyLogitsProcessor",
            "TFSuppressTokensAtBeginLogitsProcessor",
            "TFSuppressTokensLogitsProcessor",
            "TFTemperatureLogitsWarper",
            "TFTopKLogitsWarper",
            "TFTopPLogitsWarper",
        ]
    )
    _import_structure["generation_tf_utils"] = []
    _import_structure["keras_callbacks"] = ["KerasMetricCallback", "PushToHubCallback"]
    _import_structure["modeling_tf_outputs"] = []
    _import_structure["modeling_tf_utils"] = [
        "TFPreTrainedModel",
        "TFSequenceSummary",
        "TFSharedEmbeddings",
        "shape_list",
    ]
    # 更新导入结构，将 models.albert 中指定的类和常量添加到 _import_structure 中
    _import_structure["models.albert"].extend(
        [
            "TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFAlbertForMaskedLM",
            "TFAlbertForMultipleChoice",
            "TFAlbertForPreTraining",
            "TFAlbertForQuestionAnswering",
            "TFAlbertForSequenceClassification",
            "TFAlbertForTokenClassification",
            "TFAlbertMainLayer",
            "TFAlbertModel",
            "TFAlbertPreTrainedModel",
        ]
    )
    # 将 "models.auto" 下的模型名称列表扩展，添加多个 TensorFlow 模型映射
    _import_structure["models.auto"].extend(
        [
            "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",  # 音频分类模型映射
            "TF_MODEL_FOR_CAUSAL_LM_MAPPING",  # 因果语言模型映射
            "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",  # 文档问答模型映射
            "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",  # 图像分类模型映射
            "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",  # 掩膜图像建模模型映射
            "TF_MODEL_FOR_MASKED_LM_MAPPING",  # 掩膜语言模型映射
            "TF_MODEL_FOR_MASK_GENERATION_MAPPING",  # 掩膜生成模型映射
            "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",  # 多项选择模型映射
            "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",  # 下一句预测模型映射
            "TF_MODEL_FOR_PRETRAINING_MAPPING",  # 预训练模型映射
            "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",  # 问答模型映射
            "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",  # 语义分割模型映射
            "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",  # 序列到序列因果语言模型映射
            "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",  # 序列分类模型映射
            "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",  # 语音序列到序列模型映射
            "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",  # 表格问答模型映射
            "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",  # 文本编码模型映射
            "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",  # 标记分类模型映射
            "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",  # 视觉到序列模型映射
            "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",  # 零样本图像分类模型映射
            "TF_MODEL_MAPPING",  # TensorFlow 模型映射
            "TF_MODEL_WITH_LM_HEAD_MAPPING",  # 带语言模型头的 TensorFlow 模型映射
            "TFAutoModel",  # 自动选择模型
            "TFAutoModelForAudioClassification",  # 自动选择音频分类模型
            "TFAutoModelForCausalLM",  # 自动选择因果语言模型
            "TFAutoModelForDocumentQuestionAnswering",  # 自动选择文档问答模型
            "TFAutoModelForImageClassification",  # 自动选择图像分类模型
            "TFAutoModelForMaskedImageModeling",  # 自动选择掩膜图像建模模型
            "TFAutoModelForMaskedLM",  # 自动选择掩膜语言模型
            "TFAutoModelForMaskGeneration",  # 自动选择掩膜生成模型
            "TFAutoModelForMultipleChoice",  # 自动选择多项选择模型
            "TFAutoModelForNextSentencePrediction",  # 自动选择下一句预测模型
            "TFAutoModelForPreTraining",  # 自动选择预训练模型
            "TFAutoModelForQuestionAnswering",  # 自动选择问答模型
            "TFAutoModelForSemanticSegmentation",  # 自动选择语义分割模型
            "TFAutoModelForSeq2SeqLM",  # 自动选择序列到序列语言模型
            "TFAutoModelForSequenceClassification",  # 自动选择序列分类模型
            "TFAutoModelForSpeechSeq2Seq",  # 自动选择语音序列到序列模型
            "TFAutoModelForTableQuestionAnswering",  # 自动选择表格问答模型
            "TFAutoModelForTextEncoding",  # 自动选择文本编码模型
            "TFAutoModelForTokenClassification",  # 自动选择标记分类模型
            "TFAutoModelForVision2Seq",  # 自动选择视觉到序列模型
            "TFAutoModelForZeroShotImageClassification",  # 自动选择零样本图像分类模型
            "TFAutoModelWithLMHead",  # 自动选择带语言模型头的模型
        ]
    )

    # 将 "models.bart" 下的模型名称列表扩展，添加多个 TFBart 模型映射
    _import_structure["models.bart"].extend(
        [
            "TFBartForConditionalGeneration",  # 条件生成模型
            "TFBartForSequenceClassification",  # 序列分类模型
            "TFBartModel",  # BART 模型
            "TFBartPretrainedModel",  # 预训练的 BART 模型
        ]
    )
    # 将 "models.bert" 中的模块列表扩展，包括多个预训练模型和类
    _import_structure["models.bert"].extend(
        [
            "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFBertEmbeddings",
            "TFBertForMaskedLM",
            "TFBertForMultipleChoice",
            "TFBertForNextSentencePrediction",
            "TFBertForPreTraining",
            "TFBertForQuestionAnswering",
            "TFBertForSequenceClassification",
            "TFBertForTokenClassification",
            "TFBertLMHeadModel",
            "TFBertMainLayer",
            "TFBertModel",
            "TFBertPreTrainedModel",
        ]
    )
    # 将 "models.blenderbot" 中的模块列表扩展，包括条件生成、模型和预训练模型
    _import_structure["models.blenderbot"].extend(
        [
            "TFBlenderbotForConditionalGeneration",
            "TFBlenderbotModel",
            "TFBlenderbotPreTrainedModel",
        ]
    )
    # 将 "models.blenderbot_small" 中的模块列表扩展，包括条件生成、模型和预训练模型
    _import_structure["models.blenderbot_small"].extend(
        [
            "TFBlenderbotSmallForConditionalGeneration",
            "TFBlenderbotSmallModel",
            "TFBlenderbotSmallPreTrainedModel",
        ]
    )
    # 将 "models.blip" 中的模块列表扩展，包括条件生成、图像文本检索、问答、模型和预训练模型
    _import_structure["models.blip"].extend(
        [
            "TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFBlipForConditionalGeneration",
            "TFBlipForImageTextRetrieval",
            "TFBlipForQuestionAnswering",
            "TFBlipModel",
            "TFBlipPreTrainedModel",
            "TFBlipTextModel",
            "TFBlipVisionModel",
        ]
    )
    # 将 "models.camembert" 中的模块列表扩展，包括语言建模、多项选择、问答、分类、标记分类、模型和预训练模型
    _import_structure["models.camembert"].extend(
        [
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
    )
    # 将 "models.clip" 中的模块列表扩展，包括语言与图像交互的模型和预训练模型
    _import_structure["models.clip"].extend(
        [
            "TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCLIPModel",
            "TFCLIPPreTrainedModel",
            "TFCLIPTextModel",
            "TFCLIPVisionModel",
        ]
    )
    # 将 "models.convbert" 中的模块列表扩展，包括语言建模、多项选择、问答、分类、标记分类、层和预训练模型
    _import_structure["models.convbert"].extend(
        [
            "TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFConvBertForMaskedLM",
            "TFConvBertForMultipleChoice",
            "TFConvBertForQuestionAnswering",
            "TFConvBertForSequenceClassification",
            "TFConvBertForTokenClassification",
            "TFConvBertLayer",
            "TFConvBertModel",
            "TFConvBertPreTrainedModel",
        ]
    )
    # 将 "models.convnext" 中的模块列表扩展，包括图像分类、模型和预训练模型
    _import_structure["models.convnext"].extend(
        [
            "TFConvNextForImageClassification",
            "TFConvNextModel",
            "TFConvNextPreTrainedModel",
        ]
    )
    # 将 "models.convnextv2" 中的模块列表扩展，包括图像分类、模型和预训练模型
    _import_structure["models.convnextv2"].extend(
        [
            "TFConvNextV2ForImageClassification",
            "TFConvNextV2Model",
            "TFConvNextV2PreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.ctrl”部分
    _import_structure["models.ctrl"].extend(
        [
            "TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCTRLForSequenceClassification",
            "TFCTRLLMHeadModel",
            "TFCTRLModel",
            "TFCTRLPreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.cvt”部分
    _import_structure["models.cvt"].extend(
        [
            "TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFCvtForImageClassification",
            "TFCvtModel",
            "TFCvtPreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.data2vec”部分
    _import_structure["models.data2vec"].extend(
        [
            "TFData2VecVisionForImageClassification",
            "TFData2VecVisionForSemanticSegmentation",
            "TFData2VecVisionModel",
            "TFData2VecVisionPreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.deberta”部分
    _import_structure["models.deberta"].extend(
        [
            "TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDebertaForMaskedLM",
            "TFDebertaForQuestionAnswering",
            "TFDebertaForSequenceClassification",
            "TFDebertaForTokenClassification",
            "TFDebertaModel",
            "TFDebertaPreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.deberta_v2”部分
    _import_structure["models.deberta_v2"].extend(
        [
            "TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDebertaV2ForMaskedLM",
            "TFDebertaV2ForMultipleChoice",
            "TFDebertaV2ForQuestionAnswering",
            "TFDebertaV2ForSequenceClassification",
            "TFDebertaV2ForTokenClassification",
            "TFDebertaV2Model",
            "TFDebertaV2PreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.deit”部分
    _import_structure["models.deit"].extend(
        [
            "TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDeiTForImageClassification",
            "TFDeiTForImageClassificationWithTeacher",
            "TFDeiTForMaskedImageModeling",
            "TFDeiTModel",
            "TFDeiTPreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.deprecated.transfo_xl”部分
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFAdaptiveEmbedding",
            "TFTransfoXLForSequenceClassification",
            "TFTransfoXLLMHeadModel",
            "TFTransfoXLMainLayer",
            "TFTransfoXLModel",
            "TFTransfoXLPreTrainedModel",
        ]
    )
    # 将指定模块中的多个成员添加到_import_structure字典中的“models.distilbert”部分
    _import_structure["models.distilbert"].extend(
        [
            "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDistilBertForMaskedLM",
            "TFDistilBertForMultipleChoice",
            "TFDistilBertForQuestionAnswering",
            "TFDistilBertForSequenceClassification",
            "TFDistilBertForTokenClassification",
            "TFDistilBertMainLayer",
            "TFDistilBertModel",
            "TFDistilBertPreTrainedModel",
        ]
    )
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.dpr"键下
    _import_structure["models.dpr"].extend(
        [
            "TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFDPRContextEncoder",
            "TFDPRPretrainedContextEncoder",
            "TFDPRPretrainedQuestionEncoder",
            "TFDPRPretrainedReader",
            "TFDPRQuestionEncoder",
            "TFDPRReader",
        ]
    )
    
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.efficientformer"键下
    _import_structure["models.efficientformer"].extend(
        [
            "TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFEfficientFormerForImageClassification",
            "TFEfficientFormerForImageClassificationWithTeacher",
            "TFEfficientFormerModel",
            "TFEfficientFormerPreTrainedModel",
        ]
    )
    
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.electra"键下
    _import_structure["models.electra"].extend(
        [
            "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFElectraForMaskedLM",
            "TFElectraForMultipleChoice",
            "TFElectraForPreTraining",
            "TFElectraForQuestionAnswering",
            "TFElectraForSequenceClassification",
            "TFElectraForTokenClassification",
            "TFElectraModel",
            "TFElectraPreTrainedModel",
        ]
    )
    
    # 将指定模块中的"TFEncoderDecoderModel"类添加到_import_structure字典中的"models.encoder_decoder"键下
    _import_structure["models.encoder_decoder"].append("TFEncoderDecoderModel")
    
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.esm"键下
    _import_structure["models.esm"].extend(
        [
            "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFEsmForMaskedLM",
            "TFEsmForSequenceClassification",
            "TFEsmForTokenClassification",
            "TFEsmModel",
            "TFEsmPreTrainedModel",
        ]
    )
    
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.flaubert"键下
    _import_structure["models.flaubert"].extend(
        [
            "TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFFlaubertForMultipleChoice",
            "TFFlaubertForQuestionAnsweringSimple",
            "TFFlaubertForSequenceClassification",
            "TFFlaubertForTokenClassification",
            "TFFlaubertModel",
            "TFFlaubertPreTrainedModel",
            "TFFlaubertWithLMHeadModel",
        ]
    )
    
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.funnel"键下
    _import_structure["models.funnel"].extend(
        [
            "TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFFunnelBaseModel",
            "TFFunnelForMaskedLM",
            "TFFunnelForMultipleChoice",
            "TFFunnelForPreTraining",
            "TFFunnelForQuestionAnswering",
            "TFFunnelForSequenceClassification",
            "TFFunnelForTokenClassification",
            "TFFunnelModel",
            "TFFunnelPreTrainedModel",
        ]
    )
    
    # 将指定模块中的类和常量列表添加到_import_structure字典中的"models.gpt2"键下
    _import_structure["models.gpt2"].extend(
        [
            "TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFGPT2DoubleHeadsModel",
            "TFGPT2ForSequenceClassification",
            "TFGPT2LMHeadModel",
            "TFGPT2MainLayer",
            "TFGPT2Model",
            "TFGPT2PreTrainedModel",
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.gptj"].extend(
        [
            "TFGPTJForCausalLM",  # 添加 GPTJ 的条件语言模型类
            "TFGPTJForQuestionAnswering",  # 添加 GPTJ 的问答模型类
            "TFGPTJForSequenceClassification",  # 添加 GPTJ 的序列分类模型类
            "TFGPTJModel",  # 添加 GPTJ 的基础模型类
            "TFGPTJPreTrainedModel",  # 添加 GPTJ 的预训练模型基类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.groupvit"].extend(
        [
            "TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加 GroupViT 预训练模型存档列表
            "TFGroupViTModel",  # 添加 GroupViT 模型类
            "TFGroupViTPreTrainedModel",  # 添加 GroupViT 预训练模型基类
            "TFGroupViTTextModel",  # 添加 GroupViT 文本模型类
            "TFGroupViTVisionModel",  # 添加 GroupViT 视觉模型类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.hubert"].extend(
        [
            "TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加 Hubert 预训练模型存档列表
            "TFHubertForCTC",  # 添加 Hubert CTC 模型类
            "TFHubertModel",  # 添加 Hubert 模型类
            "TFHubertPreTrainedModel",  # 添加 Hubert 预训练模型基类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.layoutlm"].extend(
        [
            "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加 LayoutLM 预训练模型存档列表
            "TFLayoutLMForMaskedLM",  # 添加 LayoutLM 掩码语言模型类
            "TFLayoutLMForQuestionAnswering",  # 添加 LayoutLM 问答模型类
            "TFLayoutLMForSequenceClassification",  # 添加 LayoutLM 序列分类模型类
            "TFLayoutLMForTokenClassification",  # 添加 LayoutLM 标记分类模型类
            "TFLayoutLMMainLayer",  # 添加 LayoutLM 主层类
            "TFLayoutLMModel",  # 添加 LayoutLM 模型类
            "TFLayoutLMPreTrainedModel",  # 添加 LayoutLM 预训练模型基类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.layoutlmv3"].extend(
        [
            "TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加 LayoutLMv3 预训练模型存档列表
            "TFLayoutLMv3ForQuestionAnswering",  # 添加 LayoutLMv3 问答模型类
            "TFLayoutLMv3ForSequenceClassification",  # 添加 LayoutLMv3 序列分类模型类
            "TFLayoutLMv3ForTokenClassification",  # 添加 LayoutLMv3 标记分类模型类
            "TFLayoutLMv3Model",  # 添加 LayoutLMv3 模型类
            "TFLayoutLMv3PreTrainedModel",  # 添加 LayoutLMv3 预训练模型基类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.led"].extend(
        [
            "TFLEDForConditionalGeneration",  # 添加 LED 有条件生成模型类
            "TFLEDModel",  # 添加 LED 模型类
            "TFLEDPreTrainedModel",  # 添加 LED 预训练模型基类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.longformer"].extend(
        [
            "TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加 Longformer 预训练模型存档列表
            "TFLongformerForMaskedLM",  # 添加 Longformer 掩码语言模型类
            "TFLongformerForMultipleChoice",  # 添加 Longformer 多项选择模型类
            "TFLongformerForQuestionAnswering",  # 添加 Longformer 问答模型类
            "TFLongformerForSequenceClassification",  # 添加 Longformer 序列分类模型类
            "TFLongformerForTokenClassification",  # 添加 Longformer 标记分类模型类
            "TFLongformerModel",  # 添加 Longformer 模型类
            "TFLongformerPreTrainedModel",  # 添加 Longformer 预训练模型基类
            "TFLongformerSelfAttention",  # 添加 Longformer 自注意力类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.lxmert"].extend(
        [
            "TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加 LXMERT 预训练模型存档列表
            "TFLxmertForPreTraining",  # 添加 LXMERT 预训练模型类
            "TFLxmertMainLayer",  # 添加 LXMERT 主层类
            "TFLxmertModel",  # 添加 LXMERT 模型类
            "TFLxmertPreTrainedModel",  # 添加 LXMERT 预训练模型基类
            "TFLxmertVisualFeatureEncoder",  # 添加 LXMERT 视觉特征编码器类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.marian"].extend(
        [
            "TFMarianModel",  # 添加 Marian 模型类
            "TFMarianMTModel",  # 添加 Marian 机器翻译模型类
            "TFMarianPreTrainedModel",  # 添加 Marian 预训练模型基类
        ]
    )
    # 将指定模块下的类名添加到导入结构中
    _import_structure["models.mbart"].extend(
        [
            "TFMBartForConditionalGeneration",  # 添加 MBart 有条件生成模型类
            "TFMBartModel",  # 添加 MBart 模型类
            "TFMBartPreTrainedModel",  # 添加 MBart 预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.mobilebert"的内容
    _import_structure["models.mobilebert"].extend(
        [
            "TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加预训练模型归档列表
            "TFMobileBertForMaskedLM",  # MobileBERT的Masked Language Model
            "TFMobileBertForMultipleChoice",  # MobileBERT的多选题模型
            "TFMobileBertForNextSentencePrediction",  # MobileBERT的下一句预测模型
            "TFMobileBertForPreTraining",  # MobileBERT的预训练模型
            "TFMobileBertForQuestionAnswering",  # MobileBERT的问答模型
            "TFMobileBertForSequenceClassification",  # MobileBERT的序列分类模型
            "TFMobileBertForTokenClassification",  # MobileBERT的标记分类模型
            "TFMobileBertMainLayer",  # MobileBERT的主层
            "TFMobileBertModel",  # MobileBERT的模型
            "TFMobileBertPreTrainedModel",  # MobileBERT的预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.mobilevit"的内容
    _import_structure["models.mobilevit"].extend(
        [
            "TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加预训练模型归档列表
            "TFMobileViTForImageClassification",  # MobileViT的图像分类模型
            "TFMobileViTForSemanticSegmentation",  # MobileViT的语义分割模型
            "TFMobileViTModel",  # MobileViT的模型
            "TFMobileViTPreTrainedModel",  # MobileViT的预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.mpnet"的内容
    _import_structure["models.mpnet"].extend(
        [
            "TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加预训练模型归档列表
            "TFMPNetForMaskedLM",  # MPNet的Masked Language Model
            "TFMPNetForMultipleChoice",  # MPNet的多选题模型
            "TFMPNetForQuestionAnswering",  # MPNet的问答模型
            "TFMPNetForSequenceClassification",  # MPNet的序列分类模型
            "TFMPNetForTokenClassification",  # MPNet的标记分类模型
            "TFMPNetMainLayer",  # MPNet的主层
            "TFMPNetModel",  # MPNet的模型
            "TFMPNetPreTrainedModel",  # MPNet的预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.mt5"的内容
    _import_structure["models.mt5"].extend(
        [
            "TFMT5EncoderModel",  # MT5的编码器模型
            "TFMT5ForConditionalGeneration",  # MT5的条件生成模型
            "TFMT5Model",  # MT5的模型
        ]
    )
    # 扩展_import_structure字典中"models.openai"的内容
    _import_structure["models.openai"].extend(
        [
            "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加预训练模型归档列表
            "TFOpenAIGPTDoubleHeadsModel",  # OpenAI GPT的双头模型
            "TFOpenAIGPTForSequenceClassification",  # OpenAI GPT的序列分类模型
            "TFOpenAIGPTLMHeadModel",  # OpenAI GPT的语言模型头
            "TFOpenAIGPTMainLayer",  # OpenAI GPT的主层
            "TFOpenAIGPTModel",  # OpenAI GPT的模型
            "TFOpenAIGPTPreTrainedModel",  # OpenAI GPT的预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.opt"的内容
    _import_structure["models.opt"].extend(
        [
            "TFOPTForCausalLM",  # OPT的因果语言模型
            "TFOPTModel",  # OPT的模型
            "TFOPTPreTrainedModel",  # OPT的预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.pegasus"的内容
    _import_structure["models.pegasus"].extend(
        [
            "TFPegasusForConditionalGeneration",  # Pegasus的条件生成模型
            "TFPegasusModel",  # Pegasus的模型
            "TFPegasusPreTrainedModel",  # Pegasus的预训练模型基类
        ]
    )
    # 扩展_import_structure字典中"models.rag"的内容
    _import_structure["models.rag"].extend(
        [
            "TFRagModel",  # RAG的模型
            "TFRagPreTrainedModel",  # RAG的预训练模型基类
            "TFRagSequenceForGeneration",  # RAG用于生成序列的模型
            "TFRagTokenForGeneration",  # RAG用于生成标记的模型
        ]
    )
    # 扩展_import_structure字典中"models.regnet"的内容
    _import_structure["models.regnet"].extend(
        [
            "TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # 添加预训练模型归档列表
            "TFRegNetForImageClassification",  # RegNet的图像分类模型
            "TFRegNetModel",  # RegNet的模型
            "TFRegNetPreTrainedModel",  # RegNet的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.rembert"].extend(
        [
            "TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # REMBERT 模型的预训练模型存档列表
            "TFRemBertForCausalLM",  # 用于因果语言建模的 TFRemBert 模型
            "TFRemBertForMaskedLM",  # 用于遮蔽语言建模的 TFRemBert 模型
            "TFRemBertForMultipleChoice",  # 用于多选题的 TFRemBert 模型
            "TFRemBertForQuestionAnswering",  # 用于问答任务的 TFRemBert 模型
            "TFRemBertForSequenceClassification",  # 用于序列分类任务的 TFRemBert 模型
            "TFRemBertForTokenClassification",  # 用于标记分类任务的 TFRemBert 模型
            "TFRemBertLayer",  # REMBERT 模型的层定义
            "TFRemBertModel",  # REMBERT 模型的主模型
            "TFRemBertPreTrainedModel",  # REMBERT 模型的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.resnet"].extend(
        [
            "TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # RESNET 模型的预训练模型存档列表
            "TFResNetForImageClassification",  # 用于图像分类的 TFResNet 模型
            "TFResNetModel",  # RESNET 模型的主模型
            "TFResNetPreTrainedModel",  # RESNET 模型的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.roberta"].extend(
        [
            "TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",  # ROBERTA 模型的预训练模型存档列表
            "TFRobertaForCausalLM",  # 用于因果语言建模的 TFRoberta 模型
            "TFRobertaForMaskedLM",  # 用于遮蔽语言建模的 TFRoberta 模型
            "TFRobertaForMultipleChoice",  # 用于多选题的 TFRoberta 模型
            "TFRobertaForQuestionAnswering",  # 用于问答任务的 TFRoberta 模型
            "TFRobertaForSequenceClassification",  # 用于序列分类任务的 TFRoberta 模型
            "TFRobertaForTokenClassification",  # 用于标记分类任务的 TFRoberta 模型
            "TFRobertaMainLayer",  # ROBERTA 模型的主层定义
            "TFRobertaModel",  # ROBERTA 模型的主模型
            "TFRobertaPreTrainedModel",  # ROBERTA 模型的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",  # ROBERTA 模型带有预层标准化的预训练模型存档列表
            "TFRobertaPreLayerNormForCausalLM",  # 带有预层标准化的因果语言建模 TFRoberta 模型
            "TFRobertaPreLayerNormForMaskedLM",  # 带有预层标准化的遮蔽语言建模 TFRoberta 模型
            "TFRobertaPreLayerNormForMultipleChoice",  # 带有预层标准化的多选题 TFRoberta 模型
            "TFRobertaPreLayerNormForQuestionAnswering",  # 带有预层标准化的问答任务 TFRoberta 模型
            "TFRobertaPreLayerNormForSequenceClassification",  # 带有预层标准化的序列分类任务 TFRoberta 模型
            "TFRobertaPreLayerNormForTokenClassification",  # 带有预层标准化的标记分类任务 TFRoberta 模型
            "TFRobertaPreLayerNormMainLayer",  # 带有预层标准化的 ROBERTA 模型的主层定义
            "TFRobertaPreLayerNormModel",  # 带有预层标准化的 ROBERTA 模型的主模型
            "TFRobertaPreLayerNormPreTrainedModel",  # 带有预层标准化的 ROBERTA 模型的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.roformer"].extend(
        [
            "TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # ROFORMER 模型的预训练模型存档列表
            "TFRoFormerForCausalLM",  # 用于因果语言建模的 TFRoFormer 模型
            "TFRoFormerForMaskedLM",  # 用于遮蔽语言建模的 TFRoFormer 模型
            "TFRoFormerForMultipleChoice",  # 用于多选题的 TFRoFormer 模型
            "TFRoFormerForQuestionAnswering",  # 用于问答任务的 TFRoFormer 模型
            "TFRoFormerForSequenceClassification",  # 用于序列分类任务的 TFRoFormer 模型
            "TFRoFormerForTokenClassification",  # 用于标记分类任务的 TFRoFormer 模型
            "TFRoFormerLayer",  # ROFORMER 模型的层定义
            "TFRoFormerModel",  # ROFORMER 模型的主模型
            "TFRoFormerPreTrainedModel",  # ROFORMER 模型的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.sam"].extend(
        [
            "TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST",  # SAM 模型的预训练模型存档列表
            "TFSamModel",  # SAM 模型的主模型
            "TFSamPreTrainedModel",  # SAM 模型的预训练模型基类
        ]
    )
    # 将以下模块的多个预定义名称添加到 _import_structure 字典中的相应模块下
    _import_structure["models.segformer"].extend(
        [
            "TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # SEGFORMER 模型的预训练模型存档列表
            "TFSegformerDecodeHead",  # 用于解码头的 TFSegformer 模型
            "TFSegformerForImageClassification",  # 用于图像分类的 TFSegformer 模型
            "TFSegformerForSemanticSegmentation",  # 用于语义分割的 TFSegformer 模型
            "TFSegformerModel",  # SEGFORMER 模型的主模型
            "TFSegformerPreTrainedModel",  # SEGFORMER 模型的预训练模型基类
        ]
    )
    # 将"models.speech_to_text"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.speech_to_text"].extend(
        [
            "TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSpeech2TextForConditionalGeneration",
            "TFSpeech2TextModel",
            "TFSpeech2TextPreTrainedModel",
        ]
    )
    
    # 将"models.swin"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.swin"].extend(
        [
            "TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSwinForImageClassification",
            "TFSwinForMaskedImageModeling",
            "TFSwinModel",
            "TFSwinPreTrainedModel",
        ]
    )
    
    # 将"models.t5"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.t5"].extend(
        [
            "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    
    # 将"models.tapas"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.tapas"].extend(
        [
            "TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFTapasForMaskedLM",
            "TFTapasForQuestionAnswering",
            "TFTapasForSequenceClassification",
            "TFTapasModel",
            "TFTapasPreTrainedModel",
        ]
    )
    
    # 将"models.vision_encoder_decoder"模块中的特定名称添加到_import_structure字典中
    _import_structure["models.vision_encoder_decoder"].extend(["TFVisionEncoderDecoderModel"])
    
    # 将"models.vision_text_dual_encoder"模块中的特定名称添加到_import_structure字典中
    _import_structure["models.vision_text_dual_encoder"].extend(["TFVisionTextDualEncoderModel"])
    
    # 将"models.vit"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.vit"].extend(
        [
            "TFViTForImageClassification",
            "TFViTModel",
            "TFViTPreTrainedModel",
        ]
    )
    
    # 将"models.vit_mae"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.vit_mae"].extend(
        [
            "TFViTMAEForPreTraining",
            "TFViTMAEModel",
            "TFViTMAEPreTrainedModel",
        ]
    )
    
    # 将"models.wav2vec2"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.wav2vec2"].extend(
        [
            "TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFWav2Vec2ForCTC",
            "TFWav2Vec2ForSequenceClassification",
            "TFWav2Vec2Model",
            "TFWav2Vec2PreTrainedModel",
        ]
    )
    
    # 将"models.whisper"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.whisper"].extend(
        [
            "TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFWhisperForConditionalGeneration",
            "TFWhisperModel",
            "TFWhisperPreTrainedModel",
        ]
    )
    
    # 将"models.xglm"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.xglm"].extend(
        [
            "TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXGLMForCausalLM",
            "TFXGLMModel",
            "TFXGLMPreTrainedModel",
        ]
    )
    
    # 将"models.xlm"模块中的特定名称列表添加到_import_structure字典中
    _import_structure["models.xlm"].extend(
        [
            "TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLMForMultipleChoice",
            "TFXLMForQuestionAnsweringSimple",
            "TFXLMForSequenceClassification",
            "TFXLMForTokenClassification",
            "TFXLMMainLayer",
            "TFXLMModel",
            "TFXLMPreTrainedModel",
            "TFXLMWithLMHeadModel",
        ]
    )
    # 扩展 _import_structure 中 "models.xlm_roberta" 的模块列表
    _import_structure["models.xlm_roberta"].extend(
        [
            "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLMRobertaForCausalLM",
            "TFXLMRobertaForMaskedLM",
            "TFXLMRobertaForMultipleChoice",
            "TFXLMRobertaForQuestionAnswering",
            "TFXLMRobertaForSequenceClassification",
            "TFXLMRobertaForTokenClassification",
            "TFXLMRobertaModel",
            "TFXLMRobertaPreTrainedModel",
        ]
    )
    # 扩展 _import_structure 中 "models.xlnet" 的模块列表
    _import_structure["models.xlnet"].extend(
        [
            "TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXLNetForMultipleChoice",
            "TFXLNetForQuestionAnsweringSimple",
            "TFXLNetForSequenceClassification",
            "TFXLNetForTokenClassification",
            "TFXLNetLMHeadModel",
            "TFXLNetMainLayer",
            "TFXLNetModel",
            "TFXLNetPreTrainedModel",
        ]
    )
    # 将 "optimization_tf" 中的模块列表设置为指定的值
    _import_structure["optimization_tf"] = [
        "AdamWeightDecay",
        "GradientAccumulator",
        "WarmUp",
        "create_optimizer",
    ]
    # 将 "tf_utils" 中的模块列表设置为空列表
    _import_structure["tf_utils"] = []
# 检查是否所有的必需依赖库都可用：Librosa、Essentia、Scipy、Torch、Pretty MIDI
try:
    if not (
        is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
        and is_torch_available()
        and is_pretty_midi_available()
    ):
        # 如果有任何一个依赖库不可用，抛出OptionalDependencyNotAvailable异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到OptionalDependencyNotAvailable异常，则导入dummy模块，用于替代依赖库功能
    from .utils import (
        dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects,
    )
    # 将dummy模块中非下划线开头的对象名添加到_import_structure字典中的相应位置
    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    # 如果所有依赖库都可用，则将相关对象名添加到_import_structure字典中的"models.pop2piano"列表中
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")

# 检查是否Torchaudio库可用
try:
    if not is_torchaudio_available():
        # 如果Torchaudio库不可用，抛出OptionalDependencyNotAvailable异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到OptionalDependencyNotAvailable异常，则导入dummy模块，用于替代Torchaudio库功能
    from .utils import (
        dummy_torchaudio_objects,
    )
    # 将dummy模块中非下划线开头的对象名添加到_import_structure字典中的相应位置
    _import_structure["utils.dummy_torchaudio_objects"] = [
        name for name in dir(dummy_torchaudio_objects) if not name.startswith("_")
    ]
else:
    # 如果Torchaudio库可用，则将相关对象名添加到_import_structure字典中的"models.musicgen_melody"列表中
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyFeatureExtractor")
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyProcessor")

# 检查是否Flax库可用
try:
    if not is_flax_available():
        # 如果Flax库不可用，抛出OptionalDependencyNotAvailable异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果捕获到OptionalDependencyNotAvailable异常，则导入dummy_flax_objects模块，用于替代Flax库功能
    from .utils import dummy_flax_objects
    # 将dummy_flax_objects模块中非下划线开头的对象名添加到_import_structure字典中的相应位置
    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    # 如果Flax库可用，则扩展_import_structure中的多个列表，添加Flax相关对象名
    _import_structure["generation"].extend(
        [
            "FlaxForcedBOSTokenLogitsProcessor",
            "FlaxForcedEOSTokenLogitsProcessor",
            "FlaxForceTokensLogitsProcessor",
            "FlaxGenerationMixin",
            "FlaxLogitsProcessor",
            "FlaxLogitsProcessorList",
            "FlaxLogitsWarper",
            "FlaxMinLengthLogitsProcessor",
            "FlaxTemperatureLogitsWarper",
            "FlaxSuppressTokensAtBeginLogitsProcessor",
            "FlaxSuppressTokensLogitsProcessor",
            "FlaxTopKLogitsWarper",
            "FlaxTopPLogitsWarper",
            "FlaxWhisperTimeStampLogitsProcessor",
        ]
    )
    # 添加额外的导入结构，以及Flax相关的模块和类名到_import_structure字典中
    _import_structure["generation_flax_utils"] = []
    _import_structure["modeling_flax_outputs"] = []
    _import_structure["modeling_flax_utils"] = ["FlaxPreTrainedModel"]
    _import_structure["models.albert"].extend(
        [
            "FlaxAlbertForMaskedLM",
            "FlaxAlbertForMultipleChoice",
            "FlaxAlbertForPreTraining",
            "FlaxAlbertForQuestionAnswering",
            "FlaxAlbertForSequenceClassification",
            "FlaxAlbertForTokenClassification",
            "FlaxAlbertModel",
            "FlaxAlbertPreTrainedModel",
        ]
    )
    # 扩展 `models.auto` 模块的导入结构，添加了多个 FLAX 自动模型映射和模型类
    _import_structure["models.auto"].extend(
        [
            "FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",  # 自动模型音频分类映射
            "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",  # 自动模型因果语言建模映射
            "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",  # 自动模型图像分类映射
            "FLAX_MODEL_FOR_MASKED_LM_MAPPING",  # 自动模型掩蔽语言建模映射
            "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",  # 自动模型多选题映射
            "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",  # 自动模型下一句预测映射
            "FLAX_MODEL_FOR_PRETRAINING_MAPPING",  # 自动模型预训练映射
            "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",  # 自动模型问答映射
            "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",  # 自动模型序列到序列因果语言建模映射
            "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",  # 自动模型序列分类映射
            "FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",  # 自动模型语音序列到序列映射
            "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",  # 自动模型标记分类映射
            "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",  # 自动模型视觉到序列映射
            "FLAX_MODEL_MAPPING",  # 自动模型映射
            "FlaxAutoModel",  # FLAX 自动模型类
            "FlaxAutoModelForCausalLM",  # FLAX 自动因果语言建模模型类
            "FlaxAutoModelForImageClassification",  # FLAX 自动图像分类模型类
            "FlaxAutoModelForMaskedLM",  # FLAX 自动掩蔽语言建模模型类
            "FlaxAutoModelForMultipleChoice",  # FLAX 自动多选题模型类
            "FlaxAutoModelForNextSentencePrediction",  # FLAX 自动下一句预测模型类
            "FlaxAutoModelForPreTraining",  # FLAX 自动预训练模型类
            "FlaxAutoModelForQuestionAnswering",  # FLAX 自动问答模型类
            "FlaxAutoModelForSeq2SeqLM",  # FLAX 自动序列到序列语言建模模型类
            "FlaxAutoModelForSequenceClassification",  # FLAX 自动序列分类模型类
            "FlaxAutoModelForSpeechSeq2Seq",  # FLAX 自动语音序列到序列模型类
            "FlaxAutoModelForTokenClassification",  # FLAX 自动标记分类模型类
            "FlaxAutoModelForVision2Seq",  # FLAX 自动视觉到序列模型类
        ]
    )
    
    # Flax 模型结构，扩展 `models.bart` 模块的导入结构，包含了 BART 相关模型
    _import_structure["models.bart"].extend(
        [
            "FlaxBartDecoderPreTrainedModel",  # FLAX BART 解码器预训练模型类
            "FlaxBartForCausalLM",  # FLAX BART 因果语言建模模型类
            "FlaxBartForConditionalGeneration",  # FLAX BART 生成条件模型类
            "FlaxBartForQuestionAnswering",  # FLAX BART 问答模型类
            "FlaxBartForSequenceClassification",  # FLAX BART 序列分类模型类
            "FlaxBartModel",  # FLAX BART 模型类
            "FlaxBartPreTrainedModel",  # FLAX BART 预训练模型类
        ]
    )
    
    # 扩展 `models.beit` 模块的导入结构，添加了 BEiT 相关模型
    _import_structure["models.beit"].extend(
        [
            "FlaxBeitForImageClassification",  # FLAX BEiT 图像分类模型类
            "FlaxBeitForMaskedImageModeling",  # FLAX BEiT 掩蔽图像建模模型类
            "FlaxBeitModel",  # FLAX BEiT 模型类
            "FlaxBeitPreTrainedModel",  # FLAX BEiT 预训练模型类
        ]
    )
    
    # 扩展 `models.bert` 模块的导入结构，添加了 BERT 相关模型
    _import_structure["models.bert"].extend(
        [
            "FlaxBertForCausalLM",  # FLAX BERT 因果语言建模模型类
            "FlaxBertForMaskedLM",  # FLAX BERT 掩蔽语言建模模型类
            "FlaxBertForMultipleChoice",  # FLAX BERT 多选题模型类
            "FlaxBertForNextSentencePrediction",  # FLAX BERT 下一句预测模型类
            "FlaxBertForPreTraining",  # FLAX BERT 预训练模型类
            "FlaxBertForQuestionAnswering",  # FLAX BERT 问答模型类
            "FlaxBertForSequenceClassification",  # FLAX BERT 序列分类模型类
            "FlaxBertForTokenClassification",  # FLAX BERT 标记分类模型类
            "FlaxBertModel",  # FLAX BERT 模型类
            "FlaxBertPreTrainedModel",  # FLAX BERT 预训练模型类
        ]
    )
    
    # 扩展 `models.big_bird` 模块的导入结构，添加了 BigBird 相关模型
    _import_structure["models.big_bird"].extend(
        [
            "FlaxBigBirdForCausalLM",  # FLAX BigBird 因果语言建模模型类
            "FlaxBigBirdForMaskedLM",  # FLAX BigBird 掩蔽语言建模模型类
            "FlaxBigBirdForMultipleChoice",  # FLAX BigBird 多选题模型类
            "FlaxBigBirdForPreTraining",  # FLAX BigBird 预训练模型类
            "FlaxBigBirdForQuestionAnswering",  # FLAX BigBird 问答模型类
            "FlaxBigBirdForSequenceClassification",  # FLAX BigBird 序列分类模型类
            "FlaxBigBirdForTokenClassification",  # FLAX BigBird 标记分类模型类
            "FlaxBigBirdModel",  # FLAX BigBird 模型类
            "FlaxBigBirdPreTrainedModel",  # FLAX BigBird 预训练模型类
        ]
    )
    # 扩展 "models.blenderbot" 的导入结构，添加以下类名：
    # - FlaxBlenderbotForConditionalGeneration
    # - FlaxBlenderbotModel
    # - FlaxBlenderbotPreTrainedModel
    _import_structure["models.blenderbot"].extend(
        [
            "FlaxBlenderbotForConditionalGeneration",
            "FlaxBlenderbotModel",
            "FlaxBlenderbotPreTrainedModel",
        ]
    )
    
    # 扩展 "models.blenderbot_small" 的导入结构，添加以下类名：
    # - FlaxBlenderbotSmallForConditionalGeneration
    # - FlaxBlenderbotSmallModel
    # - FlaxBlenderbotSmallPreTrainedModel
    _import_structure["models.blenderbot_small"].extend(
        [
            "FlaxBlenderbotSmallForConditionalGeneration",
            "FlaxBlenderbotSmallModel",
            "FlaxBlenderbotSmallPreTrainedModel",
        ]
    )
    
    # 扩展 "models.bloom" 的导入结构，添加以下类名：
    # - FlaxBloomForCausalLM
    # - FlaxBloomModel
    # - FlaxBloomPreTrainedModel
    _import_structure["models.bloom"].extend(
        [
            "FlaxBloomForCausalLM",
            "FlaxBloomModel",
            "FlaxBloomPreTrainedModel",
        ]
    )
    
    # 扩展 "models.clip" 的导入结构，添加以下类名：
    # - FlaxCLIPModel
    # - FlaxCLIPPreTrainedModel
    # - FlaxCLIPTextModel
    # - FlaxCLIPTextPreTrainedModel
    # - FlaxCLIPTextModelWithProjection
    # - FlaxCLIPVisionModel
    # - FlaxCLIPVisionPreTrainedModel
    _import_structure["models.clip"].extend(
        [
            "FlaxCLIPModel",
            "FlaxCLIPPreTrainedModel",
            "FlaxCLIPTextModel",
            "FlaxCLIPTextPreTrainedModel",
            "FlaxCLIPTextModelWithProjection",
            "FlaxCLIPVisionModel",
            "FlaxCLIPVisionPreTrainedModel",
        ]
    )
    
    # 扩展 "models.distilbert" 的导入结构，添加以下类名：
    # - FlaxDistilBertForMaskedLM
    # - FlaxDistilBertForMultipleChoice
    # - FlaxDistilBertForQuestionAnswering
    # - FlaxDistilBertForSequenceClassification
    # - FlaxDistilBertForTokenClassification
    # - FlaxDistilBertModel
    # - FlaxDistilBertPreTrainedModel
    _import_structure["models.distilbert"].extend(
        [
            "FlaxDistilBertForMaskedLM",
            "FlaxDistilBertForMultipleChoice",
            "FlaxDistilBertForQuestionAnswering",
            "FlaxDistilBertForSequenceClassification",
            "FlaxDistilBertForTokenClassification",
            "FlaxDistilBertModel",
            "FlaxDistilBertPreTrainedModel",
        ]
    )
    
    # 扩展 "models.electra" 的导入结构，添加以下类名：
    # - FlaxElectraForCausalLM
    # - FlaxElectraForMaskedLM
    # - FlaxElectraForMultipleChoice
    # - FlaxElectraForPreTraining
    # - FlaxElectraForQuestionAnswering
    # - FlaxElectraForSequenceClassification
    # - FlaxElectraForTokenClassification
    # - FlaxElectraModel
    # - FlaxElectraPreTrainedModel
    _import_structure["models.electra"].extend(
        [
            "FlaxElectraForCausalLM",
            "FlaxElectraForMaskedLM",
            "FlaxElectraForMultipleChoice",
            "FlaxElectraForPreTraining",
            "FlaxElectraForQuestionAnswering",
            "FlaxElectraForSequenceClassification",
            "FlaxElectraForTokenClassification",
            "FlaxElectraModel",
            "FlaxElectraPreTrainedModel",
        ]
    )
    
    # 添加 "models.encoder_decoder" 的导入结构，包含以下类名：
    # - FlaxEncoderDecoderModel
    _import_structure["models.encoder_decoder"].append("FlaxEncoderDecoderModel")
    
    # 扩展 "models.gpt2" 的导入结构，添加以下类名：
    # - FlaxGPT2LMHeadModel
    # - FlaxGPT2Model
    # - FlaxGPT2PreTrainedModel
    _import_structure["models.gpt2"].extend(
        [
            "FlaxGPT2LMHeadModel",
            "FlaxGPT2Model",
            "FlaxGPT2PreTrainedModel",
        ]
    )
    
    # 扩展 "models.gpt_neo" 的导入结构，添加以下类名：
    # - FlaxGPTNeoForCausalLM
    # - FlaxGPTNeoModel
    # - FlaxGPTNeoPreTrainedModel
    _import_structure["models.gpt_neo"].extend(
        [
            "FlaxGPTNeoForCausalLM",
            "FlaxGPTNeoModel",
            "FlaxGPTNeoPreTrainedModel",
        ]
    )
    
    # 扩展 "models.gptj" 的导入结构，添加以下类名：
    # - FlaxGPTJForCausalLM
    # - FlaxGPTJModel
    # - FlaxGPTJPreTrainedModel
    _import_structure["models.gptj"].extend(
        [
            "FlaxGPTJForCausalLM",
            "FlaxGPTJModel",
            "FlaxGPTJPreTrainedModel",
        ]
    )
    
    # 扩展 "models.llama" 的导入结构，添加以下类名：
    # - FlaxLlamaForCausalLM
    # - FlaxLlamaModel
    # - FlaxLlamaPreTrainedModel
    _import_structure["models.llama"].extend(
        [
            "FlaxLlamaForCausalLM",
            "FlaxLlamaModel",
            "FlaxLlamaPreTrainedModel",
        ]
    )
    
    # 扩展 "models.gemma" 的导入结构，添加以下类名：
    # - FlaxGemmaForCausalLM
    # - FlaxGemmaModel
    # - FlaxGemmaPreTrainedModel
    _import_structure["models.gemma"].extend(
        [
            "FlaxGemmaForCausalLM",
            "FlaxGemmaModel",
            "FlaxGemmaPreTrainedModel",
        ]
    )
    
    # 扩展 "models.longt5" 的导入结构，添加以下类名：
    # - FlaxLongT5ForConditionalGeneration
    # - FlaxLongT5Model
    # - FlaxLongT5PreTrainedModel
    _import_structure["models.longt5"].extend(
        [
            "FlaxLongT5ForConditionalGeneration",
            "FlaxLongT5Model",
            "FlaxLongT5PreTrainedModel",
        ]
    )
    
    # 扩展 "models.marian" 的导入结构，添加以下类名：
    # - FlaxMarianModel
    # - FlaxMarianMTModel
    # - FlaxMarianPreTrainedModel
    _import_structure["models.marian"].extend(
        [
            "FlaxMarianModel",
            "FlaxMarianMTModel",
            "FlaxMarianPreTrainedModel",
        ]
    )
    # 将指定模块（models.mbart）下的类名列表扩展，包括以下类：
    # FlaxMBartForConditionalGeneration、FlaxMBartForQuestionAnswering、
    # FlaxMBartForSequenceClassification、FlaxMBartModel、FlaxMBartPreTrainedModel
    _import_structure["models.mbart"].extend(
        [
            "FlaxMBartForConditionalGeneration",
            "FlaxMBartForQuestionAnswering",
            "FlaxMBartForSequenceClassification",
            "FlaxMBartModel",
            "FlaxMBartPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.mistral）下的类名列表扩展，包括以下类：
    # FlaxMistralForCausalLM、FlaxMistralModel、FlaxMistralPreTrainedModel
    _import_structure["models.mistral"].extend(
        [
            "FlaxMistralForCausalLM",
            "FlaxMistralModel",
            "FlaxMistralPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.mt5）下的类名列表扩展，包括以下类：
    # FlaxMT5EncoderModel、FlaxMT5ForConditionalGeneration、FlaxMT5Model
    _import_structure["models.mt5"].extend(
        [
            "FlaxMT5EncoderModel",
            "FlaxMT5ForConditionalGeneration",
            "FlaxMT5Model"
        ]
    )
    
    # 将指定模块（models.opt）下的类名列表扩展，包括以下类：
    # FlaxOPTForCausalLM、FlaxOPTModel、FlaxOPTPreTrainedModel
    _import_structure["models.opt"].extend(
        [
            "FlaxOPTForCausalLM",
            "FlaxOPTModel",
            "FlaxOPTPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.pegasus）下的类名列表扩展，包括以下类：
    # FlaxPegasusForConditionalGeneration、FlaxPegasusModel、FlaxPegasusPreTrainedModel
    _import_structure["models.pegasus"].extend(
        [
            "FlaxPegasusForConditionalGeneration",
            "FlaxPegasusModel",
            "FlaxPegasusPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.regnet）下的类名列表扩展，包括以下类：
    # FlaxRegNetForImageClassification、FlaxRegNetModel、FlaxRegNetPreTrainedModel
    _import_structure["models.regnet"].extend(
        [
            "FlaxRegNetForImageClassification",
            "FlaxRegNetModel",
            "FlaxRegNetPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.resnet）下的类名列表扩展，包括以下类：
    # FlaxResNetForImageClassification、FlaxResNetModel、FlaxResNetPreTrainedModel
    _import_structure["models.resnet"].extend(
        [
            "FlaxResNetForImageClassification",
            "FlaxResNetModel",
            "FlaxResNetPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.roberta）下的类名列表扩展，包括以下类：
    # FlaxRobertaForCausalLM、FlaxRobertaForMaskedLM、FlaxRobertaForMultipleChoice、
    # FlaxRobertaForQuestionAnswering、FlaxRobertaForSequenceClassification、
    # FlaxRobertaForTokenClassification、FlaxRobertaModel、FlaxRobertaPreTrainedModel
    _import_structure["models.roberta"].extend(
        [
            "FlaxRobertaForCausalLM",
            "FlaxRobertaForMaskedLM",
            "FlaxRobertaForMultipleChoice",
            "FlaxRobertaForQuestionAnswering",
            "FlaxRobertaForSequenceClassification",
            "FlaxRobertaForTokenClassification",
            "FlaxRobertaModel",
            "FlaxRobertaPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.roberta_prelayernorm）下的类名列表扩展，包括以下类：
    # FlaxRobertaPreLayerNormForCausalLM、FlaxRobertaPreLayerNormForMaskedLM、
    # FlaxRobertaPreLayerNormForMultipleChoice、FlaxRobertaPreLayerNormForQuestionAnswering、
    # FlaxRobertaPreLayerNormForSequenceClassification、FlaxRobertaPreLayerNormForTokenClassification、
    # FlaxRobertaPreLayerNormModel、FlaxRobertaPreLayerNormPreTrainedModel
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "FlaxRobertaPreLayerNormForCausalLM",
            "FlaxRobertaPreLayerNormForMaskedLM",
            "FlaxRobertaPreLayerNormForMultipleChoice",
            "FlaxRobertaPreLayerNormForQuestionAnswering",
            "FlaxRobertaPreLayerNormForSequenceClassification",
            "FlaxRobertaPreLayerNormForTokenClassification",
            "FlaxRobertaPreLayerNormModel",
            "FlaxRobertaPreLayerNormPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.roformer）下的类名列表扩展，包括以下类：
    # FlaxRoFormerForMaskedLM、FlaxRoFormerForMultipleChoice、FlaxRoFormerForQuestionAnswering、
    # FlaxRoFormerForSequenceClassification、FlaxRoFormerForTokenClassification、
    # FlaxRoFormerModel、FlaxRoFormerPreTrainedModel
    _import_structure["models.roformer"].extend(
        [
            "FlaxRoFormerForMaskedLM",
            "FlaxRoFormerForMultipleChoice",
            "FlaxRoFormerForQuestionAnswering",
            "FlaxRoFormerForSequenceClassification",
            "FlaxRoFormerForTokenClassification",
            "FlaxRoFormerModel",
            "FlaxRoFormerPreTrainedModel",
        ]
    )
    
    # 将指定模块（models.speech_encoder_decoder）下的类名添加到列表中：
    # FlaxSpeechEncoderDecoderModel
    _import_structure["models.speech_encoder_decoder"].append("FlaxSpeechEncoderDecoderModel")
    
    # 将指定模块（models.t5）下的类名列表扩展，包括以下类：
    # FlaxT5EncoderModel、FlaxT5ForConditionalGeneration、FlaxT5Model、FlaxT5PreTrainedModel
    _import_structure["models.t5"].extend(
        [
            "FlaxT5EncoderModel",
            "FlaxT5ForConditionalGeneration",
            "FlaxT5Model",
            "FlaxT5PreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].append("FlaxVisionEncoderDecoderModel")
    # 将字符串 "FlaxVisionEncoderDecoderModel" 添加到 _import_structure 字典中 "models.vision_encoder_decoder" 键对应的列表末尾
    
    _import_structure["models.vision_text_dual_encoder"].extend(["FlaxVisionTextDualEncoderModel"])
    # 将列表 ["FlaxVisionTextDualEncoderModel"] 扩展（即添加）到 _import_structure 字典中 "models.vision_text_dual_encoder" 键对应的列表末尾
    
    _import_structure["models.vit"].extend(["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"])
    # 将包含三个字符串的列表扩展到 _import_structure 字典中 "models.vit" 键对应的列表末尾
    
    _import_structure["models.wav2vec2"].extend(
        [
            "FlaxWav2Vec2ForCTC",
            "FlaxWav2Vec2ForPreTraining",
            "FlaxWav2Vec2Model",
            "FlaxWav2Vec2PreTrainedModel",
        ]
    )
    # 将包含四个字符串的列表扩展到 _import_structure 字典中 "models.wav2vec2" 键对应的列表末尾
    
    _import_structure["models.whisper"].extend(
        [
            "FlaxWhisperForConditionalGeneration",
            "FlaxWhisperModel",
            "FlaxWhisperPreTrainedModel",
            "FlaxWhisperForAudioClassification",
        ]
    )
    # 将包含四个字符串的列表扩展到 _import_structure 字典中 "models.whisper" 键对应的列表末尾
    
    _import_structure["models.xglm"].extend(
        [
            "FlaxXGLMForCausalLM",
            "FlaxXGLMModel",
            "FlaxXGLMPreTrainedModel",
        ]
    )
    # 将包含三个字符串的列表扩展到 _import_structure 字典中 "models.xglm" 键对应的列表末尾
    
    _import_structure["models.xlm_roberta"].extend(
        [
            "FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlaxXLMRobertaForMaskedLM",
            "FlaxXLMRobertaForMultipleChoice",
            "FlaxXLMRobertaForQuestionAnswering",
            "FlaxXLMRobertaForSequenceClassification",
            "FlaxXLMRobertaForTokenClassification",
            "FlaxXLMRobertaModel",
            "FlaxXLMRobertaForCausalLM",
            "FlaxXLMRobertaPreTrainedModel",
        ]
    )
    # 将包含九个字符串的列表扩展到 _import_structure 字典中 "models.xlm_roberta" 键对应的列表末尾
# 直接导入用于类型检查的模块
if TYPE_CHECKING:
    # 配置相关
    from .configuration_utils import PretrainedConfig

    # 数据相关
    from .data import (
        DataProcessor,                     # 数据处理器
        InputExample,                      # 输入示例
        InputFeatures,                     # 输入特征
        SingleSentenceClassificationProcessor,  # 单句分类处理器
        SquadExample,                      # SQuAD 示例
        SquadFeatures,                     # SQuAD 特征
        SquadV1Processor,                  # SQuAD v1 处理器
        SquadV2Processor,                  # SQuAD v2 处理器
        glue_compute_metrics,              # GLUE 计算评估指标函数
        glue_convert_examples_to_features, # 将示例转换为 GLUE 特征的函数
        glue_output_modes,                 # GLUE 输出模式
        glue_processors,                   # GLUE 处理器列表
        glue_tasks_num_labels,             # GLUE 任务的标签数量
        squad_convert_examples_to_features,# 将示例转换为 SQuAD 特征的函数
        xnli_compute_metrics,              # XNLI 计算评估指标函数
        xnli_output_modes,                 # XNLI 输出模式
        xnli_processors,                   # XNLI 处理器列表
        xnli_tasks_num_labels,             # XNLI 任务的标签数量
    )
    from .data.data_collator import (
        DataCollator,                      # 数据收集器
        DataCollatorForLanguageModeling,   # 用于语言建模的数据收集器
        DataCollatorForPermutationLanguageModeling,  # 用于排列语言建模的数据收集器
        DataCollatorForSeq2Seq,            # 用于序列到序列的数据收集器
        DataCollatorForSOP,                # 用于SOP的数据收集器
        DataCollatorForTokenClassification,  # 用于标记分类的数据收集器
        DataCollatorForWholeWordMask,      # 用于整词掩码的数据收集器
        DataCollatorWithPadding,           # 带填充的数据收集器
        DefaultDataCollator,               # 默认数据收集器
        default_data_collator,             # 默认数据收集器的别名
    )
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor  # 序列特征提取器

    # 特征提取相关
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin  # 批次特征，特征提取混合类

    # 生成相关
    from .generation import (
        GenerationConfig,                  # 生成配置
        TextIteratorStreamer,              # 文本迭代器流
        TextStreamer,                      # 文本流
    )
    from .hf_argparser import HfArgumentParser  # Hugging Face 参数解析器

    # 集成相关
    from .integrations import (
        is_clearml_available,              # 是否支持 ClearML
        is_comet_available,                # 是否支持 Comet
        is_dvclive_available,              # 是否支持 DVCLive
        is_neptune_available,              # 是否支持 Neptune
        is_optuna_available,               # 是否支持 Optuna
        is_ray_available,                  # 是否支持 Ray
        is_ray_tune_available,             # 是否支持 Ray Tune
        is_sigopt_available,               # 是否支持 SigOpt
        is_tensorboard_available,          # 是否支持 TensorBoard
        is_wandb_available,                # 是否支持 Weights & Biases
    )

    # 模型卡相关
    from .modelcard import ModelCard       # 模型卡片

    # TF 2.0 <=> PyTorch 转换工具
    from .modeling_tf_pytorch_utils import (
        convert_tf_weight_name_to_pt_weight_name,  # 转换 TF 权重名到 PyTorch 权重名
        load_pytorch_checkpoint_in_tf2_model,     # 在 TF2 模型中加载 PyTorch 检查点
        load_pytorch_model_in_tf2_model,          # 在 TF2 模型中加载 PyTorch 模型
        load_pytorch_weights_in_tf2_model,        # 在 TF2 模型中加载 PyTorch 权重
        load_tf2_checkpoint_in_pytorch_model,     # 在 PyTorch 模型中加载 TF2 检查点
        load_tf2_model_in_pytorch_model,          # 在 PyTorch 模型中加载 TF2 模型
        load_tf2_weights_in_pytorch_model,        # 在 PyTorch 模型中加载 TF2 权重
    )

    # 模型相关
    from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig  # ALBERT 预训练配置映射，ALBERT 配置
    from .models.align import (
        ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP,      # ALIGN 预训练配置映射
        AlignConfig,                             # ALIGN 配置
        AlignProcessor,                          # ALIGN 处理器
        AlignTextConfig,                         # ALIGN 文本配置
        AlignVisionConfig,                       # ALIGN 视觉配置
    )
    from .models.altclip import (
        ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,    # ALTCLIP 预训练配置映射
        AltCLIPConfig,                           # ALTCLIP 配置
        AltCLIPProcessor,                        # ALTCLIP 处理器
        AltCLIPTextConfig,                       # ALTCLIP 文本配置
        AltCLIPVisionConfig,                     # ALTCLIP 视觉配置
    )
    from .models.audio_spectrogram_transformer import (
        AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 音频频谱变换器预训练配置映射
        ASTConfig,                               # 音频频谱变换器配置
        ASTFeatureExtractor,                     # 音频频谱变换器特征提取器
    )
    # 导入自动模型相关的所有预训练配置映射
    from .models.auto import (
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        # 自动模型的配置映射
        CONFIG_MAPPING,
        # 自动模型的特征提取器映射
        FEATURE_EXTRACTOR_MAPPING,
        # 自动模型的图像处理器映射
        IMAGE_PROCESSOR_MAPPING,
        # 自动模型的模型名称映射
        MODEL_NAMES_MAPPING,
        # 自动模型的处理器映射
        PROCESSOR_MAPPING,
        # 自动模型的分词器映射
        TOKENIZER_MAPPING,
        # 自动模型的配置类
        AutoConfig,
        # 自动模型的特征提取器类
        AutoFeatureExtractor,
        # 自动模型的图像处理器类
        AutoImageProcessor,
        # 自动模型的处理器类
        AutoProcessor,
        # 自动模型的分词器类
        AutoTokenizer,
    )
    # 导入自动形态模型相关的预训练配置映射
    from .models.autoformer import (
        AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        # 自动形态模型的配置类
        AutoformerConfig,
    )
    # 导入Bark模型相关的配置和处理器类
    from .models.bark import (
        BarkCoarseConfig,
        BarkConfig,
        BarkFineConfig,
        # Bark模型的处理器类
        BarkProcessor,
        BarkSemanticConfig,
    )
    # 导入BART模型相关的配置类和分词器类
    from .models.bart import BartConfig, BartTokenizer
    # 导入BEIT模型相关的预训练配置映射和配置类
    from .models.beit import BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BeitConfig
    # 导入BERT模型相关的预训练配置映射和分词器类
    from .models.bert import (
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        # BERT模型的基本分词器类
        BasicTokenizer,
        BertConfig,
        BertTokenizer,
        WordpieceTokenizer,
    )
    # 导入BERT生成模型的配置类
    from .models.bert_generation import BertGenerationConfig
    # 导入BERT日语模型相关的分词器类
    from .models.bert_japanese import (
        BertJapaneseTokenizer,
        # 日语分词器的字符级分词器类
        CharacterTokenizer,
        # 日语分词器的MeCab分词器类
        MecabTokenizer,
    )
    # 导入Bertweet模型的分词器类
    from .models.bertweet import BertweetTokenizer
    # 导入Big Bird模型相关的预训练配置映射和配置类
    from .models.big_bird import BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdConfig
    # 导入BigBird-Pegasus模型相关的预训练配置映射和配置类
    from .models.bigbird_pegasus import (
        BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BigBirdPegasusConfig,
    )
    # 导入BioGPT模型相关的预训练配置映射和配置类
    from .models.biogpt import (
        BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BioGptConfig,
        BioGptTokenizer,
    )
    # 导入Bit模型相关的预训练配置映射和配置类
    from .models.bit import BIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BitConfig
    # 导入Blenderbot模型相关的预训练配置映射和配置类、分词器类
    from .models.blenderbot import (
        BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlenderbotConfig,
        BlenderbotTokenizer,
    )
    # 导入Blenderbot-Small模型相关的预训练配置映射和配置类、分词器类
    from .models.blenderbot_small import (
        BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlenderbotSmallConfig,
        BlenderbotSmallTokenizer,
    )
    # 导入Blip模型相关的预训练配置映射和配置类、处理器类
    from .models.blip import (
        BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BlipConfig,
        # Blip模型的处理器类
        BlipProcessor,
        BlipTextConfig,
        BlipVisionConfig,
    )
    # 导入Blip-2模型相关的预训练配置映射和配置类、处理器类
    from .models.blip_2 import (
        BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Blip2Config,
        # Blip-2模型的处理器类
        Blip2Processor,
        Blip2QFormerConfig,
        Blip2VisionConfig,
    )
    # 导入Bloom模型相关的预训练配置映射和配置类
    from .models.bloom import BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP, BloomConfig
    # 导入BridgeTower模型相关的预训练配置映射和配置类、处理器类
    from .models.bridgetower import (
        BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BridgeTowerConfig,
        # BridgeTower模型的处理器类
        BridgeTowerProcessor,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    # 导入Bros模型相关的预训练配置映射和配置类、处理器类
    from .models.bros import (
        BROS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BrosConfig,
        BrosProcessor,
    )
    # 导入ByT5模型的分词器类
    from .models.byt5 import ByT5Tokenizer
    # 导入Camembert模型相关的预训练配置映射和配置类
    from .models.camembert import (
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CamembertConfig,
    )
    # 导入Canine模型相关的配置、配置类和分词器
    from .models.canine import (
        CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CanineConfig,
        CanineTokenizer,
    )
    
    # 导入Chinese CLIP模型相关的配置、配置类、处理器以及文本和视觉配置
    from .models.chinese_clip import (
        CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ChineseCLIPConfig,
        ChineseCLIPProcessor,
        ChineseCLIPTextConfig,
        ChineseCLIPVisionConfig,
    )
    
    # 导入CLAP模型相关的预训练模型列表、音频配置、配置类和处理器
    from .models.clap import (
        CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,
        ClapAudioConfig,
        ClapConfig,
        ClapProcessor,
        ClapTextConfig,
    )
    
    # 导入CLIP模型相关的配置、配置类、处理器、文本配置、分词器和视觉配置
    from .models.clip import (
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPConfig,
        CLIPProcessor,
        CLIPTextConfig,
        CLIPTokenizer,
        CLIPVisionConfig,
    )
    
    # 导入CLIPSeg模型相关的配置、配置类、处理器、文本配置和视觉配置
    from .models.clipseg import (
        CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CLIPSegConfig,
        CLIPSegProcessor,
        CLIPSegTextConfig,
        CLIPSegVisionConfig,
    )
    
    # 导入CLVP模型相关的配置、配置类、解码器配置、编码器配置、特征提取器、处理器和分词器
    from .models.clvp import (
        CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ClvpConfig,
        ClvpDecoderConfig,
        ClvpEncoderConfig,
        ClvpFeatureExtractor,
        ClvpProcessor,
        ClvpTokenizer,
    )
    
    # 导入CodeGen模型相关的配置、配置类和分词器
    from .models.codegen import (
        CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CodeGenConfig,
        CodeGenTokenizer,
    )
    
    # 导入Cohere模型相关的配置和配置类
    from .models.cohere import COHERE_PRETRAINED_CONFIG_ARCHIVE_MAP, CohereConfig
    
    # 导入Conditional DETR模型相关的配置和配置类
    from .models.conditional_detr import (
        CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConditionalDetrConfig,
    )
    
    # 导入ConvBERT模型相关的配置、配置类和分词器
    from .models.convbert import (
        CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConvBertConfig,
        ConvBertTokenizer,
    )
    
    # 导入ConvNext模型相关的配置
    from .models.convnext import CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextConfig
    
    # 导入ConvNextV2模型相关的配置
    from .models.convnextv2 import (
        CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConvNextV2Config,
    )
    
    # 导入CPMANT模型相关的配置、配置类和分词器
    from .models.cpmant import (
        CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CpmAntConfig,
        CpmAntTokenizer,
    )
    
    # 导入CTRL模型相关的配置、配置类和分词器
    from .models.ctrl import (
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CTRLConfig,
        CTRLTokenizer,
    )
    
    # 导入CVT模型相关的配置
    from .models.cvt import CVT_PRETRAINED_CONFIG_ARCHIVE_MAP, CvtConfig
    
    # 导入Data2Vec模型相关的文本和视觉预训练配置、音频配置、文本配置和视觉配置
    from .models.data2vec import (
        DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Data2VecAudioConfig,
        Data2VecTextConfig,
        Data2VecVisionConfig,
    )
    
    # 导入DeBERTa模型相关的配置、配置类和分词器
    from .models.deberta import (
        DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DebertaConfig,
        DebertaTokenizer,
    )
    
    # 导入DeBERTa V2模型相关的配置和配置类
    from .models.deberta_v2 import (
        DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DebertaV2Config,
    )
    
    # 导入Decision Transformer模型相关的配置和配置类
    from .models.decision_transformer import (
        DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DecisionTransformerConfig,
    )
    
    # 导入Deformable DETR模型相关的配置和配置类
    from .models.deformable_detr import (
        DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DeformableDetrConfig,
    )
    # 导入 DEIT 模型相关的预训练配置映射和配置类
    from .models.deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig
    # 导入 MCTCT 模型相关的预训练配置映射、配置类、特征提取器和处理器
    from .models.deprecated.mctct import (
        MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MCTCTConfig,
        MCTCTFeatureExtractor,
        MCTCTProcessor,
    )
    # 导入 MMBT 模型的配置类
    from .models.deprecated.mmbt import MMBTConfig
    # 导入 OpenLlama 模型相关的预训练配置映射和配置类
    from .models.deprecated.open_llama import (
        OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OpenLlamaConfig,
    )
    # 导入 RetriBert 模型相关的预训练配置映射、配置类和分词器
    from .models.deprecated.retribert import (
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RetriBertConfig,
        RetriBertTokenizer,
    )
    # 导入 Tapex 模型的分词器
    from .models.deprecated.tapex import TapexTokenizer
    # 导入 TrajectoryTransformer 模型相关的预训练配置映射和配置类
    from .models.deprecated.trajectory_transformer import (
        TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TrajectoryTransformerConfig,
    )
    # 导入 TransfoXL 模型相关的预训练配置映射、配置类、语料库和分词器
    from .models.deprecated.transfo_xl import (
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TransfoXLConfig,
        TransfoXLCorpus,
        TransfoXLTokenizer,
    )
    # 导入 VAN 模型相关的预训练配置映射和配置类
    from .models.deprecated.van import VAN_PRETRAINED_CONFIG_ARCHIVE_MAP, VanConfig
    # 导入 DepthAnything 模型相关的预训练配置映射和配置类
    from .models.depth_anything import DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP, DepthAnythingConfig
    # 导入 Deta 模型相关的预训练配置映射和配置类
    from .models.deta import DETA_PRETRAINED_CONFIG_ARCHIVE_MAP, DetaConfig
    # 导入 Detr 模型相关的预训练配置映射和配置类
    from .models.detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig
    # 导入 Dinat 模型相关的预训练配置映射和配置类
    from .models.dinat import DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP, DinatConfig
    # 导入 Dinov2 模型相关的预训练配置映射和配置类
    from .models.dinov2 import DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Dinov2Config
    # 导入 DistilBert 模型相关的预训练配置映射、配置类和分词器
    from .models.distilbert import (
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DistilBertConfig,
        DistilBertTokenizer,
    )
    # 导入 DonutSwin 模型相关的预训练配置映射、处理器和配置类
    from .models.donut import (
        DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DonutProcessor,
        DonutSwinConfig,
    )
    # 导入 DPR 模型相关的预训练配置映射、配置类和分词器
    from .models.dpr import (
        DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DPRConfig,
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )
    # 导入 DPT 模型相关的预训练配置映射和配置类
    from .models.dpt import DPT_PRETRAINED_CONFIG_ARCHIVE_MAP, DPTConfig
    # 导入 EfficientFormer 模型相关的预训练配置映射和配置类
    from .models.efficientformer import (
        EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EfficientFormerConfig,
    )
    # 导入 EfficientNet 模型相关的预训练配置映射和配置类
    from .models.efficientnet import (
        EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EfficientNetConfig,
    )
    # 导入 Electra 模型相关的预训练配置映射、配置类和分词器
    from .models.electra import (
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ElectraConfig,
        ElectraTokenizer,
    )
    # 导入 Encodec 模型相关的预训练配置映射、配置类和特征提取器
    from .models.encodec import (
        ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EncodecConfig,
        EncodecFeatureExtractor,
    )
    # 导入 EncoderDecoder 模型的配置类
    from .models.encoder_decoder import EncoderDecoderConfig
    # 导入 ERNIE 模型相关的预训练配置映射和配置类
    from .models.ernie import ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieConfig
    # 导入 ERNIE-M 模型相关的预训练配置映射和配置类
    from .models.ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig
    # 导入 ESM 模型相关的预训练配置映射、配置类和分词器
    from .models.esm import ESM_PRETRAINED_CONFIG_ARCHIVE_MAP, EsmConfig, EsmTokenizer
    # 导入 Falcon 模型相关的预训练配置映射和配置类
    from .models.falcon import FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP, FalconConfig
    # 导入 FASTSPEECH2_CONFORMER 相关模块和配置
    from .models.fastspeech2_conformer import (
        FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerTokenizer,
    )
    
    # 导入 FLAUBERT 相关模块和配置
    from .models.flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig, FlaubertTokenizer
    
    # 导入 FLAVA 相关模块和配置
    from .models.flava import (
        FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FlavaConfig,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaMultimodalConfig,
        FlavaTextConfig,
    )
    
    # 导入 FNET 相关模块和配置
    from .models.fnet import FNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FNetConfig
    
    # 导入 FOCALNET 相关模块和配置
    from .models.focalnet import FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FocalNetConfig
    
    # 导入 FSMT 相关模块和配置
    from .models.fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig, FSMTTokenizer
    
    # 导入 FUNNEL 相关模块和配置
    from .models.funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig, FunnelTokenizer
    
    # 导入 FUYU 相关模块和配置
    from .models.fuyu import FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP, FuyuConfig
    
    # 导入 GEMMA 相关模块和配置
    from .models.gemma import GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP, GemmaConfig
    
    # 导入 GIT 相关模块和配置
    from .models.git import (
        GIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GitConfig,
        GitProcessor,
        GitVisionConfig,
    )
    
    # 导入 GLPN 相关模块和配置
    from .models.glpn import GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP, GLPNConfig
    
    # 导入 GPT2 相关模块和配置
    from .models.gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2Tokenizer
    
    # 导入 GPT_BIGCODE 相关模块和配置
    from .models.gpt_bigcode import GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTBigCodeConfig
    
    # 导入 GPT_NEO 相关模块和配置
    from .models.gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig
    
    # 导入 GPT_NEOX 相关模块和配置
    from .models.gpt_neox import GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXConfig
    
    # 导入 GPT_NEOX_JAPANESE 相关模块和配置
    from .models.gpt_neox_japanese import (
        GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPTNeoXJapaneseConfig,
    )
    
    # 导入 GPTJ 相关模块和配置
    from .models.gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig
    
    # 导入 GPTSAN_JAPANESE 相关模块和配置
    from .models.gptsan_japanese import (
        GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPTSanJapaneseConfig,
        GPTSanJapaneseTokenizer,
    )
    
    # 导入 GRAPHORMER 相关模块和配置
    from .models.graphormer import GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, GraphormerConfig
    
    # 导入 GROUPVIT 相关模块和配置
    from .models.groupvit import (
        GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GroupViTConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )
    
    # 导入 HerbertTokenizer 类
    from .models.herbert import HerbertTokenizer
    
    # 导入 HUBERT 相关模块和配置
    from .models.hubert import HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, HubertConfig
    
    # 导入 IBERT 相关模块和配置
    from .models.ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig
    
    # 导入 IDEFICS 相关模块和配置
    from .models.idefics import IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP, IdeficsConfig
    # 导入所需模块和类
    
    from .models.imagegpt import IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, ImageGPTConfig
    from .models.informer import INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, InformerConfig
    from .models.instructblip import (
        INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        InstructBlipConfig,
        InstructBlipProcessor,
        InstructBlipQFormerConfig,
        InstructBlipVisionConfig,
    )
    from .models.jukebox import (
        JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP,
        JukeboxConfig,
        JukeboxPriorConfig,
        JukeboxTokenizer,
        JukeboxVQVAEConfig,
    )
    from .models.kosmos2 import (
        KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Kosmos2Config,
        Kosmos2Processor,
    )
    from .models.layoutlm import (
        LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMConfig,
        LayoutLMTokenizer,
    )
    from .models.layoutlmv2 import (
        LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMv2Config,
        LayoutLMv2FeatureExtractor,
        LayoutLMv2ImageProcessor,
        LayoutLMv2Processor,
        LayoutLMv2Tokenizer,
    )
    from .models.layoutlmv3 import (
        LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMv3Config,
        LayoutLMv3FeatureExtractor,
        LayoutLMv3ImageProcessor,
        LayoutLMv3Processor,
        LayoutLMv3Tokenizer,
    )
    from .models.layoutxlm import LayoutXLMProcessor
    from .models.led import LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LEDConfig, LEDTokenizer
    from .models.levit import LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, LevitConfig
    from .models.lilt import LILT_PRETRAINED_CONFIG_ARCHIVE_MAP, LiltConfig
    from .models.llama import LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, LlamaConfig
    from .models.llava import (
        LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LlavaConfig,
        LlavaProcessor,
    )
    from .models.llava_next import (
        LLAVA_NEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LlavaNextConfig,
        LlavaNextProcessor,
    )
    from .models.longformer import (
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LongformerConfig,
        LongformerTokenizer,
    )
    from .models.longt5 import LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, LongT5Config
    from .models.luke import LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP, LukeConfig, LukeTokenizer
    from .models.lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig, LxmertTokenizer
    from .models.m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config
    from .models.mamba import MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP, MambaConfig
    from .models.marian import MarianConfig
    from .models.markuplm import (
        MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MarkupLMConfig,
        MarkupLMFeatureExtractor,
        MarkupLMProcessor,
        MarkupLMTokenizer,
    )
    # 导入 mask2former 模块中的预训练配置映射和 Mask2FormerConfig 类
    from .models.mask2former import (
        MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Mask2FormerConfig,
    )
    
    # 导入 maskformer 模块中的预训练配置映射、MaskFormerConfig 类以及 MaskFormerSwinConfig 类
    from .models.maskformer import (
        MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MaskFormerConfig,
        MaskFormerSwinConfig,
    )
    
    # 导入 mbart 模块中的 MBartConfig 类
    from .models.mbart import MBartConfig
    
    # 导入 mega 模块中的预训练配置映射和 MegaConfig 类
    from .models.mega import MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP, MegaConfig
    
    # 导入 megatron_bert 模块中的预训练配置映射和 MegatronBertConfig 类
    from .models.megatron_bert import (
        MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MegatronBertConfig,
    )
    
    # 导入 mgp_str 模块中的预训练配置映射、MgpstrConfig 类、MgpstrProcessor 类和 MgpstrTokenizer 类
    from .models.mgp_str import (
        MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MgpstrConfig,
        MgpstrProcessor,
        MgpstrTokenizer,
    )
    
    # 导入 mistral 模块中的预训练配置映射和 MistralConfig 类
    from .models.mistral import MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MistralConfig
    
    # 导入 mixtral 模块中的预训练配置映射和 MixtralConfig 类
    from .models.mixtral import MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MixtralConfig
    
    # 导入 mobilebert 模块中的预训练配置映射、MobileBertConfig 类和 MobileBertTokenizer 类
    from .models.mobilebert import (
        MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileBertConfig,
        MobileBertTokenizer,
    )
    
    # 导入 mobilenet_v1 模块中的预训练配置映射和 MobileNetV1Config 类
    from .models.mobilenet_v1 import (
        MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileNetV1Config,
    )
    
    # 导入 mobilenet_v2 模块中的预训练配置映射和 MobileNetV2Config 类
    from .models.mobilenet_v2 import (
        MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileNetV2Config,
    )
    
    # 导入 mobilevit 模块中的预训练配置映射和 MobileViTConfig 类
    from .models.mobilevit import (
        MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileViTConfig,
    )
    
    # 导入 mobilevitv2 模块中的预训练配置映射和 MobileViTV2Config 类
    from .models.mobilevitv2 import (
        MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileViTV2Config,
    )
    
    # 导入 mpnet 模块中的预训练配置映射、MPNetConfig 类和 MPNetTokenizer 类
    from .models.mpnet import (
        MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MPNetConfig,
        MPNetTokenizer,
    )
    
    # 导入 mpt 模块中的预训练配置映射和 MptConfig 类
    from .models.mpt import MPT_PRETRAINED_CONFIG_ARCHIVE_MAP, MptConfig
    
    # 导入 mra 模块中的预训练配置映射和 MraConfig 类
    from .models.mra import MRA_PRETRAINED_CONFIG_ARCHIVE_MAP, MraConfig
    
    # 导入 mt5 模块中的 MT5Config 类
    from .models.mt5 import MT5Config
    
    # 导入 musicgen 模块中的预训练配置映射、MusicgenConfig 类和 MusicgenDecoderConfig 类
    from .models.musicgen import (
        MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MusicgenConfig,
        MusicgenDecoderConfig,
    )
    
    # 导入 musicgen_melody 模块中的预训练模型存档列表、MusicgenMelodyConfig 类和 MusicgenMelodyDecoderConfig 类
    from .models.musicgen_melody import (
        MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST,
        MusicgenMelodyConfig,
        MusicgenMelodyDecoderConfig,
    )
    
    # 导入 mvp 模块中的 MvpConfig 类和 MvpTokenizer 类
    from .models.mvp import MvpConfig, MvpTokenizer
    
    # 导入 nat 模块中的预训练配置映射和 NatConfig 类
    from .models.nat import NAT_PRETRAINED_CONFIG_ARCHIVE_MAP, NatConfig
    
    # 导入 nezha 模块中的预训练配置映射和 NezhaConfig 类
    from .models.nezha import NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP, NezhaConfig
    
    # 导入 nllb_moe 模块中的预训练配置映射和 NllbMoeConfig 类
    from .models.nllb_moe import NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP, NllbMoeConfig
    
    # 导入 nougat 模块中的 NougatProcessor 类
    from .models.nougat import NougatProcessor
    
    # 导入 nystromformer 模块中的预训练配置映射和 NystromformerConfig 类
    from .models.nystromformer import (
        NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        NystromformerConfig,
    )
    
    # 导入 oneformer 模块中的预训练配置映射、OneFormerConfig 类和 OneFormerProcessor 类
    from .models.oneformer import (
        ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OneFormerConfig,
        OneFormerProcessor,
    )
    
    # 导入 openai 模块中的预训练配置映射、OpenAIGPTConfig 类和 OpenAIGPTTokenizer 类
    from .models.openai import (
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OpenAIGPTConfig,
        OpenAIGPTTokenizer,
    )
    
    # 导入 opt 模块中的 OPTConfig 类
    from .models.opt import OPTConfig
    # 导入OWLv2模型相关的配置、处理器等
    from .models.owlv2 import (
        OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Owlv2Config,
        Owlv2Processor,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )
    
    # 导入OWLViT模型相关的配置、处理器等
    from .models.owlvit import (
        OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OwlViTConfig,
        OwlViTProcessor,
        OwlViTTextConfig,
        OwlViTVisionConfig,
    )
    
    # 导入PatchTSMixer模型相关的配置
    from .models.patchtsmixer import (
        PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PatchTSMixerConfig,
    )
    
    # 导入PatchTST模型相关的配置
    from .models.patchtst import PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP, PatchTSTConfig
    
    # 导入Pegasus模型相关的配置、配置解析器等
    from .models.pegasus import (
        PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PegasusConfig,
        PegasusTokenizer,
    )
    
    # 导入PegasusX模型相关的配置
    from .models.pegasus_x import (
        PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PegasusXConfig,
    )
    
    # 导入Perceiver模型相关的配置、配置解析器等
    from .models.perceiver import (
        PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PerceiverConfig,
        PerceiverTokenizer,
    )
    
    # 导入Persimmon模型相关的配置
    from .models.persimmon import (
        PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PersimmonConfig,
    )
    
    # 导入PHI模型相关的配置
    from .models.phi import PHI_PRETRAINED_CONFIG_ARCHIVE_MAP, PhiConfig
    
    # 导入Phobert模型的分词器
    from .models.phobert import PhobertTokenizer
    
    # 导入Pix2Struct模型相关的配置、配置解析器等
    from .models.pix2struct import (
        PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pix2StructConfig,
        Pix2StructProcessor,
        Pix2StructTextConfig,
        Pix2StructVisionConfig,
    )
    
    # 导入PLBART模型相关的配置
    from .models.plbart import PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP, PLBartConfig
    
    # 导入PoolFormer模型相关的配置
    from .models.poolformer import (
        POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PoolFormerConfig,
    )
    
    # 导入Pop2Piano模型相关的配置
    from .models.pop2piano import (
        POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pop2PianoConfig,
    )
    
    # 导入ProphetNet模型相关的配置、配置解析器等
    from .models.prophetnet import (
        PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ProphetNetConfig,
        ProphetNetTokenizer,
    )
    
    # 导入PVT模型相关的配置
    from .models.pvt import PVT_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtConfig
    
    # 导入PVT-V2模型相关的配置
    from .models.pvt_v2 import PVT_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtV2Config
    
    # 导入QDQBERT模型相关的配置
    from .models.qdqbert import QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, QDQBertConfig
    
    # 导入QWEN2模型相关的配置、配置解析器等
    from .models.qwen2 import (
        QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Qwen2Config,
        Qwen2Tokenizer,
    )
    
    # 导入RAG模型的配置、检索器、分词器等
    from .models.rag import RagConfig, RagRetriever, RagTokenizer
    
    # 导入REALM模型相关的配置、配置解析器等
    from .models.realm import (
        REALM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RealmConfig,
        RealmTokenizer,
    )
    
    # 导入Reformer模型相关的配置
    from .models.reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
    
    # 导入RegNet模型相关的配置
    from .models.regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig
    
    # 导入RemBert模型相关的配置
    from .models.rembert import REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RemBertConfig
    
    # 导入ResNet模型相关的配置
    from .models.resnet import RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ResNetConfig
    
    # 导入RoBERTa模型相关的配置、配置解析器等
    from .models.roberta import (
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaConfig,
        RobertaTokenizer,
    )
    # 导入 RoBERTa 模型的配置映射和配置类
    from .models.roberta_prelayernorm import (
        ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaPreLayerNormConfig,
    )
    
    # 导入 ROC-BERT 模型的配置映射、配置类和分词器类
    from .models.roc_bert import (
        ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RoCBertConfig,
        RoCBertTokenizer,
    )
    
    # 导入 RoFormer 模型的配置映射、配置类和分词器类
    from .models.roformer import (
        ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RoFormerConfig,
        RoFormerTokenizer,
    )
    
    # 导入 RWKV 模型的配置映射和配置类
    from .models.rwkv import RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP, RwkvConfig
    
    # 导入 SAM 模型的配置映射和多个配置类
    from .models.sam import (
        SAM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SamConfig,
        SamMaskDecoderConfig,
        SamProcessor,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )
    
    # 导入 Seamless M4T 模型的配置映射和多个配置类
    from .models.seamless_m4t import (
        SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SeamlessM4TConfig,
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
    )
    
    # 导入 Seamless M4T V2 模型的配置映射和配置类
    from .models.seamless_m4t_v2 import (
        SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SeamlessM4Tv2Config,
    )
    
    # 导入 Segformer 模型的配置映射和配置类
    from .models.segformer import SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, SegformerConfig
    
    # 导入 SegGPT 模型的配置映射和配置类
    from .models.seggpt import SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, SegGptConfig
    
    # 导入 SEW 模型的配置映射和配置类
    from .models.sew import SEW_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWConfig
    
    # 导入 SEW-D 模型的配置映射和配置类
    from .models.sew_d import SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWDConfig
    
    # 导入 Siglip 模型的配置映射和多个配置类
    from .models.siglip import (
        SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SiglipConfig,
        SiglipProcessor,
        SiglipTextConfig,
        SiglipVisionConfig,
    )
    
    # 导入语音编码解码模型的配置类
    from .models.speech_encoder_decoder import SpeechEncoderDecoderConfig
    
    # 导入语音到文本模型的配置映射、配置类、特征提取器类和处理器类
    from .models.speech_to_text import (
        SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Speech2TextConfig,
        Speech2TextFeatureExtractor,
        Speech2TextProcessor,
    )
    
    # 导入语音到文本 2 模型的配置映射、配置类、处理器类和分词器类
    from .models.speech_to_text_2 import (
        SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Speech2Text2Config,
        Speech2Text2Processor,
        Speech2Text2Tokenizer,
    )
    
    # 导入 SpeechT5 模型的配置映射、配置类、特征提取器类和处理器类
    from .models.speecht5 import (
        SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP,
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5HifiGanConfig,
        SpeechT5Processor,
    )
    
    # 导入 Splinter 模型的配置映射、配置类和分词器类
    from .models.splinter import (
        SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SplinterConfig,
        SplinterTokenizer,
    )
    
    # 导入 SqueezeBERT 模型的配置映射、配置类和分词器类
    from .models.squeezebert import (
        SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SqueezeBertConfig,
        SqueezeBertTokenizer,
    )
    
    # 导入 StableLM 模型的配置映射和配置类
    from .models.stablelm import STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP, StableLmConfig
    
    # 导入 Starcoder2 模型的配置映射和配置类
    from .models.starcoder2 import STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP, Starcoder2Config
    
    # 导入 SuperPoint 模型的配置映射和配置类
    from .models.superpoint import SUPERPOINT_PRETRAINED_CONFIG_ARCHIVE_MAP, SuperPointConfig
    
    # 导入 SwiftFormer 模型的配置映射和配置类
    from .models.swiftformer import (
        SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwiftFormerConfig,
    )
    # 导入 SWIN 模型相关的预训练配置映射和配置类
    from .models.swin import SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, SwinConfig
    # 导入 SWIN2SR 模型相关的预训练配置映射和配置类
    from .models.swin2sr import SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP, Swin2SRConfig
    # 导入 SWINV2 模型相关的预训练配置映射和配置类
    from .models.swinv2 import SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Swinv2Config
    # 导入 Switch Transformers 模型相关的预训练配置映射和配置类
    from .models.switch_transformers import (
        SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwitchTransformersConfig,
    )
    # 导入 T5 模型相关的预训练配置映射和配置类
    from .models.t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
    # 导入 Table Transformer 模型相关的预训练配置映射和配置类
    from .models.table_transformer import (
        TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TableTransformerConfig,
    )
    # 导入 TAPAS 模型相关的预训练配置映射、配置类和分词器类
    from .models.tapas import (
        TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TapasConfig,
        TapasTokenizer,
    )
    # 导入 Time Series Transformer 模型相关的预训练配置映射和配置类
    from .models.time_series_transformer import (
        TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimeSeriesTransformerConfig,
    )
    # 导入 Timesformer 模型相关的预训练配置映射和配置类
    from .models.timesformer import (
        TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimesformerConfig,
    )
    # 导入 Timm Backbone 模型的配置类
    from .models.timm_backbone import TimmBackboneConfig
    # 导入 TrOCR 模型相关的预训练配置映射、配置类和处理器类
    from .models.trocr import (
        TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TrOCRConfig,
        TrOCRProcessor,
    )
    # 导入 TVLT 模型相关的预训练配置映射、配置类和特征提取器类、处理器类
    from .models.tvlt import (
        TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TvltConfig,
        TvltFeatureExtractor,
        TvltProcessor,
    )
    # 导入 TVP 模型相关的预训练配置映射、配置类和处理器类
    from .models.tvp import (
        TVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TvpConfig,
        TvpProcessor,
    )
    # 导入 UDOP 模型相关的预训练配置映射、配置类和处理器类
    from .models.udop import UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP, UdopConfig, UdopProcessor
    # 导入 UMT5 模型的配置类
    from .models.umt5 import UMT5Config
    # 导入 UniSpeech 模型相关的预训练配置映射和配置类
    from .models.unispeech import (
        UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UniSpeechConfig,
    )
    # 导入 UniSpeech SAT 模型相关的预训练配置映射和配置类
    from .models.unispeech_sat import (
        UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UniSpeechSatConfig,
    )
    # 导入 UnivNet 模型相关的预训练配置映射、配置类和特征提取器类
    from .models.univnet import (
        UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UnivNetConfig,
        UnivNetFeatureExtractor,
    )
    # 导入 UperNet 模型的配置类
    from .models.upernet import UperNetConfig
    # 导入 VideoMAE 模型相关的预训练配置映射和配置类
    from .models.videomae import VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP, VideoMAEConfig
    # 导入 VILT 模型相关的预训练配置映射、配置类和特征提取器类、图像处理器类、处理器类
    from .models.vilt import (
        VILT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ViltConfig,
        ViltFeatureExtractor,
        ViltImageProcessor,
        ViltProcessor,
    )
    # 导入 VIPLLAVA 模型相关的预训练配置映射和配置类
    from .models.vipllava import (
        VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VipLlavaConfig,
    )
    # 导入 Vision Encoder-Decoder 模型的配置类
    from .models.vision_encoder_decoder import VisionEncoderDecoderConfig
    # 导入 Vision Text Dual Encoder 模型的配置类和处理器类
    from .models.vision_text_dual_encoder import (
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderProcessor,
    )
    # 导入 VisualBERT 模型相关的预训练配置映射和配置类
    from .models.visual_bert import (
        VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VisualBertConfig,
    )
    # 导入 ViT 模型相关的预训练配置映射和配置类
    from .models.vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig
    # 导入 ViT Hybrid 模型相关的预训练配置映射和配置类
    from .models.vit_hybrid import (
        VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ViTHybridConfig,
    )
    # 导入各个模型的预训练配置映射和配置类
    
    from .models.vit_mae import VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMAEConfig
    from .models.vit_msn import VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMSNConfig
    from .models.vitdet import VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP, VitDetConfig
    from .models.vitmatte import VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP, VitMatteConfig
    from .models.vits import (
        VITS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VitsConfig,
        VitsTokenizer,
    )
    from .models.vivit import VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, VivitConfig
    from .models.wav2vec2 import (
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
    )
    from .models.wav2vec2_bert import (
        WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2BertConfig,
        Wav2Vec2BertProcessor,
    )
    from .models.wav2vec2_conformer import (
        WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2ConformerConfig,
    )
    from .models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
    from .models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
    from .models.wavlm import WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP, WavLMConfig
    from .models.whisper import (
        WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from .models.x_clip import (
        XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XCLIPConfig,
        XCLIPProcessor,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    from .models.xglm import XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XGLMConfig
    from .models.xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMTokenizer
    from .models.xlm_prophetnet import (
        XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMProphetNetConfig,
    )
    from .models.xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
    from .models.xlm_roberta_xl import (
        XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaXLConfig,
    )
    from .models.xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
    from .models.xmod import XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP, XmodConfig
    from .models.yolos import YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP, YolosConfig
    from .models.yoso import YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP, YosoConfig
    
    # 各个模型的预训练配置映射和配置类的导入
    # 导入具体的数据处理管道类
    from .pipelines import (
        AudioClassificationPipeline,  # 音频分类管道
        AutomaticSpeechRecognitionPipeline,  # 自动语音识别管道
        Conversation,  # 对话处理
        ConversationalPipeline,  # 对话式处理管道
        CsvPipelineDataFormat,  # CSV 数据格式处理管道
        DepthEstimationPipeline,  # 深度估计管道
        DocumentQuestionAnsweringPipeline,  # 文档问答管道
        FeatureExtractionPipeline,  # 特征提取管道
        FillMaskPipeline,  # 填空管道
        ImageClassificationPipeline,  # 图像分类管道
        ImageFeatureExtractionPipeline,  # 图像特征提取管道
        ImageSegmentationPipeline,  # 图像分割管道
        ImageToImagePipeline,  # 图像到图像转换管道
        ImageToTextPipeline,  # 图像到文本转换管道
        JsonPipelineDataFormat,  # JSON 数据格式处理管道
        MaskGenerationPipeline,  # 掩码生成管道
        NerPipeline,  # 命名实体识别管道
        ObjectDetectionPipeline,  # 目标检测管道
        PipedPipelineDataFormat,  # 管道数据格式处理管道
        Pipeline,  # 通用管道基类
        PipelineDataFormat,  # 管道数据格式
        QuestionAnsweringPipeline,  # 问答管道
        SummarizationPipeline,  # 摘要生成管道
        TableQuestionAnsweringPipeline,  # 表格问答管道
        Text2TextGenerationPipeline,  # 文本到文本生成管道
        TextClassificationPipeline,  # 文本分类管道
        TextGenerationPipeline,  # 文本生成管道
        TextToAudioPipeline,  # 文本到音频生成管道
        TokenClassificationPipeline,  # 标记分类管道
        TranslationPipeline,  # 翻译管道
        VideoClassificationPipeline,  # 视频分类管道
        VisualQuestionAnsweringPipeline,  # 视觉问答管道
        ZeroShotAudioClassificationPipeline,  # 零样本音频分类管道
        ZeroShotClassificationPipeline,  # 零样本分类管道
        ZeroShotImageClassificationPipeline,  # 零样本图像分类管道
        ZeroShotObjectDetectionPipeline,  # 零样本目标检测管道
        pipeline,  # 简化调用的通用管道方法
    )

    # 导入处理工具和Mixin类
    from .processing_utils import ProcessorMixin

    # 导入分词相关工具类
    # Tokenization
    from .tokenization_utils import PreTrainedTokenizer  # 预训练分词器
    from .tokenization_utils_base import (
        AddedToken,  # 添加的特殊标记
        BatchEncoding,  # 批编码
        CharSpan,  # 字符跨度
        PreTrainedTokenizerBase,  # 预训练分词器基类
        SpecialTokensMixin,  # 特殊标记混合类
        TokenSpan,  # 标记跨度
    )

    # 导入工具类和相关函数
    # Tools
    from .tools import (
        Agent,  # 代理
        AzureOpenAiAgent,  # Azure OpenAI 代理
        HfAgent,  # Hugging Face 代理
        LocalAgent,  # 本地代理
        OpenAiAgent,  # OpenAI 代理
        PipelineTool,  # 管道工具
        RemoteTool,  # 远程工具
        Tool,  # 工具
        launch_gradio_demo,  # 启动 Gradio 演示
        load_tool,  # 载入工具
    )

    # 导入训练过程中的回调函数和状态类
    # Trainer
    from .trainer_callback import (
        DefaultFlowCallback,  # 默认流程回调
        EarlyStoppingCallback,  # 提前停止回调
        PrinterCallback,  # 打印机回调
        ProgressCallback,  # 进度回调
        TrainerCallback,  # 训练器回调基类
        TrainerControl,  # 训练控制
        TrainerState,  # 训练状态
    )

    # 导入训练过程中的实用函数和类
    from .trainer_utils import (
        EvalPrediction,  # 评估预测
        IntervalStrategy,  # 间隔策略
        SchedulerType,  # 调度器类型
        enable_full_determinism,  # 启用完全确定性
        set_seed,  # 设置随机种子
    )

    # 导入训练参数类
    from .training_args import TrainingArguments  # 训练参数
    from .training_args_seq2seq import Seq2SeqTrainingArguments  # 序列到序列训练参数
    from .training_args_tf import TFTrainingArguments  # TensorFlow 训练参数

    # 导入文件和通用工具
    # Files and general utilities
    # 从.utils模块中导入多个常量和函数
    from .utils import (
        CONFIG_NAME,                     # 导入配置名称常量
        MODEL_CARD_NAME,                 # 导入模型卡名称常量
        PYTORCH_PRETRAINED_BERT_CACHE,   # 导入PyTorch预训练BERT缓存常量
        PYTORCH_TRANSFORMERS_CACHE,      # 导入PyTorch Transformers缓存常量
        SPIECE_UNDERLINE,                # 导入SPIECE_UNDERLINE常量
        TF2_WEIGHTS_NAME,                # 导入TensorFlow 2权重名称常量
        TF_WEIGHTS_NAME,                 # 导入TensorFlow权重名称常量
        TRANSFORMERS_CACHE,              # 导入Transformers缓存常量
        WEIGHTS_NAME,                    # 导入权重名称常量
        TensorType,                      # 导入TensorType类
        add_end_docstrings,              # 导入添加文档结束注释的函数
        add_start_docstrings,            # 导入添加文档起始注释的函数
        is_apex_available,               # 导入检查Apex是否可用的函数
        is_bitsandbytes_available,       # 导入检查bitsandbytes是否可用的函数
        is_datasets_available,           # 导入检查datasets是否可用的函数
        is_decord_available,             # 导入检查decord是否可用的函数
        is_faiss_available,              # 导入检查faiss是否可用的函数
        is_flax_available,               # 导入检查flax是否可用的函数
        is_keras_nlp_available,          # 导入检查keras-nlp是否可用的函数
        is_phonemizer_available,         # 导入检查phonemizer是否可用的函数
        is_psutil_available,             # 导入检查psutil是否可用的函数
        is_py3nvml_available,            # 导入检查py3nvml是否可用的函数
        is_pyctcdecode_available,        # 导入检查pyctcdecode是否可用的函数
        is_sacremoses_available,         # 导入检查sacremoses是否可用的函数
        is_safetensors_available,        # 导入检查safetensors是否可用的函数
        is_scipy_available,              # 导入检查scipy是否可用的函数
        is_sentencepiece_available,      # 导入检查sentencepiece是否可用的函数
        is_sklearn_available,            # 导入检查scikit-learn是否可用的函数
        is_speech_available,             # 导入检查speech是否可用的函数
        is_tensorflow_text_available,    # 导入检查tensorflow-text是否可用的函数
        is_tf_available,                 # 导入检查TensorFlow是否可用的函数
        is_timm_available,               # 导入检查timm是否可用的函数
        is_tokenizers_available,         # 导入检查tokenizers是否可用的函数
        is_torch_available,              # 导入检查PyTorch是否可用的函数
        is_torch_neuroncore_available,   # 导入检查torch-neuroncore是否可用的函数
        is_torch_npu_available,          # 导入检查torch-npu是否可用的函数
        is_torch_tpu_available,          # 导入检查torch-tpu是否可用的函数
        is_torch_xla_available,          # 导入检查torch-xla是否可用的函数
        is_torch_xpu_available,          # 导入检查torch-xpu是否可用的函数
        is_torchvision_available,        # 导入检查torchvision是否可用的函数
        is_vision_available,             # 导入检查vision是否可用的函数
        logging,                         # 导入logging模块
    )

    # bitsandbytes配置
    # 从.utils.quantization_config模块导入多个配置类
    from .utils.quantization_config import AqlmConfig, AwqConfig, BitsAndBytesConfig, GPTQConfig, QuantoConfig

    try:
        # 如果sentencepiece不可用，抛出OptionalDependencyNotAvailable异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入虚拟的sentencepiece对象，以便在没有实际依赖时使用
        from .utils.dummy_sentencepiece_objects import *
    else:
        # 导入各种模型的分词器
        from .models.albert import AlbertTokenizer
        from .models.barthez import BarthezTokenizer
        from .models.bartpho import BartphoTokenizer
        from .models.bert_generation import BertGenerationTokenizer
        from .models.big_bird import BigBirdTokenizer
        from .models.camembert import CamembertTokenizer
        from .models.code_llama import CodeLlamaTokenizer
        from .models.cpm import CpmTokenizer
        from .models.deberta_v2 import DebertaV2Tokenizer
        from .models.ernie_m import ErnieMTokenizer
        from .models.fnet import FNetTokenizer
        from .models.gemma import GemmaTokenizer
        from .models.gpt_sw3 import GPTSw3Tokenizer
        from .models.layoutxlm import LayoutXLMTokenizer
        from .models.llama import LlamaTokenizer
        from .models.m2m_100 import M2M100Tokenizer
        from .models.marian import MarianTokenizer
        from .models.mbart import MBart50Tokenizer, MBartTokenizer
        from .models.mluke import MLukeTokenizer
        from .models.mt5 import MT5Tokenizer
        from .models.nllb import NllbTokenizer
        from .models.pegasus import PegasusTokenizer
        from .models.plbart import PLBartTokenizer
        from .models.reformer import ReformerTokenizer
        from .models.rembert import RemBertTokenizer
        from .models.seamless_m4t import SeamlessM4TTokenizer
        from .models.siglip import SiglipTokenizer
        from .models.speech_to_text import Speech2TextTokenizer
        from .models.speecht5 import SpeechT5Tokenizer
        from .models.t5 import T5Tokenizer
        from .models.udop import UdopTokenizer
        from .models.xglm import XGLMTokenizer
        from .models.xlm_prophetnet import XLMProphetNetTokenizer
        from .models.xlm_roberta import XLMRobertaTokenizer
        from .models.xlnet import XLNetTokenizer

    try:
        # 检查 tokenizers 库是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 tokenizers 库不可用，则导入虚拟的 tokenizers 对象
        from .utils.dummy_tokenizers_objects import *

    try:
        # 检查是否同时可用 sentencepiece 和 tokenizers 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 sentencepiece 和 tokenizers 库不可用，则导入虚拟的 sentencepiece 和 tokenizers 对象
        from .utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        # 如果以上检查都通过，则导入转换慢速分词器为快速分词器的相关函数和对象
        from .convert_slow_tokenizer import (
            SLOW_TO_FAST_CONVERTERS,
            convert_slow_tokenizer,
        )

    try:
        # 检查是否可用 tensorflow_text 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 tensorflow_text 库不可用，则导入虚拟的 tensorflow_text 对象
        from .utils.dummy_tensorflow_text_objects import *
    else:
        # 如果 tensorflow_text 库可用，则导入 TF 版本的 BERT 分词器
        from .models.bert import TFBertTokenizer

    try:
        # 检查是否可用 keras_nlp 库，如果不可用则引发 OptionalDependencyNotAvailable 异常
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 keras_nlp 库不可用，则导入虚拟的 keras_nlp 对象
        from .utils.dummy_keras_nlp_objects import *
    else:
        # 如果 keras_nlp 库可用，则导入 TF 版本的 GPT-2 分词器
        from .models.gpt2 import TFGPT2Tokenizer
    # 检查是否有视觉模块可用
    try:
        if not is_vision_available():
            # 如果视觉模块不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，从 dummy_vision_objects 模块导入所有对象
        from .utils.dummy_vision_objects import *

    # 模型部分
    # 检查是否有 Torch 库可用
    try:
        if not is_torch_available():
            # 如果 Torch 库不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，从 dummy_pt_objects 模块导入所有对象
        from .utils.dummy_pt_objects import *

    # TensorFlow 部分
    try:
        if not is_tf_available():
            # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，导入 dummy_tf_objects 模块，将对象置于命名空间中
        from .utils.dummy_tf_objects import *

    # 检查是否同时具备以下库的可用性：librosa、essentia、scipy、torch、pretty_midi
    try:
        if not (
            is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
            and is_torch_available()
            and is_pretty_midi_available()
        ):
            # 如果任何一个依赖库不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，从 dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects 模块导入所有对象
        from .utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects import *
    else:
        # 如果所有依赖库都可用，则从 pop2piano 模块导入特定对象
        from .models.pop2piano import (
            Pop2PianoFeatureExtractor,
            Pop2PianoProcessor,
            Pop2PianoTokenizer,
        )

    # 检查是否有 torchaudio 库可用
    try:
        if not is_torchaudio_available():
            # 如果 torchaudio 不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，从 dummy_torchaudio_objects 模块导入所有对象
        from .utils.dummy_torchaudio_objects import *
    else:
        # 如果 torchaudio 可用，则从 musicgen_melody 模块导入特定对象
        from .models.musicgen_melody import MusicgenMelodyFeatureExtractor, MusicgenMelodyProcessor

    # Flax 部分
    try:
        if not is_flax_available():
            # 如果 Flax 不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果异常被引发，导入 dummy_flax_objects 模块，将对象置于命名空间中
        from .utils.dummy_flax_objects import *
# 如果不是在 TensorFlow、PyTorch 或 Flax 环境下，则给出警告建议信息
if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
else:
    # 导入 sys 模块，用于后续操作
    import sys

    # 将当前模块替换为 _LazyModule 的实例，传入相关参数
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
```