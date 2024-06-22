# `.\transformers\__init__.py`

```py
# 版权声明和许可证信息

# 版权声明和许可证信息

# 检查依赖项是否满足所需的最小版本要求
from . import dependency_versions_check

# 导入类型检查模块
from .utils import (
    OptionalDependencyNotAvailable,  # 导入依赖项不可用时的异常
    _LazyModule,  # 惰性加载模块的类
    is_bitsandbytes_available,  # 检查 bitsandbytes 库是否可用
    is_essentia_available,  # 检查 essentia 库是否可用
    is_flax_available,  # 检查 flax 库是否可用
    is_g2p_en_available,  # 检查 g2p_en 库是否可用
    is_keras_nlp_available,  # 检查 keras_nlp 库是否可用
    is_librosa_available,  # 检查 librosa 库是否可用
    is_pretty_midi_available,  # 检查 pretty_midi 库是否可用
    is_scipy_available,  # 检查 scipy 库是否可用
    is_sentencepiece_available,  # 检查 sentencepiece 库是否可用
    is_speech_available,  # 检查 speech 库是否可用
    is_tensorflow_text_available,  # 检查 tensorflow_text 库是否可用
    is_tf_available,  # 检查 TensorFlow 是否可用
    is_timm_available,  # 检查 timm 库是否可用
    is_tokenizers_available,  # 检查 tokenizers 库是否可用
    is_torch_available,  # 检查 PyTorch 是否可用
    is_torchvision_available,  # 检查 torchvision 库是否可用
    is_vision_available,  # 检查 vision 库是否可用
    logging,  # 日志记录模块
)

# 获取日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 基础对象，独立于任何特定的后端
_import_structure = {
    "audio_utils": [],  # 音频处理工具
    "benchmark": [],  # 基准测试相关
    "commands": [],  # 命令行工具相关
    "configuration_utils": ["PretrainedConfig"],  # 配置相关工具，包括预训练模型配置
    "convert_graph_to_onnx": [],  # 将图形转换为 ONNX 格式相关
    "convert_slow_tokenizers_checkpoints_to_fast": [],  # 将慢速分词器的检查点转换为快速分词器相关
    "convert_tf_hub_seq_to_seq_bert_to_pytorch": [],  # 将 TF Hub Seq 转换为 Seq Bert 转换为 PyTorch 相关
    "data": [  # 数据处理相关
        "DataProcessor",  # 数据处理器
        "InputExample",  # 输入示例
        "InputFeatures",  # 输入特征
        "SingleSentenceClassificationProcessor",  # 单句分类处理器
        "SquadExample",  # Squad 示例
        "SquadFeatures",  # Squad 特征
        "SquadV1Processor",  # Squad V1 处理器
        "SquadV2Processor",  # Squad V2 处理器
        "glue_compute_metrics",  # 计算 GLUE 数据集指标
        "glue_convert_examples_to_features",  # 将 GLUE 数据集示例转换为特征
        "glue_output_modes",  # GLUE 数据集输出模式
        "glue_processors",  # GLUE 数据集处理器
        "glue_tasks_num_labels",  # GLUE 数据集任务标签数
        "squad_convert_examples_to_features",  # 将 Squad 数据集示例转换为特征
        "xnli_compute_metrics",  # 计算 XNLI 数据集指标
        "xnli_output_modes",  # XNLI 数据集输出模式
        "xnli_processors",  # XNLI 数据集处理器
        "xnli_tasks_num_labels",  # XNLI 数据集任务标签数
    ],
    # 数据相关的数据整理器类的模块列表
    "data.data_collator": [
        "DataCollator",  # 数据整理器基类
        "DataCollatorForLanguageModeling",  # 用于语言建模的数据整理器
        "DataCollatorForPermutationLanguageModeling",  # 用于置换语言建模的数据整理器
        "DataCollatorForSeq2Seq",  # 用于序列到序列任务的数据整理器
        "DataCollatorForSOP",  # 用于句子顺序预测的数据整理器
        "DataCollatorForTokenClassification",  # 用于标记分类任务的数据整理器
        "DataCollatorForWholeWordMask",  # 用于全词掩码任务的数据整理器
        "DataCollatorWithPadding",  # 带填充的数据整理器
        "DefaultDataCollator",  # 默认的数据整理器
        "default_data_collator",  # 默认的数据整理器别名
    ],
    # 数据度量模块列表
    "data.metrics": [],
    # 数据处理器模块列表
    "data.processors": [],
    # 调试工具模块列表
    "debug_utils": [],
    # DeepSpeed 模块列表
    "deepspeed": [],
    # 依赖版本检查模块列表
    "dependency_versions_check": [],
    # 依赖版本表模块列表
    "dependency_versions_table": [],
    # 动态模块工具模块列表
    "dynamic_module_utils": [],
    # 特征提取序列工具模块列表
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    # 特征提取工具模块列表
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    # 文件工具模块列表
    "file_utils": [],
    # 生成模块列表
    "generation": ["GenerationConfig", "TextIteratorStreamer", "TextStreamer"],
    # Hugging Face 参数解析器模块列表
    "hf_argparser": ["HfArgumentParser"],
    # 超参数搜索模块列表
    "hyperparameter_search": [],
    # 图像变换模块列表
    "image_transforms": [],
    # 集成模块列表
    "integrations": [
        "is_clearml_available",  # 是否可用 ClearML
        "is_comet_available",  # 是否可用 Comet
        "is_dvclive_available",  # 是否可用 DVCLive
        "is_neptune_available",  # 是否可用 Neptune
        "is_optuna_available",  # 是否可用 Optuna
        "is_ray_available",  # 是否可用 Ray
        "is_ray_tune_available",  # 是否可用 Ray Tune
        "is_sigopt_available",  # 是否可用 SigOpt
        "is_tensorboard_available",  # 是否可用 TensorBoard
        "is_wandb_available",  # 是否可用 Weights & Biases
    ],
    # 模型卡片模块列表
    "modelcard": ["ModelCard"],
    # TensorFlow 和 PyTorch 模型工具模块列表
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",  # 转换 TensorFlow 权重名为 PyTorch 权重名
        "load_pytorch_checkpoint_in_tf2_model",  # 在 TensorFlow 2 模型中加载 PyTorch 检查点
        "load_pytorch_model_in_tf2_model",  # 在 TensorFlow 2 模型中加载 PyTorch 模型
        "load_pytorch_weights_in_tf2_model",  # 在 TensorFlow 2 模型中加载 PyTorch 权重
        "load_tf2_checkpoint_in_pytorch_model",  # 在 PyTorch 模型中加载 TensorFlow 2 检查点
        "load_tf2_model_in_pytorch_model",  # 在 PyTorch 模型中加载 TensorFlow 2 模型
        "load_tf2_weights_in_pytorch_model",  # 在 PyTorch 模型中加载 TensorFlow 2 权重
    ],
    # 模型模块列表
    "models": [],
    # Albert 模型模块列表
    "models.albert": ["ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "AlbertConfig"],
    # 对齐模块列表
    "models.align": [
        "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "AlignConfig",  # 对齐配置
        "AlignProcessor",  # 对齐处理器
        "AlignTextConfig",  # 对齐文本配置
        "AlignVisionConfig",  # 对齐视觉配置
    ],
    # AltCLIP 模块列表
    "models.altclip": [
        "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "AltCLIPConfig",  # AltCLIP 配置
        "AltCLIPProcessor",  # AltCLIP 处理器
        "AltCLIPTextConfig",  # AltCLIP 文本配置
        "AltCLIPVisionConfig",  # AltCLIP 视觉配置
    ],
    # 音频频谱转换器模块列表
    "models.audio_spectrogram_transformer": [
        "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "ASTConfig",  # AST 配置
        "ASTFeatureExtractor",  # AST 特征提取器
    ],
    # 自动模块列表
    "models.auto": [
        "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 所有预训练配置映射
        "CONFIG_MAPPING",  # 配置映射
        "FEATURE_EXTRACTOR_MAPPING",  # 特征提取器映射
        "IMAGE_PROCESSOR_MAPPING",  # 图像处理器映射
        "MODEL_NAMES_MAPPING",  # 模型名称映射
        "PROCESSOR_MAPPING",  # 处理器映射
        "TOKENIZER_MAPPING",  # 分词器映射
        "AutoConfig",  # 自动配置
        "AutoFeatureExtractor",  # 自动特征提取器
        "AutoImageProcessor",  # 自动图像
    "models.bark": [
        "BarkCoarseConfig",  # BarkCoarseConfig 模型配置类
        "BarkConfig",  # BarkConfig 模型配置类
        "BarkFineConfig",  # BarkFineConfig 模型配置类
        "BarkProcessor",  # BarkProcessor 模型处理器类
        "BarkSemanticConfig",  # BarkSemanticConfig 模型配置类
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],  # BartConfig 模型配置类，BartTokenizer 模型分词器类
    "models.barthez": [],  # 空列表，未提供相关模型信息
    "models.bartpho": [],  # 空列表，未提供相关模型信息
    "models.beit": ["BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BeitConfig"],  # BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射，BeitConfig 模型配置类
    "models.bert": [
        "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BERT_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BasicTokenizer",  # BasicTokenizer 基础分词器类
        "BertConfig",  # BertConfig 模型配置类
        "BertTokenizer",  # BertTokenizer 模型分词器类
        "WordpieceTokenizer",  # WordpieceTokenizer 词块分词器类
    ],
    "models.bert_generation": ["BertGenerationConfig"],  # BertGenerationConfig 生成模型配置类
    "models.bert_japanese": [
        "BertJapaneseTokenizer",  # BertJapaneseTokenizer 日语模型分词器类
        "CharacterTokenizer",  # CharacterTokenizer 字符分词器类
        "MecabTokenizer",  # MecabTokenizer MeCab 分词器类
    ],
    "models.bertweet": ["BertweetTokenizer"],  # BertweetTokenizer 分词器类
    "models.big_bird": ["BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigBirdConfig"],  # BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射，BigBirdConfig 模型配置类
    "models.bigbird_pegasus": [
        "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BigBirdPegasusConfig",  # BigBirdPegasusConfig 模型配置类
    ],
    "models.biogpt": [
        "BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BioGptConfig",  # BioGptConfig 模型配置类
        "BioGptTokenizer",  # BioGptTokenizer 模型分词器类
    ],
    "models.bit": ["BIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BitConfig"],  # BIT_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射，BitConfig 模型配置类
    "models.blenderbot": [
        "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BlenderbotConfig",  # BlenderbotConfig 模型配置类
        "BlenderbotTokenizer",  # BlenderbotTokenizer 模型分词器类
    ],
    "models.blenderbot_small": [
        "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BlenderbotSmallConfig",  # BlenderbotSmallConfig 模型配置类
        "BlenderbotSmallTokenizer",  # BlenderbotSmallTokenizer 模型分词器类
    ],
    "models.blip": [
        "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BlipConfig",  # BlipConfig 模型配置类
        "BlipProcessor",  # BlipProcessor 模型处理器类
        "BlipTextConfig",  # BlipTextConfig 文本模型配置类
        "BlipVisionConfig",  # BlipVisionConfig 视觉模型配置类
    ],
    "models.blip_2": [
        "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "Blip2Config",  # Blip2Config 模型配置类
        "Blip2Processor",  # Blip2Processor 模型处理器类
        "Blip2QFormerConfig",  # Blip2QFormerConfig 模型配置类
        "Blip2VisionConfig",  # Blip2VisionConfig 视觉模型配置类
    ],
    "models.bloom": ["BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP", "BloomConfig"],  # BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射，BloomConfig 模型配置类
    "models.bridgetower": [
        "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BridgeTowerConfig",  # BridgeTowerConfig 模型配置类
        "BridgeTowerProcessor",  # BridgeTowerProcessor 模型处理器类
        "BridgeTowerTextConfig",  # BridgeTowerTextConfig 文本模型配置类
        "BridgeTowerVisionConfig",  # BridgeTowerVisionConfig 视觉模型配置类
    ],
    "models.bros": [
        "BROS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # BROS_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射
        "BrosConfig",  # BrosConfig 模型配置类
        "BrosProcessor",  # BrosProcessor 模型处理器类
    ],
    "models.byt5": ["ByT5Tokenizer"],  # ByT5Tokenizer 分词器类
    "models.camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig"],  # CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP 预训练配置映射，CamembertConfig 模型配置类
    "models.canine": [
        "CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "models.clip": [
        "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CLIP 预训练配置文件映射
        "CLIPConfig",  # CLIP 配置
        "CLIPProcessor",  # CLIP 处理器
        "CLIPTextConfig",  # CLIP 文本配置
        "CLIPTokenizer",  # CLIP 分词器
        "CLIPVisionConfig",  # CLIP 视觉配置
    ],
    "models.clipseg": [
        "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CLIPSEG 预训练配置文件映射
        "CLIPSegConfig",  # CLIPSeg 配置
        "CLIPSegProcessor",  # CLIPSeg 处理器
        "CLIPSegTextConfig",  # CLIPSeg 文本配置
        "CLIPSegVisionConfig",  # CLIPSeg 视觉配置
    ],
    "models.clvp": [
        "CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CLVP 预训练配置文件映射
        "ClvpConfig",  # Clvp 配置
        "ClvpDecoderConfig",  # Clvp 解码器配置
        "ClvpEncoderConfig",  # Clvp 编码器配置
        "ClvpFeatureExtractor",  # Clvp 特征提取器
        "ClvpProcessor",  # Clvp 处理器
        "ClvpTokenizer",  # Clvp 分词器
    ],
    "models.code_llama": [],  # 空列表
    "models.codegen": [
        "CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CODEGEN 预训练配置文件映射
        "CodeGenConfig",  # CodeGen 配置
        "CodeGenTokenizer",  # CodeGen 分词器
    ],
    "models.conditional_detr": [
        "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CONDITIONAL DETR 预训练配置文件映射
        "ConditionalDetrConfig",  # ConditionalDetr 配置
    ],
    "models.convbert": [
        "CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CONVBERT 预训练配置文件映射
        "ConvBertConfig",  # ConvBert 配置
        "ConvBertTokenizer",  # ConvBert 分词器
    ],
    "models.convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig"],  # CONVNEXT 预训练配置文件映射和配置
    "models.convnextv2": [
        "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CONVNEXTV2 预训练配置文件映射
        "ConvNextV2Config",  # ConvNextV2 配置
    ],
    "models.cpm": [],  # 空列表
    "models.cpmant": [
        "CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CPMANT 预训练配置文件映射
        "CpmAntConfig",  # CpmAnt 配置
        "CpmAntTokenizer",  # CpmAnt 分词器
    ],
    "models.ctrl": [
        "CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # CTRL 预训练配置文件映射
        "CTRLConfig",  # CTRL 配置
        "CTRLTokenizer",  # CTRL 分词器
    ],
    "models.cvt": ["CVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CvtConfig"],  # CVT 预训练配置文件映射和配置
    "models.data2vec": [
        "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DATA2VEC 文本预训练配置文件映射
        "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DATA2VEC 视觉预训练配置文件映射
        "Data2VecAudioConfig",  # Data2Vec 音频配置
        "Data2VecTextConfig",  # Data2Vec 文本配置
        "Data2VecVisionConfig",  # Data2Vec 视觉配置
    ],
    "models.deberta": [
        "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DEBERTA 预训练配置文件映射
        "DebertaConfig",  # Deberta 配置
        "DebertaTokenizer",  # Deberta 分词器
    ],
    "models.deberta_v2": [
        "DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DEBERTA V2 预训练配置文件映射
        "DebertaV2Config",  # Deberta V2 配置
    ],
    "models.decision_transformer": [
        "DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DECISION TRANSFORMER 预训练配置文件映射
        "DecisionTransformerConfig",  # DecisionTransformer 配置
    ],
    "models.deformable_detr": [
        "DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # DEFORMABLE DETR 预训练配置文件映射
        "DeformableDetrConfig",  # DeformableDetr 配置
    ],
    "models.deit": ["DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeiTConfig"],  # DEIT 预训练配置文件映射和配置
    "models.deprecated": [],  # 空列表
    "models.deprecated.bort": [],  # 空列表
    "models.deprecated.mctct": [
        "MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # MCTCT 预训练配置文件映射
        "MCTCTConfig",  # MCTCT 配置
        "MCTCTFeatureExtractor",  # MCTCT 特征提取器
        "MCTCTProcessor",  # MCTCT 处理器
    ],
    "models.deprecated.mmbt": ["MMBTConfig"],  # MMBT 配置
    "models.deprecated.open_llama": [
        "OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # OPEN LLAMA 预训练配置文件映射
        "OpenLlamaConfig",  # OpenLlama 配置
    ],
    "models.deprecated.retribert": [
        "RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # RETRIBERT 预训练配置文件映射
        "RetriBertConfig",  # RetriBert 配置
        "RetriBertTokenizer",  # RetriBert 分词器
    ],
    # 导入 models.deprecated.tapex 模块中的 TapexTokenizer 类
    "models.deprecated.tapex": ["TapexTokenizer"],
    # 导入 models.deprecated.trajectory_transformer 模块中的相关内容
    "models.deprecated.trajectory_transformer": [
        "TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TrajectoryTransformerConfig",
    ],
    # 导入 models.deprecated.transfo_xl 模块中的相关内容
    "models.deprecated.transfo_xl": [
        "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    # 导入 models.deprecated.van 模块中的相关内容
    "models.deprecated.van": ["VAN_PRETRAINED_CONFIG_ARCHIVE_MAP", "VanConfig"],
    # 导入 models.deta 模块中的相关内容
    "models.deta": ["DETA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetaConfig"],
    # 导入 models.detr 模块中的相关内容
    "models.detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig"],
    # 导入 models.dialogpt 模块中的内容
    "models.dialogpt": [],
    # 导入 models.dinat 模块中的相关内容
    "models.dinat": ["DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DinatConfig"],
    # 导入 models.dinov2 模块中的相关内容
    "models.dinov2": ["DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "Dinov2Config"],
    # 导入 models.distilbert 模块中的相关内容
    "models.distilbert": [
        "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DistilBertConfig",
        "DistilBertTokenizer",
    ],
    # 导入 models.dit 模块中的内容
    "models.dit": [],
    # 导入 models.donut 模块中的相关内容
    "models.donut": [
        "DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DonutProcessor",
        "DonutSwinConfig",
    ],
    # 导入 models.dpr 模块中的相关内容
    "models.dpr": [
        "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    # 导入 models.dpt 模块中的相关内容
    "models.dpt": ["DPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPTConfig"],
    # 导入 models.efficientformer 模块中的相关内容
    "models.efficientformer": [
        "EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EfficientFormerConfig",
    ],
    # 导入 models.efficientnet 模块中的相关内容
    "models.efficientnet": [
        "EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EfficientNetConfig",
    ],
    # 导入 models.electra 模块中的相关内容
    "models.electra": [
        "ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ElectraConfig",
        "ElectraTokenizer",
    ],
    # 导入 models.encodec 模块中的相关内容
    "models.encodec": [
        "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EncodecConfig",
        "EncodecFeatureExtractor",
    ],
    # 导入 models.encoder_decoder 模块中的 EncoderDecoderConfig 类
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    # 导入 models.ernie 模块中的相关内容
    "models.ernie": [
        "ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ErnieConfig",
    ],
    # 导入 models.ernie_m 模块中的相关内容
    "models.ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
    # 导入 models.esm 模块中的相关内容
    "models.esm": ["ESM_PRETRAINED_CONFIG_ARCHIVE_MAP", "EsmConfig", "EsmTokenizer"],
    # 导入 models.falcon 模块中的相关内容
    "models.falcon": ["FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP", "FalconConfig"],
    # 导入 models.fastspeech2_conformer 模块中的相关内容
    "models.fastspeech2_conformer": [
        "FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerTokenizer",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    # 导入 models.flaubert 模块中的相关内容
    "models.flaubert": ["FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FlaubertConfig", "FlaubertTokenizer"],
    "models.flava": [
        "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # FLAVA 预训练配置文件映射
        "FlavaConfig",  # FLAVA 配置
        "FlavaImageCodebookConfig",  # FLAVA 图像码书配置
        "FlavaImageConfig",  # FLAVA 图像配置
        "FlavaMultimodalConfig",  # FLAVA 多模态配置
        "FlavaTextConfig",  # FLAVA 文本配置
    ],
    "models.fnet": ["FNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FNetConfig"],  # FNET 预训练配置文件映射和配置
    "models.focalnet": ["FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FocalNetConfig"],  # FOCALNET 预训练配置文件映射和配置
    "models.fsmt": [
        "FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # FSMT 预训练配置文件映射
        "FSMTConfig",  # FSMT 配置
        "FSMTTokenizer",  # FSMT 分词器
    ],
    "models.funnel": [
        "FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # FUNNEL 预训练配置文件映射
        "FunnelConfig",  # FUNNEL 配置
        "FunnelTokenizer",  # FUNNEL 分词器
    ],
    "models.fuyu": ["FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP", "FuyuConfig"],  # FUYU 预训练配置文件映射和配置
    "models.git": [
        "GIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GIT 预训练配置文件映射
        "GitConfig",  # GIT 配置
        "GitProcessor",  # GIT 处理器
        "GitVisionConfig",  # GIT 视觉配置
    ],
    "models.glpn": ["GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP", "GLPNConfig"],  # GLPN 预训练配置文件映射和配置
    "models.gpt2": [
        "GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GPT2 预训练配置文件映射
        "GPT2Config",  # GPT2 配置
        "GPT2Tokenizer",  # GPT2 分词器
    ],
    "models.gpt_bigcode": [
        "GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GPT_BIGCODE 预训练配置文件映射
        "GPTBigCodeConfig",  # GPT_BIGCODE 配置
    ],
    "models.gpt_neo": ["GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoConfig"],  # GPT_NEO 预训练配置文件映射和配置
    "models.gpt_neox": ["GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXConfig"],  # GPT_NEOX 预训练配置文件映射和配置
    "models.gpt_neox_japanese": [
        "GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GPT_NEOX_JAPANESE 预训练配置文件映射
        "GPTNeoXJapaneseConfig",  # GPT_NEOX_JAPANESE 配置
    ],
    "models.gpt_sw3": [],  # 空列表
    "models.gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig"],  # GPTJ 预训练配置文件映射和配置
    "models.gptsan_japanese": [
        "GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GPTSAN_JAPANESE 预训练配置文件映射
        "GPTSanJapaneseConfig",  # GPTSAN_JAPANESE 配置
        "GPTSanJapaneseTokenizer",  # GPTSAN_JAPANESE 分词器
    ],
    "models.graphormer": [
        "GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GRAPHORMER 预训练配置文件映射
        "GraphormerConfig",  # GRAPHORMER 配置
    ],
    "models.groupvit": [
        "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # GROUPVIT 预训练配置文件映射
        "GroupViTConfig",  # GROUPVIT 配置
        "GroupViTTextConfig",  # GROUPVIT 文本配置
        "GroupViTVisionConfig",  # GROUPVIT 视觉配置
    ],
    "models.herbert": ["HerbertTokenizer"],  # Herbert 分词器
    "models.hubert": ["HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "HubertConfig"],  # HUBERT 预训练配置文件映射和配置
    "models.ibert": ["IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "IBertConfig"],  # IBERT 预训练配置文件映射和配置
    "models.idefics": [
        "IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # IDEFICS 预训练配置文件映射
        "IdeficsConfig",  # IDEFICS 配置
    ],
    "models.imagegpt": ["IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ImageGPTConfig"],  # IMAGEGPT 预训练配置文件映射和配置
    "models.informer": ["INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "InformerConfig"],  # INFORMER 预训练配置文件映射和配置
    "models.instructblip": [
        "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # INSTRUCTBLIP 预训练配置文件映射
        "InstructBlipConfig",  # INSTRUCTBLIP 配置
        "InstructBlipProcessor",  # INSTRUCTBLIP 处理器
        "InstructBlipQFormerConfig",  # INSTRUCTBLIP QFormer 配置
        "InstructBlipVisionConfig",  # INSTRUCTBLIP 视觉配置
    ],
    "models.jukebox": [
        "JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP",  # JUKEBOX 预训练配置文件映射
        "JukeboxConfig",  # JUKEBOX 配置
        "JukeboxPriorConfig",  # JUKEBOX 先验配置
        "JukeboxTokenizer",  # JUKEBOX 分词器
        "JukeboxVQVAEConfig",  # JUKEBOX VQVAE 配置
    ],
    "models.kosmos2": [
        "KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP
        "Kosmos2Config",  # 定义类 Kosmos2Config
        "Kosmos2Processor",  # 定义类 Kosmos2Processor
    ],
    "models.layoutlm": [
        "LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LayoutLMConfig",  # 定义类 LayoutLMConfig
        "LayoutLMTokenizer",  # 定义类 LayoutLMTokenizer
    ],
    "models.layoutlmv2": [
        "LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LayoutLMv2Config",  # 定义类 LayoutLMv2Config
        "LayoutLMv2FeatureExtractor",  # 定义类 LayoutLMv2FeatureExtractor
        "LayoutLMv2ImageProcessor",  # 定义类 LayoutLMv2ImageProcessor
        "LayoutLMv2Processor",  # 定义类 LayoutLMv2Processor
        "LayoutLMv2Tokenizer",  # 定义类 LayoutLMv2Tokenizer
    ],
    "models.layoutlmv3": [
        "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LayoutLMv3Config",  # 定义类 LayoutLMv3Config
        "LayoutLMv3FeatureExtractor",  # 定义类 LayoutLMv3FeatureExtractor
        "LayoutLMv3ImageProcessor",  # 定义类 LayoutLMv3ImageProcessor
        "LayoutLMv3Processor",  # 定义类 LayoutLMv3Processor
        "LayoutLMv3Tokenizer",  # 定义类 LayoutLMv3Tokenizer
    ],
    "models.layoutxlm": ["LayoutXLMProcessor"],  # 定义类 LayoutXLMProcessor
    "models.led": ["LED_PRETRAINED_CONFIG_ARCHIVE_MAP", "LEDConfig", "LEDTokenizer"],  # 定义变量 LED_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 LEDConfig, 类 LEDTokenizer
    "models.levit": ["LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LevitConfig"],  # 定义变量 LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 LevitConfig
    "models.lilt": ["LILT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LiltConfig"],  # 定义变量 LILT_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 LiltConfig
    "models.llama": ["LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlamaConfig"],  # 定义变量 LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 LlamaConfig
    "models.llava": [
        "LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LlavaConfig",  # 定义类 LlavaConfig
    ],
    "models.longformer": [
        "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LongformerConfig",  # 定义类 LongformerConfig
        "LongformerTokenizer",  # 定义类 LongformerTokenizer
    ],
    "models.longt5": ["LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongT5Config"],  # 定义变量 LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 LongT5Config
    "models.luke": [
        "LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LukeConfig",  # 定义类 LukeConfig
        "LukeTokenizer",  # 定义类 LukeTokenizer
    ],
    "models.lxmert": [
        "LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP
        "LxmertConfig",  # 定义类 LxmertConfig
        "LxmertTokenizer",  # 定义类 LxmertTokenizer
    ],
    "models.m2m_100": ["M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP", "M2M100Config"],  # 定义变量 M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 M2M100Config
    "models.marian": ["MarianConfig"],  # 定义类 MarianConfig
    "models.markuplm": [
        "MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP
        "MarkupLMConfig",  # 定义类 MarkupLMConfig
        "MarkupLMFeatureExtractor",  # 定义类 MarkupLMFeatureExtractor
        "MarkupLMProcessor",  # 定义类 MarkupLMProcessor
        "MarkupLMTokenizer",  # 定义类 MarkupLMTokenizer
    ],
    "models.mask2former": [
        "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
        "Mask2FormerConfig",  # 定义类 Mask2FormerConfig
    ],
    "models.maskformer": [
        "MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
        "MaskFormerConfig",  # 定义类 MaskFormerConfig
        "MaskFormerSwinConfig",  # 定义类 MaskFormerSwinConfig
    ],
    "models.mbart": ["MBartConfig"],  # 定义类 MBartConfig
    "models.mbart50": [],  # 空列表
    "models.mega": ["MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MegaConfig"],  # 定义变量 MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 MegaConfig
    "models.megatron_bert": [
        "MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
        "MegatronBertConfig",  # 定义类 MegatronBertConfig
    ],
    "models.megatron_gpt2": [],  # 空列表
    "models.mgp_str": [
        "MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义变量 MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP
        "MgpstrConfig",  # 定义类 MgpstrConfig
        "MgpstrProcessor",  # 定义类 MgpstrProcessor
        "MgpstrTokenizer",  # 定义类 MgpstrTokenizer
    ],
    "models.mistral": ["MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MistralConfig"],  # 定义变量 MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 MistralConfig
    "models.mixtral": ["MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP", "MixtralConfig"],  # 定义变量 MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, 类 MixtralConfig
    "models.mluke": [],  # 空列表
    "models.mobilebert": [
        "MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 MOBILEBERT 预训练配置文件的映射
        "MobileBertConfig",  # 定义了 MobileBertConfig 类
        "MobileBertTokenizer",  # 定义了 MobileBertTokenizer 类
    ],
    "models.mobilenet_v1": [
        "MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 MOBILENET_V1 预训练配置文件的映射
        "MobileNetV1Config",  # 定义了 MobileNetV1Config 类
    ],
    "models.mobilenet_v2": [
        "MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 MOBILENET_V2 预训练配置文件的映射
        "MobileNetV2Config",  # 定义了 MobileNetV2Config 类
    ],
    "models.mobilevit": ["MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MobileViTConfig"],  # 定义了 MOBILEVIT 预训练配置文件的映射和 MobileViTConfig 类
    "models.mobilevitv2": [
        "MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 MOBILEVITV2 预训练配置文件的映射
        "MobileViTV2Config",  # 定义了 MobileViTV2Config 类
    ],
    "models.mpnet": [
        "MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 MPNET 预训练配置文件的映射
        "MPNetConfig",  # 定义了 MPNetConfig 类
        "MPNetTokenizer",  # 定义了 MPNetTokenizer 类
    ],
    "models.mpt": ["MPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "MptConfig"],  # 定义了 MPT 预训练配置文件的映射和 MptConfig 类
    "models.mra": ["MRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "MraConfig"],  # 定义了 MRA 预训练配置文件的映射和 MraConfig 类
    "models.mt5": ["MT5Config"],  # 定义了 MT5Config 类
    "models.musicgen": [
        "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 MUSICGEN 预训练配置文件的映射
        "MusicgenConfig",  # 定义了 MusicgenConfig 类
        "MusicgenDecoderConfig",  # 定义了 MusicgenDecoderConfig 类
    ],
    "models.mvp": ["MvpConfig", "MvpTokenizer"],  # 定义了 MvpConfig 类和 MvpTokenizer 类
    "models.nat": ["NAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "NatConfig"],  # 定义了 NAT 预训练配置文件的映射和 NatConfig 类
    "models.nezha": ["NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP", "NezhaConfig"],  # 定义了 NEZHA 预训练配置文件的映射和 NezhaConfig 类
    "models.nllb": [],  # 空列表
    "models.nllb_moe": ["NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP", "NllbMoeConfig"],  # 定义了 NLLB_MOE 预训练配置文件的映射和 NllbMoeConfig 类
    "models.nougat": ["NougatProcessor"],  # 定义了 NougatProcessor 类
    "models.nystromformer": [
        "NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 NYSTROMFORMER 预训练配置文件的映射
        "NystromformerConfig",  # 定义了 NystromformerConfig 类
    ],
    "models.oneformer": [
        "ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 ONEFORMER 预训练配置文件的映射
        "OneFormerConfig",  # 定义了 OneFormerConfig 类
        "OneFormerProcessor",  # 定义了 OneFormerProcessor 类
    ],
    "models.openai": [
        "OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 OPENAI_GPT 预训练配置文件的映射
        "OpenAIGPTConfig",  # 定义了 OpenAIGPTConfig 类
        "OpenAIGPTTokenizer",  # 定义了 OpenAIGPTTokenizer 类
    ],
    "models.opt": ["OPTConfig"],  # 定义了 OPTConfig 类
    "models.owlv2": [
        "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 OWLV2 预训练配置文件的映射
        "Owlv2Config",  # 定义了 Owlv2Config 类
        "Owlv2Processor",  # 定义了 Owlv2Processor 类
        "Owlv2TextConfig",  # 定义了 Owlv2TextConfig 类
        "Owlv2VisionConfig",  # 定��了 Owlv2VisionConfig 类
    ],
    "models.owlvit": [
        "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 OWLVIT 预训练配置文件的映射
        "OwlViTConfig",  # 定义了 OwlViTConfig 类
        "OwlViTProcessor",  # 定义了 OwlViTProcessor 类
        "OwlViTTextConfig",  # 定义了 OwlViTTextConfig 类
        "OwlViTVisionConfig",  # 定义了 OwlViTVisionConfig 类
    ],
    "models.patchtsmixer": [
        "PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 PATCHTSMIXER 预训练配置文件的映射
        "PatchTSMixerConfig",  # 定义了 PatchTSMixerConfig 类
    ],
    "models.patchtst": ["PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP", "PatchTSTConfig"],  # 定义了 PATCHTST 预训练配置文件的映射和 PatchTSTConfig 类
    "models.pegasus": [
        "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 PEGASUS 预训练配置文件的映射
        "PegasusConfig",  # 定义了 PegasusConfig 类
        "PegasusTokenizer",  # 定义了 PegasusTokenizer 类
    ],
    "models.pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],  # 定义了 PEGASUS_X 预训练配置文件的映射和 PegasusXConfig 类
    "models.perceiver": [
        "PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义了 PERCEIVER 预训练配置文件的映射
        "PerceiverConfig",  # 定义了 PerceiverConfig 类
        "PerceiverTokenizer",  # 定义了 PerceiverTokenizer 类
    ],
    "models.persimmon": ["PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP", "PersimmonConfig"],  # 定义了 PERSIMMON 预训练配置文件的映射和 PersimmonConfig 类
    "models.phi": ["PHI_PRETRAINED_CONFIG_ARCHIVE_MAP", "PhiConfig"],  # 定义了 PHI 预训练配置文件的映射和 PhiConfig 类
    "models.phobert": ["PhobertTokenizer"],  # 定义了 PhobertTokenizer 类
    # PIX2STRUCT 模型相关的预训练配置文件映射
    "models.pix2struct": [
        "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # PIX2STRUCT 模型的配置类
        "Pix2StructConfig",
        # PIX2STRUCT 模型的处理器
        "Pix2StructProcessor",
        # PIX2STRUCT 文本配置类
        "Pix2StructTextConfig",
        # PIX2STRUCT 视觉配置类
        "Pix2StructVisionConfig",
    ],
    # PLBART 模型相关的预训练配置文件映射
    "models.plbart": ["PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "PLBartConfig"],
    # POOLFORMER 模型相关的预训练配置文件映射
    "models.poolformer": [
        "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # POOLFORMER 模型的配置类
        "PoolFormerConfig",
    ],
    # POP2PIANO 模型相关的预训练配置文件映射
    "models.pop2piano": [
        "POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # POP2PIANO 模型的配置类
        "Pop2PianoConfig",
    ],
    # PROPHETNET 模型相关的预训练配置文件映射
    "models.prophetnet": [
        "PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # PROPHETNET 模型的配置类
        "ProphetNetConfig",
        # PROPHETNET 模型的分词器
        "ProphetNetTokenizer",
    ],
    # PVT 模型相关的预训练配置文件映射
    "models.pvt": ["PVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtConfig"],
    # QDQBERT 模型相关的预训练配置文件映射
    "models.qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"],
    # QWEN2 模型相关的预训练配置文件映射
    "models.qwen2": [
        "QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # QWEN2 模型的配置类
        "Qwen2Config",
        # QWEN2 模型的分词器
        "Qwen2Tokenizer",
    ],
    # RAG 模型相关的配置类和工具
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    # REALM 模型相关的预训练配置文件映射
    "models.realm": [
        "REALM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # REALM 模型的配置类
        "RealmConfig",
        # REALM 模型的分词器
        "RealmTokenizer",
    ],
    # REFORMER 模型相关的预训练配置文件映射
    "models.reformer": ["REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "ReformerConfig"],
    # REGNET 模型相关的预训练配置文件映射
    "models.regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"],
    # REMBERT 模型相关的预训练配置文件映射
    "models.rembert": ["REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RemBertConfig"],
    # RESNET 模型相关的预训练配置文件映射
    "models.resnet": ["RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetConfig"],
    # ROBERTA 模型相关的预训练配置文件映射
    "models.roberta": [
        "ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # ROBERTA 模型的配置类
        "RobertaConfig",
        # ROBERTA 模型的分词器
        "RobertaTokenizer",
    ],
    # ROBERTA 模型的预层归一化相关的预训练配置文件映射
    "models.roberta_prelayernorm": [
        "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # ROBERTA 预层归一化模型的配置类
        "RobertaPreLayerNormConfig",
    ],
    # ROC_BERT 模型相关的预训练配置文件映射
    "models.roc_bert": [
        "ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # ROC_BERT 模型的配置类
        "RoCBertConfig",
        # ROC_BERT 模型的分词器
        "RoCBertTokenizer",
    ],
    # ROFORMER 模型相关的预训练配置文件映射
    "models.roformer": [
        "ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # ROFORMER 模型的配置类
        "RoFormerConfig",
        # ROFORMER 模型的分词器
        "RoFormerTokenizer",
    ],
    # RWKV 模型相关的预训练配置文件映射
    "models.rwkv": ["RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP", "RwkvConfig"],
    # SAM 模型相关的预训练配置文件映射
    "models.sam": [
        "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SAM 模型的配置类
        "SamConfig",
        # SAM 模型的蒙版解码器配置类
        "SamMaskDecoderConfig",
        # SAM 模型的处理器
        "SamProcessor",
        # SAM 模型的提示编码器配置类
        "SamPromptEncoderConfig",
        # SAM 模型的视觉配置类
        "SamVisionConfig",
    ],
    # SEAMLESS_M4T 模型相关的预训练配置文件映射
    "models.seamless_m4t": [
        "SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SEAMLESS_M4T 模型的配置类
        "SeamlessM4TConfig",
        # SEAMLESS_M4T 模型的特征提取器
        "SeamlessM4TFeatureExtractor",
        # SEAMLESS_M4T 模型的处理器
        "SeamlessM4TProcessor",
    ],
    # SEAMLESS_M4T_V2 模型相关的预训练配置文件映射
    "models.seamless_m4t_v2
    # models.siglip 模块的内容
    "models.siglip": [
        # SIGLIP 预训练配置文件存档映射
        "SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SiglipConfig 类
        "SiglipConfig",
        # SiglipProcessor 类
        "SiglipProcessor",
        # SiglipTextConfig 类
        "SiglipTextConfig",
        # SiglipTokenizer 类
        "SiglipTokenizer",
        # SiglipVisionConfig 类
        "SiglipVisionConfig",
    ],
    
    # models.speech_encoder_decoder 模块的内容
    "models.speech_encoder_decoder": [
        # SpeechEncoderDecoderConfig 类
        "SpeechEncoderDecoderConfig",
    ],
    
    # models.speech_to_text 模块的内容
    "models.speech_to_text": [
        # 语音转文本预训练配置文件存档映射
        "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # Speech2TextConfig 类
        "Speech2TextConfig",
        # Speech2TextFeatureExtractor 类
        "Speech2TextFeatureExtractor",
        # Speech2TextProcessor 类
        "Speech2TextProcessor",
    ],
    
    # models.speech_to_text_2 模块的内容
    "models.speech_to_text_2": [
        # 语音转文本2预训练配置文件存档映射
        "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # Speech2Text2Config 类
        "Speech2Text2Config",
        # Speech2Text2Processor 类
        "Speech2Text2Processor",
        # Speech2Text2Tokenizer 类
        "Speech2Text2Tokenizer",
    ],
    
    # models.speecht5 模块的内容
    "models.speecht5": [
        # SpeechT5 预训练配置文件存档映射
        "SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SpeechT5 HifiGan 预训练配置文件存档映射
        "SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP",
        # SpeechT5Config 类
        "SpeechT5Config",
        # SpeechT5FeatureExtractor 类
        "SpeechT5FeatureExtractor",
        # SpeechT5HifiGanConfig 类
        "SpeechT5HifiGanConfig",
        # SpeechT5Processor 类
        "SpeechT5Processor",
    ],
    
    # models.splinter 模块的内容
    "models.splinter": [
        # Splinter 预训练配置文件存档映射
        "SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SplinterConfig 类
        "SplinterConfig",
        # SplinterTokenizer 类
        "SplinterTokenizer",
    ],
    
    # models.squeezebert 模块的内容
    "models.squeezebert": [
        # SqueezeBert 预训练配置文件存档映射
        "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SqueezeBertConfig 类
        "SqueezeBertConfig",
        # SqueezeBertTokenizer 类
        "SqueezeBertTokenizer",
    ],
    
    # models.swiftformer 模块的内容
    "models.swiftformer": [
        # SwiftFormer 预训练配置文件存档映射
        "SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SwiftFormerConfig 类
        "SwiftFormerConfig",
    ],
    
    # models.swin 模块的内容
    "models.swin": [
        # Swin 预训练配置文件存档映射
        "SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SwinConfig 类
        "SwinConfig",
    ],
    
    # models.swin2sr 模块的内容
    "models.swin2sr": [
        # Swin2SR 预训练配置文件存档映射
        "SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # Swin2SRConfig 类
        "Swin2SRConfig",
    ],
    
    # models.swinv2 模块的内容
    "models.swinv2": [
        # Swinv2 预训练配置文件存档映射
        "SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # Swinv2Config 类
        "Swinv2Config",
    ],
    
    # models.switch_transformers 模块的内容
    "models.switch_transformers": [
        # SwitchTransformers 预训练配置文件存档映射
        "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # SwitchTransformersConfig 类
        "SwitchTransformersConfig",
    ],
    
    # models.t5 模块的内容
    "models.t5": [
        # T5 预训练配置文件存档映射
        "T5_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # T5Config 类
        "T5Config",
    ],
    
    # models.table_transformer 模块的内容
    "models.table_transformer": [
        # 表格变换器预训练配置文件存档映射
        "TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # TableTransformerConfig 类
        "TableTransformerConfig",
    ],
    
    # models.tapas 模块的内容
    "models.tapas": [
        # TAPAS 预训练配置文件存档映射
        "TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # TapasConfig 类
        "TapasConfig",
        # TapasTokenizer 类
        "TapasTokenizer",
    ],
    
    # models.time_series_transformer 模块的内容
    "models.time_series_transformer": [
        # 时间序列变换器预训练配置文件存档映射
        "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # TimeSeriesTransformerConfig 类
        "TimeSeriesTransformerConfig",
    ],
    
    # models.timesformer 模块的内容
    "models.timesformer": [
        # Timesformer 预训练配置文件存档映射
        "TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # TimesformerConfig 类
        "TimesformerConfig",
    ],
    
    # models.timm_backbone 模块的内容
    "models.timm_backbone": [
        # TimmBackboneConfig 类
        "TimmBackboneConfig",
    ],
    
    # models.trocr 模块的内容
    "models.trocr": [
        # TROCR 预训练配置文件存档映射
        "TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        # TrOCRConfig 类
        "TrOCRConfig",
        # TrOCRProcessor 类
        "TrOCRProcessor",
    ],
    
    #
    "models.unispeech_sat": [
        "UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 UNISPEECH_SAT 预训练配置文件映射
        "UniSpeechSatConfig",  # 定义 UniSpeechSatConfig 类
    ],
    "models.univnet": [
        "UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 UNIVNET 预训练配置文件映射
        "UnivNetConfig",  # 定义 UnivNetConfig 类
        "UnivNetFeatureExtractor",  # 定义 UnivNetFeatureExtractor 类
    ],
    "models.upernet": ["UperNetConfig"],  # 定义 UperNetConfig 类
    "models.videomae": ["VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VideoMAEConfig"],  # 定义 VIDEOMAE 预训练配置文件映射和 VideoMAEConfig 类
    "models.vilt": [
        "VILT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 VILT 预训练配置文件映射
        "ViltConfig",  # 定义 ViltConfig 类
        "ViltFeatureExtractor",  # 定义 ViltFeatureExtractor 类
        "ViltImageProcessor",  # 定义 ViltImageProcessor 类
        "ViltProcessor",  # 定义 ViltProcessor 类
    ],
    "models.vipllava": [
        "VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 VIPLLAVA 预训练配置文件映射
        "VipLlavaConfig",  # 定义 VipLlavaConfig 类
    ],
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],  # 定义 VisionEncoderDecoderConfig 类
    "models.vision_text_dual_encoder": [
        "VisionTextDualEncoderConfig",  # 定义 VisionTextDualEncoderConfig 类
        "VisionTextDualEncoderProcessor",  # 定义 VisionTextDualEncoderProcessor 类
    ],
    "models.visual_bert": [
        "VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 VISUAL_BERT 预训练配置文件映射
        "VisualBertConfig",  # 定义 VisualBertConfig 类
    ],
    "models.vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig"],  # 定义 VIT 预训练配置文件映射和 ViTConfig 类
    "models.vit_hybrid": [
        "VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 VIT_HYBRID 预训练配置文件映射
        "ViTHybridConfig",  # 定义 ViTHybridConfig 类
    ],
    "models.vit_mae": ["VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMAEConfig"],  # 定义 VIT_MAE 预训练配置文件映射和 ViTMAEConfig 类
    "models.vit_msn": ["VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMSNConfig"],  # 定义 VIT_MSN 预训练配置文件映射和 ViTMSNConfig 类
    "models.vitdet": ["VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitDetConfig"],  # 定义 VITDET 预训练配置文件映射和 VitDetConfig 类
    "models.vitmatte": ["VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitMatteConfig"],  # 定义 VITMATTE 预训练配置文件映射和 VitMatteConfig 类
    "models.vits": [
        "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 VITS 预训练配置文件映射
        "VitsConfig",  # 定义 VitsConfig 类
        "VitsTokenizer",  # 定义 VitsTokenizer 类
    ],
    "models.vivit": [
        "VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 VIVIT 预训练配置文件映射
        "VivitConfig",  # 定义 VivitConfig 类
    ],
    "models.wav2vec2": [
        "WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 WAV_2_VEC_2 预训练配置文件映射
        "Wav2Vec2Config",  # 定义 Wav2Vec2Config 类
        "Wav2Vec2CTCTokenizer",  # 定义 Wav2Vec2CTCTokenizer 类
        "Wav2Vec2FeatureExtractor",  # 定义 Wav2Vec2FeatureExtractor 类
        "Wav2Vec2Processor",  # 定义 Wav2Vec2Processor 类
        "Wav2Vec2Tokenizer",  # 定义 Wav2Vec2Tokenizer 类
    ],
    "models.wav2vec2_bert": [
        "WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 WAV2VEC2_BERT 预训练配置文件映射
        "Wav2Vec2BertConfig",  # 定义 Wav2Vec2BertConfig 类
        "Wav2Vec2BertProcessor",  # 定义 Wav2Vec2BertProcessor 类
    ],
    "models.wav2vec2_conformer": [
        "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 WAV2VEC2_CONFORMER 预训练配置文件映射
        "Wav2Vec2ConformerConfig",  # 定义 Wav2Vec2ConformerConfig 类
    ],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],  # 定义 Wav2Vec2PhonemeCTCTokenizer 类
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],  # 定义 Wav2Vec2ProcessorWithLM 类
    "models.wavlm": [
        "WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 WAVLM 预训练配置文件映射
        "WavLMConfig",  # 定义 WavLMConfig 类
    ],
    "models.whisper": [
        "WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 WHISPER 预训练配置文件映射
        "WhisperConfig",  # 定义 WhisperConfig 类
        "WhisperFeatureExtractor",  # 定义 WhisperFeatureExtractor 类
        "WhisperProcessor",  # 定义 WhisperProcessor 类
        "WhisperTokenizer",  # 定义 WhisperTokenizer 类
    ],
    "models.x_clip": [
        "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 定义 XCLIP 预训练配置文件映射
        "XCLIPConfig",  # 定义 XCLIPConfig 类
        "XCLIPProcessor",  # 定义 XCLIPProcessor 类
        "XCLIPTextConfig",  # 定义 XCLIPTextConfig 类
        "XCLIPVisionConfig",  # 定义 XCLIPVisionConfig 类
    ],
    "models.xglm": ["XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XGLMConfig"],  # 定义 XGLM 预训练配置文件映射和 XGLMConfig 类
    "models.xlm": ["XLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLMConfig", "XLMTokenizer"],  # 定义了模型 XLM 的相关信息，包括预训练配置映射、配置类和分词器类
    "models.xlm_prophetnet": [  # 模型 XLM-ProphetNet 相关信息
        "XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "XLMProphetNetConfig",  # XLM-ProphetNet 的配置类
    ],
    "models.xlm_roberta": [  # 模型 XLM-RoBERTa 相关信息
        "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "XLMRobertaConfig",  # XLM-RoBERTa 的配置类
    ],
    "models.xlm_roberta_xl": [  # 模型 XLM-RoBERTa-XL 相关信息
        "XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置映射
        "XLMRobertaXLConfig",  # XLM-RoBERTa-XL 的配置类
    ],
    "models.xlnet": ["XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "XLNetConfig"],  # 模型 XLNet 相关信息，包括预训练配置映射和配置类
    "models.xmod": ["XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP", "XmodConfig"],  # 模型 Xmod 相关信息，包括预训练配置映射和配置类
    "models.yolos": ["YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP", "YolosConfig"],  # 模型 YOLOS 相关信息，包括预训练配置映射和配置类
    "models.yoso": ["YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP", "YosoConfig"],  # 模型 YOSO 相关信息，包括预训练配置映射和配置类
    "onnx": [],  # ONNX 的相关信息为空列表
    "pipelines": [  # 定义了各种处理管道的类
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "Conversation",
        "ConversationalPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageSegmentationPipeline",
        "ImageToImagePipeline",
        "ImageToTextPipeline",
        "JsonPipelineDataFormat",
        "MaskGenerationPipeline",
        "NerPipeline",
        "ObjectDetectionPipeline",
        "PipedPipelineDataFormat",
        "Pipeline",
        "PipelineDataFormat",
        "QuestionAnsweringPipeline",
        "SummarizationPipeline",
        "TableQuestionAnsweringPipeline",
        "Text2TextGenerationPipeline",
        "TextClassificationPipeline",
        "TextGenerationPipeline",
        "TextToAudioPipeline",
        "TokenClassificationPipeline",
        "TranslationPipeline",
        "VideoClassificationPipeline",
        "VisualQuestionAnsweringPipeline",
        "ZeroShotAudioClassificationPipeline",
        "ZeroShotClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "pipeline",  # 管道类
    ],
    "processing_utils": ["ProcessorMixin"],  # 定义了处理工具的混合类
    "testing_utils": [],  # 测试工具相关信息为空列表
    "tokenization_utils": ["PreTrainedTokenizer"],  # 定义了分词工具的基类
    "tokenization_utils_base": [  # 分词工具基类相关信息
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "tools": [  # 工具相关信息，包括各种工具类和方法
        "Agent",
        "AzureOpenAiAgent",
        "HfAgent",
        "LocalAgent",
        "OpenAiAgent",
        "PipelineTool",
        "RemoteTool",
        "Tool",
        "launch_gradio_demo",
        "load_tool",
    ],
    "trainer_callback": [  # 训练回调相关信息，包括各种训练回调类和状态类
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    # 定义了一些模块和类的名称，用于在代码中引用
    "trainer_utils": [
        "EvalPrediction",  # 评估预测结果的类
        "IntervalStrategy",  # 训练策略的类
        "SchedulerType",  # 调度器类型
        "enable_full_determinism",  # 启用完全确定性
        "set_seed",  # 设置随机种子
    ],
    "training_args": ["TrainingArguments"],  # 训练参数类
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],  # 序列到序列训练参数类
    "training_args_tf": ["TFTrainingArguments"],  # TensorFlow训练参数类
    "utils": [
        "CONFIG_NAME",  # 配置文件名
        "MODEL_CARD_NAME",  # 模型卡片名称
        "PYTORCH_PRETRAINED_BERT_CACHE",  # PyTorch预训练BERT缓存
        "PYTORCH_TRANSFORMERS_CACHE",  # PyTorch Transformers缓存
        "SPIECE_UNDERLINE",  # 分词符号下划线
        "TF2_WEIGHTS_NAME",  # TensorFlow 2权重名称
        "TF_WEIGHTS_NAME",  # TensorFlow权重名称
        "TRANSFORMERS_CACHE",  # Transformers缓存
        "WEIGHTS_NAME",  # 权重名称
        "TensorType",  # 张量类型
        "add_end_docstrings",  # 添加结束文档字符串
        "add_start_docstrings",  # 添加开始文档字符串
        "is_apex_available",  # 是否可用Apex
        "is_bitsandbytes_available",  # 是否可用BitsAndBytes
        "is_datasets_available",  # 是否可用数据集
        "is_decord_available",  # 是否可用Decord
        "is_faiss_available",  # 是否可用Faiss
        "is_flax_available",  # 是否可用Flax
        "is_keras_nlp_available",  # 是否可用Keras NLP
        "is_phonemizer_available",  # 是否可用Phonemizer
        "is_psutil_available",  # 是否可用psutil
        "is_py3nvml_available",  # 是否可用py3nvml
        "is_pyctcdecode_available",  # 是否可用pyctcdecode
        "is_safetensors_available",  # 是否可用SafeTensors
        "is_scipy_available",  # 是否可用SciPy
        "is_sentencepiece_available",  # 是否可用SentencePiece
        "is_sklearn_available",  # 是否可用scikit-learn
        "is_speech_available",  # 是否可用语音
        "is_tensorflow_text_available",  # 是否可用TensorFlow文本
        "is_tf_available",  # 是否可用TensorFlow
        "is_timm_available",  # 是否可用timm
        "is_tokenizers_available",  # 是否可用Tokenizers
        "is_torch_available",  # 是否可用PyTorch
        "is_torch_neuroncore_available",  # 是否可用PyTorch NeuronCore
        "is_torch_npu_available",  # 是否可用PyTorch NPU
        "is_torch_tpu_available",  # 是否可用PyTorch TPU
        "is_torchvision_available",  # 是否可用PyTorch Vision
        "is_torch_xpu_available",  # 是否可用PyTorch XPU
        "is_vision_available",  # 是否可用视觉
        "logging",  # 日志记录
    ],
    "utils.quantization_config": ["AwqConfig", "BitsAndBytesConfig", "GPTQConfig"],  # 量化配置类
}
# 结束一个代码块

# 检查是否安装了 sentencepiece 库
try:
    # 如果未安装，抛出异常
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入 dummy_sentencepiece_objects 模块
    from .utils import dummy_sentencepiece_objects

    # 将 dummy_sentencepiece_objects 模块中非私有的对象添加到 _import_structure 中
    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    # 如果安装了 sentencepiece 库，则将相应的 Tokenizer 添加到对应的模型中
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
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
    _import_structure["models.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")

# 检查是否安装了 tokenizers 库
try:
    # 如果未安装，抛出异常
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入 dummy_tokenizers_objects 模块
    from .utils import dummy_tokenizers_objects

    # 将 dummy_tokenizers_objects 模块中非私有的对象添加到 _import_structure 中
    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # 如果安装了 tokenizers 库，则进行相应操作
    # Fast tokenizers structure
    # 将模块和类别添加到导入结构中
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.bart"].append("BartTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.bert"].append("BertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.big_bird"].append("BigBirdTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.blenderbot"].append("BlenderbotTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.blenderbot_small"].append("BlenderbotSmallTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.bloom"].append("BloomTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.clip"].append("CLIPTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.code_llama"].append("CodeLlamaTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.codegen"].append("CodeGenTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoderTokenizerFast",
            "DPRQuestionEncoderTokenizerFast",
            "DPRReaderTokenizerFast",
        ]
    )
    # 将模块和类别添加到导入结构中
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.gpt_neox"].append("GPTNeoXTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.gpt_neox_japanese"].append("GPTNeoXJapaneseTokenizer")
    # 将模块和类别添加到导入结构中
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.layoutlmv2"].append("LayoutLMv2TokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.layoutlmv3"].append("LayoutLMv3TokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.led"].append("LEDTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.markuplm"].append("MarkupLMTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.mbart50"].append("MBart50TokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    # 将模块和类别添加到导入结构中
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    # 将 "models.mt5" 的值添加到 _import_structure 字典中的键 "MT5TokenizerFast" 的列表中
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    # 将 "models.mvp" 的值添加到 _import_structure 字典中的键 "MvpTokenizerFast" 的列表中
    _import_structure["models.mvp"].append("MvpTokenizerFast")
    # 将 "models.nllb" 的值添加到 _import_structure 字典中的键 "NllbTokenizerFast" 的列表中
    _import_structure["models.nllb"].append("NllbTokenizerFast")
    # 将 "models.nougat" 的值添加到 _import_structure 字典中的键 "NougatTokenizerFast" 的列表中
    _import_structure["models.nougat"].append("NougatTokenizerFast")
    # 将 "models.openai" 的值添加到 _import_structure 字典中的键 "OpenAIGPTTokenizerFast" 的列表中
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    # 将 "models.pegasus" 的值添加到 _import_structure 字典中的键 "PegasusTokenizerFast" 的列表中
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    # 将 "models.qwen2" 的值添加到 _import_structure 字典中的键 "Qwen2TokenizerFast" 的列表中
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    # 将 "models.realm" 的值添加到 _import_structure 字典中的键 "RealmTokenizerFast" 的列表中
    _import_structure["models.realm"].append("RealmTokenizerFast")
    # 将 "models.reformer" 的值添加到 _import_structure 字典中的键 "ReformerTokenizerFast" 的列表中
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    # 将 "models.rembert" 的值添加到 _import_structure 字典中的键 "RemBertTokenizerFast" 的列表中
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    # 将 "models.roberta" 的值添加到 _import_structure 字典中的键 "RobertaTokenizerFast" 的列表中
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    # 将 "models.roformer" 的值添加到 _import_structure 字典中的键 "RoFormerTokenizerFast" 的列表中
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    # 将 "models.seamless_m4t" 的值添加到 _import_structure 字典中的键 "SeamlessM4TTokenizerFast" 的列表中
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    # 将 "models.splinter" 的值添加到 _import_structure 字典中的键 "SplinterTokenizerFast" 的列表中
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    # 将 "models.squeezebert" 的值添加到 _import_structure 字典中的键 "SqueezeBertTokenizerFast" 的列表中
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    # 将 "models.t5" 的值添加到 _import_structure 字典中的键 "T5TokenizerFast" 的列表中
    _import_structure["models.t5"].append("T5TokenizerFast")
    # 将 "models.whisper" 的值添加到 _import_structure 字典中的键 "WhisperTokenizerFast" 的列表中
    _import_structure["models.whisper"].append("WhisperTokenizerFast")
    # 将 "models.xglm" 的值添加到 _import_structure 字典中的键 "XGLMTokenizerFast" 的列表中
    _import_structure["models.xglm"].append("XGLMTokenizerFast")
    # 将 "models.xlm_roberta" 的值添加到 _import_structure 字典中的键 "XLMRobertaTokenizerFast" 的列表中
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    # 将 "models.xlnet" 的值添加到 _import_structure 字典中的键 "XLNetTokenizerFast" 的列表中
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    # 将 "tokenization_utils_fast" 的值设置为 _import_structure 字典中的键 "PreTrainedTokenizerFast" 的列表
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]
# 尝试检查是否句子拼接和分词器可用
try:
    # 如果句子拼接或分词器不可用，则引发可选依赖项不可用异常
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚拟的句子拼接和分词器对象
    from .utils import dummy_sentencepiece_and_tokenizers_objects

    # 将虚拟对象添加到导入结构中
    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # 如果句子拼接和分词器可用，则将慢速分词器转换相关对象添加到导入结构中
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

# Tensorflow-text-specific 对象
try:
    # 检查是否可用Tensorflow-text
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚拟Tensorflow-text对象
    from .utils import dummy_tensorflow_text_objects

    # 将虚拟对象添加到导入结构中
    _import_structure["utils.dummy_tensorflow_text_objects"] = [
        name for name in dir(dummy_tensorflow_text_objects) if not name.startswith("_")
    ]
else:
    # 如果Tensorflow-text可用，则将TFBertTokenizer添加到导入结构中的models.bert部分
    _import_structure["models.bert"].append("TFBertTokenizer")

# keras-nlp-specific 对象
try:
    # 检查是否可用keras-nlp
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚拟keras-nlp对象
    from .utils import dummy_keras_nlp_objects

    # 将虚拟对象添加到导入结构中
    _import_structure["utils.dummy_keras_nlp_objects"] = [
        name for name in dir(dummy_keras_nlp_objects) if not name.startswith("_")
    ]
else:
    # 如果keras-nlp可用，则将TFGPT2Tokenizer添加到导入结构中的models.gpt2部分
    _import_structure["models.gpt2"].append("TFGPT2Tokenizer")

# Vision-specific 对象
try:
    # 检查是否可用Vision
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 导入虚拟Vision对象
    from .utils import dummy_vision_objects

    # 将虚拟对象添加到导入结构中
    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    # 如果Vision可用，则将以下对象添加到相应的导入结构中
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
    # 将 "models.detr" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor"])
    # 将 "models.donut" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    # 将 "models.dpt" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    # 将 "models.efficientformer" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.efficientformer"].append("EfficientFormerImageProcessor")
    # 将 "models.efficientnet" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    # 将 "models.flava" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    # 将 "models.fuyu" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    # 将 "models.glpn" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    # 将 "models.idefics" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    # 将 "models.imagegpt" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    # 将 "models.layoutlmv2" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    # 将 "models.layoutlmv3" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    # 将 "models.levit" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    # 将 "models.mask2former" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    # 将 "models.maskformer" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    # 将 "models.mobilenet_v1" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    # 将 "models.mobilenet_v2" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    # 将 "models.mobilevit" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    # 将 "models.nougat" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.nougat"].append("NougatImageProcessor")
    # 将 "models.oneformer" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    # 将 "models.owlv2" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    # 将 "models.owlvit" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    # 将 "models.perceiver" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    # 将 "models.pix2struct" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    # 将 "models.poolformer" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    # 将 "models.pvt" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    # 将 "models.sam" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    # 将 "models.segformer" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    # 将 "models.siglip" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.siglip"].append("SiglipImageProcessor")
    # 将 "models.swin2sr" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    # 将 "models.tvlt" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.tvlt"].append("TvltImageProcessor")
    # 将 "models.tvp" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.tvp"].append("TvpImageProcessor")
    # 将 "models.videomae" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    # 将 "models.vilt" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    # 将 "models.vit" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    # 将 "models.vit_hybrid" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    # 将 "models.vitmatte" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    # 将 "models.vivit" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.vivit"].append("VivitImageProcessor")
    # 将 "models.yolos" 模块下的类添加到 _import_structure 字典中
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])
# 尝试导入 PyTorch 相关对象，如果 PyTorch 不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果 PyTorch 不可用，则导入虚拟的 PyTorch 对象
    from .utils import dummy_pt_objects

    # 将 dummy_pt_objects 模块中的非私有成员添加到 _import_structure 中
    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    # 如果 PyTorch 可用，则添加以下模块到 _import_structure 中
    _import_structure["activations"] = []
    _import_structure["benchmark.benchmark"] = ["PyTorchBenchmark"]
    _import_structure["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
    _import_structure["cache_utils"] = ["Cache", "DynamicCache", "SinkCache"]
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
    # 将以下 generation 模块中的成员添加到 _import_structure 中
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
            "top_k_top_p_filtering",
        ]
    )
    _import_structure["generation_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_utils"] = ["PreTrainedModel"]

    # PyTorch 模型结构
    # 将模块导入结构中的"models.albert"列表扩展，添加了一系列 ALBERT 模型相关的字符串
    _import_structure["models.albert"].extend(
        [
            "ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AlbertForMaskedLM",
            "AlbertForMultipleChoice",
            "AlbertForPreTraining",
            "AlbertForQuestionAnswering",
            "AlbertForSequenceClassification",
            "AlbertForTokenClassification",
            "AlbertModel",
            "AlbertPreTrainedModel",
            "load_tf_weights_in_albert",
        ]
    )
    
    # 将模块导入结构中的"models.align"列表扩展，添加了一系列 ALIGN 模型相关的字符串
    _import_structure["models.align"].extend(
        [
            "ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AlignModel",
            "AlignPreTrainedModel",
            "AlignTextModel",
            "AlignVisionModel",
        ]
    )
    
    # 将模块导入结构中的"models.altclip"列表扩展，添加了一系列 ALTCLIP 模型相关的字符串
    _import_structure["models.altclip"].extend(
        [
            "ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AltCLIPModel",
            "AltCLIPPreTrainedModel",
            "AltCLIPTextModel",
            "AltCLIPVisionModel",
        ]
    )
    
    # 将模块导入结构中的"models.audio_spectrogram_transformer"列表扩展，添加了一系列音频频谱变换器模型相关的字符串
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
            "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ASTForAudioClassification",
            "ASTModel",
            "ASTPreTrainedModel",
        ]
    )
    
    # 将模块导入结构中的"models.autoformer"列表扩展，添加了一系列 AUTOFORMER 模型相关的字符串
    _import_structure["models.autoformer"].extend(
        [
            "AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "AutoformerForPrediction",
            "AutoformerModel",
            "AutoformerPreTrainedModel",
        ]
    )
    
    # 将模块导入结构中的"models.bark"列表扩展，添加了一系列 BARK 模型相关的字符串
    _import_structure["models.bark"].extend(
        [
            "BARK_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BarkCausalModel",
            "BarkCoarseModel",
            "BarkFineModel",
            "BarkModel",
            "BarkPreTrainedModel",
            "BarkSemanticModel",
        ]
    )
    
    # 将模块导入结构中的"models.bart"列表扩展，添加了一系列 BART 模型相关的字符串
    _import_structure["models.bart"].extend(
        [
            "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BartForCausalLM",
            "BartForConditionalGeneration",
            "BartForQuestionAnswering",
            "BartForSequenceClassification",
            "BartModel",
            "BartPretrainedModel",
            "BartPreTrainedModel",
            "PretrainedBartModel",
        ]
    )
    
    # 将模块导入结构中的"models.beit"列表扩展，添加了一系列 BEIT 模型相关的字符串
    _import_structure["models.beit"].extend(
        [
            "BEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BeitBackbone",
            "BeitForImageClassification",
            "BeitForMaskedImageModeling",
            "BeitForSemanticSegmentation",
            "BeitModel",
            "BeitPreTrainedModel",
        ]
    )
    # 将 models.bert 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.bert"].extend(
        [
            "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertLayer",
            "BertLMHeadModel",
            "BertModel",
            "BertPreTrainedModel",
            "load_tf_weights_in_bert",
        ]
    )
    # 将 models.bert_generation 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",
            "BertGenerationEncoder",
            "BertGenerationPreTrainedModel",
            "load_tf_weights_in_bert_generation",
        ]
    )
    # 将 models.big_bird 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.big_bird"].extend(
        [
            "BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BigBirdForCausalLM",
            "BigBirdForMaskedLM",
            "BigBirdForMultipleChoice",
            "BigBirdForPreTraining",
            "BigBirdForQuestionAnswering",
            "BigBirdForSequenceClassification",
            "BigBirdForTokenClassification",
            "BigBirdLayer",
            "BigBirdModel",
            "BigBirdPreTrainedModel",
            "load_tf_weights_in_big_bird",
        ]
    )
    # 将 models.bigbird_pegasus 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.bigbird_pegasus"].extend(
        [
            "BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BigBirdPegasusForCausalLM",
            "BigBirdPegasusForConditionalGeneration",
            "BigBirdPegasusForQuestionAnswering",
            "BigBirdPegasusForSequenceClassification",
            "BigBirdPegasusModel",
            "BigBirdPegasusPreTrainedModel",
        ]
    )
    # 将 models.biogpt 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.biogpt"].extend(
        [
            "BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BioGptForCausalLM",
            "BioGptForSequenceClassification",
            "BioGptForTokenClassification",
            "BioGptModel",
            "BioGptPreTrainedModel",
        ]
    )
    # 将 models.bit 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.bit"].extend(
        [
            "BIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BitBackbone",
            "BitForImageClassification",
            "BitModel",
            "BitPreTrainedModel",
        ]
    )
    # 将 models.blenderbot 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.blenderbot"].extend(
        [
            "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlenderbotForCausalLM",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotPreTrainedModel",
        ]
    )
    # 将 models.blenderbot_small 模块中的指定类和常量添加到 _import_structure 字典中
    _import_structure["models.blenderbot_small"].extend(
        [
            "BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlenderbotSmallForCausalLM",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallPreTrainedModel",
        ]
    )
    # 将 models.blip 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.blip"].extend(
        [
            "BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BlipForConditionalGeneration",
            "BlipForImageTextRetrieval",
            "BlipForQuestionAnswering",
            "BlipModel",
            "BlipPreTrainedModel",
            "BlipTextModel",
            "BlipVisionModel",
        ]
    )
    # 将 models.blip_2 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.blip_2"].extend(
        [
            "BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Blip2ForConditionalGeneration",
            "Blip2Model",
            "Blip2PreTrainedModel",
            "Blip2QFormerModel",
            "Blip2VisionModel",
        ]
    )
    # 将 models.bloom 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.bloom"].extend(
        [
            "BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BloomForCausalLM",
            "BloomForQuestionAnswering",
            "BloomForSequenceClassification",
            "BloomForTokenClassification",
            "BloomModel",
            "BloomPreTrainedModel",
        ]
    )
    # 将 models.bridgetower 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.bridgetower"].extend(
        [
            "BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BridgeTowerForContrastiveLearning",
            "BridgeTowerForImageAndTextRetrieval",
            "BridgeTowerForMaskedLM",
            "BridgeTowerModel",
            "BridgeTowerPreTrainedModel",
        ]
    )
    # 将 models.bros 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.bros"].extend(
        [
            "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "BrosForTokenClassification",
            "BrosModel",
            "BrosPreTrainedModel",
            "BrosProcessor",
            "BrosSpadeEEForTokenClassification",
            "BrosSpadeELForTokenClassification",
        ]
    )
    # 将 models.camembert 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.camembert"].extend(
        [
            "CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CamembertForCausalLM",
            "CamembertForMaskedLM",
            "CamembertForMultipleChoice",
            "CamembertForQuestionAnswering",
            "CamembertForSequenceClassification",
            "CamembertForTokenClassification",
            "CamembertModel",
            "CamembertPreTrainedModel",
        ]
    )
    # 将 models.canine 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.canine"].extend(
        [
            "CANINE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CanineForMultipleChoice",
            "CanineForQuestionAnswering",
            "CanineForSequenceClassification",
            "CanineForTokenClassification",
            "CanineLayer",
            "CanineModel",
            "CaninePreTrainedModel",
            "load_tf_weights_in_canine",
        ]
    )
    # 将 models.chinese_clip 模块中的内容添加到 _import_structure 字典中
    _import_structure["models.chinese_clip"].extend(
        [
            "CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ChineseCLIPModel",
            "ChineseCLIPPreTrainedModel",
            "ChineseCLIPTextModel",
            "ChineseCLIPVisionModel",
        ]
    )
    # 将 models.clap 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.clip 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.clip"].extend(
        [
            "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CLIPModel",
            "CLIPPreTrainedModel",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
        ]
    )
    # 将 models.clipseg 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.clvp 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.codegen 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.codegen"].extend(
        [
            "CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    # 将 models.conditional_detr 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.conditional_detr"].extend(
        [
            "CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    # 将 models.convbert 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.convnext 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.convnext"].extend(
        [
            "CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    # 将 models.convnextv2 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.convnextv2"].extend(
        [
            "CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ConvNextV2Backbone",
            "ConvNextV2ForImageClassification",
            "ConvNextV2Model",
            "ConvNextV2PreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.cpmant"].extend(
        [
            "CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CpmAntForCausalLM",
            "CpmAntModel",
            "CpmAntPreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.ctrl"].extend(
        [
            "CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.cvt"].extend(
        [
            "CVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "CvtForImageClassification",
            "CvtModel",
            "CvtPreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.data2vec"].extend(
        [
            "DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Data2VecAudioForAudioFrameClassification",
            "Data2VecAudioForCTC",
            "Data2VecAudioForSequenceClassification",
            "Data2VecAudioForXVector",
            "Data2VecAudioModel",
            "Data2VecAudioPreTrainedModel",
            "Data2VecTextForCausalLM",
            "Data2VecTextForMaskedLM",
            "Data2VecTextForMultipleChoice",
            "Data2VecTextForQuestionAnswering",
            "Data2VecTextForSequenceClassification",
            "Data2VecTextForTokenClassification",
            "Data2VecTextModel",
            "Data2VecTextPreTrainedModel",
            "Data2VecVisionForImageClassification",
            "Data2VecVisionForSemanticSegmentation",
            "Data2VecVisionModel",
            "Data2VecVisionPreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.deberta"].extend(
        [
            "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DebertaForMaskedLM",
            "DebertaForQuestionAnswering",
            "DebertaForSequenceClassification",
            "DebertaForTokenClassification",
            "DebertaModel",
            "DebertaPreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.deberta_v2"].extend(
        [
            "DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DebertaV2ForMaskedLM",
            "DebertaV2ForMultipleChoice",
            "DebertaV2ForQuestionAnswering",
            "DebertaV2ForSequenceClassification",
            "DebertaV2ForTokenClassification",
            "DebertaV2Model",
            "DebertaV2PreTrainedModel",
        ]
    )
    # 将指定模块下的模型名称列表扩展，包括预训练模型的存档列表以及各种模型类
    _import_structure["models.decision_transformer"].extend(
        [
            "DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    # 将 models.deformable_detr 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deformable_detr"].extend(
        [
            "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    # 将 models.deit 模块下的指定内容添加到 _import_structure 字典中
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
    # 将 models.deprecated.mctct 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    # 将 models.deprecated.mmbt 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deprecated.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    # 将 models.deprecated.open_llama 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deprecated.open_llama"].extend(
        [
            "OpenLlamaForCausalLM",
            "OpenLlamaForSequenceClassification",
            "OpenLlamaModel",
            "OpenLlamaPreTrainedModel",
        ]
    )
    # 将 models.deprecated.retribert 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deprecated.retribert"].extend(
        [
            "RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RetriBertModel",
            "RetriBertPreTrainedModel",
        ]
    )
    # 将 models.deprecated.trajectory_transformer 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    # 将 models.deprecated.transfo_xl 模块下的指定内容添加到 _import_structure 字典中
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
    # 将 models.deprecated.van 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deprecated.van"].extend(
        [
            "VAN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    # 将 models.deta 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.deta"].extend(
        [
            "DETA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DetaForObjectDetection",
            "DetaModel",
            "DetaPreTrainedModel",
        ]
    )
    # 将 models.detr 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.detr"].extend(
        [
            "DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DetrForObjectDetection",
            "DetrForSegmentation",
            "DetrModel",
            "DetrPreTrainedModel",
        ]
    )
    # 将 models.dinat 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.dinat"].extend(
        [
            "DINAT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DinatBackbone",
            "DinatForImageClassification",
            "DinatModel",
            "DinatPreTrainedModel",
        ]
    )
    # 将 models.dinov2 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.dinov2"].extend(
        [
            "DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Dinov2Backbone",
            "Dinov2ForImageClassification",
            "Dinov2Model",
            "Dinov2PreTrainedModel",
        ]
    )
    # 将 models.distilbert 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.distilbert"].extend(
        [
            "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DistilBertForMaskedLM",
            "DistilBertForMultipleChoice",
            "DistilBertForQuestionAnswering",
            "DistilBertForSequenceClassification",
            "DistilBertForTokenClassification",
            "DistilBertModel",
            "DistilBertPreTrainedModel",
        ]
    )
    # 将 models.donut 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.donut"].extend(
        [
            "DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DonutSwinModel",
            "DonutSwinPreTrainedModel",
        ]
    )
    # 将 models.dpr 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.dpr"].extend(
        [
            "DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPRContextEncoder",
            "DPRPretrainedContextEncoder",
            "DPRPreTrainedModel",
            "DPRPretrainedQuestionEncoder",
            "DPRPretrainedReader",
            "DPRQuestionEncoder",
            "DPRReader",
        ]
    )
    # 将 models.dpt 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.dpt"].extend(
        [
            "DPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "DPTForDepthEstimation",
            "DPTForSemanticSegmentation",
            "DPTModel",
            "DPTPreTrainedModel",
        ]
    )
    # 将 models.efficientformer 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.efficientformer"].extend(
        [
            "EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EfficientFormerForImageClassification",
            "EfficientFormerForImageClassificationWithTeacher",
            "EfficientFormerModel",
            "EfficientFormerPreTrainedModel",
        ]
    )
    # 将 models.efficientnet 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.efficientnet"].extend(
        [
            "EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EfficientNetForImageClassification",
            "EfficientNetModel",
            "EfficientNetPreTrainedModel",
        ]
    )
    # 将 models.electra 模块下的指定内容添加到 _import_structure 字典中
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
    # 将 models.encodec 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.encodec"].extend(
        [
            "ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST",
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    # 将 "EncoderDecoderModel" 添加到 _import_structure 字典中的 "models.encoder_decoder" 键对应的值列表中
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.ernie" 键对应的值列表中
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
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.ernie_m" 键对应的值列表中
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
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.esm" 键对应的值列表中
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
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.falcon" 键对应的值列表中
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
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.fastspeech2_conformer" 键对应的值列表中
    _import_structure["models.fastspeech2_conformer"].extend(
        [
            "FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FastSpeech2ConformerHifiGan",
            "FastSpeech2ConformerModel",
            "FastSpeech2ConformerPreTrainedModel",
            "FastSpeech2ConformerWithHifiGan",
        ]
    )
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.flaubert" 键对应的值列表中
    _import_structure["models.flaubert"].extend(
        [
            "FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlaubertForMultipleChoice",
            "FlaubertForQuestionAnswering",
            "FlaubertForQuestionAnsweringSimple",
            "FlaubertForSequenceClassification",
            "FlaubertForTokenClassification",
            "FlaubertModel",
            "FlaubertPreTrainedModel",
            "FlaubertWithLMHeadModel",
        ]
    )
    
    # 将多个字符串依次添加到 _import_structure 字典中的 "models.flava" 键对应的值列表中
    _import_structure["models.flava"].extend(
        [
            "FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "FlavaForPreTraining",
            "FlavaImageCodebook",
            "FlavaImageModel",
            "FlavaModel",
            "FlavaMultimodalModel",
            "FlavaPreTrainedModel",
            "FlavaTextModel",
        ]
    )
    # 导入模块结构中的 "models.fnet"，扩展其中包含的模块和类
    _import_structure["models.fnet"].extend(
        [
            "FNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # FNet 预训练模型存档列表
            "FNetForMaskedLM",  # 用于 Masked Language Modeling 的 FNet 模型
            "FNetForMultipleChoice",  # 用于多选题的 FNet 模型
            "FNetForNextSentencePrediction",  # 用于下一个句子预测的 FNet 模型
            "FNetForPreTraining",  # 用于预训练的 FNet 模型
            "FNetForQuestionAnswering",  # 用于问答任务的 FNet 模型
            "FNetForSequenceClassification",  # 用于序列分类任务的 FNet 模型
            "FNetForTokenClassification",  # 用于标记分类任务的 FNet 模型
            "FNetLayer",  # FNet 模型的层
            "FNetModel",  # FNet 模型
            "FNetPreTrainedModel",  # FNet 预训练模型
        ]
    )
    
    # 导入模块结构中的 "models.focalnet"，扩展其中包含的模块和类
    _import_structure["models.focalnet"].extend(
        [
            "FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # FocalNet 预训练模型存档列表
            "FocalNetBackbone",  # FocalNet 模型的骨干网络
            "FocalNetForImageClassification",  # 用于图像分类的 FocalNet 模型
            "FocalNetForMaskedImageModeling",  # 用于图像蒙版建模的 FocalNet 模型
            "FocalNetModel",  # FocalNet 模型
            "FocalNetPreTrainedModel",  # FocalNet 预训练模型
        ]
    )
    
    # 导入模块结构中的 "models.fsmt"，扩展其中包含的模块和类
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    
    # 导入模块结构中的 "models.funnel"，扩展其中包含的模块和类
    _import_structure["models.funnel"].extend(
        [
            "FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",  # Funnel 预训练模型存档列表
            "FunnelBaseModel",  # Funnel 模型的基础模型
            "FunnelForMaskedLM",  # 用于 Masked Language Modeling 的 Funnel 模型
            "FunnelForMultipleChoice",  # 用于多选题的 Funnel 模型
            "FunnelForPreTraining",  # 用于预训练的 Funnel 模型
            "FunnelForQuestionAnswering",  # 用于问答任务的 Funnel 模型
            "FunnelForSequenceClassification",  # 用于序列分类任务的 Funnel 模型
            "FunnelForTokenClassification",  # 用于标记分类任务的 Funnel 模型
            "FunnelModel",  # Funnel 模型
            "FunnelPreTrainedModel",  # Funnel 预训练模型
            "load_tf_weights_in_funnel",  # 载入 Funnel 模型的 TensorFlow 权重
        ]
    )
    
    # 导入模块结构中的 "models.fuyu"，扩展其中包含的模块和类
    _import_structure["models.fuyu"].extend(["FuyuForCausalLM", "FuyuPreTrainedModel"])
    
    # 导入模块结构中的 "models.git"，扩展其中包含的模块和类
    _import_structure["models.git"].extend(
        [
            "GIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Git 预训练模型存档列表
            "GitForCausalLM",  # 用于因果语言建模的 Git 模型
            "GitModel",  # Git 模型
            "GitPreTrainedModel",  # Git 预训练模型
            "GitVisionModel",  # Git 视觉模型
        ]
    )
    
    # 导入模块结构中的 "models.glpn"，扩展其中包含的模块和类
    _import_structure["models.glpn"].extend(
        [
            "GLPN_PRETRAINED_MODEL_ARCHIVE_LIST",  # GLPN 预训练模型存档列表
            "GLPNForDepthEstimation",  # 用于深度估计的 GLPN 模型
            "GLPNModel",  # GLPN 模型
            "GLPNPreTrainedModel",  # GLPN 预训练模型
        ]
    )
    
    # 导入模块结构中的 "models.gpt2"，扩展其中包含的模块和类
    _import_structure["models.gpt2"].extend(
        [
            "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",  # GPT-2 预训练模型存档列表
            "GPT2DoubleHeadsModel",  # 双头 GPT-2 模型
            "GPT2ForQuestionAnswering",  # 用于问答任务的 GPT-2 模型
            "GPT2ForSequenceClassification",  # 用于序列分类任务的 GPT-2 模型
            "GPT2ForTokenClassification",  # 用于标记分类任务的 GPT-2 模型
            "GPT2LMHeadModel",  # GPT-2 语言模型头
            "GPT2Model",  # GPT-2 模型
            "GPT2PreTrainedModel",  # GPT-2 预训练模型
            "load_tf_weights_in_gpt2",  # 载入 GPT-2 模型的 TensorFlow 权重
        ]
    )
    
    # 导入模块结构中的 "models.gpt_bigcode"，扩展其中包含的模块和类
    _import_structure["models.gpt_bigcode"].extend(
        [
            "GPT_BIGCODE_PRETRAINED_MODEL
    # 将一系列模型的相关信息添加到_import_structure字典中，以便后续导入使用
    _import_structure["models.gpt_neo"].extend(
        [
            "GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST",  # GPT-Neo预训练模型的存档列表
            "GPTNeoForCausalLM",  # 用于因果语言建模的GPT-Neo模型
            "GPTNeoForQuestionAnswering",  # 用于问答任务的GPT-Neo模型
            "GPTNeoForSequenceClassification",  # 用于序列分类任务的GPT-Neo模型
            "GPTNeoForTokenClassification",  # 用于标记分类任务的GPT-Neo模型
            "GPTNeoModel",  # GPT-Neo的基础模型
            "GPTNeoPreTrainedModel",  # GPT-Neo的预训练模型基类
            "load_tf_weights_in_gpt_neo",  # 加载TensorFlow权重到GPT-Neo模型的函数
        ]
    )
    # 继续添加GPT-NeoX模型的相关信息到_import_structure字典中
    _import_structure["models.gpt_neox"].extend(
        [
            "GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST",  # GPT-NeoX预训练模型的存档列表
            "GPTNeoXForCausalLM",  # 用于因果语言建模的GPT-NeoX模型
            "GPTNeoXForQuestionAnswering",  # 用于问答任务的GPT-NeoX模型
            "GPTNeoXForSequenceClassification",  # 用于序列分类任务的GPT-NeoX模型
            "GPTNeoXForTokenClassification",  # 用于标记分类任务的GPT-NeoX模型
            "GPTNeoXLayer",  # GPT-NeoX的层
            "GPTNeoXModel",  # GPT-NeoX的基础模型
            "GPTNeoXPreTrainedModel",  # GPT-NeoX的预训练模型基类
        ]
    )
    # 继续添加GPT-NeoX日语模型的相关信息到_import_structure字典中
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",  # GPT-NeoX日语预训练模型的存档列表
            "GPTNeoXJapaneseForCausalLM",  # 用于因果语言建模的GPT-NeoX日语模型
            "GPTNeoXJapaneseLayer",  # GPT-NeoX日语的层
            "GPTNeoXJapaneseModel",  # GPT-NeoX日语的基础模型
            "GPTNeoXJapanesePreTrainedModel",  # GPT-NeoX日语的预训练模型基类
        ]
    )
    # 继续添加GPT-J模型的相关信息到_import_structure字典中
    _import_structure["models.gptj"].extend(
        [
            "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",  # GPT-J预训练模型的存档列表
            "GPTJForCausalLM",  # 用于因果语言建模的GPT-J模型
            "GPTJForQuestionAnswering",  # 用于问答任务的GPT-J模型
            "GPTJForSequenceClassification",  # 用于序列分类任务的GPT-J模型
            "GPTJModel",  # GPT-J的基础模型
            "GPTJPreTrainedModel",  # GPT-J的预训练模型基类
        ]
    )
    # 继续添加GPT-SAN日语模型的相关信息到_import_structure字典中
    _import_structure["models.gptsan_japanese"].extend(
        [
            "GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",  # GPT-SAN日语预训练模型的存档列表
            "GPTSanJapaneseForConditionalGeneration",  # 用于条件生成任务的GPT-SAN日语模型
            "GPTSanJapaneseModel",  # GPT-SAN日语的基础模型
            "GPTSanJapanesePreTrainedModel",  # GPT-SAN日语的预训练模型基类
        ]
    )
    # 继续添加Graphormer模型的相关信息到_import_structure字典中
    _import_structure["models.graphormer"].extend(
        [
            "GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # Graphormer预训练模型的存档列表
            "GraphormerForGraphClassification",  # 用于图分类任务的Graphormer模型
            "GraphormerModel",  # Graphormer的基础模型
            "GraphormerPreTrainedModel",  # Graphormer的预训练模型基类
        ]
    )
    # 继续添加GroupViT模型的相关信息到_import_structure字典中
    _import_structure["models.groupvit"].extend(
        [
            "GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # GroupViT预训练模型的存档列表
            "GroupViTModel",  # GroupViT的基础模型
            "GroupViTPreTrainedModel",  # GroupViT的预训练模型基类
            "GroupViTTextModel",  # 用于文本任务的GroupViT模型
            "GroupViTVisionModel",  # 用于视觉任务的GroupViT模型
        ]
    )
    # 继续添加Hubert模型的相关信息到_import_structure字典中
    _import_structure["models.hubert"].extend(
        [
            "HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Hubert预训练模型的存档列表
            "HubertForCTC",  # 用于CTC任务的Hubert模型
            "HubertForSequenceClassification",  # 用于序列分类任务的Hubert模型
            "HubertModel",  # Hubert的基础模型
            "HubertPreTrainedModel",  # Hubert的预训练模型基类
        ]
    )
    # 继续添加IBert模型的相关信息到_import_structure字
    # 导入结构的扩展，包含了模型的命名空间和预训练模型列表
    _import_structure["models.idefics"].extend(
        [
            "IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST",  # IDEFICS 预训练模型存档列表
            "IdeficsForVisionText2Text",  # IDEFICS 用于视觉文本到文本任务的模型
            "IdeficsModel",  # IDEFICS 模型
            "IdeficsPreTrainedModel",  # IDEFICS 预训练模型
            "IdeficsProcessor",  # IDEFICS 处理器
        ]
    )
    _import_structure["models.imagegpt"].extend(
        [
            "IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # IMAGEGPT 预训练模型存档列表
            "ImageGPTForCausalImageModeling",  # IMAGEGPT 用于因果图像建模任务的模型
            "ImageGPTForImageClassification",  # IMAGEGPT 用于图像分类任务的模型
            "ImageGPTModel",  # IMAGEGPT 模型
            "ImageGPTPreTrainedModel",  # IMAGEGPT 预训练模型
            "load_tf_weights_in_imagegpt",  # 加载 TensorFlow 权重到 IMAGEGPT 模型中
        ]
    )
    _import_structure["models.informer"].extend(
        [
            "INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # INFORMER 预训练模型存档列表
            "InformerForPrediction",  # INFORMER 用于预测任务的模型
            "InformerModel",  # INFORMER 模型
            "InformerPreTrainedModel",  # INFORMER 预训练模型
        ]
    )
    _import_structure["models.instructblip"].extend(
        [
            "INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # INSTRUCTBLIP 预训练模型存档列表
            "InstructBlipForConditionalGeneration",  # INSTRUCTBLIP 用于条件生成任务的模型
            "InstructBlipPreTrainedModel",  # INSTRUCTBLIP 预训练模型
            "InstructBlipQFormerModel",  # INSTRUCTBLIP QFormer 模型
            "InstructBlipVisionModel",  # INSTRUCTBLIP 视觉模型
        ]
    )
    _import_structure["models.jukebox"].extend(
        [
            "JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST",  # JUKEBOX 预训练模型存档列表
            "JukeboxModel",  # JUKEBOX 模型
            "JukeboxPreTrainedModel",  # JUKEBOX 预训练模型
            "JukeboxPrior",  # JUKEBOX Prior 模型
            "JukeboxVQVAE",  # JUKEBOX VQVAE 模型
        ]
    )
    _import_structure["models.kosmos2"].extend(
        [
            "KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST",  # KOSMOS2 预训练模型存档列表
            "Kosmos2ForConditionalGeneration",  # KOSMOS2 用于条件生成任务的模型
            "Kosmos2Model",  # KOSMOS2 模型
            "Kosmos2PreTrainedModel",  # KOSMOS2 预训练模型
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",  # LAYOUTLM 预训练模型存档列表
            "LayoutLMForMaskedLM",  # LAYOUTLM 用于遮蔽语言建模任务的模型
            "LayoutLMForQuestionAnswering",  # LAYOUTLM 用于问答任务的模型
            "LayoutLMForSequenceClassification",  # LAYOUTLM 用于序列分类任务的模型
            "LayoutLMForTokenClassification",  # LAYOUTLM 用于标记分类任务的模型
            "LayoutLMModel",  # LAYOUTLM 模型
            "LayoutLMPreTrainedModel",  # LAYOUTLM 预训练模型
        ]
    )
    _import_structure["models.layoutlmv2"].extend(
        [
            "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # LAYOUTLMV2 预训练模型存档列表
            "LayoutLMv2ForQuestionAnswering",  # LAYOUTLMV2 用于问答任务的模型
            "LayoutLMv2ForSequenceClassification",  # LAYOUTLMV2 用于序列分类任务的模型
            "LayoutLMv2ForTokenClassification",  # LAYOUTLMV2 用于标记分类任务的模型
            "LayoutLMv2Model",  # LAYOUTLMV2 模型
            "LayoutLMv2PreTrainedModel",  # LAYOUTLMV2 预训练模型
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",  # LAYOUTLMV3 预训练模型存档列表
            "LayoutLMv3ForQuestionAnswering",  # LAYOUTLMV3 用于问答任务的模型
            "LayoutLMv3ForSequenceClassification",  # LAYOUTLMV3 用于序列分类任务的模型
            "LayoutLMv3ForTokenClassification",  # LAYOUTLMV3 用于标记分类任务的模型
            "LayoutLMv3Model",  # LAYOUTLMV3 模型
            "LayoutLMv3PreTrainedModel",  # LAYOUTLMV3 预训练模型
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LED_PRETRAINED_MODEL_ARCHIVE_LIST",  # LED 预训练模型存档列表
            "LEDForConditionalGeneration",  # LED 用于条件生成任务的模型
            "LEDForQuestionAnswering",  # LED 用于问答任务的模型
            "LEDForSequenceClassification",  # LED 用于序列分类任务的模型
            "LEDModel",  # LED 模型
            "LEDPreTrainedModel",  # LED 预训练
    )
    # 扩展_import_structure字典中"models.levit"对应的值列表
    _import_structure["models.levit"].extend(
        [
            "LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # LEVIT模型的预训练模型存档列表
            "LevitForImageClassification",          # 用于图像分类的Levit模型
            "LevitForImageClassificationWithTeacher",  # 带有教师的图像分类Levit模型
            "LevitModel",                           # Levit模型
            "LevitPreTrainedModel",                 # Levit预训练模型
        ]
    )
    # 扩展_import_structure字典中"models.lilt"对应的值列表
    _import_structure["models.lilt"].extend(
        [
            "LILT_PRETRAINED_MODEL_ARCHIVE_LIST",   # LILT模型的预训练模型存档列表
            "LiltForQuestionAnswering",             # 用于问答的Lilt模型
            "LiltForSequenceClassification",       # 用于序列分类的Lilt模型
            "LiltForTokenClassification",           # 用于标记分类的Lilt模型
            "LiltModel",                            # Lilt模型
            "LiltPreTrainedModel",                  # Lilt预训练模型
        ]
    )
    # 扩展_import_structure字典中"models.llama"对应的值列表
    _import_structure["models.llama"].extend(
        [
            "LlamaForCausalLM",                     # 用于因果语言模型的Llama模型
            "LlamaForSequenceClassification",      # 用于序列分类的Llama模型
            "LlamaModel",                           # Llama模型
            "LlamaPreTrainedModel",                 # Llama预训练模型
        ]
    )
    # 扩展_import_structure字典中"models.llava"对应的值列表
    _import_structure["models.llava"].extend(
        [
            "LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",  # LLAVA模型的预训练模型存档列表
            "LlavaForConditionalGeneration",        # 用于条件生成的Llava模型
            "LlavaPreTrainedModel",                 # Llava预训练模型
            "LlavaProcessor",                       # Llava处理器
        ]
    )
    # 扩展_import_structure字典中"models.longformer"对应的值列表
    _import_structure["models.longformer"].extend(
        [
            "LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # LONGFORMER模型的预训练模型存档列表
            "LongformerForMaskedLM",                     # 用于掩码语言建模的Longformer模型
            "LongformerForMultipleChoice",               # 用于多项选择的Longformer模型
            "LongformerForQuestionAnswering",            # 用于问答的Longformer模型
            "LongformerForSequenceClassification",       # 用于序列分类的Longformer模型
            "LongformerForTokenClassification",          # 用于标记分类的Longformer模型
            "LongformerModel",                           # Longformer模型
            "LongformerPreTrainedModel",                 # Longformer预训练模型
            "LongformerSelfAttention",                   # Longformer自注意力机制
        ]
    )
    # 扩展_import_structure字典中"models.longt5"对应的值列表
    _import_structure["models.longt5"].extend(
        [
            "LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",  # LONGT5模型的预训练模型存档列表
            "LongT5EncoderModel",                    # LongT5编码器模型
            "LongT5ForConditionalGeneration",        # 用于条件生成的LongT5模型
            "LongT5Model",                           # LongT5模型
            "LongT5PreTrainedModel",                 # LongT5预训练模型
        ]
    )
    # 扩展_import_structure字典中"models.luke"对应的值列表
    _import_structure["models.luke"].extend(
        [
            "LUKE_PRETRAINED_MODEL_ARCHIVE_LIST",   # LUKE模型的预训练模型存档列表
            "LukeForEntityClassification",          # 用于实体分类的Luke模型
            "LukeForEntityPairClassification",      # 用于实体对分类的Luke模型
            "LukeForEntitySpanClassification",      # 用于实体跨度分类的Luke模型
            "LukeForMaskedLM",                      # 用于掩码语言建模的Luke模型
            "LukeForMultipleChoice",                # 用于多项选择的Luke模型
            "LukeForQuestionAnswering",             # 用于问答的Luke模型
            "LukeForSequenceClassification",        # 用于序列分类的Luke模型
            "LukeForTokenClassification",           # 用于标记分类的Luke模型
            "LukeModel",                            # Luke模型
            "LukePreTrainedModel",                  # Luke预训练模型
        ]
    )
    # 扩展_import_structure字典中"models.lxmert"对应的值列表
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",                        # Lxmert编码器
            "LxmertForPreTraining",                 # 用于预训练的Lxmert模型
            "LxmertForQuestionAnswering",           # 用于问答的Lxmert模型
            "LxmertModel",                          # Lxmert模型
            "LxmertPreTrainedModel",                # Lxmert预训练模型
            "LxmertVisualFeatureEncoder",           # Lxmert视觉特征编码器
            "LxmertXLayer",                         # Lxmert X层
        ]
    )
    # 扩展_import_structure字典中"models.m2m_100"对应的值列表
    _import_structure["models.m2m_100"].extend(
        [
            "M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST",  # M2M_100模型的预训练模型存档列表
            "M2M100ForConditionalGeneration",         # 用于条件生成的M2M100模型
            "
    # 将模块 "models.markuplm" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.markuplm"].extend(
        [
            "MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型归档列表
            "MarkupLMForQuestionAnswering",            # 用于问题回答的 MarkupLM 模型
            "MarkupLMForSequenceClassification",      # 用于序列分类的 MarkupLM 模型
            "MarkupLMForTokenClassification",          # 用于标记分类的 MarkupLM 模型
            "MarkupLMModel",                           # MarkupLM 模型
            "MarkupLMPreTrainedModel",                 # MarkupLM 预训练模型
        ]
    )
    # 将模块 "models.mask2former" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.mask2former"].extend(
        [
            "MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST",   # 预训练模型归档列表
            "Mask2FormerForUniversalSegmentation",         # 用于通用分割的 Mask2Former 模型
            "Mask2FormerModel",                            # Mask2Former 模型
            "Mask2FormerPreTrainedModel",                  # Mask2Former 预训练模型
        ]
    )
    # 将模块 "models.maskformer" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.maskformer"].extend(
        [
            "MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",    # 预训练模型归档列表
            "MaskFormerForInstanceSegmentation",           # 用于实例分割的 MaskFormer 模型
            "MaskFormerModel",                             # MaskFormer 模型
            "MaskFormerPreTrainedModel",                   # MaskFormer 预训练模型
            "MaskFormerSwinBackbone",                      # MaskFormer Swin 骨干模型
        ]
    )
    # 将模块 "models.mbart" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",                        # 用于因果语言建模的 MBart 模型
            "MBartForConditionalGeneration",           # 用于条件生成的 MBart 模型
            "MBartForQuestionAnswering",               # 用于问题回答的 MBart 模型
            "MBartForSequenceClassification",          # 用于序列分类的 MBart 模型
            "MBartModel",                              # MBart 模型
            "MBartPreTrainedModel",                    # MBart 预训练模型
        ]
    )
    # 将模块 "models.mega" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.mega"].extend(
        [
            "MEGA_PRETRAINED_MODEL_ARCHIVE_LIST",      # 预训练模型归档列表
            "MegaForCausalLM",                         # 用于因果语言建模的 Mega 模型
            "MegaForMaskedLM",                         # 用于遮蔽语言建模的 Mega 模型
            "MegaForMultipleChoice",                   # 用于多选题的 Mega 模型
            "MegaForQuestionAnswering",                # 用于问题回答的 Mega 模型
            "MegaForSequenceClassification",           # 用于序列分类的 Mega 模型
            "MegaForTokenClassification",              # 用于标记分类的 Mega 模型
            "MegaModel",                               # Mega 模型
            "MegaPreTrainedModel",                     # Mega 预训练模型
        ]
    )
    # 将模块 "models.megatron_bert" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.megatron_bert"].extend(
        [
            "MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",          # 预训练模型归档列表
            "MegatronBertForCausalLM",                             # 用于因果语言建模的 Megatron Bert 模型
            "MegatronBertForMaskedLM",                             # 用于遮蔽语言建模的 Megatron Bert 模型
            "MegatronBertForMultipleChoice",                       # 用于多选题的 Megatron Bert 模型
            "MegatronBertForNextSentencePrediction",                # 用于下一句预测的 Megatron Bert 模型
            "MegatronBertForPreTraining",                           # 用于预训练的 Megatron Bert 模型
            "MegatronBertForQuestionAnswering",                     # 用于问题回答的 Megatron Bert 模型
            "MegatronBertForSequenceClassification",               # 用于序列分类的 Megatron Bert 模型
            "MegatronBertForTokenClassification",                   # 用于标记分类的 Megatron Bert 模型
            "MegatronBertModel",                                    # Megatron Bert 模型
            "MegatronBertPreTrainedModel",                          # Megatron Bert 预训练模型
        ]
    )
    # 将模块 "models.mgp_str" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.mgp_str"].extend(
        [
            "MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST",              # 预训练模型归档列表
            "MgpstrForSceneTextRecognition",                     # 用于场景文本识别的 Mgpstr 模型
            "MgpstrModel",                                        # Mgpstr 模型
            "MgpstrPreTrainedModel",                              # Mgpstr 预训练模型
        ]
    )
    # 将模块 "models.mistral" 中的各项内容添加到 _import_structure 字典中
    _import_structure["models.mistral"].extend(
        [
            "MistralForCausalLM",                           # 用于因果语言建模的 Mistral 模型
            "MistralForSequenceClassification",             # 用于序列分类的 Mistral 模型
            "MistralModel",                                 # Mistral 模型
            "MistralPreTrainedModel",                       # Mistral 预训练模型
        ]
    )
    # 将
    # 将模块"models.mobilebert"的内容扩展到_import_structure字典中
    _import_structure["models.mobilebert"].extend(
        [
            "MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 移动BERT的预训练模型归档列表
            "MobileBertForMaskedLM",  # 用于遮盖语言建模的移动BERT模型
            "MobileBertForMultipleChoice",  # 用于多项选择任务的移动BERT模型
            "MobileBertForNextSentencePrediction",  # 用于下一个句子预测任务的移动BERT模型
            "MobileBertForPreTraining",  # 用于预训练任务的移动BERT模型
            "MobileBertForQuestionAnswering",  # 用于问答任务的移动BERT模型
            "MobileBertForSequenceClassification",  # 用于序列分类任务的移动BERT模型
            "MobileBertForTokenClassification",  # 用于令牌分类任务的移动BERT模型
            "MobileBertLayer",  # 移动BERT的层模型
            "MobileBertModel",  # 移动BERT的模型
            "MobileBertPreTrainedModel",  # 移动BERT的预训练模型
            "load_tf_weights_in_mobilebert",  # 加载TensorFlow权重的函数，用于移动BERT
        ]
    )
    
    # 将模块"models.mobilenet_v1"的内容扩展到_import_structure字典中
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST",  # 移动网V1的预训练模型归档列表
            "MobileNetV1ForImageClassification",  # 用于图像分类任务的移动网V1模型
            "MobileNetV1Model",  # 移动网V1的模型
            "MobileNetV1PreTrainedModel",  # 移动网V1的预训练模型
            "load_tf_weights_in_mobilenet_v1",  # 加载TensorFlow权重的函数，用于移动网V1
        ]
    )
    
    # 将模块"models.mobilenet_v2"的内容扩展到_import_structure字典中
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST",  # 移动网V2的预训练模型归档列表
            "MobileNetV2ForImageClassification",  # 用于图像分类任务的移动网V2模型
            "MobileNetV2ForSemanticSegmentation",  # 用于语义分割任务的移动网V2模型
            "MobileNetV2Model",  # 移动网V2的模型
            "MobileNetV2PreTrainedModel",  # 移动网V2的预训练模型
            "load_tf_weights_in_mobilenet_v2",  # 加载TensorFlow权重的函数，用于移动网V2
        ]
    )
    
    # 将模块"models.mobilevit"的内容扩展到_import_structure字典中
    _import_structure["models.mobilevit"].extend(
        [
            "MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 移动ViT的预训练模型归档列表
            "MobileViTForImageClassification",  # 用于图像分类任务的移动ViT模型
            "MobileViTForSemanticSegmentation",  # 用于语义分割任务的移动ViT模型
            "MobileViTModel",  # 移动ViT的模型
            "MobileViTPreTrainedModel",  # 移动ViT的预训练模型
        ]
    )
    
    # 将模块"models.mobilevitv2"的内容扩展到_import_structure字典中
    _import_structure["models.mobilevitv2"].extend(
        [
            "MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # 移动ViT V2的预训练模型归档列表
            "MobileViTV2ForImageClassification",  # 用于图像分类任务的移动ViT V2模型
            "MobileViTV2ForSemanticSegmentation",  # 用于语义分割任务的移动ViT V2模型
            "MobileViTV2Model",  # 移动ViT V2的模型
            "MobileViTV2PreTrainedModel",  # 移动ViT V2的预训练模型
        ]
    )
    
    # 将模块"models.mpnet"的内容扩展到_import_structure字典中
    _import_structure["models.mpnet"].extend(
        [
            "MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # MPNet的预训练模型归档列表
            "MPNetForMaskedLM",  # 用于遮盖语言建模的MPNet模型
            "MPNetForMultipleChoice",  # 用于多项选择任务的MPNet模型
            "MPNetForQuestionAnswering",  # 用于问答任务的MPNet模型
            "MPNetForSequenceClassification",  # 用于序列分类任务的MPNet模型
            "MPNetForTokenClassification",  # 用于令牌分类任务的MPNet模型
            "MPNetLayer",  # MPNet的层模型
            "MPNetModel",  # MPNet的模型
            "MPNetPreTrainedModel",  # MPNet的预训练模型
        ]
    )
    
    # 将模块"models.mpt"的内容扩展到_import_structure字典中
    _import_structure["models.mpt"].extend(
        [
            "MPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # MPT的预训练模型归档列表
            "MptForCausalLM",  # 用于因果语言建模的MPT模型
            "MptForQuestionAnswering",  # 用于问答任务的MPT模型
            "MptForSequenceClassification",  # 用于序列分类任务的MPT模型
            "MptForTokenClassification",  # 用于令牌分类任务的MPT模型
            "MptModel",  # MPT的模型
    # 导入结构的模块路径为 "models.mt5"，扩展该路径下的模块列表
    _import_structure["models.mt5"].extend(
        [
            "MT5EncoderModel",  # MT5 编码器模型
            "MT5ForConditionalGeneration",  # 用于条件生成的 MT5 模型
            "MT5ForQuestionAnswering",  # 用于问答任务的 MT5 模型
            "MT5ForSequenceClassification",  # 用于序列分类任务的 MT5 模型
            "MT5Model",  # MT5 模型
            "MT5PreTrainedModel",  # MT5 预训练模型
        ]
    )
    # 导入结构的模块路径为 "models.musicgen"，扩展该路径下的模块列表
    _import_structure["models.musicgen"].extend(
        [
            "MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST",  # MUSICGEN 预训练模型存档列表
            "MusicgenForCausalLM",  # 用于因果语言建模的 MUSICGEN 模型
            "MusicgenForConditionalGeneration",  # 用于条件生成的 MUSICGEN 模型
            "MusicgenModel",  # MUSICGEN 模型
            "MusicgenPreTrainedModel",  # MUSICGEN 预训练模型
            "MusicgenProcessor",  # MUSICGEN 处理器
        ]
    )
    # 导入结构的模块路径为 "models.mvp"，扩展该路径下的模块列表
    _import_structure["models.mvp"].extend(
        [
            "MVP_PRETRAINED_MODEL_ARCHIVE_LIST",  # MVP 预训练模型存档列表
            "MvpForCausalLM",  # 用于因果语言建模的 MVP 模型
            "MvpForConditionalGeneration",  # 用于条件生成的 MVP 模型
            "MvpForQuestionAnswering",  # 用于问答任务的 MVP 模型
            "MvpForSequenceClassification",  # 用于序列分类任务的 MVP 模型
            "MvpModel",  # MVP 模型
            "MvpPreTrainedModel",  # MVP 预训练模型
        ]
    )
    # 导入结构的模块路径为 "models.nat"，扩展该路径下的模块列表
    _import_structure["models.nat"].extend(
        [
            "NAT_PRETRAINED_MODEL_ARCHIVE_LIST",  # NAT 预训练模型存档列表
            "NatBackbone",  # NAT 模型的主干网络
            "NatForImageClassification",  # 用于图像分类的 NAT 模型
            "NatModel",  # NAT 模型
            "NatPreTrainedModel",  # NAT 预训练模型
        ]
    )
    # 导入结构的模块路径为 "models.nezha"，扩展该路径下的模块列表
    _import_structure["models.nezha"].extend(
        [
            "NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST",  # NEZHA 预训练模型存档列表
            "NezhaForMaskedLM",  # 用于遮蔽语言建模的 NEZHA 模型
            "NezhaForMultipleChoice",  # 用于多项选择任务的 NEZHA 模型
            "NezhaForNextSentencePrediction",  # 用于下一句预测任务的 NEZHA 模型
            "NezhaForPreTraining",  # 用于预训练的 NEZHA 模型
            "NezhaForQuestionAnswering",  # 用于问答任务的 NEZHA 模型
            "NezhaForSequenceClassification",  # 用于序列分类任务的 NEZHA 模型
            "NezhaForTokenClassification",  # 用于标记分类任务的 NEZHA 模型
            "NezhaModel",  # NEZHA 模型
            "NezhaPreTrainedModel",  # NEZHA 预训练模型
        ]
    )
    # 导入结构的模块路径为 "models.nllb_moe"，扩展该路径下的模块列表
    _import_structure["models.nllb_moe"].extend(
        [
            "NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST",  # NLLB_MOE 预训练模型存档列表
            "NllbMoeForConditionalGeneration",  # 用于条件生成的 NLLB_MOE 模型
            "NllbMoeModel",  # NLLB_MOE 模型
            "NllbMoePreTrainedModel",  # NLLB_MOE 预训练模型
            "NllbMoeSparseMLP",  # NLLB_MOE 稀疏 MLP
            "NllbMoeTop2Router",  # NLLB_MOE Top-2 路由器
        ]
    )
    # 导入结构的模块路径为 "models.nystromformer"，扩展该路径下的模块列表
    _import_structure["models.nystromformer"].extend(
        [
            "NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # NYSTROMFORMER 预训练模型存档列表
            "NystromformerForMaskedLM",  # 用于遮蔽语言建模的 NYSTROMFORMER 模型
            "NystromformerForMultipleChoice",  # 用于多项选择任务的 NYSTROMFORMER 模型
            "NystromformerForQuestionAnswering",  # 用于问答任务的 NYSTROMFORMER 模型
            "NystromformerForSequenceClassification",  # 用于序列分类任务的 NYSTROMFORMER 模型
            "NystromformerForTokenClassification",  # 用于标记分类任务的 NYSTROMFORMER 模型
            "NystromformerLayer",  # NYSTROMFORMER 层
            "NystromformerModel",  # NYSTROMFORMER 模型
            "NystromformerPreTrainedModel",  # NYSTROMFORMER
    # 扩展 models.openai 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.openai"].extend(
        [
            "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # OpenAI GPT 预训练模型的存档列表
            "OpenAIGPTDoubleHeadsModel",  # OpenAI GPT 双头模型
            "OpenAIGPTForSequenceClassification",  # 用于序列分类的 OpenAI GPT 模型
            "OpenAIGPTLMHeadModel",  # OpenAI GPT 语言模型头模型
            "OpenAIGPTModel",  # OpenAI GPT 模型
            "OpenAIGPTPreTrainedModel",  # OpenAI GPT 预训练模型基类
            "load_tf_weights_in_openai_gpt",  # 在 OpenAI GPT 中加载 TensorFlow 权重的函数
        ]
    )
    # 扩展 models.opt 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.opt"].extend(
        [
            "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # OPT 模型预训练模型的存档列表
            "OPTForCausalLM",  # 用于因果语言建模的 OPT 模型
            "OPTForQuestionAnswering",  # 用于问答任务的 OPT 模型
            "OPTForSequenceClassification",  # 用于序列分类任务的 OPT 模型
            "OPTModel",  # OPT 模型
            "OPTPreTrainedModel",  # OPT 预训练模型基类
        ]
    )
    # 扩展 models.owlv2 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.owlv2"].extend(
        [
            "OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # OWLV2 模型预训练模型的存档列表
            "Owlv2ForObjectDetection",  # 用于目标检测任务的 OWLV2 模型
            "Owlv2Model",  # OWLV2 模型
            "Owlv2PreTrainedModel",  # OWLV2 预训练模型基类
            "Owlv2TextModel",  # 用于文本任务的 OWLV2 模型
            "Owlv2VisionModel",  # 用于视觉任务的 OWLV2 模型
        ]
    )
    # 扩展 models.owlvit 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.owlvit"].extend(
        [
            "OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST",  # OWLVIT 模型预训练模型的存档列表
            "OwlViTForObjectDetection",  # 用于目标检测任务的 OWLVIT 模型
            "OwlViTModel",  # OWLVIT 模型
            "OwlViTPreTrainedModel",  # OWLVIT 预训练模型基类
            "OwlViTTextModel",  # 用于文本任务的 OWLVIT 模型
            "OwlViTVisionModel",  # 用于视觉任务的 OWLVIT 模型
        ]
    )
    # 扩展 models.patchtsmixer 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.patchtsmixer"].extend(
        [
            "PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST",  # PATCHTSMIXER 模型预训练模型的存档列表
            "PatchTSMixerForPrediction",  # 用于预测任务的 PATCHTSMIXER 模型
            "PatchTSMixerForPretraining",  # 用于预训练任务的 PATCHTSMIXER 模型
            "PatchTSMixerForRegression",  # 用于回归任务的 PATCHTSMIXER 模型
            "PatchTSMixerForTimeSeriesClassification",  # 用于时间序列分类任务的 PATCHTSMIXER 模型
            "PatchTSMixerModel",  # PATCHTSMIXER 模型
            "PatchTSMixerPreTrainedModel",  # PATCHTSMIXER 预训练模型基类
        ]
    )
    # 扩展 models.patchtst 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.patchtst"].extend(
        [
            "PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST",  # PATCHTST 模型预训练模型的存档列表
            "PatchTSTForClassification",  # 用于分类任务的 PATCHTST 模型
            "PatchTSTForPrediction",  # 用于预测任务的 PATCHTST 模型
            "PatchTSTForPretraining",  # 用于预训练任务的 PATCHTST 模型
            "PatchTSTForRegression",  # 用于回归任务的 PATCHTST 模型
            "PatchTSTModel",  # PATCHTST 模型
            "PatchTSTPreTrainedModel",  # PATCHTST 预训练模型基类
        ]
    )
    # 扩展 models.pegasus 模块的导入结构，添加指定的模型类
    _import_structure["models.pegasus"].extend(
        [
            "PegasusForCausalLM",  # 用于因果语言建模的 PEGASUS 模型
            "PegasusForConditionalGeneration",  # 用于条件生成任务的 PEGASUS 模型
            "PegasusModel",  # PEGASUS 模型
            "PegasusPreTrainedModel",  # PEGASUS 预训练模型基类
        ]
    )
    # 扩展 models.pegasus_x 模块的导入结构，添加指定的模型类和常量列表
    _import_structure["models.pegasus_x"].extend(
        [
            "PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",  # PEGASUS_X 模型预训练模型的存档列表
            "PegasusXForConditionalGeneration",  # 用于条件生成任务的 PEGASUS_X 模型
            "PegasusXModel",
    # 将 models.perceiver 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.persimmon 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.persimmon"].extend(
        [
            "PersimmonForCausalLM",
            "PersimmonForSequenceClassification",
            "PersimmonModel",
            "PersimmonPreTrainedModel",
        ]
    )
    # 将 models.phi 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.pix2struct 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.pix2struct"].extend(
        [
            "PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    # 将 models.plbart 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.poolformer 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.poolformer"].extend(
        [
            "POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    # 将 models.pop2piano 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.pop2piano"].extend(
        [
            "POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    # 将 models.prophetnet 模块中的指定内容添加到 _import_structure 字典中
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
    # 将 models.pvt 模块中的指定内容添加到 _import_structure 字典中
    _import_structure["models.pvt"].extend(
        [
            "PVT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    # 导入结构模块下的 qdqbert 模块，并扩展其列表
    _import_structure["models.qdqbert"].extend(
        [
            "QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型档案列表
            "QDQBertForMaskedLM",  # 用于掩码语言建模任务的 QDQBert 模型
            "QDQBertForMultipleChoice",  # 用于多项选择任务的 QDQBert 模型
            "QDQBertForNextSentencePrediction",  # 用于下一句预测任务的 QDQBert 模型
            "QDQBertForQuestionAnswering",  # 用于问答任务的 QDQBert 模型
            "QDQBertForSequenceClassification",  # 用于序列分类任务的 QDQBert 模型
            "QDQBertForTokenClassification",  # 用于标记分类任务的 QDQBert 模型
            "QDQBertLayer",  # QDQBert 的层
            "QDQBertLMHeadModel",  # 带有语言模型头的 QDQBert 模型
            "QDQBertModel",  # QDQBert 模型
            "QDQBertPreTrainedModel",  # QDQBert 预训练模型
            "load_tf_weights_in_qdqbert",  # 加载 QDQBert 模型中的 TensorFlow 权重
        ]
    )
    # 导入结构模块下的 qwen2 模块，并扩展其列表
    _import_structure["models.qwen2"].extend(
        [
            "Qwen2ForCausalLM",  # 用于因果语言建模任务的 Qwen2 模型
            "Qwen2ForSequenceClassification",  # 用于序列分类任务的 Qwen2 模型
            "Qwen2Model",  # Qwen2 模型
            "Qwen2PreTrainedModel",  # Qwen2 预训练模型
        ]
    )
    # 导入结构模块下的 rag 模块，并扩展其列表
    _import_structure["models.rag"].extend(
        [
            "RagModel",  # RAG 模型
            "RagPreTrainedModel",  # RAG 预训练模型
            "RagSequenceForGeneration",  # 用于生成任务的 RAG 序列
            "RagTokenForGeneration",  # 用于生成任务的 RAG 标记
        ]
    )
    # 导入结构模块下的 realm 模块，并扩展其列表
    _import_structure["models.realm"].extend(
        [
            "REALM_PRETRAINED_MODEL_ARCHIVE_LIST",  # REALM 预训练模型档案列表
            "RealmEmbedder",  # Realm 嵌入器
            "RealmForOpenQA",  # 用于开放式问答任务的 Realm 模型
            "RealmKnowledgeAugEncoder",  # 用于知识增强编码的 Realm 模型
            "RealmPreTrainedModel",  # Realm 预训练模型
            "RealmReader",  # Realm 读取器
            "RealmRetriever",  # Realm 检索器
            "RealmScorer",  # Realm 评分器
            "load_tf_weights_in_realm",  # 加载 Realm 模型中的 TensorFlow 权重
        ]
    )
    # 导入结构模块下的 reformer 模块，并扩展其列表
    _import_structure["models.reformer"].extend(
        [
            "REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # Reformer 预训练模型档案列表
            "ReformerAttention",  # Reformer 注意力机制
            "ReformerForMaskedLM",  # 用于掩码语言建模任务的 Reformer 模型
            "ReformerForQuestionAnswering",  # 用于问答任务的 Reformer 模型
            "ReformerForSequenceClassification",  # 用于序列分类任务的 Reformer 模型
            "ReformerLayer",  # Reformer 的层
            "ReformerModel",  # Reformer 模型
            "ReformerModelWithLMHead",  # 带有语言模型头的 Reformer 模型
            "ReformerPreTrainedModel",  # Reformer 预训练模型
        ]
    )
    # 导入结构模块下的 regnet 模块，并扩展其列表
    _import_structure["models.regnet"].extend(
        [
            "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # RegNet 预训练模型档案列表
            "RegNetForImageClassification",  # 用于图像分类任务的 RegNet 模型
            "RegNetModel",  # RegNet 模型
            "RegNetPreTrainedModel",  # RegNet 预训练模型
        ]
    )
    # 导入结构模块下的 rembert 模块，并扩展其列表
    _import_structure["models.rembert"].extend(
        [
            "REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # RemBert 预训练模型档案列表
            "RemBertForCausalLM",  # 用于因果语言建模任务的 RemBert 模型
            "RemBertForMaskedLM",  # 用于掩码语言建模任务的 RemBert 模型
            "RemBertForMultipleChoice",  # 用于多项选择任务的 RemBert 模型
            "RemBertForQuestionAnswering",  # 用于问答任务的 RemBert 模型
            "RemBertForSequenceClassification",  # 用于序列分类任务的 RemBert 模型
            "RemBertForTokenClassification",  # 用于标记分类任务的 RemBert 模型
            "RemBertLayer",  # RemBert 的层
            "RemBertModel",  # RemBert 模型
            "RemBertPreTrainedModel",  # RemBert 预训练模型
            "load_tf_weights_in_rembert",  # 加载 RemBert 模型中的 TensorFlow 权重
        ]
    )
    # 导
    # 将 models.roberta 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.roberta"].extend(
        [
            "ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RobertaForCausalLM",
            "RobertaForMaskedLM",
            "RobertaForMultipleChoice",
            "RobertaForQuestionAnswering",
            "RobertaForSequenceClassification",
            "RobertaForTokenClassification",
            "RobertaModel",
            "RobertaPreTrainedModel",
        ]
    )
    # 将 models.roberta_prelayernorm 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RobertaPreLayerNormForCausalLM",
            "RobertaPreLayerNormForMaskedLM",
            "RobertaPreLayerNormForMultipleChoice",
            "RobertaPreLayerNormForQuestionAnswering",
            "RobertaPreLayerNormForSequenceClassification",
            "RobertaPreLayerNormForTokenClassification",
            "RobertaPreLayerNormModel",
            "RobertaPreLayerNormPreTrainedModel",
        ]
    )
    # 将 models.roc_bert 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.roc_bert"].extend(
        [
            "ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RoCBertForCausalLM",
            "RoCBertForMaskedLM",
            "RoCBertForMultipleChoice",
            "RoCBertForPreTraining",
            "RoCBertForQuestionAnswering",
            "RoCBertForSequenceClassification",
            "RoCBertForTokenClassification",
            "RoCBertLayer",
            "RoCBertModel",
            "RoCBertPreTrainedModel",
            "load_tf_weights_in_roc_bert",
        ]
    )
    # 将 models.roformer 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.roformer"].extend(
        [
            "ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RoFormerForCausalLM",
            "RoFormerForMaskedLM",
            "RoFormerForMultipleChoice",
            "RoFormerForQuestionAnswering",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerLayer",
            "RoFormerModel",
            "RoFormerPreTrainedModel",
            "load_tf_weights_in_roformer",
        ]
    )
    # 将 models.rwkv 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.rwkv"].extend(
        [
            "RWKV_PRETRAINED_MODEL_ARCHIVE_LIST",
            "RwkvForCausalLM",
            "RwkvModel",
            "RwkvPreTrainedModel",
        ]
    )
    # 将 models.sam 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.sam"].extend(
        [
            "SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SamModel",
            "SamPreTrainedModel",
        ]
    )
    # 将 models.seamless_m4t 模块下的相关类和常量添加到 _import_structure 字典中
    _import_structure["models.seamless_m4t"].extend(
        [
            "SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST",
            "SeamlessM4TCodeHifiGan",
            "SeamlessM4TForSpeechToSpeech",
            "SeamlessM4TForSpeechToText",
            "SeamlessM4TForTextToSpeech",
            "SeamlessM4TForTextToText",
            "SeamlessM4THifiGan",
            "SeamlessM4TModel",
            "SeamlessM4TPreTrainedModel",
            "SeamlessM4TTextToUnitForConditionalGeneration",
            "SeamlessM4TTextToUnitModel",
        ]
    )
    # 导入模块的结构字典中添加 SEAMLESS_M4T_V2 模块相关内容
    _import_structure["models.seamless_m4t_v2"].extend(
        [
            "SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SeamlessM4Tv2ForSpeechToSpeech",  # 语音到语音的 SeamlessM4Tv2 模型
            "SeamlessM4Tv2ForSpeechToText",  # 语音到文本的 SeamlessM4Tv2 模型
            "SeamlessM4Tv2ForTextToSpeech",  # 文本到语音的 SeamlessM4Tv2 模型
            "SeamlessM4Tv2ForTextToText",  # 文本到文本的 SeamlessM4Tv2 模型
            "SeamlessM4Tv2Model",  # SeamlessM4Tv2 模型
            "SeamlessM4Tv2PreTrainedModel",  # 预训练的 SeamlessM4Tv2 模型
        ]
    )
    
    # 导入模块的结构字典中添加 SEGFORMER 模块相关内容
    _import_structure["models.segformer"].extend(
        [
            "SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SegformerDecodeHead",  # Segformer 解码头
            "SegformerForImageClassification",  # 用于图像分类的 Segformer 模型
            "SegformerForSemanticSegmentation",  # 用于语义分割的 Segformer 模型
            "SegformerLayer",  # Segformer 层
            "SegformerModel",  # Segformer 模型
            "SegformerPreTrainedModel",  # 预训练的 Segformer 模型
        ]
    )
    
    # 导入模块的结构字典中添加 SEW 模块相关内容
    _import_structure["models.sew"].extend(
        [
            "SEW_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SEWForCTC",  # 用于 CTC 的 SEW 模型
            "SEWForSequenceClassification",  # 用于序列分类的 SEW 模型
            "SEWModel",  # SEW 模型
            "SEWPreTrainedModel",  # 预训练的 SEW 模型
        ]
    )
    
    # 导入模块的结构字典中添加 SEW_D 模块相关内容
    _import_structure["models.sew_d"].extend(
        [
            "SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SEWDForCTC",  # 用于 CTC 的 SEW_D 模型
            "SEWDForSequenceClassification",  # 用于序列分类的 SEW_D 模型
            "SEWDModel",  # SEW_D 模型
            "SEWDPreTrainedModel",  # 预训练的 SEW_D 模型
        ]
    )
    
    # 导入模块的结构字典中添加 SIGLIP 模块相关内容
    _import_structure["models.siglip"].extend(
        [
            "SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SiglipModel",  # Siglip 模型
            "SiglipPreTrainedModel",  # 预训练的 Siglip 模型
            "SiglipTextModel",  # 文本模型
            "SiglipVisionModel",  # 视觉模型
        ]
    )
    
    # 导入模块的结构字典中添加 speech_encoder_decoder 模块相关内容
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    
    # 导入模块的结构字典中添加 speech_to_text 模块相关内容
    _import_structure["models.speech_to_text"].extend(
        [
            "SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "Speech2TextForConditionalGeneration",  # 用于条件生成的语音到文本模型
            "Speech2TextModel",  # 语音到文本模型
            "Speech2TextPreTrainedModel",  # 预训练的语音到文本模型
        ]
    )
    
    # 导入模块的结构字典中添加 speech_to_text_2 模块相关内容
    _import_structure["models.speech_to_text_2"].extend(["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"])
    
    # 导入模块的结构字典中添加 speecht5 模块相关内容
    _import_structure["models.speecht5"].extend(
        [
            "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SpeechT5ForSpeechToSpeech",  # 语音到语音的 SpeechT5 模型
            "SpeechT5ForSpeechToText",  # 语音到文本的 SpeechT5 模型
            "SpeechT5ForTextToSpeech",  # 文本到语音的 SpeechT5 模型
            "SpeechT5HifiGan",  # HiFi-GAN 模型
            "SpeechT5Model",  # SpeechT5 模型
            "SpeechT5PreTrainedModel",  # 预训练的 SpeechT5 模型
        ]
    )
    
    # 导入模块的结构字典中添加 splinter 模块相关内容
    _import_structure["models.splinter"].extend(
        [
            "SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
            "SplinterForPreTraining",  # 用于预训练的 Splinter 模型
            "SplinterForQuestionAnswering",  # 用于问答的 Splinter 模型
            "SplinterLayer",  # Splinter 层
            "SplinterModel",  # Splinter 模型
            "SplinterPreTrainedModel",  # 预训练的 Splinter 模型
        ]
    )
    # 将模型结构中的SqueezeBert相关内容添加到_import_structure字典中
    _import_structure["models.squeezebert"].extend(
        [
            "SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # SqueezeBert预训练模型存档列表
            "SqueezeBertForMaskedLM",  # 用于遮蔽语言模型的SqueezeBert模型
            "SqueezeBertForMultipleChoice",  # 用于多选题的SqueezeBert模型
            "SqueezeBertForQuestionAnswering",  # 用于问答任务的SqueezeBert模型
            "SqueezeBertForSequenceClassification",  # 用于序列分类任务的SqueezeBert模型
            "SqueezeBertForTokenClassification",  # 用于标记分类任务的SqueezeBert模型
            "SqueezeBertModel",  # SqueezeBert模型
            "SqueezeBertModule",  # SqueezeBert模块
            "SqueezeBertPreTrainedModel",  # SqueezeBert预训练模型
        ]
    )
    # 将模型结构中的SwiftFormer相关内容添加到_import_structure字典中
    _import_structure["models.swiftformer"].extend(
        [
            "SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # SwiftFormer预训练模型存档列表
            "SwiftFormerForImageClassification",  # 用于图像分类任务的SwiftFormer模型
            "SwiftFormerModel",  # SwiftFormer模型
            "SwiftFormerPreTrainedModel",  # SwiftFormer预训练模型
        ]
    )
    # 将模型结构中的Swin相关内容添加到_import_structure字典中
    _import_structure["models.swin"].extend(
        [
            "SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",  # Swin预训练模型存档列表
            "SwinBackbone",  # Swin骨干网络
            "SwinForImageClassification",  # 用于图像分类任务的Swin模型
            "SwinForMaskedImageModeling",  # 用于图像处理任务的Swin模型
            "SwinModel",  # Swin模型
            "SwinPreTrainedModel",  # Swin预训练模型
        ]
    )
    # 将模型结构中的Swin2SR相关内容添加到_import_structure字典中
    _import_structure["models.swin2sr"].extend(
        [
            "SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST",  # Swin2SR预训练模型存档列表
            "Swin2SRForImageSuperResolution",  # 用于图像超分辨率任务的Swin2SR模型
            "Swin2SRModel",  # Swin2SR模型
            "Swin2SRPreTrainedModel",  # Swin2SR预训练模型
        ]
    )
    # 将模型结构中的Swinv2相关内容添加到_import_structure字典中
    _import_structure["models.swinv2"].extend(
        [
            "SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST",  # Swinv2预训练模型存档列表
            "Swinv2Backbone",  # Swinv2骨干网络
            "Swinv2ForImageClassification",  # 用于图像分类任务的Swinv2模型
            "Swinv2ForMaskedImageModeling",  # 用于图像处理任务的Swinv2模型
            "Swinv2Model",  # Swinv2模型
            "Swinv2PreTrainedModel",  # Swinv2预训练模型
        ]
    )
    # 将模型结构中的SwitchTransformers相关内容添加到_import_structure字典中
    _import_structure["models.switch_transformers"].extend(
        [
            "SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",  # SwitchTransformers预训练模型存档列表
            "SwitchTransformersEncoderModel",  # SwitchTransformers编码器模型
            "SwitchTransformersForConditionalGeneration",  # 用于条件生成任务的SwitchTransformers模型
            "SwitchTransformersModel",  # SwitchTransformers模型
            "SwitchTransformersPreTrainedModel",  # SwitchTransformers预训练模型
            "SwitchTransformersSparseMLP",  # SwitchTransformers稀疏多层感知机模型
            "SwitchTransformersTop1Router",  # SwitchTransformers Top-1 路由器
        ]
    )
    # 将模型结构中的T5相关内容添加到_import_structure字典中
    _import_structure["models.t5"].extend(
        [
            "T5_PRETRAINED_MODEL_ARCHIVE_LIST",  # T5预训练模型存档列表
            "T5EncoderModel",  # T5编码器模型
            "T5ForConditionalGeneration",  # 用于条件生成任务的T5模型
            "T5ForQuestionAnswering",  # 用于问答任务的T5模型
            "T5ForSequenceClassification",  # 用于序列分类任务的T5模型
            "T5Model",  # T5模型
            "T5PreTrainedModel",  # T5预训练模型
            "load_tf_weights_in_t5",  # 从TensorFlow加载权重到T5模型中
        ]
    )
    # 将模型结构中的TableTransformer相关内容添加到_import_structure字典中
    _import_structure["models.table_transformer"].extend(
        [
            "TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # TableTransformer预训练模型存档列表
            "TableTransformerForObjectDetection",  # 用于目标检测任务的TableTransformer模型
            "TableTransformerModel",  # TableTransformer模型
            "TableTransformerPreTrainedModel",  # TableTransformer预训练模型
        ]
    )
    # 将模型结构中的Tapas相关内容添加到_import_structure字典中
    _import_structure["models.tapas"].extend(
        [
            "TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",  # Tapas预训练模型存档列表
            "TapasForMaskedLM",  # 用于遮蔽语言建模任务的Tapas模型
            "TapasForQuestionAnswering",  # 用于问答任务的Tapas模型
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.time_series_transformer"].extend(
        [
            "TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # 时间序列变换器预训练模型存档列表
            "TimeSeriesTransformerForPrediction",  # 用于预测的时间序列变换器
            "TimeSeriesTransformerModel",  # 时间序列变换器模型
            "TimeSeriesTransformerPreTrainedModel",  # 时间序列变换器预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.timesformer"].extend(
        [
            "TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # Timesformer预训练模型存档列表
            "TimesformerForVideoClassification",  # 用于视频分类的Timesformer
            "TimesformerModel",  # Timesformer模型
            "TimesformerPreTrainedModel",  # Timesformer预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])  # TimmBackbone模块
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.trocr"].extend(
        [
            "TROCR_PRETRAINED_MODEL_ARCHIVE_LIST",  # TROCR预训练模型存档列表
            "TrOCRForCausalLM",  # 用于因果语言建模的TrOCR
            "TrOCRPreTrainedModel",  # TrOCR预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.tvlt"].extend(
        [
            "TVLT_PRETRAINED_MODEL_ARCHIVE_LIST",  # TVLT预训练模型存档列表
            "TvltForAudioVisualClassification",  # 用于音频视觉分类的Tvlt
            "TvltForPreTraining",  # 用于预训练的Tvlt
            "TvltModel",  # Tvlt模型
            "TvltPreTrainedModel",  # Tvlt预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.tvp"].extend(
        [
            "TVP_PRETRAINED_MODEL_ARCHIVE_LIST",  # TVP预训练模型存档列表
            "TvpForVideoGrounding",  # 用于视频定位的Tvp
            "TvpModel",  # Tvp模型
            "TvpPreTrainedModel",  # Tvp预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",  # UMT5编码器模型
            "UMT5ForConditionalGeneration",  # 用于条件生成的UMT5
            "UMT5ForQuestionAnswering",  # 用于问答的UMT5
            "UMT5ForSequenceClassification",  # 用于序列分类的UMT5
            "UMT5Model",  # UMT5模型
            "UMT5PreTrainedModel",  # UMT5预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.unispeech"].extend(
        [
            "UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST",  # UNISPEECH预训练模型存档列表
            "UniSpeechForCTC",  # 用于CTC的UniSpeech
            "UniSpeechForPreTraining",  # 用于预训练的UniSpeech
            "UniSpeechForSequenceClassification",  # 用于序列分类的UniSpeech
            "UniSpeechModel",  # UniSpeech模型
            "UniSpeechPreTrainedModel",  # UniSpeech预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.unispeech_sat"].extend(
        [
            "UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST",  # UNISPEECH_SAT预训练模型存档列表
            "UniSpeechSatForAudioFrameClassification",  # 用于音频帧分类的UniSpeechSat
            "UniSpeechSatForCTC",  # 用于CTC的UniSpeechSat
            "UniSpeechSatForPreTraining",  # 用于预训练的UniSpeechSat
            "UniSpeechSatForSequenceClassification",  # 用于序列分类的UniSpeechSat
            "UniSpeechSatForXVector",  # 用于X向量的UniSpeechSat
            "UniSpeechSatModel",  # UniSpeechSat模型
            "UniSpeechSatPreTrainedModel",  # UniSpeechSat预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.univnet"].extend(
        [
            "UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # UNIVNET预训练模型存档列表
            "UnivNetModel",  # UnivNet模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",  # 用于语义分割的UperNet
            "UperNetPreTrainedModel",  # UperNet预训练模型
        ]
    )
    # 导入模块结构中的特定模块列表并扩展其内容
    _import_structure["models.videomae"].extend(
        [
            "VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST",  # VIDEOMAE预训练模型存档列表
            "VideoMAEForPreTraining",  # 用于预训练的VideoMAE
            "VideoMAEForVideoClassification",  # 用于视频分类的VideoMAE
            "VideoMAEModel", 
    # 将 models.vilt 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vilt"].extend(
        [
            "VILT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViltForImageAndTextRetrieval",
            "ViltForImagesAndTextClassification",
            "ViltForMaskedLM",
            "ViltForQuestionAnswering",
            "ViltForTokenClassification",
            "ViltLayer",
            "ViltModel",
            "ViltPreTrainedModel",
        ]
    )
    # 将 models.vipllava 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vipllava"].extend(
        [
            "VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VipLlavaForConditionalGeneration",
            "VipLlavaPreTrainedModel",
        ]
    )
    # 将 models.vision_encoder_decoder 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    # 将 models.vision_text_dual_encoder 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    # 将 models.visual_bert 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.visual_bert"].extend(
        [
            "VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VisualBertForMultipleChoice",
            "VisualBertForPreTraining",
            "VisualBertForQuestionAnswering",
            "VisualBertForRegionToPhraseAlignment",
            "VisualBertForVisualReasoning",
            "VisualBertLayer",
            "VisualBertModel",
            "VisualBertPreTrainedModel",
        ]
    )
    # 将 models.vit 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vit"].extend(
        [
            "VIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTForImageClassification",
            "ViTForMaskedImageModeling",
            "ViTModel",
            "ViTPreTrainedModel",
        ]
    )
    # 将 models.vit_hybrid 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vit_hybrid"].extend(
        [
            "VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTHybridForImageClassification",
            "ViTHybridModel",
            "ViTHybridPreTrainedModel",
        ]
    )
    # 将 models.vit_mae 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vit_mae"].extend(
        [
            "VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTMAEForPreTraining",
            "ViTMAELayer",
            "ViTMAEModel",
            "ViTMAEPreTrainedModel",
        ]
    )
    # 将 models.vit_msn 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vit_msn"].extend(
        [
            "VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "ViTMSNForImageClassification",
            "ViTMSNModel",
            "ViTMSNPreTrainedModel",
        ]
    )
    # 将 models.vitdet 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vitdet"].extend(
        [
            "VITDET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VitDetBackbone",
            "VitDetModel",
            "VitDetPreTrainedModel",
        ]
    )
    # 将 models.vitmatte 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vitmatte"].extend(
        [
            "VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VitMatteForImageMatting",
            "VitMattePreTrainedModel",
        ]
    )
    # 将 models.vits 模块下的相关内容添加到 _import_structure 字典中
    _import_structure["models.vits"].extend(
        [
            "VITS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VitsModel",
            "VitsPreTrainedModel",
        ]
    )
    # 导入结构的字典中对应 "models.vivit" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.vivit"].extend(
        [
            "VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "VivitForVideoClassification",
            "VivitModel",
            "VivitPreTrainedModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.wav2vec2" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.wav2vec2"].extend(
        [
            "WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Wav2Vec2ForAudioFrameClassification",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForMaskedLM",
            "Wav2Vec2ForPreTraining",
            "Wav2Vec2ForSequenceClassification",
            "Wav2Vec2ForXVector",
            "Wav2Vec2Model",
            "Wav2Vec2PreTrainedModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.wav2vec2_bert" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.wav2vec2_bert"].extend(
        [
            "WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Wav2Vec2BertForAudioFrameClassification",
            "Wav2Vec2BertForCTC",
            "Wav2Vec2BertForSequenceClassification",
            "Wav2Vec2BertForXVector",
            "Wav2Vec2BertModel",
            "Wav2Vec2BertPreTrainedModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.wav2vec2_conformer" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.wav2vec2_conformer"].extend(
        [
            "WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "Wav2Vec2ConformerForAudioFrameClassification",
            "Wav2Vec2ConformerForCTC",
            "Wav2Vec2ConformerForPreTraining",
            "Wav2Vec2ConformerForSequenceClassification",
            "Wav2Vec2ConformerForXVector",
            "Wav2Vec2ConformerModel",
            "Wav2Vec2ConformerPreTrainedModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.wavlm" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.wavlm"].extend(
        [
            "WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "WavLMForAudioFrameClassification",
            "WavLMForCTC",
            "WavLMForSequenceClassification",
            "WavLMForXVector",
            "WavLMModel",
            "WavLMPreTrainedModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.whisper" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.whisper"].extend(
        [
            "WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "WhisperForAudioClassification",
            "WhisperForCausalLM",
            "WhisperForConditionalGeneration",
            "WhisperModel",
            "WhisperPreTrainedModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.x_clip" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.x_clip"].extend(
        [
            "XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XCLIPModel",
            "XCLIPPreTrainedModel",
            "XCLIPTextModel",
            "XCLIPVisionModel",
        ]
    )
    
    # 导入结构的字典中对应 "models.xglm" 模块的列表扩展，包括预训练模型存档列表和各个类的定义
    _import_structure["models.xglm"].extend(
        [
            "XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "XGLMForCausalLM",
            "XGLMModel",
            "XGLMPreTrainedModel",
        ]
    )
    # 将指定模块下的类名列表添加到_import_structure字典中
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
    # 将指定模块下的类名列表添加到_import_structure字典中
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
    # 将指定模块下的类名列表添加到_import_structure字典中
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
    # 将指定模块下的类名列表添加到_import_structure字典中
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
    # 将指定模块下的类名列表添加到_import_structure字典中
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
    # 将指���模块下的类名列表添加到_import_structure字典中
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
    # 将指定模块下的类名列表添加到_import_structure字典中
    _import_structure["models.yolos"].extend(
        [
            "YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST",
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    # 将指定模块中的类、函数等添加到_import_structure字典中的相应键对应的列表中
    _import_structure["models.yoso"].extend(
        [
            "YOSO_PRETRAINED_MODEL_ARCHIVE_LIST",  # YOSO预训练模型的归档列表
            "YosoForMaskedLM",                      # 用于掩码语言建模任务的YOSO模型
            "YosoForMultipleChoice",                # 用于多项选择任务的YOSO模型
            "YosoForQuestionAnswering",             # 用于问答任务的YOSO模型
            "YosoForSequenceClassification",        # 用于序列分类任务的YOSO模型
            "YosoForTokenClassification",           # 用于标记分类任务的YOSO模型
            "YosoLayer",                            # YOSO模型中的自注意力层
            "YosoModel",                            # YOSO模型
            "YosoPreTrainedModel",                  # YOSO预训练模型的基类
        ]
    )
    # 将优化相关的类、函数等添加到_import_structure字典中的相应键对应的列表中
    _import_structure["optimization"] = [
        "Adafactor",                                        # Adafactor优化器
        "AdamW",                                            # AdamW优化器
        "get_constant_schedule",                            # 获取常数学习率调度器
        "get_constant_schedule_with_warmup",                # 获取带预热的常数学习率调度器
        "get_cosine_schedule_with_warmup",                  # 获取带预热的余弦退火学习率调度器
        "get_cosine_with_hard_restarts_schedule_with_warmup",  # 获取带预热的带硬重启的余弦退火学习率调度器
        "get_inverse_sqrt_schedule",                        # 获取倒数平方根学习率调度器
        "get_linear_schedule_with_warmup",                  # 获取带预热的线性学习率调度器
        "get_polynomial_decay_schedule_with_warmup",        # 获取带预热的多项式衰减学习率调度器
        "get_scheduler",                                    # 获取调度器
    ]
    # 将PyTorch工具相关的类、函数等添加到_import_structure字典中的相应键对应的列表中
    _import_structure["pytorch_utils"] = [
        "Conv1D",                                           # 1维卷积层
        "apply_chunking_to_forward",                        # 将前向传播应用分块
        "prune_layer",                                      # 剪枝层
    ]
    # 将Sagemaker相关的类、函数等添加到_import_structure字典中的相应键对应的列表中
    _import_structure["sagemaker"] = []
    # 将时间序列工具相关的类、函数等添加到_import_structure字典中的相应键对应的列表中
    _import_structure["time_series_utils"] = []
    # 将训练器相关的类、函数等添加到_import_structure字典中的相应键对应的列表中
    _import_structure["trainer"] = ["Trainer"]
    # 将训练器PyTorch工具相关的函数添加到_import_structure字典中的相应键对应的列表中
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    # 将序列到序列训练器相关的类添加到_import_structure字典中的相应键对应的列表中
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]
# 尝试导入 TensorFlow 相关对象
try:
    # 检查 TensorFlow 是否可用
    if not is_tf_available():
        # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果出现 OptionalDependencyNotAvailable 异常，则导入虚拟的 TensorFlow 对象
    from .utils import dummy_tf_objects

    # 更新导入结构字典，包含 dummy_tf_objects 中的所有非下划线开头的名称
    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    # 如果 TensorFlow 可用，则更新导入结构字典
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
            "tf_top_k_top_p_filtering",
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
    # TensorFlow 模型结构
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
    # 将 models.auto 模块下的一组模型名称添加到 _import_structure 字典中
    _import_structure["models.auto"].extend(
        [
            "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",  # 音频分类模型映射
            "TF_MODEL_FOR_CAUSAL_LM_MAPPING",  # 因果语言模型映射
            "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",  # 文档问答模型映射
            "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",  # 图像分类模型映射
            "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",  # 掩码图像建模模型映射
            "TF_MODEL_FOR_MASKED_LM_MAPPING",  # 掩码语言建模模型映射
            "TF_MODEL_FOR_MASK_GENERATION_MAPPING",  # 掩码生成模型映射
            "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",  # 多选题模型映射
            "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",  # 下一个句子预测模型映射
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
            "TF_MODEL_MAPPING",  # 模型映射
            "TF_MODEL_WITH_LM_HEAD_MAPPING",  # 具有语言模型头部的模型映射
            "TFAutoModel",  # 自动模型
            "TFAutoModelForAudioClassification",  # 音频分类自动模型
            "TFAutoModelForCausalLM",  # 因果语言模型自动模型
            "TFAutoModelForDocumentQuestionAnswering",  # 文档问答自动模型
            "TFAutoModelForImageClassification",  # 图像分类自动模型
            "TFAutoModelForMaskedImageModeling",  # 掩码图像建模自动模型
            "TFAutoModelForMaskedLM",  # 掩码语言建模自动模型
            "TFAutoModelForMaskGeneration",  # 掩码生成自动模型
            "TFAutoModelForMultipleChoice",  # 多选题自动模型
            "TFAutoModelForNextSentencePrediction",  # 下一个句子预测自动模型
            "TFAutoModelForPreTraining",  # 预训练自动模型
            "TFAutoModelForQuestionAnswering",  # 问答自动模型
            "TFAutoModelForSemanticSegmentation",  # 语义分割自动模型
            "TFAutoModelForSeq2SeqLM",  # 序列到序列语言模型自动模型
            "TFAutoModelForSequenceClassification",  # 序列分类自动模型
            "TFAutoModelForSpeechSeq2Seq",  # 语音序列到序列自动模型
            "TFAutoModelForTableQuestionAnswering",  # 表格问答自动模型
            "TFAutoModelForTextEncoding",  # 文本编码自动模型
            "TFAutoModelForTokenClassification",  # 标记分类自动模型
            "TFAutoModelForVision2Seq",  # 视觉到序列自动模型
            "TFAutoModelForZeroShotImageClassification",  # 零样本图像分类自动模型
            "TFAutoModelWithLMHead",  # 具有语言模型头部的自动模型
        ]
    )
    # 将 models.bart 模块下的一组模型名称添加到 _import_structure 字典中
    _import_structure["models.bart"].extend(
        [
            "TFBartForConditionalGeneration",  # 生成条件BART模型
            "TFBartForSequenceClassification",  # 序列分类BART模型
            "TFBartModel",  # BART模型
            "TFBartPretrainedModel",  # 预训练BART模型
        ]
    )
    # 扩展模型结构导入字典中的 "models.bert" 键，添加以下 BERT 模型相关的项
    _import_structure["models.bert"].extend(
        [
            "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # BERT 预训练模型存档列表
            "TFBertEmbeddings",  # BERT 嵌入层
            "TFBertForMaskedLM",  # 用于遮蔽语言建模的 BERT 模型
            "TFBertForMultipleChoice",  # 用于多项选择任务的 BERT 模型
            "TFBertForNextSentencePrediction",  # 用于下一句预测任务的 BERT 模型
            "TFBertForPreTraining",  # 用于预训练任务的 BERT 模型
            "TFBertForQuestionAnswering",  # 用于问答任务的 BERT 模型
            "TFBertForSequenceClassification",  # 用于序列分类任务的 BERT 模型
            "TFBertForTokenClassification",  # 用于标记分类任务的 BERT 模型
            "TFBertLMHeadModel",  # BERT 语言建模头部模型
            "TFBertMainLayer",  # BERT 主层
            "TFBertModel",  # BERT 模型
            "TFBertPreTrainedModel",  # BERT 预训练模型
        ]
    )
    
    # 扩展模型结构导入字典中的 "models.blenderbot" 键，添加以下 Blenderbot 相关的项
    _import_structure["models.blenderbot"].extend(
        [
            "TFBlenderbotForConditionalGeneration",  # 用于条件生成的 Blenderbot 模型
            "TFBlenderbotModel",  # Blenderbot 模型
            "TFBlenderbotPreTrainedModel",  # Blenderbot 预训练模型
        ]
    )
    
    # 扩展模型结构导入字典中的 "models.blenderbot_small" 键，添加以下 Blenderbot Small 相关的项
    _import_structure["models.blenderbot_small"].extend(
        [
            "TFBlenderbotSmallForConditionalGeneration",  # 用于条件生成的 Blenderbot Small 模型
            "TFBlenderbotSmallModel",  # Blenderbot Small 模型
            "TFBlenderbotSmallPreTrainedModel",  # Blenderbot Small 预训练模型
        ]
    )
    
    # 扩展模型结构导入字典中的 "models.blip" 键，添加以下 BLIP 相关的项
    _import_structure["models.blip"].extend(
        [
            "TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # BLIP 预训练模型存档列表
            "TFBlipForConditionalGeneration",  # 用于条件生成的 BLIP 模型
            "TFBlipForImageTextRetrieval",  # 用于图像文本检索任务的 BLIP 模型
            "TFBlipForQuestionAnswering",  # 用于问答任务的 BLIP 模型
            "TFBlipModel",  # BLIP 模型
            "TFBlipPreTrainedModel",  # BLIP 预训练模型
            "TFBlipTextModel",  # BLIP 文本模型
            "TFBlipVisionModel",  # BLIP 视觉模型
        ]
    )
    
    # 扩展模型结构导入字典中的 "models.camembert" 键，添加以下 Camembert 相关的项
    _import_structure["models.camembert"].extend(
        [
            "TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Camembert 预训练模型存档列表
            "TFCamembertForCausalLM",  # 用于因果语言建模的 Camembert 模型
            "TFCamembertForMaskedLM",  # 用于遮蔽语言建模的 Camembert 模型
            "TFCamembertForMultipleChoice",  # 用于多项选择任务的 Camembert 模型
            "TFCamembertForQuestionAnswering",  # 用于问答任务的 Camembert 模型
            "TFCamembertForSequenceClassification",  # 用于序列分类任务的 Camembert 模型
            "TFCamembertForTokenClassification",  # 用于标记分类任务的 Camembert 模型
            "TFCamembertModel",  # Camembert 模型
            "TFCamembertPreTrainedModel",  # Camembert 预训练模型
        ]
    )
    
    # 扩展模型结构导入字典中的 "models.clip" 键，添加以下 CLIP 相关的项
    _import_structure["models.clip"].extend(
        [
            "TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",  # CLIP 预训练模型存档列表
            "TFCLIPModel",  # CLIP 模型
            "TFCLIPPreTrainedModel",  # CLIP 预训练模型
            "TFCLIPTextModel",  # CLIP 文本模型
            "TFCLIPVisionModel",  # CLIP 视觉模型
        ]
    )
    
    # 扩展模型结构导入字典中的 "models.convbert" 键，添加以下 ConvBERT 相关的项
    _import_structure["models.convbert"].extend(
        [
            "TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # ConvBERT 预训练模型存档列表
            "TFConvBertForMaskedLM",  # 用于遮蔽语言建模的 ConvBERT 模型
            "TFConvBertForMultipleChoice",  # 用于多项选择任务的 ConvBERT 模型
            "TFConvBertForQuestionAnswering",  # 用于问答任务的 ConvBERT 模型
            "TFConvBertForSequenceClassification",  # 用于序列分类任务的 ConvBERT 模型
            "TFConvBertForTokenClassification",  # 用于标记分类任务的 ConvBERT 模型
            "TFConvBertLayer",  # ConvBERT 层
            "TFConvBertModel",  # ConvBERT 模型
            "TF
    # 将一系列控制器模型相关的标识符添加到模块导入结构中
    _import_structure["models.ctrl"].extend(
        [
            "TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型归档列表
            "TFCTRLForSequenceClassification",         # 序列分类的 TFCTRL 模型
            "TFCTRLLMHeadModel",                       # 语言模型头的 TFCTRL 模型
            "TFCTRLModel",                             # TFCTRL 模型
            "TFCTRLPreTrainedModel",                   # 预训练的 TFCTRL 模型
        ]
    )
    # 将一系列 CVT 模型相关的标识符添加到模块导入结构中
    _import_structure["models.cvt"].extend(
        [
            "TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST",   # CVT 模型的预训练模型归档列表
            "TFCvtForImageClassification",            # 图像分类的 TFCvt 模型
            "TFCvtModel",                              # TFCvt 模型
            "TFCvtPreTrainedModel",                    # 预训练的 TFCvt 模型
        ]
    )
    # 将一系列 Data2Vec 模型相关的标识符添加到模块导入结构中
    _import_structure["models.data2vec"].extend(
        [
            "TFData2VecVisionForImageClassification",            # 图像分类的 TFData2Vec 模型
            "TFData2VecVisionForSemanticSegmentation",           # 语义分割的 TFData2Vec 模型
            "TFData2VecVisionModel",                            # TFData2Vec 模型
            "TFData2VecVisionPreTrainedModel",                  # 预训练的 TFData2Vec 模型
        ]
    )
    # 将一系列 DeBERTa 模型相关的标识符添加到模块导入结构中
    _import_structure["models.deberta"].extend(
        [
            "TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",         # DeBERTa 模型的预训练模型归档列表
            "TFDebertaForMaskedLM",                             # 掩码语言模型的 TFDeberta 模型
            "TFDebertaForQuestionAnswering",                     # 问答任务的 TFDeberta 模型
            "TFDebertaForSequenceClassification",               # 序列分类的 TFDeberta 模型
            "TFDebertaForTokenClassification",                  # 标记分类的 TFDeberta 模型
            "TFDebertaModel",                                   # TFDeberta 模型
            "TFDebertaPreTrainedModel",                         # 预训练的 TFDeberta 模型
        ]
    )
    # 将一系列 DeBERTa v2 模型相关的标识符添加到模块导入结构中
    _import_structure["models.deberta_v2"].extend(
        [
            "TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST",       # DeBERTa v2 模型的预训练模型归档列表
            "TFDebertaV2ForMaskedLM",                            # 掩码语言模型的 TFDeberta v2 模型
            "TFDebertaV2ForMultipleChoice",                      # 多选题任务的 TFDeberta v2 模型
            "TFDebertaV2ForQuestionAnswering",                  # 问答任务的 TFDeberta v2 模型
            "TFDebertaV2ForSequenceClassification",             # 序列分类的 TFDeberta v2 模型
            "TFDebertaV2ForTokenClassification",                # 标记分类的 TFDeberta v2 模型
            "TFDebertaV2Model",                                  # TFDeberta v2 模型
            "TFDebertaV2PreTrainedModel",                        # 预训练的 TFDeberta v2 模型
        ]
    )
    # 将一系列 DeiT 模型相关的标识符添加到模块导入结构中
    _import_structure["models.deit"].extend(
        [
            "TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST",             # DeiT 模型的预训练模型归档列表
            "TFDeiTForImageClassification",                      # 图像分类的 TFDeiT 模型
            "TFDeiTForImageClassificationWithTeacher",           # 带有教师的图像分类的 TFDeiT 模型
            "TFDeiTForMaskedImageModeling",                      # 带掩码图像建模的 TFDeiT 模型
            "TFDeiTModel",                                       # TFDeiT 模型
            "TFDeiTPreTrainedModel",                             # 预训练的 TFDeiT 模型
        ]
    )
    # 将一系列已弃用的 TransfoXL 模型相关的标识符添加到模块导入结构中
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST",        # TransfoXL 模型的预训练模型归档列表
            "TFAdaptiveEmbedding",                                # 自适应嵌入的 TFTransfoXL 模型
            "TFTransfoXLForSequenceClassification",               # 序列分类的 TFTransfoXL 模型
            "TFTransfoXLLMHeadModel",                             # 语言模型头的 TFTransfoXL 模型
            "TFTransfoXLMainLayer",                                # TFTransfoXL 主层
            "TFTransfoXLModel",                                   # TFTransfoXL 模型
            "TFTransfoXLPreTrainedModel",                         # 预训练的 TFTransfoXL 模型
        ]
    )
    # 将一系列 DistilBERT 模型相关的标识符添加到模块导入结构中
    _import_structure["models.distilbert"].extend(
        [
            "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",        # DistilBERT 模型的预训练模型归档列表
            "TFDistilBertForMaskedLM",                            # 掩码语言模型的 TFDistilBert 模型
            "TFDistilBertForMultipleChoice",                      # 多选题任务的
    # 将模型的导入结构更新，扩展包含的模型和相关预训练模型的列表
    _import_structure["models.dpr"].extend(
        [
            "TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPR 上下文编码器的预训练模型存档列表
            "TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPR 问题编码器的预训练模型存档列表
            "TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",  # DPR 阅读器的预训练模型存档列表
            "TFDPRContextEncoder",  # DPR 上下文编码器的 TensorFlow 模型
            "TFDPRPretrainedContextEncoder",  # DPR 上下文编码器的预训练 TensorFlow 模型
            "TFDPRPretrainedQuestionEncoder",  # DPR 问题编码器的预训练 TensorFlow 模型
            "TFDPRPretrainedReader",  # DPR 阅读器的预训练 TensorFlow 模型
            "TFDPRQuestionEncoder",  # DPR 问题编码器的 TensorFlow 模型
            "TFDPRReader",  # DPR 阅读器的 TensorFlow 模型
        ]
    )
    # 扩展包含的 EfficientFormer 模型和相关预训练模型的列表
    _import_structure["models.efficientformer"].extend(
        [
            "TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",  # EfficientFormer 的预训练模型存档列表
            "TFEfficientFormerForImageClassification",  # 用于图像分类的 EfficientFormer TensorFlow 模型
            "TFEfficientFormerForImageClassificationWithTeacher",  # 带有教师的图像分类 EfficientFormer TensorFlow 模型
            "TFEfficientFormerModel",  # EfficientFormer 的 TensorFlow 模型
            "TFEfficientFormerPreTrainedModel",  # EfficientFormer 的预训练 TensorFlow 模型
        ]
    )
    # 扩展包含的 Electra 模型和相关预训练模型的列表
    _import_structure["models.electra"].extend(
        [
            "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",  # Electra 的预训练模型存档列表
            "TFElectraForMaskedLM",  # 用于 Masked Language Modeling 的 Electra TensorFlow 模型
            "TFElectraForMultipleChoice",  # 用于多项选择任务的 Electra TensorFlow 模型
            "TFElectraForPreTraining",  # 用于预训练任务的 Electra TensorFlow 模型
            "TFElectraForQuestionAnswering",  # 用于问答任务的 Electra TensorFlow 模型
            "TFElectraForSequenceClassification",  # 用于序列分类任务的 Electra TensorFlow 模型
            "TFElectraForTokenClassification",  # 用于标记分类任务的 Electra TensorFlow 模型
            "TFElectraModel",  # Electra 的 TensorFlow 模型
            "TFElectraPreTrainedModel",  # Electra 的预训练 TensorFlow 模型
        ]
    )
    # 将 Encoder-Decoder 模型添加到模型导入结构中
    _import_structure["models.encoder_decoder"].append("TFEncoderDecoderModel")  # Encoder-Decoder 模型的 TensorFlow 实现
    # 扩展包含的 ESM 模型和相关预训练模型的列表
    _import_structure["models.esm"].extend(
        [
            "ESM_PRETRAINED_MODEL_ARCHIVE_LIST",  # ESM 模型的预训练模型存档列表
            "TFEsmForMaskedLM",  # 用于 Masked Language Modeling 的 ESM TensorFlow 模型
            "TFEsmForSequenceClassification",  # 用于序列分类任务的 ESM TensorFlow 模型
            "TFEsmForTokenClassification",  # 用于标记分类任务的 ESM TensorFlow 模型
            "TFEsmModel",  # ESM 模型的 TensorFlow 模型
            "TFEsmPreTrainedModel",  # ESM 模型的预训练 TensorFlow 模型
        ]
    )
    # 扩展包含的 Flaubert 模型和相关预训练模型的列表
    _import_structure["models.flaubert"].extend(
        [
            "TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",  # Flaubert 模型的预训练模型存档列表
            "TFFlaubertForMultipleChoice",  # 用于多项选择任务的 Flaubert TensorFlow 模型
            "TFFlaubertForQuestionAnsweringSimple",  # 用于简单问答任务的 Flaubert TensorFlow 模型
            "TFFlaubertForSequenceClassification",  # 用于序列分类任务的 Flaubert TensorFlow 模型
            "TFFlaubertForTokenClassification",  # 用于标记分类任务的 Flaubert TensorFlow 模型
            "TFFlaubertModel",  # Flaubert 的 TensorFlow 模型
            "TFFlaubertPreTrainedModel",  # Flaubert 的预训练 TensorFlow 模型
            "TFFlaubertWithLMHeadModel",  # 带有语言模型头的 Flaubert TensorFlow 模型
        ]
    )
    # 扩展包含的 Funnel 模型和相关预训练模型的列表
    _import_structure["models.funnel"].extend(
        [
            "TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST",  # Funnel 模型的预训练模型存档列表
            "TFFunnelBaseModel",  # Funnel 基础模型的 TensorFlow 实现
            "TFFunnelForMaskedLM",  # 用于 Masked Language Modeling 的 Funnel TensorFlow 模型
            "TFFunnelForMultipleChoice",  # 用于多项选择任务的 Funnel TensorFlow 模型
            "TFFunnelForPreTraining",  # 用于预训练任务的 Funnel TensorFlow 模型
            "TFFunnelForQuestionAnswering",  # 用于问答任务的 Funnel TensorFlow 模型
            "TFFunnelForSequenceClassification",  # 用于序列分类任务的 Funnel TensorFlow 模型
            "TFFunnelForTokenClassification",  # 用于标记分类任务的 Funnel TensorFlow 模型
            "TFFunnelModel",  # Funnel
    # 扩展 models.gptj 模块的导入结构
    _import_structure["models.gptj"].extend(
        [
            "TFGPTJForCausalLM",
            "TFGPTJForQuestionAnswering",
            "TFGPTJForSequenceClassification",
            "TFGPTJModel",
            "TFGPTJPreTrainedModel",
        ]
    )
    # 扩展 models.groupvit 模块的导入结构
    _import_structure["models.groupvit"].extend(
        [
            "TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFGroupViTModel",
            "TFGroupViTPreTrainedModel",
            "TFGroupViTTextModel",
            "TFGroupViTVisionModel",
        ]
    )
    # 扩展 models.hubert 模块的导入结构
    _import_structure["models.hubert"].extend(
        [
            "TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFHubertForCTC",
            "TFHubertModel",
            "TFHubertPreTrainedModel",
        ]
    )
    # 扩展 models.layoutlm 模块的导入结构
    _import_structure["models.layoutlm"].extend(
        [
            "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLayoutLMForMaskedLM",
            "TFLayoutLMForQuestionAnswering",
            "TFLayoutLMForSequenceClassification",
            "TFLayoutLMForTokenClassification",
            "TFLayoutLMMainLayer",
            "TFLayoutLMModel",
            "TFLayoutLMPreTrainedModel",
        ]
    )
    # 扩展 models.layoutlmv3 模块的导入结构
    _import_structure["models.layoutlmv3"].extend(
        [
            "TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLayoutLMv3ForQuestionAnswering",
            "TFLayoutLMv3ForSequenceClassification",
            "TFLayoutLMv3ForTokenClassification",
            "TFLayoutLMv3Model",
            "TFLayoutLMv3PreTrainedModel",
        ]
    )
    # 扩展 models.led 模块的导入结构
    _import_structure["models.led"].extend(["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"])
    # 扩展 models.longformer 模块的导入结构
    _import_structure["models.longformer"].extend(
        [
            "TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLongformerForMaskedLM",
            "TFLongformerForMultipleChoice",
            "TFLongformerForQuestionAnswering",
            "TFLongformerForSequenceClassification",
            "TFLongformerForTokenClassification",
            "TFLongformerModel",
            "TFLongformerPreTrainedModel",
            "TFLongformerSelfAttention",
        ]
    )
    # 扩展 models.lxmert 模块的导入结构
    _import_structure["models.lxmert"].extend(
        [
            "TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFLxmertForPreTraining",
            "TFLxmertMainLayer",
            "TFLxmertModel",
            "TFLxmertPreTrainedModel",
            "TFLxmertVisualFeatureEncoder",
        ]
    )
    # 扩展 models.marian 模块的导入结构
    _import_structure["models.marian"].extend(["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"])
    # 扩展 models.mbart 模块的导入结构
    _import_structure["models.mbart"].extend(
        ["TFMBartForConditionalGeneration", "TFMBartModel", "TFMBartPreTrainedModel"]
    )
    # 扩展 models.mobilebert 模块的导入结构
    _import_structure["models.mobilebert"].extend(
        [
            "TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMobileBertForMaskedLM",
            "TFMobileBertForMultipleChoice",
            "TFMobileBertForNextSentencePrediction",
            "TFMobileBertForPreTraining",
            "TFMobileBertForQuestionAnswering",
            "TFMobileBertForSequenceClassification",
            "TFMobileBertForTokenClassification",
            "TFMobileBertMainLayer",
            "TFMobileBertModel",
            "TFMobileBertPreTrainedModel",
        ]
    )
    # 扩展 models.mobilevit 模块的导入结构
    _import_structure["models.mobilevit"].extend(
        [
            "TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMobileViTForImageClassification",
            "TFMobileViTForSemanticSegmentation",
            "TFMobileViTModel",
            "TFMobileViTPreTrainedModel",
        ]
    )
    # 扩展 models.mpnet 模块的导入结构
    _import_structure["models.mpnet"].extend(
        [
            "TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFMPNetForMaskedLM",
            "TFMPNetForMultipleChoice",
            "TFMPNetForQuestionAnswering",
            "TFMPNetForSequenceClassification",
            "TFMPNetForTokenClassification",
            "TFMPNetMainLayer",
            "TFMPNetModel",
            "TFMPNetPreTrainedModel",
        ]
    )
    # 扩展 models.mt5 模块的导入结构
    _import_structure["models.mt5"].extend(["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"])
    # 扩展 models.openai 模块的导入结构
    _import_structure["models.openai"].extend(
        [
            "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFOpenAIGPTDoubleHeadsModel",
            "TFOpenAIGPTForSequenceClassification",
            "TFOpenAIGPTLMHeadModel",
            "TFOpenAIGPTMainLayer",
            "TFOpenAIGPTModel",
            "TFOpenAIGPTPreTrainedModel",
        ]
    )
    # 扩展 models.opt 模块的导入结构
    _import_structure["models.opt"].extend(
        [
            "TFOPTForCausalLM",
            "TFOPTModel",
            "TFOPTPreTrainedModel",
        ]
    )
    # 扩展 models.pegasus 模块的导入结构
    _import_structure["models.pegasus"].extend(
        [
            "TFPegasusForConditionalGeneration",
            "TFPegasusModel",
            "TFPegasusPreTrainedModel",
        ]
    )
    # 扩展 models.rag 模块的导入结构
    _import_structure["models.rag"].extend(
        [
            "TFRagModel",
            "TFRagPreTrainedModel",
            "TFRagSequenceForGeneration",
            "TFRagTokenForGeneration",
        ]
    )
    # 扩展 models.regnet 模块的导入结构
    _import_structure["models.regnet"].extend(
        [
            "TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRegNetForImageClassification",
            "TFRegNetModel",
            "TFRegNetPreTrainedModel",
        ]
    )
    # 将 models.rembert 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.rembert"].extend(
        [
            "TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRemBertForCausalLM",
            "TFRemBertForMaskedLM",
            "TFRemBertForMultipleChoice",
            "TFRemBertForQuestionAnswering",
            "TFRemBertForSequenceClassification",
            "TFRemBertForTokenClassification",
            "TFRemBertLayer",
            "TFRemBertModel",
            "TFRemBertPreTrainedModel",
        ]
    )
    # 将 models.resnet 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.resnet"].extend(
        [
            "TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFResNetForImageClassification",
            "TFResNetModel",
            "TFResNetPreTrainedModel",
        ]
    )
    # 将 models.roberta 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.roberta"].extend(
        [
            "TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRobertaForCausalLM",
            "TFRobertaForMaskedLM",
            "TFRobertaForMultipleChoice",
            "TFRobertaForQuestionAnswering",
            "TFRobertaForSequenceClassification",
            "TFRobertaForTokenClassification",
            "TFRobertaMainLayer",
            "TFRobertaModel",
            "TFRobertaPreTrainedModel",
        ]
    )
    # 将 models.roberta_prelayernorm 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRobertaPreLayerNormForCausalLM",
            "TFRobertaPreLayerNormForMaskedLM",
            "TFRobertaPreLayerNormForMultipleChoice",
            "TFRobertaPreLayerNormForQuestionAnswering",
            "TFRobertaPreLayerNormForSequenceClassification",
            "TFRobertaPreLayerNormForTokenClassification",
            "TFRobertaPreLayerNormMainLayer",
            "TFRobertaPreLayerNormModel",
            "TFRobertaPreLayerNormPreTrainedModel",
        ]
    )
    # 将 models.roformer 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.roformer"].extend(
        [
            "TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFRoFormerForCausalLM",
            "TFRoFormerForMaskedLM",
            "TFRoFormerForMultipleChoice",
            "TFRoFormerForQuestionAnswering",
            "TFRoFormerForSequenceClassification",
            "TFRoFormerForTokenClassification",
            "TFRoFormerLayer",
            "TFRoFormerModel",
            "TFRoFormerPreTrainedModel",
        ]
    )
    # 将 models.sam 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.sam"].extend(
        [
            "TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSamModel",
            "TFSamPreTrainedModel",
        ]
    )
    # 将 models.segformer 模块下的指定内容添加到 _import_structure 字典中
    _import_structure["models.segformer"].extend(
        [
            "TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSegformerDecodeHead",
            "TFSegformerForImageClassification",
            "TFSegformerForSemanticSegmentation",
            "TFSegformerModel",
            "TFSegformerPreTrainedModel",
        ]
    )
    # 扩展 models.speech_to_text 模块的导入结构
    _import_structure["models.speech_to_text"].extend(
        [
            "TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSpeech2TextForConditionalGeneration",
            "TFSpeech2TextModel",
            "TFSpeech2TextPreTrainedModel",
        ]
    )
    # 扩展 models.swin 模块的导入结构
    _import_structure["models.swin"].extend(
        [
            "TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFSwinForImageClassification",
            "TFSwinForMaskedImageModeling",
            "TFSwinModel",
            "TFSwinPreTrainedModel",
        ]
    )
    # 扩展 models.t5 模块的导入结构
    _import_structure["models.t5"].extend(
        [
            "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    # 扩展 models.tapas 模块的导入结构
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
    # 扩展 models.vision_encoder_decoder 模块的导入结构
    _import_structure["models.vision_encoder_decoder"].extend(["TFVisionEncoderDecoderModel"])
    # 扩展 models.vision_text_dual_encoder 模块的导入结构
    _import_structure["models.vision_text_dual_encoder"].extend(["TFVisionTextDualEncoderModel"])
    # 扩展 models.vit 模块的导入结构
    _import_structure["models.vit"].extend(
        [
            "TFViTForImageClassification",
            "TFViTModel",
            "TFViTPreTrainedModel",
        ]
    )
    # 扩展 models.vit_mae 模块的导入结构
    _import_structure["models.vit_mae"].extend(
        [
            "TFViTMAEForPreTraining",
            "TFViTMAEModel",
            "TFViTMAEPreTrainedModel",
        ]
    )
    # 扩展 models.wav2vec2 模块的导入结构
    _import_structure["models.wav2vec2"].extend(
        [
            "TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFWav2Vec2ForCTC",
            "TFWav2Vec2ForSequenceClassification",
            "TFWav2Vec2Model",
            "TFWav2Vec2PreTrainedModel",
        ]
    )
    # 扩展 models.whisper 模块的导入结构
    _import_structure["models.whisper"].extend(
        [
            "TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFWhisperForConditionalGeneration",
            "TFWhisperModel",
            "TFWhisperPreTrainedModel",
        ]
    )
    # 扩展 models.xglm 模块的导入结构
    _import_structure["models.xglm"].extend(
        [
            "TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST",
            "TFXGLMForCausalLM",
            "TFXGLMModel",
            "TFXGLMPreTrainedModel",
        ]
    )
    # 扩展 models.xlm 模块的导入结构
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
    # 扩展模块的导入结构，将指定模块下的类名或变量名添加到_import_structure字典中
    _import_structure["models.xlm_roberta"].extend(
        [
            "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",  # XLM-RoBERTa预训练模型存档列表
            "TFXLMRobertaForCausalLM",  # 用于因果语言建模任务的XLM-RoBERTa模型
            "TFXLMRobertaForMaskedLM",  # 用于遮蔽语言建模任务的XLM-RoBERTa模型
            "TFXLMRobertaForMultipleChoice",  # 用于多项选择任务的XLM-RoBERTa模型
            "TFXLMRobertaForQuestionAnswering",  # 用于问答任务的XLM-RoBERTa模型
            "TFXLMRobertaForSequenceClassification",  # 用于序列分类任务的XLM-RoBERTa模型
            "TFXLMRobertaForTokenClassification",  # 用于标记分类任务的XLM-RoBERTa模型
            "TFXLMRobertaModel",  # XLM-RoBERTa模型
            "TFXLMRobertaPreTrainedModel",  # XLM-RoBERTa预训练模型
        ]
    )
    # 扩展模块的导入结构，将指定模块下的类名或变量名添加到_import_structure字典中
    _import_structure["models.xlnet"].extend(
        [
            "TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST",  # XLNet预训练模型存档列表
            "TFXLNetForMultipleChoice",  # 用于多项选择任务的XLNet模型
            "TFXLNetForQuestionAnsweringSimple",  # 用于简单问答任务的XLNet模型
            "TFXLNetForSequenceClassification",  # 用于序列分类任务的XLNet模型
            "TFXLNetForTokenClassification",  # 用于标记分类任务的XLNet模型
            "TFXLNetLMHeadModel",  # XLNet语言建模头模型
            "TFXLNetMainLayer",  # XLNet主层
            "TFXLNetModel",  # XLNet模型
            "TFXLNetPreTrainedModel",  # XLNet预训练模型
        ]
    )
    # 将优化相关的类和函数名添加到_import_structure字典中
    _import_structure["optimization_tf"] = [
        "AdamWeightDecay",  # Adam权重衰减优化器
        "GradientAccumulator",  # 梯度累加器
        "WarmUp",  # 学习率WarmUp策略
        "create_optimizer",  # 创建优化器函数
    ]
    # 将tf_utils模块下的类和函数名添加到_import_structure字典中
    _import_structure["tf_utils"] = []
# 尝试检查是否所有必要的依赖库都可用，如果有任何一个不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    # 检查是否所有必要的依赖库都可用
    if not (
        is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
        and is_torch_available()
        and is_pretty_midi_available()
    ):
        # 如果有任何一个依赖库不可用，则抛出 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果出现 OptionalDependencyNotAvailable 异常，则导入 dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects
    from .utils import (
        dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects,
    )

    # 将 dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects 中不以 "_" 开头的对象名添加到 _import_structure 中
    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    # 如果所有必要的依赖库都可用，则将以下对象添加到 _import_structure 中
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")


# FLAX-backed objects
# 检查是否 Flax 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果出现 OptionalDependencyNotAvailable 异常，则导入 dummy_flax_objects
    from .utils import dummy_flax_objects

    # 将 dummy_flax_objects 中不以 "_" 开头的对象名添加到 _import_structure 中
    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    # 如果 Flax 可用，则将以下对象添加到 _import_structure 中
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
    # 扩展 models.auto 模块的导入结构
    _import_structure["models.auto"].extend(
        [
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
    )

    # Flax 模型结构

    # 扩展 models.bart 模块的导入结构
    _import_structure["models.bart"].extend(
        [
            "FlaxBartDecoderPreTrainedModel",
            "FlaxBartForCausalLM",
            "FlaxBartForConditionalGeneration",
            "FlaxBartForQuestionAnswering",
            "FlaxBartForSequenceClassification",
            "FlaxBartModel",
            "FlaxBartPreTrainedModel",
        ]
    )
    
    # 扩展 models.beit 模块的导入结构
    _import_structure["models.beit"].extend(
        [
            "FlaxBeitForImageClassification",
            "FlaxBeitForMaskedImageModeling",
            "FlaxBeitModel",
            "FlaxBeitPreTrainedModel",
        ]
    )

    # 扩展 models.bert 模块的导入结构
    _import_structure["models.bert"].extend(
        [
            "FlaxBertForCausalLM",
            "FlaxBertForMaskedLM",
            "FlaxBertForMultipleChoice",
            "FlaxBertForNextSentencePrediction",
            "FlaxBertForPreTraining",
            "FlaxBertForQuestionAnswering",
            "FlaxBertForSequenceClassification",
            "FlaxBertForTokenClassification",
            "FlaxBertModel",
            "FlaxBertPreTrainedModel",
        ]
    )
    
    # 扩展 models.big_bird 模块的导入结构
    _import_structure["models.big_bird"].extend(
        [
            "FlaxBigBirdForCausalLM",
            "FlaxBigBirdForMaskedLM",
            "FlaxBigBirdForMultipleChoice",
            "FlaxBigBirdForPreTraining",
            "FlaxBigBirdForQuestionAnswering",
            "FlaxBigBirdForSequenceClassification",
            "FlaxBigBirdForTokenClassification",
            "FlaxBigBirdModel",
            "FlaxBigBirdPreTrainedModel",
        ]
    )
    # 扩展_import_structure字典中"models.blenderbot"键对应的值，添加Blenderbot模型相关的类名
    _import_structure["models.blenderbot"].extend(
        [
            "FlaxBlenderbotForConditionalGeneration",
            "FlaxBlenderbotModel",
            "FlaxBlenderbotPreTrainedModel",
        ]
    )
    # 扩展_import_structure字典中"models.blenderbot_small"键对应的值，添加Blenderbot Small模型相关的类名
    _import_structure["models.blenderbot_small"].extend(
        [
            "FlaxBlenderbotSmallForConditionalGeneration",
            "FlaxBlenderbotSmallModel",
            "FlaxBlenderbotSmallPreTrainedModel",
        ]
    )
    # 扩展_import_structure字典中"models.bloom"键对应的值，添加Bloom模型相关的类名
    _import_structure["models.bloom"].extend(
        [
            "FlaxBloomForCausalLM",
            "FlaxBloomModel",
            "FlaxBloomPreTrainedModel",
        ]
    )
    # 扩展_import_structure字典中"models.clip"键对应的值，添加CLIP模型相关的类名
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
    # 扩展_import_structure字典中"models.distilbert"键对应的值，添加DistilBERT模型相关的类名
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
    # 扩展_import_structure字典中"models.electra"键对应的值，添加Electra模型相关的类名
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
    # 添加"FlaxEncoderDecoderModel"类名到_import_structure字典中"models.encoder_decoder"键对应的值
    _import_structure["models.encoder_decoder"].append("FlaxEncoderDecoderModel")
    # 扩展_import_structure字典中"models.gpt2"键对应的值，添加GPT-2模型相关的类名
    _import_structure["models.gpt2"].extend(["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"])
    # 扩展_import_structure字典中"models.gpt_neo"键对应的值，添加GPT-Neo模型相关的类名
    _import_structure["models.gpt_neo"].extend(
        ["FlaxGPTNeoForCausalLM", "FlaxGPTNeoModel", "FlaxGPTNeoPreTrainedModel"]
    )
    # 扩展_import_structure字典中"models.gptj"键对应的值，添加GPT-J模型相关的类名
    _import_structure["models.gptj"].extend(["FlaxGPTJForCausalLM", "FlaxGPTJModel", "FlaxGPTJPreTrainedModel"])
    # 扩展_import_structure字典中"models.llama"键对应的值，添加Llama模型相关的类名
    _import_structure["models.llama"].extend(["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"])
    # 扩展_import_structure字典中"models.longt5"键对应的值，添加LongT5模型相关的类名
    _import_structure["models.longt5"].extend(
        [
            "FlaxLongT5ForConditionalGeneration",
            "FlaxLongT5Model",
            "FlaxLongT5PreTrainedModel",
        ]
    )
    # 扩展_import_structure字典中"models.marian"键对应的值，添加Marian模型相关的类名
    _import_structure["models.marian"].extend(
        [
            "FlaxMarianModel",
            "FlaxMarianMTModel",
            "FlaxMarianPreTrainedModel",
        ]
    )
    # 扩展 models.mbart 模块的导入结构，添加以下类名
    _import_structure["models.mbart"].extend(
        [
            "FlaxMBartForConditionalGeneration",  # 用于条件生成的 MBart 模型
            "FlaxMBartForQuestionAnswering",      # 用于问答任务的 MBart 模型
            "FlaxMBartForSequenceClassification", # 用于序列分类任务的 MBart 模型
            "FlaxMBartModel",                     # MBart 模型
            "FlaxMBartPreTrainedModel",           # MBart 预训练模型
        ]
    )
    # 扩展 models.mt5 模块的导入结构，添加以下类名
    _import_structure["models.mt5"].extend(
        [
            "FlaxMT5EncoderModel",                # MT5 编码器模型
            "FlaxMT5ForConditionalGeneration",    # 用于条件生成的 MT5 模型
            "FlaxMT5Model",                       # MT5 模型
        ]
    )
    # 扩展 models.opt 模块的导入结构，添加以下类名
    _import_structure["models.opt"].extend(
        [
            "FlaxOPTForCausalLM",                 # 用于因果语言建模的 OPT 模型
            "FlaxOPTModel",                       # OPT 模型
            "FlaxOPTPreTrainedModel",             # OPT 预训练模型
        ]
    )
    # 扩展 models.pegasus 模块的导入结构，添加以下类名
    _import_structure["models.pegasus"].extend(
        [
            "FlaxPegasusForConditionalGeneration",# 用于条件生成的 Pegasus 模型
            "FlaxPegasusModel",                   # Pegasus 模型
            "FlaxPegasusPreTrainedModel",         # Pegasus 预训练模型
        ]
    )
    # 扩展 models.regnet 模块的导入结构，添加以下类名
    _import_structure["models.regnet"].extend(
        [
            "FlaxRegNetForImageClassification",   # 用于图像分类的 RegNet 模型
            "FlaxRegNetModel",                     # RegNet 模型
            "FlaxRegNetPreTrainedModel",           # RegNet 预训练模型
        ]
    )
    # 扩展 models.resnet 模块的导入结构，添加以下类名
    _import_structure["models.resnet"].extend(
        [
            "FlaxResNetForImageClassification",   # 用于图像分类的 ResNet 模型
            "FlaxResNetModel",                     # ResNet 模型
            "FlaxResNetPreTrainedModel",           # ResNet 预训练模型
        ]
    )
    # 扩展 models.roberta 模块的导入结构，添加以下类名
    _import_structure["models.roberta"].extend(
        [
            "FlaxRobertaForCausalLM",             # 用于因果语言建模的 RoBERTa 模型
            "FlaxRobertaForMaskedLM",             # 用于遮蔽语言建模的 RoBERTa 模型
            "FlaxRobertaForMultipleChoice",       # 用于多项选择任务的 RoBERTa 模型
            "FlaxRobertaForQuestionAnswering",    # 用于问答任务的 RoBERTa 模型
            "FlaxRobertaForSequenceClassification",# 用于序列分类任务的 RoBERTa 模型
            "FlaxRobertaForTokenClassification",   # 用于标记分类任务的 RoBERTa 模型
            "FlaxRobertaModel",                    # RoBERTa 模型
            "FlaxRobertaPreTrainedModel",          # RoBERTa 预训练模型
        ]
    )
    # 扩展 models.roberta_prelayernorm 模块的导入结构，添加以下类名
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "FlaxRobertaPreLayerNormForCausalLM",             # 用于因果语言建模的 RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormForMaskedLM",             # 用于遮蔽语言建模的 RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormForMultipleChoice",       # 用于多项选择任务的 RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormForQuestionAnswering",    # 用于问答任务的 RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormForSequenceClassification",# 用于序列分类任务的 RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormForTokenClassification",   # 用于标记分类任务的 RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormModel",                    # RoBERTa PreLayerNorm 模型
            "FlaxRobertaPreLayerNormPreTrainedModel",          # RoBERTa PreLayerNorm 预训练模型
        ]
    )
    # 扩展 models.roformer 模块的导入结构，添加以下类名
    _import_structure["models.roformer"].extend(
        [
            "FlaxRoFormerForMaskedLM",             # 用于遮蔽语言建模的 RoFormer 模型
            "FlaxRoFormerForMultipleChoice",       # 用于多项选择任务的 RoFormer 模型
            "FlaxRoFormerForQuestionAnswering",    # 用于问答任务的 RoFormer 模型
            "FlaxRoFormerForSequenceClassification",# 用于序列分类任务的 RoFormer 模型
            "FlaxRoFormerForTokenClassification",   # 用于标记分类任务的 RoFormer 模型
            "FlaxRoFormerModel",                    # RoFormer 模型
            "FlaxRoFormerPreTrainedModel",          # RoFormer 预训练模型
        ]
    )
    # 将 FlaxSpeechEncoderDecoderModel 添加到 models.speech_encoder_decoder 模块的导入结构
    _import_structure["models.speech_encoder_decoder"].append("FlaxSpeechEncoderDecoderModel")
    # 扩展 models.t5 模块的导入结构，添加以下类名
    _import_structure["models.t5"].extend(
        [
            "FlaxT5EncoderModel",                   # T5 编码器模型
            "FlaxT5
    # 导入模块结构中的 models.vit 下的特定类
    _import_structure["models.vit"].extend(["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"])
    # 导入模块结构中的 models.wav2vec2 下的特定类
    _import_structure["models.wav2vec2"].extend(
        [
            "FlaxWav2Vec2ForCTC",
            "FlaxWav2Vec2ForPreTraining",
            "FlaxWav2Vec2Model",
            "FlaxWav2Vec2PreTrainedModel",
        ]
    )
    # 导入模块结构中的 models.whisper 下的特定类
    _import_structure["models.whisper"].extend(
        [
            "FlaxWhisperForConditionalGeneration",
            "FlaxWhisperModel",
            "FlaxWhisperPreTrainedModel",
            "FlaxWhisperForAudioClassification",
        ]
    )
    # 导入模块结构中的 models.xglm 下的特定类
    _import_structure["models.xglm"].extend(
        [
            "FlaxXGLMForCausalLM",
            "FlaxXGLMModel",
            "FlaxXGLMPreTrainedModel",
        ]
    )
    # 导入模块结构中的 models.xlm_roberta 下的特定类
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
# 如果 TYPE_CHECKING 为真，则进行直接导入以进行类型检查
if TYPE_CHECKING:
    # 导入配置相关的类
    from .configuration_utils import PretrainedConfig

    # 导入数据相关的类
    from .data import (
        DataProcessor,
        InputExample,
        InputFeatures,
        SingleSentenceClassificationProcessor,
        SquadExample,
        SquadFeatures,
        SquadV1Processor,
        SquadV2Processor,
        glue_compute_metrics,
        glue_convert_examples_to_features,
        glue_output_modes,
        glue_processors,
        glue_tasks_num_labels,
        squad_convert_examples_to_features,
        xnli_compute_metrics,
        xnli_output_modes,
        xnli_processors,
        xnli_tasks_num_labels,
    )
    # 导入数据收集器相关的类
    from .data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
        DefaultDataCollator,
        default_data_collator,
    )
    # 导入序列特征提取器相关的类
    from .feature_extraction_sequence_utils import SequenceFeatureExtractor

    # 导入特征提取相关的类
    from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # 导入生成相关的类
    from .generation import GenerationConfig, TextIteratorStreamer, TextStreamer
    # 导入 HF 参数解析器
    from .hf_argparser import HfArgumentParser

    # 导入集成相关的类
    from .integrations import (
        is_clearml_available,
        is_comet_available,
        is_dvclive_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
    )

    # 导入模型卡片相关的类
    from .modelcard import ModelCard

    # 导入 TF 2.0 <=> PyTorch 转换工具类
    from .modeling_tf_pytorch_utils import (
        convert_tf_weight_name_to_pt_weight_name,
        load_pytorch_checkpoint_in_tf2_model,
        load_pytorch_model_in_tf2_model,
        load_pytorch_weights_in_tf2_model,
        load_tf2_checkpoint_in_pytorch_model,
        load_tf2_model_in_pytorch_model,
        load_tf2_weights_in_pytorch_model,
    )
    # 导入 ALBERT 相关的类和配置
    from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
    # 导入对齐模型相关的类和配置
    from .models.align import (
        ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AlignConfig,
        AlignProcessor,
        AlignTextConfig,
        AlignVisionConfig,
    )
    # 导入 AltCLIP 相关的类和配置
    from .models.altclip import (
        ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AltCLIPConfig,
        AltCLIPProcessor,
        AltCLIPTextConfig,
        AltCLIPVisionConfig,
    )
    # 导入音频频谱变换器相关的类和配置
    from .models.audio_spectrogram_transformer import (
        AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ASTConfig,
        ASTFeatureExtractor,
    )
    # 导入所有自动模型相关的模块和内容
    from .models.auto import (
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 所有预训练配置存档映射
        CONFIG_MAPPING,  # 配置映射
        FEATURE_EXTRACTOR_MAPPING,  # 特征提取器映射
        IMAGE_PROCESSOR_MAPPING,  # 图像处理器映射
        MODEL_NAMES_MAPPING,  # 模型名称映射
        PROCESSOR_MAPPING,  # 处理器映射
        TOKENIZER_MAPPING,  # 分词器映射
        AutoConfig,  # 自动配置类
        AutoFeatureExtractor,  # 自动特征提取器类
        AutoImageProcessor,  # 自动图像处理器类
        AutoProcessor,  # 自动处理器类
        AutoTokenizer,  # 自动分词器类
    )
    # 导入 Autoformer 相关模块和内容
    from .models.autoformer import (
        AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 自动格式预训练配置存档映射
        AutoformerConfig,  # Autoformer 配置类
    )
    # 导入 Bark 相关模块和内容
    from .models.bark import (
        BarkCoarseConfig,  # Bark 粗配置类
        BarkConfig,  # Bark 配置类
        BarkFineConfig,  # Bark 细配置类
        BarkProcessor,  # Bark 处理器类
        BarkSemanticConfig,  # Bark 语义配置类
    )
    # 导入 Bart 相关模块和内容
    from .models.bart import BartConfig, BartTokenizer  # Bart 配置类、Bart 分词器类
    # 导入 Beit 相关模块和内容
    from .models.beit import (
        BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Beit 预训练配置存档映射
        BeitConfig,  # Beit 配置类
    )
    # 导入 Bert 相关模块和内容
    from .models.bert import (
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Bert 预训练配置存档映射
        BasicTokenizer,  # 基础分词器类
        BertConfig,  # Bert 配置类
        BertTokenizer,  # Bert 分词器类
        WordpieceTokenizer,  # WordPiece 分词器类
    )
    # 导入 BertGeneration 相关模块和内容
    from .models.bert_generation import BertGenerationConfig  # Bert 生成配置类
    # 导入 BertJapanese 相关模块和内容
    from .models.bert_japanese import (
        BertJapaneseTokenizer,  # Bert 日语分词器类
        CharacterTokenizer,  # 字符分词器类
        MecabTokenizer,  # Mecab 分词器类
    )
    # 导入 Bertweet 相关模块和内容
    from .models.bertweet import BertweetTokenizer  # Bertweet 分词器类
    # 导入 BigBird 相关模块和内容
    from .models.big_bird import (
        BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP,  # BigBird 预训练配置存档映射
        BigBirdConfig,  # BigBird 配置类
    )
    # 导入 BigBirdPegasus 相关模块和内容
    from .models.bigbird_pegasus import (
        BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,  # BigBirdPegasus 预训练配置存档映射
        BigBirdPegasusConfig,  # BigBirdPegasus 配置类
    )
    # 导入 BioGPT 相关模块和内容
    from .models.biogpt import (
        BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # BioGPT 预训练配置存档映射
        BioGptConfig,  # BioGPT 配置类
        BioGptTokenizer,  # BioGPT 分词器类
    )
    # 导入 Bit 相关模块和内容
    from .models.bit import (
        BIT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Bit 预训练配置存档映射
        BitConfig,  # Bit 配置类
    )
    # 导入 Blenderbot 相关模块和内容
    from .models.blenderbot import (
        BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Blenderbot 预训练配置存档映射
        BlenderbotConfig,  # Blenderbot 配置类
        BlenderbotTokenizer,  # Blenderbot 分词器类
    )
    # 导入 BlenderbotSmall 相关模块和内容
    from .models.blenderbot_small import (
        BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,  # BlenderbotSmall 预训练配置存档映射
        BlenderbotSmallConfig,  # BlenderbotSmall 配置类
        BlenderbotSmallTokenizer,  # BlenderbotSmall 分词器类
    )
    # 导入 Blip 相关模块和内容
    from .models.blip import (
        BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Blip 预训练配置存档映射
        BlipConfig,  # Blip 配置类
        BlipProcessor,  # Blip 处理器类
        BlipTextConfig,  # Blip 文本配置类
        BlipVisionConfig,  # Blip 视觉配置类
    )
    # 导入 Blip2 相关模块和内容
    from .models.blip_2 import (
        BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Blip2 预训练配置存档映射
        Blip2Config,  # Blip2 配置类
        Blip2Processor,  # Blip2 处理器类
        Blip2QFormerConfig,  # Blip2 QFormer 配置类
        Blip2VisionConfig,  # Blip2 视觉配置类
    )
    # 导入 Bloom 相关模块和内容
    from .models.bloom import (
        BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Bloom 预训练配置存档映射
        BloomConfig,  # Bloom 配置类
    )
    # 导入 BridgeTower 相关模块和内容
    # 从.models.canine模块中导入相关内容
    from .models.canine import (
        CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CANINE预训练配置映射
        CanineConfig,  # 导入Canine配置类
        CanineTokenizer,  # 导入Canine分词器
    )
    # 从.models.chinese_clip模块中导入相关内容
    from .models.chinese_clip import (
        CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CHINESE_CLIP预训练配置映射
        ChineseCLIPConfig,  # 导入ChineseCLIP配置类
        ChineseCLIPProcessor,  # 导入ChineseCLIP处理器
        ChineseCLIPTextConfig,  # 导入ChineseCLIP文本配置类
        ChineseCLIPVisionConfig,  # 导入ChineseCLIP视觉配置类
    )
    # 从.models.clap模块中导入相关内容
    from .models.clap import (
        CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入CLAP预训练模型归档列表
        ClapAudioConfig,  # 导入Clap音频配置类
        ClapConfig,  # 导入Clap配置类
        ClapProcessor,  # 导入Clap处理器
        ClapTextConfig,  # 导入Clap文本配置类
    )
    # 从.models.clip模块中导入相关内容
    from .models.clip import (
        CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CLIP预训练配置映射
        CLIPConfig,  # 导入CLIP配置类
        CLIPProcessor,  # 导入CLIP处理器
        CLIPTextConfig,  # 导入CLIP文本配置类
        CLIPTokenizer,  # 导入CLIP分词器
        CLIPVisionConfig,  # 导入CLIP视觉配置类
    )
    # 从.models.clipseg模块中导入相关内容
    from .models.clipseg import (
        CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CLIPSEG预训练配置映射
        CLIPSegConfig,  # 导入CLIPSeg配置类
        CLIPSegProcessor,  # 导入CLIPSeg处理器
        CLIPSegTextConfig,  # 导入CLIPSeg文本配置类
        CLIPSegVisionConfig,  # 导入CLIPSeg视觉配置类
    )
    # 从.models.clvp模块中导入相关内容
    from .models.clvp import (
        CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CLVP预训练配置映射
        ClvpConfig,  # 导入Clvp配置类
        ClvpDecoderConfig,  # 导入Clvp解码器配置类
        ClvpEncoderConfig,  # 导入Clvp编码器配置类
        ClvpFeatureExtractor,  # 导入Clvp特征提取器
        ClvpProcessor,  # 导入Clvp处理器
        ClvpTokenizer,  # 导入Clvp分词器
    )
    # 从.models.codegen模块中导入相关内容
    from .models.codegen import (
        CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CODEGEN预训练配置映射
        CodeGenConfig,  # 导入CodeGen配置类
        CodeGenTokenizer,  # 导入CodeGen分词器
    )
    # 从.models.conditional_detr模块中导入相关内容
    from .models.conditional_detr import (
        CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CONDITIONAL_DETR预训练配置映射
        ConditionalDetrConfig,  # 导入ConditionalDetr配置类
    )
    # 从.models.convbert模块中导入相关内容
    from .models.convbert import (
        CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CONVBERT预训练配置映射
        ConvBertConfig,  # 导入ConvBert配置类
        ConvBertTokenizer,  # 导入ConvBert分词器
    )
    # 从.models.convnext模块中导入相关内容
    from .models.convnext import CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextConfig
    # 从.models.convnextv2模块中导入相关内容
    from .models.convnextv2 import (
        CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CONVNEXTV2预训练配置映射
        ConvNextV2Config,  # 导入ConvNextV2配置类
    )
    # 从.models.cpmant模块中导入相关内容
    from .models.cpmant import (
        CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CPMANT预训练配置映射
        CpmAntConfig,  # 导入CpmAnt配置类
        CpmAntTokenizer,  # 导入CpmAnt分词器
    )
    # 从.models.ctrl模块中导入相关内容
    from .models.ctrl import (
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入CTRL预训练配置映射
        CTRLConfig,  # 导入CTRL配置类
        CTRLTokenizer,  # 导入CTRL分词器
    )
    # 从.models.cvt模块中导入相关内容
    from .models.cvt import CVT_PRETRAINED_CONFIG_ARCHIVE_MAP, CvtConfig
    # 从.models.data2vec模块中导入相关内容
    from .models.data2vec import (
        DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入DATA2VEC_TEXT预训练配置映射
        DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入DATA2VEC_VISION预训练配置映射
        Data2VecAudioConfig,  # 导入Data2Vec音频配置类
        Data2VecTextConfig,  # 导入Data2Vec文本配置类
        Data2VecVisionConfig,  # 导入Data2Vec视觉配置类
    )
    # 从.models.deberta模块中导入相关内容
    from .models.deberta import (
        DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 导入DEBERTA预训练配置映射
        DebertaConfig,  # 导入Deberta
    # 导入 MCTCT 相关模块
    from .models.deprecated.mctct import (
        MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MCTCTConfig,
        MCTCTFeatureExtractor,
        MCTCTProcessor,
    )
    # 导入 MMBTConfig 模块
    from .models.deprecated.mmbt import MMBTConfig
    # 导入 OpenLlama 相关模块
    from .models.deprecated.open_llama import (
        OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OpenLlamaConfig,
    )
    # 导入 RetriBert 相关模块
    from .models.deprecated.retribert import (
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RetriBertConfig,
        RetriBertTokenizer,
    )
    # 导入 TapexTokenizer 模块
    from .models.deprecated.tapex import TapexTokenizer
    # 导入 TrajectoryTransformer 相关模块
    from .models.deprecated.trajectory_transformer import (
        TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TrajectoryTransformerConfig,
    )
    # 导入 TransfoXL 相关模块
    from .models.deprecated.transfo_xl import (
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TransfoXLConfig,
        TransfoXLCorpus,
        TransfoXLTokenizer,
    )
    # 导入 VanConfig 模块
    from .models.deprecated.van import VAN_PRETRAINED_CONFIG_ARCHIVE_MAP, VanConfig
    # 导入 DetaConfig 模块
    from .models.deta import DETA_PRETRAINED_CONFIG_ARCHIVE_MAP, DetaConfig
    # 导入 DetrConfig 模块
    from .models.detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig
    # 导入 DinatConfig 模块
    from .models.dinat import DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP, DinatConfig
    # 导入 Dinov2Config 模块
    from .models.dinov2 import DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Dinov2Config
    # 导入 DistilBert 相关模块
    from .models.distilbert import (
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DistilBertConfig,
        DistilBertTokenizer,
    )
    # 导入 Donut 相关模块
    from .models.donut import (
        DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DonutProcessor,
        DonutSwinConfig,
    )
    # 导入 DPR 相关模块
    from .models.dpr import (
        DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DPRConfig,
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )
    # 导入 DPTConfig 模块
    from .models.dpt import DPT_PRETRAINED_CONFIG_ARCHIVE_MAP, DPTConfig
    # 导入 EfficientFormer 相关模块
    from .models.efficientformer import (
        EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EfficientFormerConfig,
    )
    # 导入 EfficientNetConfig 模块
    from .models.efficientnet import (
        EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EfficientNetConfig,
    )
    # 导入 Electra 相关模块
    from .models.electra import (
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ElectraConfig,
        ElectraTokenizer,
    )
    # 导入 Encodec 相关模块
    from .models.encodec import (
        ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EncodecConfig,
        EncodecFeatureExtractor,
    )
    # 导入 EncoderDecoderConfig 模块
    from .models.encoder_decoder import EncoderDecoderConfig
    # 导入 ErnieConfig 模块
    from .models.ernie import ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieConfig
    # 导入 ErnieMConfig 模块
    from .models.ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig
    # 导入 Esm 相关模块
    from .models.esm import ESM_PRETRAINED_CONFIG_ARCHIVE_MAP, EsmConfig, EsmTokenizer
    # 导入 FalconConfig 模块
    from .models.falcon import FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP, FalconConfig
    # 从自定义的模块中导入 FASTSPEECH2_CONFORMER 相关模块和配置
    from .models.fastspeech2_conformer import (
        FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerTokenizer,
    )
    # 从自定义的模块中导入 FLAUBERT 相关模块和配置
    from .models.flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig, FlaubertTokenizer
    # 从自定义的模块中导入 FLAVA 相关模块和配置
    from .models.flava import (
        FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FlavaConfig,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaMultimodalConfig,
        FlavaTextConfig,
    )
    # 从自定义的模块中导入 FNET 相关模块和配置
    from .models.fnet import FNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FNetConfig
    # 从自定义的模块中导入 FOCALNET 相关模块和配置
    from .models.focalnet import FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FocalNetConfig
    # 从自定义的模块中导入 FSMT 相关模块和配置
    from .models.fsmt import (
        FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FSMTConfig,
        FSMTTokenizer,
    )
    # 从自定义的模块中导入 FUNNEL 相关模块和配置
    from .models.funnel import (
        FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FunnelConfig,
        FunnelTokenizer,
    )
    # 从自定义的模块中导入 FUYU 相关模块和配置
    from .models.fuyu import FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP, FuyuConfig
    # 从自定义的模块中导入 GIT 相关模块和配置
    from .models.git import (
        GIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GitConfig,
        GitProcessor,
        GitVisionConfig,
    )
    # 从自定义的模块中导入 GLPN 相关模块和配置
    from .models.glpn import GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP, GLPNConfig
    # 从自定义的模块中导入 GPT2 相关模块和配置
    from .models.gpt2 import (
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPT2Config,
        GPT2Tokenizer,
    )
    # 从自定义的模块中导入 GPT_BIGCODE 相关模块和配置
    from .models.gpt_bigcode import (
        GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPTBigCodeConfig,
    )
    # 从自定义的模块中导入 GPT_NEO 相关模块和配置
    from .models.gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig
    # 从自定义的模块中导入 GPT_NEOX 相关模块和配置
    from .models.gpt_neox import GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXConfig
    # 从自定义的模块中导入 GPT_NEOX_JAPANESE 相关模块和配置
    from .models.gpt_neox_japanese import (
        GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPTNeoXJapaneseConfig,
    )
    # 从自定义的模块中导入 GPTJ 相关模块和配置
    from .models.gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig
    # 从自定义的模块中导入 GPTSAN_JAPANESE 相关模块和配置
    from .models.gptsan_japanese import (
        GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPTSanJapaneseConfig,
        GPTSanJapaneseTokenizer,
    )
    # 从自定义的模块中导入 GRAPHORMER 相关模块和配置
    from .models.graphormer import (
        GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GraphormerConfig,
    )
    # 从自定义的模块中导入 GROUPVIT 相关模块和配置
    from .models.groupvit import (
        GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GroupViTConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )
    # 从自定义的模块中导入 HerbertTokenizer
    from .models.herbert import HerbertTokenizer
    # 从自定义的模块中导入 HUBERT 相关模块和配置
    from .models.hubert import HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, HubertConfig
    # 从自定义的模块中导入 IBERT 相关模块和配置
    from .models.ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig
    # 从自定义的模块中导入 IDEFICS 相关模块和配置
    from .models.idefics import (
        IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        IdeficsConfig,
    )
    # 从自定义的模块中导入 IMAGEGPT 相关模块和配置
    from .models.imagegpt import IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, ImageGPTConfig
    # 导入从指定模块中所需的类和对象
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
    )
    from .models.longformer import (
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LongformerConfig,
        LongformerTokenizer,
    )
    from .models.longt5 import LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, LongT5Config
    from .models.luke import (
        LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LukeConfig,
        LukeTokenizer,
    )
    from .models.lxmert import (
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LxmertConfig,
        LxmertTokenizer,
    )
    from .models.m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config
    from .models.marian import MarianConfig
    from .models.markuplm import (
        MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MarkupLMConfig,
        MarkupLMFeatureExtractor,
        MarkupLMProcessor,
        MarkupLMTokenizer,
    )
    from .models.mask2former import (
        MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Mask2FormerConfig,
    )
    from .models.maskformer import (
        MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MaskFormerConfig,
        MaskFormerSwinConfig,
    )
    from .models.mbart import MBartConfig
    from .models.mega import MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP, MegaConfig
    # 导入 Megatron-BERT 模型相关的配置、类和方法
    from .models.megatron_bert import (
        MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Megatron-BERT 预训练配置映射
        MegatronBertConfig,  # Megatron-BERT 配置类
    )
    # 导入 MGP-STR 模型相关的配置、类和方法
    from .models.mgp_str import (
        MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MGP-STR 预训练配置映射
        MgpstrConfig,  # MGP-STR 配置类
        MgpstrProcessor,  # MGP-STR 处理器类
        MgpstrTokenizer,  # MGP-STR 分词器类
    )
    # 导入 Mistral 模型相关的配置和类
    from .models.mistral import MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MistralConfig  # Mistral 预训练配置映射、Mistral 配置类
    # 导入 Mixtral 模型相关的配置和类
    from .models.mixtral import MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP, MixtralConfig  # Mixtral 预训练配置映射、Mixtral 配置类
    # 导入 MobileBERT 模型相关的配置、类和方法
    from .models.mobilebert import (
        MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileBERT 预训练配置映射
        MobileBertConfig,  # MobileBERT 配置类
        MobileBertTokenizer,  # MobileBERT 分词器类
    )
    # 导入 MobileNetV1 模型相关的配置和类
    from .models.mobilenet_v1 import (
        MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileNetV1 预训练配置映射
        MobileNetV1Config,  # MobileNetV1 配置类
    )
    # 导入 MobileNetV2 模型相关的配置和类
    from .models.mobilenet_v2 import (
        MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileNetV2 预训练配置映射
        MobileNetV2Config,  # MobileNetV2 配置类
    )
    # 导入 MobileViT 模型相关的配置和类
    from .models.mobilevit import (
        MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileViT 预训练配置映射
        MobileViTConfig,  # MobileViT 配置类
    )
    # 导入 MobileViTV2 模型相关的配置和类
    from .models.mobilevitv2 import (
        MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MobileViTV2 预训练配置映射
        MobileViTV2Config,  # MobileViTV2 配置类
    )
    # 导入 MPNet 模型相关的配置、类和方法
    from .models.mpnet import (
        MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MPNet 预训练配置映射
        MPNetConfig,  # MPNet 配置类
        MPNetTokenizer,  # MPNet 分词器类
    )
    # 导入 MPT 模型相关的配置和类
    from .models.mpt import MPT_PRETRAINED_CONFIG_ARCHIVE_MAP, MptConfig  # MPT 预训练配置映射、MPT 配置类
    # 导入 MRA 模型相关的配置和类
    from .models.mra import MRA_PRETRAINED_CONFIG_ARCHIVE_MAP, MraConfig  # MRA 预训练配置映射、MRA 配置类
    # 导入 MT5 模型相关的配置类
    from .models.mt5 import MT5Config  # MT5 配置类
    # 导入 MusicGen 模型相关的配置、类和方法
    from .models.musicgen import (
        MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,  # MusicGen 预训练配置映射
        MusicgenConfig,  # MusicGen 配置类
        MusicgenDecoderConfig,  # MusicGen 解码器配置类
    )
    # 导入 MVP 模型相关的配置和类
    from .models.mvp import MvpConfig, MvpTokenizer  # MVP 配置类、MVP 分词器类
    # 导入 NAT 模型相关的配置和类
    from .models.nat import NAT_PRETRAINED_CONFIG_ARCHIVE_MAP, NatConfig  # NAT 预训练配置映射、NAT 配置类
    # 导入 Nezha 模型相关的配置和类
    from .models.nezha import NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP, NezhaConfig  # Nezha 预训练配置映射、Nezha 配置类
    # 导入 NLLB-MoE 模型相关的配置和类
    from .models.nllb_moe import (
        NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP,  # NLLB-MoE 预训练配置映射
        NllbMoeConfig,  # NLLB-MoE 配置类
    )
    # 导入 Nougat 模型相关的处理器类
    from .models.nougat import NougatProcessor  # Nougat 处理器类
    # 导入 Nystromformer 模型相关的配置和类
    from .models.nystromformer import (
        NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # Nystromformer 预训练配置映射
        NystromformerConfig,  # Nystromformer 配置类
    )
    # 导入 OneFormer 模型相关的配置、类和方法
    from .models.oneformer import (
        ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,  # OneFormer 预训练配置映射
        OneFormerConfig,  # OneFormer 配置类
        OneFormerProcessor,  # OneFormer 处理器类
    )
    # 导入 OpenAI-GPT 模型相关的配置、类和方法
    from .models.openai import (
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # OpenAI-GPT 预训练配置映射
        OpenAIGPTConfig,  # OpenAI-GPT 配置类
        OpenAIGPT
    # 从自定义模块中导入 PEGASUS 相关的内容：预训练配置映射、配置、分词器
    from .models.pegasus import (
        PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PegasusConfig,
        PegasusTokenizer,
    )
    # 从自定义模块中导入 PEGASUS-X 相关的内容：预训练配置映射、配置
    from .models.pegasus_x import (
        PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PegasusXConfig,
    )
    # 从自定义模块中导入 Perceiver 相关的内容：预训练配置映射、配置、分词器
    from .models.perceiver import (
        PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PerceiverConfig,
        PerceiverTokenizer,
    )
    # 从自定义模块中导入 Persimmon 相关的内容：预训练配置映射、配置
    from .models.persimmon import (
        PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PersimmonConfig,
    )
    # 从自定义模块中导入 PHI 相关的内容：预训练配置映射、配置
    from .models.phi import PHI_PRETRAINED_CONFIG_ARCHIVE_MAP, PhiConfig
    # 从自定义模块中导入 Phobert 的分词器
    from .models.phobert import PhobertTokenizer
    # 从自定义模块中导入 Pix2Struct 相关的内容：预训练配置映射、配置、处理器、文本配置、视觉配置
    from .models.pix2struct import (
        PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pix2StructConfig,
        Pix2StructProcessor,
        Pix2StructTextConfig,
        Pix2StructVisionConfig,
    )
    # 从自定义模块中导入 PLBART 相关的内容：预训练配置映射、配置
    from .models.plbart import PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP, PLBartConfig
    # 从自定义模块中导入 PoolFormer 相关的内容：预训练配置映射、配置
    from .models.poolformer import (
        POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PoolFormerConfig,
    )
    # 从自定义模块中导入 Pop2Piano 相关的内容：预训练配置映射、配置
    from .models.pop2piano import (
        POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Pop2PianoConfig,
    )
    # 从自定义模块中导入 ProphetNet 相关的内容：预训练配置映射、配置、分词器
    from .models.prophetnet import (
        PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ProphetNetConfig,
        ProphetNetTokenizer,
    )
    # 从自定义模块中导入 PVT 相关的内容：预训练配置映射、配置
    from .models.pvt import PVT_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtConfig
    # 从自定义模块中导入 QDQBERT 相关的内容：预训练配置映射、配置
    from .models.qdqbert import QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, QDQBertConfig
    # 从自定义模块中导入 QWEN2 相关的内容：预训练配置映射、配置、分词器
    from .models.qwen2 import QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP, Qwen2Config, Qwen2Tokenizer
    # 从自定义模块中导入 RAG 相关的内容：配置、检索器、分词器
    from .models.rag import RagConfig, RagRetriever, RagTokenizer
    # 从自定义模块中导入 REALM 相关的内容：预训练配置映射、配置、分词器
    from .models.realm import (
        REALM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RealmConfig,
        RealmTokenizer,
    )
    # 从自定义模块中导入 REFORMER 相关的内容：预训练配置映射、配置
    from .models.reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
    # 从自定义模块中导入 REGNET 相关的内容：预训练配置映射、配置
    from .models.regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig
    # 从自定义模块中导入 REMBERT 相关的内容：预训练配置映射、配置
    from .models.rembert import REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RemBertConfig
    # 从自定义模块中导入 RESNET 相关的内容：预训练配置映射、配置
    from .models.resnet import RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ResNetConfig
    # 从自定义模块中导入 ROBERTA 相关的内容：预训练配置映射、配置、分词器
    from .models.roberta import (
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaConfig,
        RobertaTokenizer,
    )
    # 从自定义模块中导入 ROBERTA_PRELAYERNORM 相关的内容：预训练配置映射、配置
    from .models.roberta_prelayernorm import (
        ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaPreLayerNormConfig,
    )
    # 从自定义模块中导入 ROC_BERT 相关的内容：预训练配置映射、配置、分词器
    from .models.roc_bert import (
        ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RoCBertConfig,
        RoCBertTokenizer,
    )
    # 从自定义模块中导入 ROFORMER 相关的内容：预训练配置映射、配置、分词器
    from .models.roformer import (
        ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RoFormerConfig,
        RoFormerTokenizer,
    )
    # 从自定义模块中导入 RWKV 相关的内容：预训练配置映射、配置
    from .models.rwkv import RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP, RwkvConfig
    # 从自定义模块中导入 SAM 相关的内容：预训练配置映射、配置、遮
    # 导入 SEAMLESS_M4T 相关模块
    from .models.seamless_m4t import (
        SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SeamlessM4TConfig,
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
    )
    # 导入 SEAMLESS_M4T_V2 相关模块
    from .models.seamless_m4t_v2 import (
        SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SeamlessM4Tv2Config,
    )
    # 导入 SEGFORMER 相关模块
    from .models.segformer import (
        SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SegformerConfig,
    )
    # 导入 SEW 相关模块
    from .models.sew import SEW_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWConfig
    # 导入 SEW_D 相关模块
    from .models.sew_d import SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP, SEWDConfig
    # 导入 SIGLIP 相关模块
    from .models.siglip import (
        SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SiglipConfig,
        SiglipProcessor,
        SiglipTextConfig,
        SiglipTokenizer,
        SiglipVisionConfig,
    )
    # 导入 SpeechEncoderDecoderConfig 模块
    from .models.speech_encoder_decoder import SpeechEncoderDecoderConfig
    # 导入 SPEECH_TO_TEXT 相关模块
    from .models.speech_to_text import (
        SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Speech2TextConfig,
        Speech2TextFeatureExtractor,
        Speech2TextProcessor,
    )
    # 导入 SPEECH_TO_TEXT_2 相关模块
    from .models.speech_to_text_2 import (
        SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Speech2Text2Config,
        Speech2Text2Processor,
        Speech2Text2Tokenizer,
    )
    # 导入 SPEECHT5 相关模块
    from .models.speecht5 import (
        SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP,
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5HifiGanConfig,
        SpeechT5Processor,
    )
    # 导入 SPLINTER 相关模块
    from .models.splinter import (
        SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SplinterConfig,
        SplinterTokenizer,
    )
    # 导入 SQUEEZEBERT 相关模块
    from .models.squeezebert import (
        SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SqueezeBertConfig,
        SqueezeBertTokenizer,
    )
    # 导入 SWIFTFORMER 相关模块
    from .models.swiftformer import (
        SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwiftFormerConfig,
    )
    # 导入 SWIN 相关模块
    from .models.swin import SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, SwinConfig
    # 导入 SWIN2SR 相关模块
    from .models.swin2sr import SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP, Swin2SRConfig
    # 导入 SWINV2 相关模块
    from .models.swinv2 import SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP, Swinv2Config
    # 导入 SWITCH_TRANSFORMERS 相关模块
    from .models.switch_transformers import (
        SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwitchTransformersConfig,
    )
    # 导入 T5 相关模块
    from .models.t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
    # 导入 TABLE_TRANSFORMER 相关模块
    from .models.table_transformer import (
        TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TableTransformerConfig,
    )
    # 导入 TAPAS 相关模块
    from .models.tapas import (
        TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TapasConfig,
        TapasTokenizer,
    )
    # 导入 TIME_SERIES_TRANSFORMER 相关模块
    from .models.time_series_transformer import (
        TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimeSeriesTransformerConfig,
    )
    # 导入 TIMESFORMER 相关模块
    from .models.timesformer import (
        TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TimesformerConfig,
    )
    # 导入 TimmBackboneConfig 模块
    from .models.timm_backbone import TimmBackboneConfig
    # 导入 TROCR 模块相关内容
    from .models.trocr import (
        TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TrOCRConfig,
        TrOCRProcessor,
    )
    # 导入 TVLT 模块相关内容
    from .models.tvlt import (
        TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TvltConfig,
        TvltFeatureExtractor,
        TvltProcessor,
    )
    # 导入 TVP 模块相关内容
    from .models.tvp import (
        TVP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TvpConfig,
        TvpProcessor,
    )
    # 导入 UMT5 模块相关内容
    from .models.umt5 import UMT5Config
    # 导入 UNISPEECH 模块相关内容
    from .models.unispeech import (
        UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UniSpeechConfig,
    )
    # 导入 UNISPEECH_SAT 模块相关内容
    from .models.unispeech_sat import (
        UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UniSpeechSatConfig,
    )
    # 导入 UNIVNET 模块相关内容
    from .models.univnet import (
        UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UnivNetConfig,
        UnivNetFeatureExtractor,
    )
    # 导入 UperNet 模块相关内容
    from .models.upernet import UperNetConfig
    # 导入 VIDEOMAE 模块相关内容
    from .models.videomae import VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP, VideoMAEConfig
    # 导入 VILT 模块相关内容
    from .models.vilt import (
        VILT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ViltConfig,
        ViltFeatureExtractor,
        ViltImageProcessor,
        ViltProcessor,
    )
    # 导入 VIPLLAVA 模块相关内容
    from .models.vipllava import (
        VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VipLlavaConfig,
    )
    # 导入 VisionEncoderDecoder 模块相关内容
    from .models.vision_encoder_decoder import VisionEncoderDecoderConfig
    # 导入 VisionTextDualEncoder 模块相关内容
    from .models.vision_text_dual_encoder import (
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderProcessor,
    )
    # 导入 VisualBert 模块相关内容
    from .models.visual_bert import (
        VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VisualBertConfig,
    )
    # 导入 ViT 模块相关内容
    from .models.vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig
    # 导入 ViT_HYBRID 模块相关内容
    from .models.vit_hybrid import (
        VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ViTHybridConfig,
    )
    # 导入 ViT_MAE 模块相关内容
    from .models.vit_mae import VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMAEConfig
    # 导入 ViT_MSN 模块相关内容
    from .models.vit_msn import VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMSNConfig
    # 导入 ViTDET 模块相关内容
    from .models.vitdet import VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP, VitDetConfig
    # 导入 ViT_MATTE 模块相关内容
    from .models.vitmatte import VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP, VitMatteConfig
    # 导入 VITS 模块相关内容
    from .models.vits import (
        VITS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VitsConfig,
        VitsTokenizer,
    )
    # 导入 VIVIT 模块相关内容
    from .models.vivit import VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, VivitConfig
    # 导入 WAV2VEC2 模块相关内容
    from .models.wav2vec2 import (
        WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
    )
    # 导入 WAV2VEC2_BERT 模块相关内容
    from .models.wav2vec2_bert import (
        WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2BertConfig,
        Wav2Vec2BertProcessor,
    )
    # 导入 WAV2VEC2_CONFORMER 模块相关内容
    from .models.wav2vec2_conformer import (
        WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Wav2Vec2ConformerConfig,
    )
    # 导入 WAV2VEC2_PHONEME 模块相关内容
    from .models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
    # 导入 WAV2VEC2_WITH_LM 模块相关内容
    from .models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
    # 导入 WAVLM 模型相关内容
    from .models.wavlm import WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP, WavLMConfig
    # 导入 Whisper 模型相关内容
    from .models.whisper import (
        WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )
    # 导入 XCLIP 模型相关内容
    from .models.x_clip import (
        XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XCLIPConfig,
        XCLIPProcessor,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    # 导入 XGLM 模型相关内容
    from .models.xglm import XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XGLMConfig
    # 导入 XLM 模型相关内容
    from .models.xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMTokenizer
    # 导入 XLM-ProphetNet 模型相关内容
    from .models.xlm_prophetnet import (
        XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMProphetNetConfig,
    )
    # 导入 XLM-Roberta 模型相关内容
    from .models.xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaConfig,
    )
    # 导入 XLM-Roberta-XL 模型相关内容
    from .models.xlm_roberta_xl import (
        XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaXLConfig,
    )
    # 导入 XLNet 模型相关内容
    from .models.xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
    # 导入 XMOD 模型相关内容
    from .models.xmod import XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP, XmodConfig
    # 导入 YOLOS 模型相关内容
    from .models.yolos import YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP, YolosConfig
    # 导入 YOSO 模型相关内容
    from .models.yoso import YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP, YosoConfig
    
    # 导入管道相关内容
    from .pipelines import (
        AudioClassificationPipeline,
        AutomaticSpeechRecognitionPipeline,
        Conversation,
        ConversationalPipeline,
        CsvPipelineDataFormat,
        DepthEstimationPipeline,
        DocumentQuestionAnsweringPipeline,
        FeatureExtractionPipeline,
        FillMaskPipeline,
        ImageClassificationPipeline,
        ImageSegmentationPipeline,
        ImageToImagePipeline,
        ImageToTextPipeline,
        JsonPipelineDataFormat,
        MaskGenerationPipeline,
        NerPipeline,
        ObjectDetectionPipeline,
        PipedPipelineDataFormat,
        Pipeline,
        PipelineDataFormat,
        QuestionAnsweringPipeline,
        SummarizationPipeline,
        TableQuestionAnsweringPipeline,
        Text2TextGenerationPipeline,
        TextClassificationPipeline,
        TextGenerationPipeline,
        TextToAudioPipeline,
        TokenClassificationPipeline,
        TranslationPipeline,
        VideoClassificationPipeline,
        VisualQuestionAnsweringPipeline,
        ZeroShotAudioClassificationPipeline,
        ZeroShotClassificationPipeline,
        ZeroShotImageClassificationPipeline,
        ZeroShotObjectDetectionPipeline,
        pipeline,
    )
    # 导入处理工具混合类
    from .processing_utils import ProcessorMixin
    
    # 导入 Tokenizer 基类
    from .tokenization_utils import PreTrainedTokenizer
    # 导入 Tokenizer 基础工具类
    from .tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )
    
    # 工具
    # 从.tools模块导入所需工具类和函数
    from .tools import (
        Agent,
        AzureOpenAiAgent,
        HfAgent,
        LocalAgent,
        OpenAiAgent,
        PipelineTool,
        RemoteTool,
        Tool,
        launch_gradio_demo,
        load_tool,
    )
    
    # Trainer
    # 从.trainer_callback模块导入训练器相关的回调函数和类
    from .trainer_callback import (
        DefaultFlowCallback,
        EarlyStoppingCallback,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    # 从.trainer_utils模块导入训练器相关的实用函数和枚举类型
    from .trainer_utils import (
        EvalPrediction,
        IntervalStrategy,
        SchedulerType,
        enable_full_determinism,
        set_seed,
    )
    # 从.training_args模块导入训练参数相关的类
    from .training_args import TrainingArguments
    # 从.training_args_seq2seq模块导入序列到序列模型训练参数相关的类
    from .training_args_seq2seq import Seq2SeqTrainingArguments
    # 从.training_args_tf模块导入TensorFlow模型训练参数相关的类
    from .training_args_tf import TFTrainingArguments
    
    # Files and general utilities
    # 从.utils模块导入通用工具类和函数，以及一些常量
    from .utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        is_apex_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_decord_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_safetensors_available,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_speech_available,
        is_tensorflow_text_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_tpu_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )
    
    # bitsandbytes config
    # 从.utils.quantization_config模块导入位和字节相关的配置类
    from .utils.quantization_config import AwqConfig, BitsAndBytesConfig, GPTQConfig
    
    # 尝试检查是否安装了sentencepiece库，如果未安装则抛出异常
    try:
        # 如果sentencepiece库不可用，则抛出OptionalDependencyNotAvailable异常
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出OptionalDependencyNotAvailable异常，则执行以下操作
    except OptionalDependencyNotAvailable:
        # 从.utils.dummy_sentencepiece_objects模块导入虚拟的sentencepiece对象
        from .utils.dummy_sentencepiece_objects import *
    # 导入不同模型的对应的 Tokenizer
    else:
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
        from .models.speech_to_text import Speech2TextTokenizer
        from .models.speecht5 import SpeechT5Tokenizer
        from .models.t5 import T5Tokenizer
        from .models.xglm import XGLMTokenizer
        from .models.xlm_prophetnet import XLMProphetNetTokenizer
        from .models.xlm_roberta import XLMRobertaTokenizer
        from .models.xlnet import XLNetTokenizer

    # 检查是否 Tokenizers 可用，若不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入虚拟的 Tokenizers 对象
        from .utils.dummy_tokenizers_objects import *

    # 检查是否 SentencePiece 和 Tokenizers 可用，若不可用则引发异常
    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入虚拟的 SentencePiece 和 Tokenizers 对象
        from .utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        # 导入转换慢速 Tokenizer 的模块
        from .convert_slow_tokenizer import (
            SLOW_TO_FAST_CONVERTERS,
            convert_slow_tokenizer,
        )

    # 检查是否 TensorFlow Text 可用，若不可用则引发异常
    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入虚拟的 TensorFlow Text 对象
        from .utils.dummy_tensorflow_text_objects import *
    else:
        # 导入 TF Bert Tokenizer
        from .models.bert import TFBertTokenizer

    # 检查是否 Keras NLP 可用，若不可用则引发异常
    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 导入虚拟的 Keras NLP 对象
        from .utils.dummy_keras_nlp_objects import *
    else:
        # 导入 TF GPT2 Tokenizer
        from .models.gpt2 import TFGPT2Tokenizer

    # 检查是否 Vision 可用，若不可用则引发异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入 OptionalDependencyNotAvailable 异常，用于处理可选依赖项不可用的情况
    except OptionalDependencyNotAvailable:
        # 如果可选依赖项不可用，则从 .utils.dummy_vision_objects 模块导入所有对象
        from .utils.dummy_vision_objects import *
    # Modeling
    # 尝试检查是否 torch 可用
    try:
        # 如果 torch 不可用，则抛出 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 torch 不可用，则从 .utils.dummy_pt_objects 模块导入所有对象
        from .utils.dummy_pt_objects import *
    # TensorFlow
    # 尝试检查是否 TensorFlow 可用
    try:
        # 如果 TensorFlow 不可用，则抛出 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 TensorFlow 不可用，则从 .utils.dummy_tf_objects 模块导入所有对象
        # 以将它们置于命名空间中。如果用户尝试实例化/使用它们，它们将引发导入错误。
        from .utils.dummy_tf_objects import *
    # 尝试检查是否所有必需的依赖项均可用
    try:
        # 如果有任何一个必需的依赖项不可用，则抛出 OptionalDependencyNotAvailable 异常
        if not (
            is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
            and is_torch_available()
            and is_pretty_midi_available()
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果有任何一个必需的依赖项不可用，则从 .utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects 模块导入所有对象
        from .utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects import *
    else:
        # 如果所有必需的依赖项都可用，则从 .models.pop2piano 模块导入以下对象
        from .models.pop2piano import (
            Pop2PianoFeatureExtractor,
            Pop2PianoProcessor,
            Pop2PianoTokenizer,
        )

    # 尝试检查是否 Flax 可用
    try:
        # 如果 Flax 不可用，则抛出 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 Flax 不可用，则从 .utils.dummy_flax_objects 模块导入所有对象
        # 以将它们置于命名空间中。如果用户尝试实例化/使用它们，它们将引发导入错误。
        from .utils.dummy_flax_objects import *
# 如果前面的条件都不满足，即未找到 TensorFlow、PyTorch >= 2.0 或 Flax，则执行以下操作
else:
    # 导入 sys 模块
    import sys

    # 使用 LazyModule 类将当前模块替换为惰性加载的模块，其中包括模块名、模块文件、导入结构、模块规范以及额外对象
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )

# 如果既没有 TensorFlow 可用，也没有 PyTorch 可用，也没有 Flax 可用，则发出警告
if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
```