# `.\transformers\models\oneformer\configuration_oneformer.py`

```
# 导入必要的模块和类
from typing import Dict, Optional  # 导入用于类型提示的模块和类
# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging
# 从自动模块中导入配置映射
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置存档映射
ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/oneformer_ade20k_swin_tiny": (
        "https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny/blob/main/config.json"
    ),
    # 查看所有 OneFormer 模型的链接
    # 在这里 https://huggingface.co/models?filter=oneformer
}

# OneFormer 配置类，继承自预训练配置类
class OneFormerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`OneFormerModel`] 的配置。根据指定的参数，它用于实例化一个 OneFormer 模型，
    定义模型的架构。使用默认参数实例化一个配置将会产生一个类似于
    [shi-labs/oneformer_ade20k_swin_tiny](https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny)
    架构的配置，该架构在 [ADE20k-150](https://huggingface.co/datasets/scene_parse_150) 上进行了训练。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读
    [`PretrainedConfig`] 的文档以获取更多信息。

    Examples:
    ```python
    >>> from transformers import OneFormerConfig, OneFormerModel

    >>> # 初始化一个 OneFormer shi-labs/oneformer_ade20k_swin_tiny 配置
    >>> configuration = OneFormerConfig()
    >>> # 使用该配置初始化一个（具有随机权重）shi-labs/oneformer_ade20k_swin_tiny 风格的模型
    >>> model = OneFormerModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型
    model_type = "oneformer"
    # 属性映射
    attribute_map = {"hidden_size": "hidden_dim"}
    # 定义模型的初始化方法，初始化模型参数和超参数
    def __init__(
        self,
        backbone_config: Optional[Dict] = None,  # 设置骨干网络的配置字典，默认为空
        ignore_value: int = 255,  # 定义忽略值，默认为255
        num_queries: int = 150,  # 定义查询数，默认为150
        no_object_weight: int = 0.1,  # 定义无对象权重，默认为0.1
        class_weight: float = 2.0,  # 定义类别权重，默认为2.0
        mask_weight: float = 5.0,  # 定义掩码权重，默认为5.0
        dice_weight: float = 5.0,  # 定义 Dice 损失权重，默认为5.0
        contrastive_weight: float = 0.5,  # 定义对比损失权重，默认为0.5
        contrastive_temperature: float = 0.07,  # 定义对比温度，默认为0.07
        train_num_points: int = 12544,  # 定义训练点数，默认为12544
        oversample_ratio: float = 3.0,  # 定义过采样比率，默认为3.0
        importance_sample_ratio: float = 0.75,  # 定义重要性采样比率，默认为0.75
        init_std: float = 0.02,  # 定义初始化标准差，默认为0.02
        init_xavier_std: float = 1.0,  # 定义 Xavier 初始化的标准差，默认为1.0
        layer_norm_eps: float = 1e-05,  # 定义层归一化的 epsilon，默认为1e-05
        is_training: bool = False,  # 定义是否为训练模式，默认为False
        use_auxiliary_loss: bool = True,  # 定义是否使用辅助损失，默认为True
        output_auxiliary_logits: bool = True,  # 定义是否输出辅助的 logits，默认为True
        strides: Optional[list] = [4, 8, 16, 32],  # 定义步长列表，默认为[4, 8, 16, 32]
        task_seq_len: int = 77,  # 定义任务序列长度，默认为77
        text_encoder_width: int = 256,  # 定义文本编码器宽度，默认为256
        text_encoder_context_length: int = 77,  # 定义文本编码器上下文长度，默认为77
        text_encoder_num_layers: int = 6,  # 定义文本编码器层数，默认为6
        text_encoder_vocab_size: int = 49408,  # 定义文本编码器词汇表大小，默认为49408
        text_encoder_proj_layers: int = 2,  # 定义文本编码器投影层数，默认为2
        text_encoder_n_ctx: int = 16,  # 定义文本编码器 n_ctx，默认为16
        conv_dim: int = 256,  # 定义卷积维度，默认为256
        mask_dim: int = 256,  # 定义掩码维度，默认为256
        hidden_dim: int = 256,  # 定义隐藏层维度，默认为256
        encoder_feedforward_dim: int = 1024,  # 定义编码器前馈网络维度，默认为1024
        norm: str = "GN",  # 定义归一化方式，默认为"GN" (Group Normalization)
        encoder_layers: int = 6,  # 定义编码器层数，默认为6
        decoder_layers: int = 10,  # 定义解码器层数，默认为10
        use_task_norm: bool = True,  # 定义是否使用任务归一化，默认为True
        num_attention_heads: int = 8,  # 定义注意力头数，默认为8
        dropout: float = 0.1,  # 定义 dropout 概率，默认为0.1
        dim_feedforward: int = 2048,  # 定义前馈网络维度，默认为2048
        pre_norm: bool = False,  # 定义是否在层归一化之前应用 dropout，默认为False
        enforce_input_proj: bool = False,  # 定义是否强制输入投影，默认为False
        query_dec_layers: int = 2,  # 定义查询解码器层数，默认为2
        common_stride: int = 4,  # 定义公共步长，默认为4
        **kwargs,  # 其他未指定的参数
    # 根据传入的 backbone_config 参数初始化配置
    def __init__(
        self,
        backbone_config=None,
        ignore_value=-100,
        num_queries=100,
        no_object_weight=0.1,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        contrastive_weight=1.0,
        contrastive_temperature=0.07,
        train_num_points=12,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        init_std=0.02,
        init_xavier_std=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        use_auxiliary_loss=True,
        output_auxiliary_logits=True,
        strides=[4, 8, 16, 32],
        task_seq_len=77,
        text_encoder_width=768,
        text_encoder_context_length=77,
        text_encoder_num_layers=12,
        text_encoder_vocab_size=50257,
        text_encoder_proj_layers=2,
        text_encoder_n_ctx=77,
        conv_dim=256,
        mask_dim=256,
        hidden_dim=256,
        encoder_feedforward_dim=1024,
        norm="layer_norm",
        encoder_layers=6,
        decoder_layers=6,
        use_task_norm=True,
        num_attention_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pre_norm=True,
        enforce_input_proj=True,
        query_dec_layers=1,
        common_stride=4,
        num_hidden_layers=6,
        **kwargs
    ):
        # 如果未传入 backbone_config，则使用默认的 Swin 配置
        if backbone_config is None:
            logger.info("`backbone_config` is unset. Initializing the config with the default `Swin` backbone.")
            backbone_config = CONFIG_MAPPING["swin"](
                image_size=224,
                in_channels=3,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                use_absolute_embeddings=False,
                out_features=["stage1", "stage2", "stage3", "stage4"],
            )
        # 如果 backbone_config 是字典格式，则使用其中的 model_type 从 CONFIG_MAPPING 中获取对应的配置类
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)
    
        # 将 backbone_config 赋值给 self.backbone_config
        self.backbone_config = backbone_config
    
        # 将其他参数赋值给对应属性
        self.ignore_value = ignore_value
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.layer_norm_eps = layer_norm_eps
        self.is_training = is_training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.output_auxiliary_logits = output_auxiliary_logits
        self.strides = strides
        self.task_seq_len = task_seq_len
        self.text_encoder_width = text_encoder_width
        self.text_encoder_context_length = text_encoder_context_length
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_vocab_size = text_encoder_vocab_size
        self.text_encoder_proj_layers = text_encoder_proj_layers
        self.text_encoder_n_ctx = text_encoder_n_ctx
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.norm = norm
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_task_norm = use_task_norm
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_proj = enforce_input_proj
        self.query_dec_layers = query_dec_layers
        self.common_stride = common_stride
        self.num_hidden_layers = decoder_layers
    
        # 调用父类的 __init__ 方法
        super().__init__(**kwargs)
```