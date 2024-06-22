# `.\transformers\models\upernet\configuration_upernet.py`

```py
# 代码文件的编码格式为UTF-8
# 版权声明
# 这是一个UperNet模型的配置类

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具
from ..auto.configuration_auto import CONFIG_MAPPING  # 导入自动配置映射

logger = logging.get_logger(__name__)  # 获取日志记录器

class UperNetConfig(PretrainedConfig):  # 创建UperNetConfig类，继承自PretrainedConfig类
    r"""  # 类的文档字符串
    This is the configuration class to store the configuration of an [`UperNetForSemanticSegmentation`]. It is used to
    instantiate an UperNet model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UperNet
    [openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:  # 参数说明
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `ResNetConfig()`):  # 背骨模型的配置
            The configuration of the backbone model.
        hidden_size (`int`, *optional*, defaults to 512):  # 隐藏层的单元数
            The number of hidden units in the convolutional layers.
        initializer_range (`float`, *optional*, defaults to 0.02):  # 截断正态分布初始化权重矩阵的标准差
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        pool_scales (`Tuple[int]`, *optional*, defaults to `[1, 2, 3, 6]`):  # 应用于最后特征图的池化标度
            Pooling scales used in Pooling Pyramid Module applied on the last feature map.
        use_auxiliary_head (`bool`, *optional*, defaults to `True`):  # 是否在训练时使用辅助头
            Whether to use an auxiliary head during training.
        auxiliary_loss_weight (`float`, *optional*, defaults to 0.4):  # 辅助头的交叉熵损失权重
            Weight of the cross-entropy loss of the auxiliary head.
        auxiliary_channels (`int`, *optional*, defaults to 256):  # 辅助头中要使用的通道数
            Number of channels to use in the auxiliary head.
        auxiliary_num_convs (`int`, *optional*, defaults to 1):  # 辅助头中要使用的卷积层数
            Number of convolutional layers to use in the auxiliary head.
        auxiliary_concat_input (`bool`, *optional*, defaults to `False`):  # 是否在分类层前将辅助头的输出与输入进行连接
            Whether to concatenate the output of the auxiliary head with the input before the classification layer.
        loss_ignore_index (`int`, *optional*, defaults to 255):  # 损失函数忽略的索引
            The index that is ignored by the loss function.
    Examples:
    
    >>> from transformers import UperNetConfig, UperNetForSemanticSegmentation
    
    >>> # Initializing a configuration
    >>> configuration = UperNetConfig()
    
    >>> # Initializing a model (with random weights) from the configuration
    >>> model = UperNetForSemanticSegmentation(configuration)
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    
    
    model_type = "upernet"
    
    def __init__(
        self,
        backbone_config=None,
        hidden_size=512,
        initializer_range=0.02,
        pool_scales=[1, 2, 3, 6],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_in_channels=384,
        auxiliary_channels=256,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        loss_ignore_index=255,
        **kwargs,
    ):
        # 使用继承的构造函数并传递关键字参数
        super().__init__(**kwargs)
    
        # 如果没有传入 backbone_config，则使用默认的 ResNet backbone
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
            backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage1", "stage2", "stage3", "stage4"])
        # 如果传入的 backbone_config 是字典类型，则初始化对应的配置类
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)
    
        # 设置实例变量
        self.backbone_config = backbone_config
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.pool_scales = pool_scales
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_in_channels = auxiliary_in_channels
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.loss_ignore_index = loss_ignore_index
```