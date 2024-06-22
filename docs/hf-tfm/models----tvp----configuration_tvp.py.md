# `.\transformers\models\tvp\configuration_tvp.py`

```py
# 该代码定义了一个 TvpConfig 类，用于存储和配置 TvpModel 模型的相关参数
# 这个类继承自 PretrainedConfig 类，提供了配置模型架构的功能
# 通过实例化该类并设置对应的参数，可以创建一个 TvpModel 模型并进行初始化

# 导入必要的模块和类
import copy
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置的映射
TVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/tvp-base": "https://huggingface.co/Intel/tvp-base/resolve/main/config.json",
}

# 定义 TvpConfig 类
class TvpConfig(PretrainedConfig):
    # 设置模型类型为 "tvp"
    model_type = "tvp"

    # 在初始化时设置各种配置参数的默认值
    def __init__(self,
                 backbone_config=None,
                 distance_loss_weight=1.0,
                 duration_loss_weight=0.1,
                 visual_prompter_type="framepad",
                 visual_prompter_apply="replace",
                 visual_prompt_size=96,
                 max_img_size=448,
                 num_frames=48,
                 vocab_size=30522,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=512,
                 max_grid_col_position_embeddings=100,
                 max_grid_row_position_embeddings=100,
                 hidden_dropout_prob=0.1,
                 hidden_act="gelu",
                 layer_norm_eps=1e-12,
                 initializer_range=0.02,
                 attention_probs_dropout_prob=0.1,
                 **kwargs):
        pass


这段代码定义了一个 `TvpConfig` 类,用于存储和配置 `TvpModel` 模型的相关参数。该类继承自 `PretrainedConfig` 类,提供了配置模型架构的功能。通过实例化该类并设置对应的参数,可以创建一个 `TvpModel` 模型并进行初始化。这个类中包含了很多配置参数,如输入图像大小、帧数、词表大小、模型大小等,可以根据实际需求进行调整。此外,还定义了一个 `TVP_PRETRAINED_CONFIG_ARCHIVE_MAP` 字典,用于存储预训练模型配置的映射。
    ):
        # 调用父类的构造函数，传入所有关键字参数
        super().__init__(**kwargs)

        # 如果没有指定 backbone_config，则使用默认的 ResNet 骨干网络配置
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
            backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
        # 如果 backbone_config 是字典类型，则将其转换为相应的配置类对象
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        # 设置各种属性值
        self.backbone_config = backbone_config
        self.distance_loss_weight = distance_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.visual_prompter_type = visual_prompter_type
        self.visual_prompter_apply = visual_prompter_apply
        self.visual_prompt_size = visual_prompt_size
        self.max_img_size = max_img_size
        self.num_frames = num_frames
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.max_grid_col_position_embeddings = max_grid_col_position_embeddings
        self.max_grid_row_position_embeddings = max_grid_row_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

    @classmethod
    def from_backbone_config(cls, backbone_config: PretrainedConfig, **kwargs):
        """Instantiate a [`TvpConfig`] (or a derived class) from a pre-trained backbone model configuration.

        Args:
            backbone_config ([`PretrainedConfig`]):
                The backbone configuration.
        Returns:
            [`TvpConfig`]: An instance of a configuration object
        """
        # 从预训练的骨干网络模型配置实例化一个 TvpConfig 对象
        return cls(backbone_config=backbone_config, **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将当前实例序列化为 Python 字典
        output = copy.deepcopy(self.__dict__)
        # 如果 backbone_config 不为 None，则将其转换为字典
        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()
        # 添加 model_type 属性到输出字典中
        output["model_type"] = self.__class__.model_type
        return output
```