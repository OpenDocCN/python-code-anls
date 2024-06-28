# `.\models\vision_text_dual_encoder\configuration_vision_text_dual_encoder.py`

```py
# 设置文件编码为 UTF-8

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..chinese_clip.configuration_chinese_clip import ChineseCLIPVisionConfig
from ..clip.configuration_clip import CLIPVisionConfig
from ..siglip.configuration_siglip import SiglipVisionConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义不同视觉模型配置类的映射关系
VISION_MODEL_CONFIGS = {
    "clip_vision_model": CLIPVisionConfig,
    "chinese_clip_vision_model": ChineseCLIPVisionConfig,
    "siglip_vision_model": SiglipVisionConfig,
}

# VisionTextDualEncoderConfig 类继承自 PretrainedConfig 类，用于存储 VisionTextDualEncoderModel 的配置信息
class VisionTextDualEncoderConfig(PretrainedConfig):
    r"""
    [`VisionTextDualEncoderConfig`] 是一个配置类，用于存储 [`VisionTextDualEncoderModel`] 的配置信息。
    根据指定的参数实例化 [`VisionTextDualEncoderModel`] 模型，定义了文本模型和视觉模型的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    Args:
        projection_dim (`int`, *optional*, defaults to 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值。默认值按照原始 CLIP 实现使用。
        kwargs (*optional*):
            字典形式的关键字参数。

    Examples:

    ```
    >>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

    >>> # 初始化 BERT 和 ViT 的配置
    >>> config_vision = ViTConfig()
    >>> config_text = BertConfig()

    >>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

    >>> # 初始化一个带有随机权重的 BERT 和 ViT 模型
    >>> model = VisionTextDualEncoderModel(config=config)

    >>> # 访问模型配置
    >>> config_vision = model.config.vision_config
    >>> config_text = model.config.text_config

    >>> # 保存模型及其配置
    >>> model.save_pretrained("vit-bert")

    >>> # 从预训练文件夹加载模型和配置
    ```
    # 从预训练模型“vit-bert”加载视觉文本双编码器配置
    vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
    # 使用加载的配置实例化视觉文本双编码器模型
    model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)



    # 设定模型类型为“vision-text-dual-encoder”
    model_type = "vision-text-dual-encoder"
    # 表示这个类是一个复合类
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查是否提供了视觉配置参数
        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")
        
        # 检查是否提供了文本配置参数
        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")
        
        # 弹出并获取视觉配置参数
        vision_config = kwargs.pop("vision_config")
        # 弹出并获取文本配置参数
        text_config = kwargs.pop("text_config")
        
        # 获取视觉模型类型
        vision_model_type = vision_config.pop("model_type")
        # 获取文本模型类型
        text_model_type = text_config.pop("model_type")
        
        # 根据视觉模型类型获取对应的配置类
        vision_config_class = VISION_MODEL_CONFIGS.get(vision_model_type)
        # 如果找到了对应的配置类，则使用提供的视觉配置参数实例化它
        if vision_config_class is not None:
            self.vision_config = vision_config_class(**vision_config)
        # 否则，根据视觉模型类型和参数自动创建一个配置实例
        else:
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)
            # 如果这个配置实例本身有一个名为`vision_config`的属性，则将其设置为当前实例的`vision_config`
            if hasattr(self.vision_config, "vision_config"):
                self.vision_config = self.vision_config.vision_config
        
        # 根据文本模型类型和参数自动创建一个文本配置实例
        self.text_config = AutoConfig.for_model(text_model_type, **text_config)
        
        # 设置投影维度参数
        self.projection_dim = projection_dim
        # 设置对数尺度初始化值参数
        self.logit_scale_init_value = logit_scale_init_value



    @classmethod
    def from_vision_text_configs(cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs):
        """
        从视觉模型配置和文本模型配置实例化一个[`VisionTextDualEncoderConfig`]（或其派生类）。

        Args:
            vision_config (PretrainedConfig): 视觉模型配置的实例
            text_config (PretrainedConfig): 文本模型配置的实例
            **kwargs: 其他参数

        Returns:
            VisionTextDualEncoderConfig: 配置对象的一个实例
        """
        return cls(vision_config=vision_config.to_dict(), text_config=text_config.to_dict(), **kwargs)


这些注释为每行代码提供了详细的解释，包括代码的目的、参数的作用以及返回值的说明。
```