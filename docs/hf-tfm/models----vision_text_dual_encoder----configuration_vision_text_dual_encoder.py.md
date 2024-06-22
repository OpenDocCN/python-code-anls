# `.\transformers\models\vision_text_dual_encoder\configuration_vision_text_dual_encoder.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关特定语言的权限和限制

""" VisionTextDualEncoder 模型配置"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig
from ..clip.configuration_clip import CLIPVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# VisionTextDualEncoderConfig 类继承自 PretrainedConfig 类
class VisionTextDualEncoderConfig(PretrainedConfig):
    r"""
    [`VisionTextDualEncoderConfig`] 是用于存储 [`VisionTextDualEncoderModel`] 配置的类。
    根据指定的参数实例化 [`VisionTextDualEncoderModel`] 模型，定义文本模型和视觉模型配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        projection_dim (`int`, *optional*, 默认为 512):
            文本和视觉投影层的维度。
        logit_scale_init_value (`float`, *optional*, 默认为 2.6592):
            *logit_scale* 参数的初始值。默认值根据原始 CLIP 实现使用。
        kwargs (*optional*):
            关键字参数的字典。

    Examples:

    ```python
    >>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

    >>> # 初始化一个 BERT 和 ViT 配置
    >>> config_vision = ViTConfig()
    >>> config_text = BertConfig()

    >>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

    >>> # 初始化一个 BERT 和 ViT 模型（带有随机权重）
    >>> model = VisionTextDualEncoderModel(config=config)

    >>> # 访问模型配置
    >>> config_vision = model.config.vision_config
    >>> config_text = model.config.text_config

    >>> # 保存模型，包括其配置
    >>> model.save_pretrained("vit-bert")

    >>> # 从预训练文件夹加载模型和配置
    >>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
    >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)
    ```py"""

    model_type = "vision-text-dual-encoder"
    is_composition = True
    # 初始化方法，设置投影维度和logit缩放初始值
    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 检查是否传入了vision_config参数
        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")

        # 检查是否传入了text_config参数
        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")

        # 从kwargs中弹出vision_config和text_config参数
        vision_config = kwargs.pop("vision_config")
        text_config = kwargs.pop("text_config")

        # 弹出vision_config和text_config中的model_type参数
        vision_model_type = vision_config.pop("model_type")
        text_model_type = text_config.pop("model_type")

        # 根据vision_model_type的值选择不同的配置方式
        if vision_model_type == "clip":
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config).vision_config
        elif vision_model_type == "clip_vision_model":
            self.vision_config = CLIPVisionConfig(**vision_config)
        else:
            self.vision_config = AutoConfig.for_model(vision_model_type, **vision_config)

        # 根据text_model_type的值选择配置方式
        self.text_config = AutoConfig.for_model(text_model_type, **text_config)

        # 设置投影维度和logit缩放初始值
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    # 类方法，从vision和text的配置中实例化一个VisionTextDualEncoderConfig对象
    @classmethod
    def from_vision_text_configs(cls, vision_config: PretrainedConfig, text_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        # 返回一个VisionTextDualEncoderConfig对象，传入vision_config和text_config的字典形式
        return cls(vision_config=vision_config.to_dict(), text_config=text_config.to_dict(), **kwargs)
```