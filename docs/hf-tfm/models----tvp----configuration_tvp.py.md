# `.\models\tvp\configuration_tvp.py`

```py
# coding=utf-8
# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" TVP model configuration"""

import copy  # 导入copy模块，用于复制对象

from ...configuration_utils import PretrainedConfig  # 导入预训练配置的基类
from ...utils import logging  # 导入日志工具
from ..auto import CONFIG_MAPPING  # 导入自动配置映射


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


TVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/tvp-base": "https://huggingface.co/Intel/tvp-base/resolve/main/config.json",
}  # TVP预训练模型配置文件的映射表，指定模型名称和对应的配置文件URL


class TvpConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TvpModel`]. It is used to instantiate an Tvp
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Tvp
    [Intel/tvp-base](https://huggingface.co/Intel/tvp-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """

    model_type = "tvp"  # 模型类型标识为"tvp"

    def __init__(
        self,
        backbone_config=None,  # 背景配置（backbone）的配置参数，默认为None
        backbone=None,  # 背景模型，默认为None
        use_pretrained_backbone=False,  # 是否使用预训练的背景模型，默认为False
        use_timm_backbone=False,  # 是否使用timm库中的背景模型，默认为False
        backbone_kwargs=None,  # 背景模型的其他参数配置，默认为None
        distance_loss_weight=1.0,  # 距离损失的权重，默认为1.0
        duration_loss_weight=0.1,  # 时长损失的权重，默认为0.1
        visual_prompter_type="framepad",  # 视觉提示器的类型，默认为"framepad"
        visual_prompter_apply="replace",  # 视觉提示器的应用方式，默认为"replace"
        visual_prompt_size=96,  # 视觉提示的大小，默认为96
        max_img_size=448,  # 最大图像尺寸，默认为448
        num_frames=48,  # 图像中的帧数，默认为48
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        intermediate_size=3072,  # 中间层大小，默认为3072
        num_hidden_layers=12,  # 隐藏层数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        max_position_embeddings=512,  # 最大位置嵌入数，默认为512
        max_grid_col_position_embeddings=100,  # 最大网格列位置嵌入数，默认为100
        max_grid_row_position_embeddings=100,  # 最大网格行位置嵌入数，默认为100
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为"gelu"
        layer_norm_eps=1e-12,  # 层归一化的epsilon值，默认为1e-12
        initializer_range=0.02,  # 初始化范围，默认为0.02
        attention_probs_dropout_prob=0.1,  # 注意力概率dropout概率，默认为0.1
        **kwargs,
    ):
        """
        Initialize the TvpConfig with specific model configuration parameters.

        Args:
            backbone_config (Optional): Configuration for the backbone, default is None.
            backbone (Optional): Backbone model, default is None.
            use_pretrained_backbone (bool): Whether to use a pretrained backbone model, default is False.
            use_timm_backbone (bool): Whether to use a backbone model from the timm library, default is False.
            backbone_kwargs (Optional): Additional parameters for the backbone model, default is None.
            distance_loss_weight (float): Weight for the distance loss, default is 1.0.
            duration_loss_weight (float): Weight for the duration loss, default is 0.1.
            visual_prompter_type (str): Type of visual prompter, default is "framepad".
            visual_prompter_apply (str): Application method of visual prompter, default is "replace".
            visual_prompt_size (int): Size of the visual prompt, default is 96.
            max_img_size (int): Maximum image size, default is 448.
            num_frames (int): Number of frames in the image, default is 48.
            vocab_size (int): Size of the vocabulary, default is 30522.
            hidden_size (int): Size of the hidden layers, default is 768.
            intermediate_size (int): Size of the intermediate layers, default is 3072.
            num_hidden_layers (int): Number of hidden layers, default is 12.
            num_attention_heads (int): Number of attention heads, default is 12.
            max_position_embeddings (int): Maximum position embeddings, default is 512.
            max_grid_col_position_embeddings (int): Maximum grid column position embeddings, default is 100.
            max_grid_row_position_embeddings (int): Maximum grid row position embeddings, default is 100.
            hidden_dropout_prob (float): Dropout probability for hidden layers, default is 0.1.
            hidden_act (str): Activation function for hidden layers, default is "gelu".
            layer_norm_eps (float): Epsilon value for layer normalization, default is 1e-12.
            initializer_range (float): Range for weight initialization, default is 0.02.
            attention_probs_dropout_prob (float): Dropout probability for attention probabilities, default is 0.1.
            **kwargs: Additional keyword arguments for potential future updates.
        """
        super().__init__(**kwargs)  # 调用父类PretrainedConfig的初始化方法，传入额外的关键字参数
    ):
        # 调用父类的初始化方法，传入所有的关键字参数
        super().__init__(**kwargs)
        # 如果使用预训练的主干网络，抛出值错误异常
        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        # 如果既指定了主干网络配置又指定了主干网络模型，抛出值错误异常
        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        # 如果既未指定主干网络配置又未指定主干网络模型，记录日志并使用默认的 ResNet 主干网络配置进行初始化
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
            backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage4"])
        # 如果主干网络配置是字典类型，则根据模型类型从字典创建配置类实例
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        # 如果既指定了主干网络配置参数又指定了主干网络关键字参数，抛出值错误异常
        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

        # 初始化对象的各个属性
        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
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
        # 使用给定的主干网络配置实例化一个 [`TvpConfig`]（或其派生类）对象
        return cls(backbone_config=backbone_config, **kwargs)
    def to_dict(self):
        """
        将当前实例序列化为一个 Python 字典。重写默认的 [`~PretrainedConfig.to_dict`] 方法。

        Returns:
            `Dict[str, any]`: 包含此配置实例所有属性的字典，
        """
        # 深拷贝当前实例的所有属性到 output 变量中
        output = copy.deepcopy(self.__dict__)
        
        # 如果 backbone_config 属性不为 None，则将其转换为字典形式
        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()
        
        # 将 model_type 属性设置为当前类的模型类型
        output["model_type"] = self.__class__.model_type
        
        # 返回序列化后的字典
        return output
```