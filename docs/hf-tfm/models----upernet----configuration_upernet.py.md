# `.\models\upernet\configuration_upernet.py`

```
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按现状”提供的，不附带任何明示或暗示的担保或条件。
# 请参阅许可证获取详细信息。
""" UperNet 模型配置"""


from ...configuration_utils import PretrainedConfig  # 导入预配置类
from ...utils import logging  # 导入日志工具
from ..auto.configuration_auto import CONFIG_MAPPING  # 导入自动配置映射


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class UperNetConfig(PretrainedConfig):
    r"""
    这是用于存储 [`UperNetForSemanticSegmentation`] 配置的类。它用于根据指定的参数实例化 UperNet 模型，
    定义模型的架构。使用默认值实例化配置会产生类似于 UperNet
    [openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读
    [`PretrainedConfig`] 的文档获取更多信息。
    """
    # 设置模型类型为 "upernet"
    model_type = "upernet"
    def __init__(
        self,
        backbone_config=None,  # 初始化函数的参数：用于指定主干网络的配置
        backbone=None,  # 初始化函数的参数：用于指定主干网络的实例
        use_pretrained_backbone=False,  # 初始化函数的参数：是否使用预训练的主干网络
        use_timm_backbone=False,  # 初始化函数的参数：是否使用timm库中的主干网络
        backbone_kwargs=None,  # 初始化函数的参数：主干网络的额外参数
        hidden_size=512,  # 初始化函数的参数：隐藏层的大小
        initializer_range=0.02,  # 初始化函数的参数：权重初始化的范围
        pool_scales=[1, 2, 3, 6],  # 初始化函数的参数：池化操作的尺度
        use_auxiliary_head=True,  # 初始化函数的参数：是否使用辅助头部
        auxiliary_loss_weight=0.4,  # 初始化函数的参数：辅助损失的权重
        auxiliary_in_channels=384,  # 初始化函数的参数：辅助头部的输入通道数
        auxiliary_channels=256,  # 初始化函数的参数：辅助头部的通道数
        auxiliary_num_convs=1,  # 初始化函数的参数：辅助头部的卷积层数量
        auxiliary_concat_input=False,  # 初始化函数的参数：辅助头部是否将输入进行拼接
        loss_ignore_index=255,  # 初始化函数的参数：损失函数中需要忽略的索引值
        **kwargs,  # 初始化函数的参数：其他未命名参数
    ):
        super().__init__(**kwargs)  # 调用父类的初始化函数

        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")  # 如果使用预训练的主干网络，则抛出错误

        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")  # 如果同时指定了主干网络实例和配置，则抛出错误

        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
            backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage1", "stage2", "stage3", "stage4"])
            # 如果主干网络配置为空且主干网络实例也为空，则使用默认的ResNet主干网络配置进行初始化
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)
            # 如果主干网络配置是一个字典，则根据字典中的信息初始化主干网络配置

        if backbone_kwargs is not None and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")
            # 如果同时指定了主干网络的额外参数和主干网络配置，则抛出错误

        # 将所有初始化的参数保存到类的属性中
        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
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