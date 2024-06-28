# `.\models\deformable_detr\configuration_deformable_detr.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明版权及许可条款
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 如果适用法律要求或书面同意，软件按“原样”分发，不提供任何形式的担保或条件
# 请查看许可证了解特定语言下的权限和限制
""" Deformable DETR 模型配置 """

# 导入必要的配置和日志模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置映射字典，将模型名称映射到其预训练配置文件的 URL
DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "SenseTime/deformable-detr": "https://huggingface.co/sensetime/deformable-detr/resolve/main/config.json",
    # 查看所有 Deformable DETR 模型，请访问 https://huggingface.co/models?filter=deformable-detr
}

# DeformableDetrConfig 类，继承自 PretrainedConfig 类
class DeformableDetrConfig(PretrainedConfig):
    r"""
    这是用于存储 [`DeformableDetrModel`] 配置的类。它用于根据指定参数实例化 Deformable DETR 模型，定义模型架构。
    使用默认配置来实例化对象将会生成类似于 Deformable DETR [SenseTime/deformable-detr]
    (https://huggingface.co/SenseTime/deformable-detr) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    Examples:

    ```python
    >>> from transformers import DeformableDetrConfig, DeformableDetrModel

    >>> # 初始化一个 Deformable DETR SenseTime/deformable-detr 风格的配置
    >>> configuration = DeformableDetrConfig()

    >>> # 从指定配置文件初始化一个（带有随机权重）SenseTime/deformable-detr 风格的模型
    >>> model = DeformableDetrModel(configuration)

    >>> # 访问模型的配置
    >>> configuration = model.config
    ```
    """

    # 模型类型
    model_type = "deformable_detr"

    # 属性映射字典，将配置文件中的属性名称映射到 Deformable DETR 模型中相应的属性名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化函数，用于创建一个新的对象实例，初始化各种参数和属性
    def __init__(
        self,
        use_timm_backbone=True,  # 是否使用timm库中的backbone模型作为特征提取器，默认为True
        backbone_config=None,  # backbone模型的配置参数，默认为None
        num_channels=3,  # 输入图像的通道数，默认为3（RGB图像）
        num_queries=300,  # 查询数量，用于查询Transformer解码器输出的对象位置，默认为300
        max_position_embeddings=1024,  # 最大位置编码数，默认为1024
        encoder_layers=6,  # Transformer编码器层数，默认为6
        encoder_ffn_dim=1024,  # Transformer编码器中FeedForward层的维度，默认为1024
        encoder_attention_heads=8,  # Transformer编码器中注意力头的数量，默认为8
        decoder_layers=6,  # Transformer解码器层数，默认为6
        decoder_ffn_dim=1024,  # Transformer解码器中FeedForward层的维度，默认为1024
        decoder_attention_heads=8,  # Transformer解码器中注意力头的数量，默认为8
        encoder_layerdrop=0.0,  # Transformer编码器中每层dropout的比例，默认为0.0（不使用dropout）
        is_encoder_decoder=True,  # 是否使用编码-解码结构，默认为True
        activation_function="relu",  # 激活函数的类型，默认为ReLU
        d_model=256,  # Transformer模型中的隐藏层维度，默认为256
        dropout=0.1,  # 模型中的普通dropout比例，默认为0.1
        attention_dropout=0.0,  # 注意力机制中的dropout比例，默认为0.0（不使用dropout）
        activation_dropout=0.0,  # 激活函数中的dropout比例，默认为0.0（不使用dropout）
        init_std=0.02,  # 初始化模型参数的标准差，默认为0.02
        init_xavier_std=1.0,  # Xavier初始化中的标准差，默认为1.0
        return_intermediate=True,  # 是否返回中间层的输出，默认为True
        auxiliary_loss=False,  # 是否使用辅助损失，默认为False
        position_embedding_type="sine",  # 位置编码的类型，默认为"sine"（正弦位置编码）
        backbone="resnet50",  # 使用的backbone模型，默认为"resnet50"
        use_pretrained_backbone=True,  # 是否使用预训练的backbone模型，默认为True
        backbone_kwargs=None,  # backbone模型的其他参数，默认为None
        dilation=False,  # 是否使用空洞卷积（dilation convolution），默认为False
        num_feature_levels=4,  # 特征级别的数量，默认为4
        encoder_n_points=4,  # 编码器中位置嵌入的点数，默认为4
        decoder_n_points=4,  # 解码器中位置嵌入的点数，默认为4
        two_stage=False,  # 是否使用两阶段检测器，默认为False
        two_stage_num_proposals=300,  # 第二阶段的提议数量，默认为300
        with_box_refine=False,  # 是否使用边界框细化，默认为False
        class_cost=1,  # 类别损失的系数，默认为1
        bbox_cost=5,  # 边界框损失的系数，默认为5
        giou_cost=2,  # GIoU损失的系数，默认为2
        mask_loss_coefficient=1,  # 掩膜损失的系数，默认为1
        dice_loss_coefficient=1,  # Dice损失的系数，默认为1
        bbox_loss_coefficient=5,  # 边界框损失的系数，默认为5
        giou_loss_coefficient=2,  # GIoU损失的系数，默认为2
        eos_coefficient=0.1,  # EOS（结束符）损失的系数，默认为0.1
        focal_alpha=0.25,  # Focal损失的alpha参数，默认为0.25
        disable_custom_kernels=False,  # 是否禁用自定义内核，默认为False
        **kwargs,  # 其他未列出的关键字参数
    ):
        # 继承类的初始化方法
        super().__init__(**kwargs)

    @property
    def num_attention_heads(self) -> int:
        # 返回编码器中的注意力头数量
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        # 返回模型中的隐藏层维度
        return self.d_model
```