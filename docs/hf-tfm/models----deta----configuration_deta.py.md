# `.\models\deta\configuration_deta.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可证声明，告知代码使用者版权和许可条件
# 仅在遵守 Apache 许可证 Version 2.0 的情况下可使用本文件
# 可以从指定的网址获取完整的许可证文本
# 根据适用法律或书面同意，本软件以"原样"提供，不带任何明示或暗示的担保或条件
# 详细许可证信息请参见指定的网址

""" DETA model configuration"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# DETA 预训练配置文件映射，指定了模型名称和其对应的配置文件 URL
DETA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ut/deta": "https://huggingface.co/ut/deta/resolve/main/config.json",
}

# DetaConfig 类继承自 PretrainedConfig 类，用于存储 DETA 模型的配置信息
class DetaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DetaModel`]. It is used to instantiate a DETA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DETA
    [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```
    >>> from transformers import DetaConfig, DetaModel

    >>> # Initializing a DETA SenseTime/deformable-detr style configuration
    >>> configuration = DetaConfig()

    >>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
    >>> model = DetaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 指定模型类型为 "deta"
    model_type = "deta"
    # 定义属性映射，将通用名称映射到具体模型配置参数名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化方法，用于初始化模型对象
    def __init__(
        self,
        backbone_config=None,  # 设置用于骨干网络的配置
        backbone=None,  # 设置用于骨干网络的具体实现
        use_pretrained_backbone=False,  # 是否使用预训练的骨干网络参数
        use_timm_backbone=False,  # 是否使用timm库提供的骨干网络
        backbone_kwargs=None,  # 骨干网络的额外参数
        num_queries=900,  # 查询的数量
        max_position_embeddings=2048,  # 最大位置嵌入数
        encoder_layers=6,  # 编码器层数
        encoder_ffn_dim=2048,  # 编码器中FFN层的维度
        encoder_attention_heads=8,  # 编码器中注意力头的数量
        decoder_layers=6,  # 解码器层数
        decoder_ffn_dim=1024,  # 解码器中FFN层的维度
        decoder_attention_heads=8,  # 解码器中注意力头的数量
        encoder_layerdrop=0.0,  # 编码器层的dropout率
        is_encoder_decoder=True,  # 模型是否为编码-解码结构
        activation_function="relu",  # 激活函数类型
        d_model=256,  # 模型的维度
        dropout=0.1,  # 全局的dropout率
        attention_dropout=0.0,  # 注意力机制的dropout率
        activation_dropout=0.0,  # 激活函数的dropout率
        init_std=0.02,  # 初始化的标准差
        init_xavier_std=1.0,  # Xavier初始化的标准差
        return_intermediate=True,  # 是否返回中间结果
        auxiliary_loss=False,  # 是否使用辅助损失
        position_embedding_type="sine",  # 位置嵌入的类型
        num_feature_levels=5,  # 特征金字塔的层数
        encoder_n_points=4,  # 编码器中的采样点数
        decoder_n_points=4,  # 解码器中的采样点数
        two_stage=True,  # 是否使用两阶段检测器
        two_stage_num_proposals=300,  # 第一阶段提议的数量
        with_box_refine=True,  # 是否进行框细化
        assign_first_stage=True,  # 是否进行第一阶段的指派
        assign_second_stage=True,  # 是否进行第二阶段的指派
        class_cost=1,  # 类别损失的权重
        bbox_cost=5,  # 边界框损失的权重
        giou_cost=2,  # GIoU损失的权重
        mask_loss_coefficient=1,  # 掩膜损失的系数
        dice_loss_coefficient=1,  # Dice损失的系数
        bbox_loss_coefficient=5,  # 边界框损失的系数
        giou_loss_coefficient=2,  # GIoU损失的系数
        eos_coefficient=0.1,  # EOS损失的系数
        focal_alpha=0.25,  # Focal损失的alpha参数
        disable_custom_kernels=True,  # 是否禁用自定义内核
        **kwargs,  # 其他未列出的关键字参数
    ):
        # 属性访问器，返回编码器注意力头的数量
        @property
        def num_attention_heads(self) -> int:
            return self.encoder_attention_heads

        # 属性访问器，返回模型隐藏层的维度
        @property
        def hidden_size(self) -> int:
            return self.d_model
```