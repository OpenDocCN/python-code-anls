# `.\models\oneformer\configuration_oneformer.py`

```
# 设置文件编码为UTF-8，确保能够正确处理中文等特殊字符
# 版权声明，指出版权归属于SHI Labs和HuggingFace Inc.团队，并保留所有权利
# 使用Apache许可证2.0版本，允许按照此许可证使用和分发本软件
# 获取Apache许可证2.0的详细信息，请访问指定的URL
# 除非法律要求或书面同意，否则不得使用此文件
# 本软件基于"按原样"提供，没有任何明示或暗示的保证或条件
# 更多信息请参见许可证
"""OneFormer模型配置"""

# 从typing库导入Dict和Optional类，用于类型提示
from typing import Dict, Optional

# 导入预训练配置类PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志工具logging
from ...utils import logging
# 从自动导入中导入配置映射CONFIG_MAPPING
from ..auto import CONFIG_MAPPING

# 获取logger对象，用于记录日志
logger = logging.get_logger(__name__)

# 定义OneFormer预训练配置文件的存档映射
ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/oneformer_ade20k_swin_tiny": (
        "https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny/blob/main/config.json"
    ),
    # 查看所有OneFormer模型，请访问指定的URL
}


class OneFormerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储[`OneFormerModel`]的配置。根据指定的参数实例化一个OneFormer模型，定义模型架构。
    使用默认值实例化配置会产生一个类似于OneFormer[shi-labs/oneformer_ade20k_swin_tiny]架构的配置，
    它在ADE20k-150数据集上进行了训练。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅[`PretrainedConfig`]的文档。

    Examples:
    ```python
    >>> from transformers import OneFormerConfig, OneFormerModel

    >>> # 初始化一个OneFormer shi-labs/oneformer_ade20k_swin_tiny配置
    >>> configuration = OneFormerConfig()
    >>> # 使用该配置初始化一个模型（带有随机权重）
    >>> model = OneFormerModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型设为"oneformer"
    model_type = "oneformer"
    # 属性映射表，将"hidden_size"映射为"hidden_dim"
    attribute_map = {"hidden_size": "hidden_dim"}
    # 初始化函数，用于创建一个对象实例，并接受多个可选参数
    def __init__(
        self,
        backbone_config: Optional[Dict] = None,  # 设置骨干网络的配置字典，可选，默认为None
        backbone: Optional[str] = None,  # 指定使用的骨干网络的名称，可选，默认为None
        use_pretrained_backbone: bool = False,  # 是否使用预训练的骨干网络模型，默认为False
        use_timm_backbone: bool = False,  # 是否使用timm库提供的骨干网络模型，默认为False
        backbone_kwargs: Optional[Dict] = None,  # 骨干网络的额外参数字典，可选，默认为None
        ignore_value: int = 255,  # 忽略值，用于特定任务中指定像素值，默认为255
        num_queries: int = 150,  # 查询的数量，用于某些任务中的查询数量设定，默认为150
        no_object_weight: int = 0.1,  # 无对象的权重，用于某些任务中的权重设定，默认为0.1
        class_weight: float = 2.0,  # 类别权重，用于某些任务中的类别权重设定，默认为2.0
        mask_weight: float = 5.0,  # 掩码权重，用于某些任务中的掩码权重设定，默认为5.0
        dice_weight: float = 5.0,  # Dice损失的权重，用于某些任务中的Dice损失权重设定，默认为5.0
        contrastive_weight: float = 0.5,  # 对比损失的权重，用于某些任务中的对比损失权重设定，默认为0.5
        contrastive_temperature: float = 0.07,  # 对比损失的温度参数，用于某些任务中的对比损失温度设定，默认为0.07
        train_num_points: int = 12544,  # 训练点的数量，用于某些任务中的训练点数量设定，默认为12544
        oversample_ratio: float = 3.0,  # 过采样比率，用于某些任务中的过采样比率设定，默认为3.0
        importance_sample_ratio: float = 0.75,  # 重要样本比率，用于某些任务中的重要样本比率设定，默认为0.75
        init_std: float = 0.02,  # 初始化标准差，用于某些初始化操作中的标准差设定，默认为0.02
        init_xavier_std: float = 1.0,  # Xavier初始化中的标准差，用于某些初始化操作中的Xavier标准差设定，默认为1.0
        layer_norm_eps: float = 1e-05,  # 层归一化中的epsilon参数，用于某些层归一化操作中的epsilon设定，默认为1e-05
        is_training: bool = False,  # 是否处于训练模式，用于指示当前是否在训练模型，默认为False
        use_auxiliary_loss: bool = True,  # 是否使用辅助损失，用于某些任务中的辅助损失设定，默认为True
        output_auxiliary_logits: bool = True,  # 是否输出辅助Logits，用于某些任务中是否输出辅助Logits，默认为True
        strides: Optional[list] = [4, 8, 16, 32],  # 步长列表，用于某些网络结构中的步长设定，默认为[4, 8, 16, 32]
        task_seq_len: int = 77,  # 任务序列长度，用于某些任务中的任务序列长度设定，默认为77
        text_encoder_width: int = 256,  # 文本编码器的宽度，用于某些任务中的文本编码器宽度设定，默认为256
        text_encoder_context_length: int = 77,  # 文本编码器的上下文长度，用于某些任务中的文本编码器上下文长度设定，默认为77
        text_encoder_num_layers: int = 6,  # 文本编码器的层数，用于某些任务中的文本编码器层数设定，默认为6
        text_encoder_vocab_size: int = 49408,  # 文本编码器的词汇表大小，用于某些任务中的文本编码器词汇表大小设定，默认为49408
        text_encoder_proj_layers: int = 2,  # 文本编码器的投影层数，用于某些任务中的文本编码器投影层数设定，默认为2
        text_encoder_n_ctx: int = 16,  # 文本编码器的上下文数，用于某些任务中的文本编码器上下文数设定，默认为16
        conv_dim: int = 256,  # 卷积层的维度，用于某些任务中的卷积层维度设定，默认为256
        mask_dim: int = 256,  # 掩码的维度，用于某些任务中的掩码维度设定，默认为256
        hidden_dim: int = 256,  # 隐藏层的维度，用于某些任务中的隐藏层维度设定，默认为256
        encoder_feedforward_dim: int = 1024,  # 编码器前馈层的维度，用于某些任务中的编码器前馈层维度设定，默认为1024
        norm: str = "GN",  # 标准化方法，用于某些任务中的标准化方法选择，默认为"GN"
        encoder_layers: int = 6,  # 编码器的层数，用于某些任务中的编码器层数设定，默认为6
        decoder_layers: int = 10,  # 解码器的层数，用于某些任务中的解码器层数设定，默认为10
        use_task_norm: bool = True,  # 是否使用任务归一化，用于某些任务中的任务归一化设定，默认为True
        num_attention_heads: int = 8,  # 注意力头的数量，用于某些任务中的注意力头数量设定，默认为8
        dropout: float = 0.1,  # 丢弃率，用于某些层中的丢弃率设定，默认为0.1
        dim_feedforward: int = 2048,  # 前馈层的维度，用于某些任务中的前馈层维度设定，默认为2048
        pre_norm: bool = False,  # 是否在层归一化之前应用归一化，用于某些任务中的层归一化顺序设定，默认为False
        enforce_input_proj: bool = False,  # 是否强制输入投影，用于某些任务中的输入投影设定，默认为False
        query_dec_layers: int = 2,  # 查询解码器的层数，用于某些任务中的查询解码器层数设定，默认为2
        common_stride: int = 4,  # 公共步长，用于某些任务中的公共步长设定，默认为4
        **kwargs,  # 其他未列出的关键字参数，用于接收任意其他参数
```