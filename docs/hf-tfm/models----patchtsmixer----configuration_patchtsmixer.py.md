# `.\models\patchtsmixer\configuration_patchtsmixer.py`

```py
# 设置编码格式为 UTF-8
# 版权声明和许可协议，指定代码使用许可
# 导入所需的模块和函数
from typing import List, Optional, Union

# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型名称到配置文件的映射字典
PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtsmixer-etth1-pretrain": "https://huggingface.co/ibm/patchtsmixer-etth1-pretrain/resolve/main/config.json",
}

# PatchTSMixerConfig 类，用于存储 PatchTSMixer 模型的配置信息
class PatchTSMixerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PatchTSMixerModel`]. It is used to instantiate a
    PatchTSMixer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PatchTSMixer
    [ibm/patchtsmixer-etth1-pretrain](https://huggingface.co/ibm/patchtsmixer-etth1-pretrain) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import PatchTSMixerConfig, PatchTSMixerModel

    >>> # Initializing a default PatchTSMixer configuration
    >>> configuration = PatchTSMixerConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = PatchTSMixerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "patchtsmixer"
    model_type = "patchtsmixer"

    # 属性映射字典，将 PatchTSMixerConfig 的属性名映射到其他标准属性名
    attribute_map = {
        "hidden_size": "d_model",               # 隐藏层大小映射到 d_model
        "num_hidden_layers": "num_layers",      # 隐藏层数量映射到 num_layers
    }

# 注释结束
    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 32,  # 定义时间序列的上下文长度，默认为32
        patch_length: int = 8,  # 定义用于处理的补丁长度，默认为8
        num_input_channels: int = 1,  # 输入的通道数，默认为1
        patch_stride: int = 8,  # 补丁的步长，默认为8
        num_parallel_samples: int = 100,  # 并行采样的数量，默认为100
        # General model configuration
        d_model: int = 8,  # 模型的隐藏单元数，默认为8
        expansion_factor: int = 2,  # 扩展因子，默认为2
        num_layers: int = 3,  # 模型层数，默认为3
        dropout: float = 0.2,  # Dropout 的比率，默认为0.2
        mode: str = "common_channel",  # 模型的工作模式，默认为"common_channel"
        gated_attn: bool = True,  # 是否使用门控注意力，默认为True
        norm_mlp: str = "LayerNorm",  # MLP 归一化类型，默认为"LayerNorm"
        self_attn: bool = False,  # 是否使用自注意力，默认为False
        self_attn_heads: int = 1,  # 自注意力头的数量，默认为1
        use_positional_encoding: bool = False,  # 是否使用位置编码，默认为False
        positional_encoding_type: str = "sincos",  # 位置编码的类型，默认为"sincos"
        scaling: Optional[Union[str, bool]] = "std",  # 缩放的方式，默认为"std"
        loss: str = "mse",  # 损失函数类型，默认为"mse"
        init_std: float = 0.02,  # 初始化标准差，默认为0.02
        post_init: bool = False,  # 是否在初始化后执行后处理，默认为False
        norm_eps: float = 1e-5,  # 归一化的小常数，默认为1e-5
        # Pretrain model configuration
        mask_type: str = "random",  # 掩码类型，默认为"random"
        random_mask_ratio: float = 0.5,  # 随机掩码的比率，默认为0.5
        num_forecast_mask_patches: Optional[Union[List[int], int]] = [2],  # 预测掩码的补丁数量，默认为[2]
        mask_value: int = 0,  # 掩码值，默认为0
        masked_loss: bool = True,  # 是否使用掩码损失，默认为True
        channel_consistent_masking: bool = True,  # 是否通道一致的掩码，默认为True
        unmasked_channel_indices: Optional[List[int]] = None,  # 未掩码的通道索引，默认为None
        # General head configuration
        head_dropout: float = 0.2,  # 头部的Dropout比率，默认为0.2
        distribution_output: str = "student_t",  # 分布输出类型，默认为"student_t"
        # Prediction head configuration
        prediction_length: int = 16,  # 预测长度，默认为16
        prediction_channel_indices: list = None,  # 预测的通道索引，默认为None
        # Classification/Regression configuration
        num_targets: int = 3,  # 目标数量，默认为3
        output_range: list = None,  # 输出范围，默认为None
        head_aggregation: str = "max_pool",  # 头部聚合方法，默认为"max_pool"
        **kwargs,
        ):
        self.num_input_channels = num_input_channels
        # 输入通道数，用于模型输入数据的通道数目
        self.context_length = context_length
        # 上下文长度，表示模型处理输入数据时的上下文窗口大小
        self.patch_length = patch_length
        # 补丁长度，指定模型用于处理输入数据的每个补丁的长度
        self.patch_stride = patch_stride
        # 补丁步长，指定模型在输入数据上滑动补丁时的步长
        self.d_model = d_model
        # 模型维度，表示模型中注意力机制的向量维度
        self.expansion_factor = expansion_factor
        # 扩展因子，用于指定模型在进行特征映射时的扩展因子大小
        self.num_layers = num_layers
        # 层数，表示模型中堆叠的自注意力层或前馈网络层的数量
        self.dropout = dropout
        # 丢弃率，指定模型在训练时用于防止过拟合的丢弃率
        self.mode = mode
        # 模式，指定模型的操作模式，如训练模式或推理模式
        self.gated_attn = gated_attn
        # 门控注意力，指定模型是否使用门控机制增强注意力机制
        self.norm_mlp = norm_mlp
        # 归一化MLP，指定模型是否使用归一化操作来规范MLP层
        self.scaling = scaling
        # 缩放，指定模型中注意力机制的缩放因子
        self.head_dropout = head_dropout
        # 头部丢弃率，指定模型中多头注意力机制的丢弃率
        self.num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
        # 补丁数量，根据上下文长度、补丁长度和步长计算模型需要处理的补丁数量
        self.mask_type = mask_type
        # 掩码类型，指定模型中使用的掩码类型，如随机掩码或预测掩码
        self.random_mask_ratio = random_mask_ratio
        # 随机掩码比率，指定模型中随机掩码的比率
        self.num_forecast_mask_patches = num_forecast_mask_patches
        # 预测掩码补丁数量，指定模型中用于预测的掩码补丁的数量
        self.mask_value = mask_value
        # 掩码值，指定模型中用于掩码的特定数值
        self.channel_consistent_masking = channel_consistent_masking
        # 通道一致掩码，指定模型中是否进行通道一致的掩码处理
        self.masked_loss = masked_loss
        # 掩码损失，指定模型中是否使用掩码损失函数
        self.patch_last = True
        # 补丁最后，指定模型是否将补丁处理放在最后执行
        self.use_positional_encoding = use_positional_encoding
        # 使用位置编码，指定模型是否使用位置编码来增强输入数据的位置信息
        self.positional_encoding_type = positional_encoding_type
        # 位置编码类型，指定模型中使用的位置编码的类型
        self.prediction_length = prediction_length
        # 预测长度，指定模型中输出的预测长度
        self.prediction_channel_indices = prediction_channel_indices
        # 预测通道索引，指定模型中用于预测的通道索引
        self.num_targets = num_targets
        # 目标数量，指定模型中预测的目标数量
        self.output_range = output_range
        # 输出范围，指定模型中预测输出的范围
        self.head_aggregation = head_aggregation
        # 头部聚合，指定模型中多头注意力机制的聚合方式
        self.self_attn = self_attn
        # 自注意力，指定模型是否使用自注意力机制
        self.self_attn_heads = self_attn_heads
        # 自注意力头数，指定模型中自注意力机制的头数
        self.init_std = init_std
        # 初始化标准差，指定模型中参数初始化的标准差
        self.post_init = post_init
        # 后初始化，指定模型在初始化后执行的操作
        self.distribution_output = distribution_output
        # 分布输出，指定模型输出的分布类型
        self.loss = loss
        # 损失函数，指定模型中使用的损失函数
        self.num_parallel_samples = num_parallel_samples
        # 并行样本数，指定模型中每次推理时的并行样本数量
        self.unmasked_channel_indices = unmasked_channel_indices
        # 未掩码通道索引，指定模型中不需要进行掩码处理的通道索引
        self.norm_eps = norm_eps
        # 归一化epsilon，指定模型中归一化操作的epsilon值
        super().__init__(**kwargs)
        # 调用父类初始化方法，传入额外的关键字参数
```