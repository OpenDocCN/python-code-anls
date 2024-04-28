# `.\transformers\models\patchtsmixer\configuration_patchtsmixer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，说明代码版权归 IBM 和 HuggingFace Inc. 团队所有，采用 Apache License 2.0 授权
# 详细版权信息和许可证可在 http://www.apache.org/licenses/LICENSE-2.0 找到
# 如果没有适用的法律要求或书面同意，本软件将按“原样”提供，不提供任何形式的担保
# 请查看许可证以了解更多信息

# 导入所需的类型和模块
from typing import List, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典
PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtsmixer-etth1-pretrain": "https://huggingface.co/ibm/patchtsmixer-etth1-pretrain/resolve/main/config.json",
}

# PatchTSMixer 模型的配置类，用于存储 PatchTSMixer 模型的配置信息
class PatchTSMixerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`PatchTSMixerModel`] 配置的配置类。根据指定的参数，定义模型架构。通过默认实例化一个配置，可以得到与 PatchTSMixer
    [ibm/patchtsmixer-etth1-pretrain](https://huggingface.co/ibm/patchtsmixer-etth1-pretrain) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    示例:

    ```python
    >>> from transformers import PatchTSMixerConfig, PatchTSMixerModel

    >>> # 初始化一个默认的 PatchTSMixer 配置
    >>> configuration = PatchTSMixerConfig()

    >>> # 从配置随机初始化一个模型（具有随机权重）
    >>> model = PatchTSMixerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    
    # 模型类型为 PatchTSMixer
    model_type = "patchtsmixer"
    # 属性映射字典，将配置中的属性名称映射到 PatchTSMixer 模型中对应的属性名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }
    def __init__(
        self,
        # 时间序列特定配置
        context_length: int = 32,  # 上下文长度，默认为32
        patch_len: int = 8,  # 补丁长度，默认为8
        num_input_channels: int = 1,  # 输入通道数，默认为1
        patch_stride: int = 8,  # 补丁步长，默认为8
        num_parallel_samples: int = 100,  # 并行采样数，默认为100
        # 一般模型配置
        d_model: int = 8,  # 模型维度，默认为8
        expansion_factor: int = 2,  # 扩展因子，默认为2
        num_layers: int = 3,  # 层数，默认为3
        dropout: float = 0.2,  # 丢弃率，默认为0.2
        mode: str = "common_channel",  # 模式，默认为"common_channel"
        gated_attn: bool = True,  # 是否有门控注意力，默认为True
        norm_mlp: str = "LayerNorm",  # 归一化MLP，默认为"LayerNorm"
        self_attn: bool = False,  # 自注意力，默认为False
        self_attn_heads: int = 1,  # 自注意力头数，默认为1
        use_positional_encoding: bool = False,  # 是否使用位置编码，默认为False
        positional_encoding_type: str = "sincos",  # 位置编码类型，默认为"sincos"
        scaling: Optional[Union[str, bool]] = "std",  # 缩放，默认为"std"
        loss: str = "mse",  # 损失函数，默认为"mse"
        init_std: float = 0.02,  # 初始化标准差，默认为0.02
        post_init: bool = False,  # 是否后初始化，默认为False
        norm_eps: float = 1e-5,  # 归一化参数，默认为1e-5
        # 预训练模型配置
        mask_type: str = "random",  # 掩码类型，默认为"random"
        random_mask_ratio: float = 0.5,  # 随机掩码比率，默认为0.5
        num_forecast_mask_patches: Optional[Union[List[int], int]] = [2],  # 预测掩码补丁数，默认为[2]
        mask_value: int = 0,  # 掩码值，默认为0
        masked_loss: bool = True,  # 掩码损失，默认为True
        channel_consistent_masking: bool = True,  # 通道一致掩码，默认为True
        unmasked_channel_indices: Optional[List[int]] = None,  # 未掩码通道索引，默认为None
        # 一般头配置
        head_dropout: float = 0.2,  # 头部丢弃率，默认为0.2
        distribution_output: str = "student_t",  # 分布输出，默认为"student_t"
        # 预测头配置
        prediction_length: int = 16,  # 预测长度，默认为16
        prediction_channel_indices: list = None,  # 预测通道索引，默认为None
        # 分类/回归配置
        num_targets: int = 3,  # 目标数，默认为3
        output_range: list = None,  # 输出范围，默认为None
        head_aggregation: str = "max_pool",  # 头部聚合方式，默认为"max_pool"
        **kwargs,  # 其他参数
        # 初始化模型的各种参数
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_len
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout
        # 计算图像切片的数量
        self.num_patches = (max(context_length, patch_len) - patch_len) // patch_stride + 1
        self.mask_type = mask_type
        self.random_mask_ratio = random_mask_ratio
        self.num_forecast_mask_patches = num_forecast_mask_patches
        self.mask_value = mask_value
        self.channel_consistent_masking = channel_consistent_masking
        self.masked_loss = masked_loss
        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.prediction_length = prediction_length
        self.prediction_channel_indices = prediction_channel_indices
        self.num_targets = num_targets
        self.output_range = output_range
        self.head_aggregation = head_aggregation
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.distribution_output = distribution_output
        self.loss = loss
        self.num_parallel_samples = num_parallel_samples
        self.unmasked_channel_indices = unmasked_channel_indices
        self.norm_eps = norm_eps
        # 调用父类的构造函数
        super().__init__(**kwargs)
```