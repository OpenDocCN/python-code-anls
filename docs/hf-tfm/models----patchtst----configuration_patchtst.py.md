# `.\transformers\models\patchtst\configuration_patchtst.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 授权，除非您遵守许可证的条款，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件根据许可证按"原样"分发，没有任何明示或暗示的担保或条件
# 有关特定语言的模型输出的许可证，请参阅许可证的详细信息和限制
"""PatchTST 模型配置"""

# 导入必要的模块
from typing import List, Optional, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射
PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ibm/patchtst-base": "https://huggingface.co/ibm/patchtst-base/resolve/main/config.json",
    # 查看所有 PatchTST 模型：https://huggingface.co/ibm/models?filter=patchtst
}

class PatchTSTConfig(PretrainedConfig):
    r"""
    这是一个用于存储 [`PatchTSTModel`] 配置信息的配置类。根据指定的参数实例化 PatchTST 模型，定义模型架构。
    [ibm/patchtst](https://huggingface.co/ibm/patchtst) 架构。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    ```python
    >>> from transformers import PatchTSTConfig, PatchTSTModel

    >>> # 使用 12 个时间步长进行预测初始化一个 PatchTST 配置
    >>> configuration = PatchTSTConfig(prediction_length=12)

    >>> # 根据配置随机初始化一个模型（带有随机权重）
    >>> model = PatchTSTModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    model_type = "patchtst"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }
    def __init__(
        # 初始化函数，用于创建 PatchTST 模型实例
        self,
        # 时间序列特定配置
        num_input_channels: int = 1,  # 输入通道数，默认为1
        context_length: int = 32,  # 上下文长度，默认为32
        distribution_output: str = "student_t",  # 分布输出类型，默认为“student_t”
        loss: str = "mse",  # 损失函数类型，默认为"mse"
        # PatchTST 参数
        patch_length: int = 1,  # 补丁长度，默认为1
        patch_stride: int = 1,  # 补丁步幅，默认为1
        # Transformer 架构配置
        num_hidden_layers: int = 3,  # 隐藏层层数，默认为3
        d_model: int = 128,  # 模型维度，默认为128
        num_attention_heads: int = 4,  # 注意力头数，默认为4
        share_embedding: bool = True,  # 是否共享嵌入，默认为True
        channel_attention: bool = False,  # 是否通道注意力，默认为False
        ffn_dim: int = 512,  # FeedForward网络维度，默认为512
        norm_type: str = "batchnorm",  # 规范化类型，默认为“batchnorm”
        norm_eps: float = 1e-05,  # 规范化参数，默认为1e-05
        attention_dropout: float = 0.0,  # 注意力机制的dropout率，默认为0.0
        dropout: float = 0.0,  # 普通的dropout率，默认为0.0
        positional_dropout: float = 0.0,  # 位置编码的dropout率，默认为0.0
        path_dropout: float = 0.0,  # 路径dropout率，默认为0.0
        ff_dropout: float = 0.0,  # FeedForward网络的dropout率，默认为0.0
        bias: bool = True,  # 是否使用偏置，默认为True
        activation_function: str = "gelu",  # 激活函数类型，默认为“gelu”
        pre_norm: bool = True,  # 是否在层归一化之前应用激活函数，默认为True
        positional_encoding_type: str = "sincos",  # 位置编码类型，默认为“sincos”
        use_cls_token: bool = False,  # 是否使用分类标记，默认为False
        init_std: float = 0.02,  # 初始化标准差，默认为0.02
        share_projection: bool = True,  # 是否共享投影，默认为True
        scaling: Optional[Union[str, bool]] = "std",  # 缩放类型，默认为“std”
        # 掩码预训练
        do_mask_input: Optional[bool] = None,  # 是否进行输入掩码预训练，默认为None
        mask_type: str = "random",  # 掩码类型，默认为“random”
        random_mask_ratio: float = 0.5,  # 随机掩码比例，默认为0.5
        num_forecast_mask_patches: Optional[Union[List[int], int]] = [2],  # 预测掩码补丁数，默认为[2]
        channel_consistent_masking: Optional[bool] = False,  # 是否进行一致性通道掩码，默认为False
        unmasked_channel_indices: Optional[List[int]] = None,  # 未掩码通道索引��默认为None
        mask_value: int = 0,  # 掩码值，默认为0
        # 头部
        pooling_type: str = "mean",  # 池化类型，默认为“mean”
        head_dropout: float = 0.0,  # 头部的dropout率，默认为0.0
        prediction_length: int = 24,  # 预测长度，默认为24
        num_targets: int = 1,  # 目标数，默认为1
        output_range: Optional[List] = None,  # 输出范围，默认为None
        # 分布头
        num_parallel_samples: int = 100,  # 并行采样数，默认为100
        **kwargs,  # 其他参数
        # 时间序列特定配置
        self.context_length = context_length  # 上下文长度
        self.num_input_channels = num_input_channels  # 输入通道数，即变量数
        self.loss = loss  # 损失函数
        self.distribution_output = distribution_output  # 分布输出
        self.num_parallel_samples = num_parallel_samples  # 并行采样数量

        # Transformer 架构配置
        self.d_model = d_model  # 模型维度
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.ffn_dim = ffn_dim  # 前馈网络维度
        self.num_hidden_layers = num_hidden_layers  # 隐藏层数
        self.dropout = dropout  # dropout 比率
        self.attention_dropout = attention_dropout  # 注意力 dropout 比率
        self.share_embedding = share_embedding  # 是否共享嵌入层
        self.channel_attention = channel_attention  # 通道注意力
        self.norm_type = norm_type  # 归一化类型
        self.norm_eps = norm_eps  # 归一化 epsilon
        self.positional_dropout = positional_dropout  # 位置 dropout 比率
        self.path_dropout = path_dropout  # 路径 dropout 比率
        self.ff_dropout = ff_dropout  # 前馈网络 dropout 比率
        self.bias = bias  # 是否使用偏置
        self.activation_function = activation_function  # 激活函数
        self.pre_norm = pre_norm  # 是否预归一化
        self.positional_encoding_type = positional_encoding_type  # 位置编码类型
        self.use_cls_token = use_cls_token  # 是否使用 CLS token
        self.init_std = init_std  # 初始化标准差
        self.scaling = scaling  # 是否进行缩放

        # PatchTST 参数
        self.patch_length = patch_length  # Patch 长度
        self.patch_stride = patch_stride  # Patch 步长

        # Mask 预训练
        self.do_mask_input = do_mask_input  # 是否进行输入遮罩
        self.mask_type = mask_type  # 遮罩类型
        self.random_mask_ratio = random_mask_ratio  # 随机遮罩比例
        self.num_forecast_mask_patches = num_forecast_mask_patches  # 预测遮罩块数
        self.channel_consistent_masking = channel_consistent_masking  # 通道一致遮罩
        self.unmasked_channel_indices = unmasked_channel_indices  # 未遮罩通道索引
        self.mask_value = mask_value  # 遮罩值

        # 一般头参数
        self.pooling_type = pooling_type  # 池化类型
        self.head_dropout = head_dropout  # 头 dropout 比率

        # 预测头
        self.share_projection = share_projection  # 是否共享投影
        self.prediction_length = prediction_length  # 预测长度

        # 预测和回归头
        self.num_parallel_samples = num_parallel_samples  # 并行采样数量

        # 回归
        self.num_targets = num_targets  # 目标数量
        self.output_range = output_range  # 输出范围

        # 调用父类的初始化方法
        super().__init__(**kwargs)
```