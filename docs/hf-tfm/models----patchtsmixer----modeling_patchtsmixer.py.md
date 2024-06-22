# `.\transformers\models\patchtsmixer\modeling_patchtsmixer.py`

```py
# 设置文件编码和版权信息的注释
# 导入需要的模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# 导入自定义的模型输出类和工具函数
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
# 导入自定义的日志模块
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入模型配置
from .configuration_patchtsmixer import PatchTSMixerConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 配置文档说明
_CONFIG_FOR_DOC = "PatchTSMixerConfig"

# 预训练模型列表
PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtsmixer-etth1-pretrain",
    # See all PatchTSMixer models at https://huggingface.co/models?filter=patchtsmixer
]

# 模型文档说明
PATCHTSMIXER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        mask_input (`bool`, *optional*, defaults to `False`):
            If True, Masking will be enabled. False otherwise.
"""

# 输入文档说明
PATCHTSMIXER_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            # 过去时间序列的上下文值。对于预训练任务，这表示预测被屏蔽部分的输入时间序列。对于预测任务，这表示历史/过去时间序列的值。同样地，对于分类或回归任务，它表示时间序列的适当上下文值。
            对于单变量时间序列，`num_input_channels` 维度应为 1。对于多变量时间序列，它大于 1。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。

        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
# PatchTSMixerGatedAttention类，应用带门控注意力机制到输入数据中
class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        # 使用线性层实现注意力机制
        self.attn_layer = nn.Linear(in_size, out_size)
        # 应用softmax函数进行归一化
        self.attn_softmax = nn.Softmax(dim=-1)

    # 前向传播函数
    def forward(self, inputs):
        # 计算注意力权重
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        # 对输入进行加权
        inputs = inputs * attn_weight
        return inputs


# 从transformers.models.patchtst.modeling_patchtst.PatchTSTBatchNorm复制而来，将PatchTST改为PatchTSMixer
class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # 使用BatchNorm1d进行批标准化
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        # 调整维度顺序，将时间维度放到第二维
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


# 位置编码类
class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # 如果使用位置编码
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            # 使用全零向量作为位置编码
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: PatchTSMixerConfig) -> nn.Parameter:
        # 初始化位置编码
        # 如果位置编码类型为随机
        if config.positional_encoding_type == "random":
            # 创建一个形状为(config.num_patches, config.d_model)的张量，并赋予随机值作为位置编码
            position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
        # 如果位置编码类型为sin-cos
        elif config.positional_encoding_type == "sincos":
            # 创建一个形状为(config.num_patches, config.d_model)的零张量作为位置编码的容器
            position_enc = torch.zeros(config.num_patches, config.d_model)
            # 创建一个形状为(config.num_patches, 1)的张量，表示位置索引
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            # 计算sin-cos位置编码
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            # 对位置编码进行归一化处理
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            # 将位置编码转换为可学习参数
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            # 如果位置编码类型不是'random'或'sincos'，则引发值错误
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        # 返回位置编码
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # 前向传播函数，用于处理输入张量中的位置编码
        # 隐藏状态：[批大小 x 通道数 x 补丁数 x d_model]
        # 将输入的补丁张量与位置编码相加，以获取最终的隐藏状态
        hidden_state = patch_input + self.position_enc
        return hidden_state
# 定义一个名为 PatchTSMixerNormLayer 的 PyTorch 模块类
class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 获取配置中的归一化方式
        self.norm_mlp = config.norm_mlp

        # 根据配置选择使用批归一化或者层归一化
        if "batch" in config.norm_mlp.lower():
            self.norm = PatchTSMixerBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        # 如果使用批归一化
        if "batch" in self.norm_mlp.lower():
            # 将输入数据重塑为 [batch_size*num_channels, num_patches, d_model] 的形状
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )
            # 对重塑后的数据进行批归一化
            inputs_reshaped = self.norm(inputs_reshaped)
            # 将数据恢复到原始形状
            inputs = torch.reshape(inputs_reshaped, inputs.shape)
        # 如果使用层归一化
        else:
            inputs = self.norm(inputs)

        return inputs


# 定义一个名为 PatchTSMixerMLP 的 PyTorch 模块类
class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        # 计算隐藏层大小
        num_hidden = in_features * config.expansion_factor
        # 定义全连接层和dropout层
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        # 经过第一个全连接层和激活函数GELU，并应用dropout
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        # 经过第二个全连接层并应用dropout
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


# 定义一个名为 PatchTSMixerChannelFeatureMixerBlock 的 PyTorch 模块类
class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    # 初始化函数，接收一个配置对象作为参数
    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类的初始化函数
        super().__init__()
        
        # 初始化 PatchTSMixerNormLayer
        self.norm = PatchTSMixerNormLayer(config)
        # 获取是否使用门控注意力的配置信息
        self.gated_attn = config.gated_attn
        # 初始化 PatchTSMixerMLP
        self.mlp = PatchTSMixerMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        # 如果使用门控注意力
        if config.gated_attn:
            # 初始化 PatchTSMixerGatedAttention
            self.gating_block = PatchTSMixerGatedAttention(
                in_size=config.num_input_channels, out_size=config.num_input_channels
            )

    # 前向传播函数
    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        # 保存输入的残差连接
        residual = inputs
        # 对输入进行归一化
        inputs = self.norm(inputs)

        # 调整输入的维度顺序
        inputs = inputs.permute(0, 3, 2, 1)

        # 如果使用门控注意力
        if self.gated_attn:
            # 使用门控注意力模块处理输入
            inputs = self.gating_block(inputs)

        # 经过 MLP 模块处理输入
        inputs = self.mlp(inputs)

        # 调整输出的维度顺序
        inputs = inputs.permute(0, 3, 2, 1)

        # 将处理后的输入和之前保存的残差连接相加得到最终输出
        out = inputs + residual
        return out
# 定义了一个 PatchTSMixerAttention 类，继承自 nn.Module
# 这个类实现了多头注意力机制，可以用于 PatchTSMixer 模型
class PatchTSMixerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PatchTSMixerConfig] = None,
    ):
        super().__init__()
        # 设置注意力机制的一些超参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 确保 embed_dim 可以被 num_heads 整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 定义线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重新整形为多头注意力需要的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 在这里实现多头注意力的前向计算过程
        pass

# 定义了一个 PatchMixerBlock 类，继承自 nn.Module
# 这个类实现了 PatchTSMixer 模型中的 Patch Mixer 块
class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 定义 Patch Mixer 块中的各个组件
        self.norm = PatchTSMixerNormLayer(config)
        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn
        self.mlp = PatchTSMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        # 如果使用门控注意力，则定义门控注意力层
        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.num_patches, out_size=config.num_patches)

        # 如果使用自注意力，则定义自注意力层和归一化层
        if config.self_attn:
            self.self_attn_layer = PatchTSMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = PatchTSMixerNormLayer(config)
    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor. 输入张量

        Returns:
            `torch.Tensor`: Transformed tensor. 转换后的张量
        """
        # 保存残差连接
        residual = hidden_state

        # 应用 Layer Normalization
        hidden_state = self.norm(hidden_state)

        # 如果使用自注意力机制
        if self.self_attn:
            # 获取输入张量的形状信息
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            # 将输入张量重塑为二维形状以进行自注意力计算
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            # 进行自注意力计算
            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            # 将自注意力计算结果重塑为原始形状
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # 将张量的维度进行转置，使得 num_patches 变为最后一个维度
        hidden_state = hidden_state.transpose(2, 3)
        # 应用 MLP 层
        hidden_state = self.mlp(hidden_state)

        # 如果使用门控注意力机制
        if self.gated_attn:
            # 应用门控注意力机制
            hidden_state = self.gating_block(hidden_state)

        # 将维度转置回原始形状
        hidden_state = hidden_state.transpose(2, 3)

        # 如果使用自注意力机制，再次应用 Layer Normalization
        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        # 添加残差连接
        out = hidden_state + residual
        return out
class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 初始化层归一化模块
        self.norm = PatchTSMixerNormLayer(config)

        # 是否使用门控注意力
        self.gated_attn = config.gated_attn

        # 初始化 MLP 模块
        self.mlp = PatchTSMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        # 如果配置中指定使用门控注意力，则初始化门控注意力模块
        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.d_model, out_size=config.d_model)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        # 保存残差连接
        residual = hidden
        # 执行层归一化
        hidden = self.norm(hidden)
        # 执行 MLP 操作
        hidden = self.mlp(hidden)

        # 如果配置中指定使用门控注意力，则执行门控注意力操作
        if self.gated_attn:
            hidden = self.gating_block(hidden)

        # 残差连接和处理后的 hidden 相加，得到输出
        out = hidden + residual
        return out


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 初始化 patch_mixer 模块
        self.patch_mixer = PatchMixerBlock(config=config)
        # 初始化 feature_mixer 模块
        self.feature_mixer = FeatureMixerBlock(config=config)

        # 设置模式
        self.mode = config.mode

        # 如果模式为 "mix_channel"，则初始化 channel_feature_mixer 模块
        if config.mode == "mix_channel":
            self.channel_feature_mixer = PatchTSMixerChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        # 如果模式为 "mix_channel"，则执行 channel_feature_mixer 操作
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        # 执行 patch_mixer 操作
        hidden = self.patch_mixer(hidden)
        # 执行 feature_mixer 操作
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        num_layers = config.num_layers

        # 初始化多个 PatchTSMixerLayer 模块，组成 mixers
        self.mixers = nn.ModuleList([PatchTSMixerLayer(config=config) for _ in range(num_layers)])
    # 定义了一个前向传播的函数
    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor. 输入的张量
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well. 是否输出隐藏状态

        Returns:
            `torch.Tensor`: The embedding. 嵌入向量（嵌入表示）
            `list`: List of all hidden states if `output_hidden_states` is set to `True`. 如果设置输出隐藏状态为True，则返回所有隐藏状态的列表
        """
        # 用于存储所有隐藏状态
        all_hidden_states = []
        # 初始化embedding为输入的张量
        embedding = hidden_state

        # 遍历所有的mixer（混合）模块
        for mod in self.mixers:
            # 将输入的embedding经过mod处理
            embedding = mod(embedding)
            # 如果需要输出隐藏状态，则将当前embedding添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states.append(embedding)

        # 如果需要输出隐藏状态，则返回embedding和all_hidden_states，否则返回embedding和None
        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None
class PatchTSMixerForPredictionHead(nn.Module):
    """用于预测的预测头模块

    Args:
        config (`PatchTSMixerConfig`, *required*): 配置参数.
    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()

        self.dropout_layer = nn.Dropout(config.head_dropout)  # 随机失活层
        if distribution_output is None:
            self.base_forecast_block = nn.Linear((config.num_patches * config.d_model), config.prediction_length)  # 全连接层
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(
                config.num_patches * config.d_model
            )

        self.flatten = nn.Flatten(start_dim=-2)  # 将输入展平

    def forward(self, hidden_features):
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): 输入的隐藏特征.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """

        hidden_features = self.flatten(hidden_features)  # 将隐藏特征展平为 [batch_size x n_vars x num_patch * d_model]
        hidden_features = self.dropout_layer(hidden_features)  # 使用随机失活层
        forecast = self.base_forecast_block(hidden_features)  # 基础预测块进行预测
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)
        else:
            forecast = forecast.transpose(-1, -2)  # 转置操作

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[..., self.prediction_channel_indices] for z in forecast)
            else:
                forecast = forecast[..., self.prediction_channel_indices]  # 根据预测通道索引进行预测

        return forecast


class PatchTSMixerLinearHead(nn.Module):
    """用于分类和回归的线性头模块.

    Args:
        config (`PatchTSMixerConfig`, *required*): 配置参数.

    """
    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()  # 调用父类的初始化方法

        # 从传入的配置中获取头部聚合方式和输出范围
        self.head_aggregation = config.head_aggregation
        self.output_range = config.output_range

        # 根据头部聚合方式确定乘数因子
        if config.head_aggregation is None:
            mul_factor = config.num_patches
        else:
            mul_factor = 1
        
        # 存储分布输出
        self.distribution_output = distribution_output
        
        # 如果没有提供分布输出，使用线性投影进行预测
        if distribution_output is None:
            self.projection = nn.Linear(
                config.d_model * config.num_input_channels * mul_factor,
                config.num_targets,
            )
        else:  # 否则使用分布输出提供的参数投影
            self.projection = distribution_output.get_parameter_projection(
                config.d_model * config.num_input_channels * mul_factor
            )

        # 根据头部聚合方式确定展平操作的维度
        if config.head_aggregation is None:
            self.flatten = nn.Flatten(start_dim=-3)
        else:
            self.flatten = nn.Flatten(start_dim=-2)

        # 初始化一个dropout层
        self.dropout = nn.Dropout(config.head_dropout)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """

        # 转置隐藏特征张量，以适应不同的聚合方式
        hidden_features = hidden_features.transpose(-1, -2)
        
        # 根据头部聚合方式对隐藏特征进行处理
        if self.head_aggregation == "use_last":
            # 选择最后一个时间步的隐藏特征
            hidden_features = hidden_features[..., -1]
        elif self.head_aggregation == "max_pool":
            # 对每个变量的隐藏特征进行最大池化
            hidden_features = hidden_features.max(dim=-1).values
        elif self.head_aggregation == "avg_pool":
            # 对每个变量的隐藏特征进行平均池化
            hidden_features = hidden_features.mean(dim=-1)

        # 如果需要，对隐藏特征进行展平操作
        if self.flatten:
            hidden_features = self.flatten(hidden_features)
        
        # 对隐藏特征进行dropout
        hidden_features = self.dropout(hidden_features)
        
        # 使用线性投影层进行预测
        hidden_features = self.projection(hidden_features)  # batch_size x num_targets

        # 如果没有提供分布输出并且输出范围不为空，则进行范围缩放
        if (self.distribution_output is None) and (self.output_range is not None):
            hidden_features = (
                torch.sigmoid(hidden_features) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            )
        return hidden_features
class PatchTSMixerPreTrainedModel(PreTrainedModel):
    # PatchTSMixerPreTrainedModel类继承自PreTrainedModel类
    # 定义config_class为PatchTSMixerConfig类
    config_class = PatchTSMixerConfig
    # 定义base_model_prefix为"model"
    base_model_prefix = "model"
    # 定义main_input_name为"past_values"
    main_input_name = "past_values"
    # 设置supports_gradient_checkpointing为False，不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        # 初始化权重
        if isinstance(module, PatchTSMixerPositionalEncoding):
            # 如果module是PatchTSMixerPositionalEncoding类的实例
            # 初始化位置编码
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            # 如果module是nn.LayerNorm或nn.BatchNorm1d类的实例
            # 将偏置初始化为零，权重初始化为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSMixerBatchNorm):
            # 如果module是PatchTSMixerBatchNorm类的实例
            # 将batchnorm的偏置初始化为零，权重初始化为1.0
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            # 如果module是nn.Linear类的实例
            # 将权重初始化为正态分布，均值为0.0，标准差为config.init_std
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                # 如果存在偏置，将偏置初始化为零
                module.bias.data.zero_()


class PatchTSMixerPretrainHead(nn.Module):
    """Pretraining head.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # 初始化方法
        self.dropout_layer = nn.Dropout(config.head_dropout)
        self.base_pt_block = nn.Linear(config.d_model, config.patch_length)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x n_vars x num_patch x patch_length)`.
        """
        # 前向传播方法
        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_pt_block(hidden_features)  # [batch_size x n_vars x num_patch x patch_length]
        return forecast


# Copied from transformers.models.patchtst.modeling_patchtst.random_masking
def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value: int = 0,
):
    """random_masking: Mask the input considering the control variables.
    # 定义random_masking函数，实现根据控制变量掩码输入
    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            输入张量，要进行遮盖的输入数据。
        mask_ratio (`float`):
            在随机预训练期间应用于遮盖输入数据的遮盖比例。它是介于 0 和 1 之间的数字。
        unmasked_channel_indices (list, *optional*):
            不会被遮盖的通道的索引。
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            当为 true 时，遮盖将在时间序列的所有通道上相同。否则，遮盖位置将在通道之间变化。
        mask_value (int, *optional*, defaults to 0):
            用于预训练的遮盖补丁的值。

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, 掩盖后的输入，与输入张量具有相同的形状，以及形状为 [bs x c x n] 的遮盖张量。
    """
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"遮盖比例 {mask_ratio} 必须在 0 和 1 之间。")

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # 在 [0, 1] 内的噪声，bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        # 在 [0, 1] 内的噪声，bs x num_channels x L
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # mask: [bs x num_channels x num_patch]
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    mask[:, :, :len_keep] = 0

    # 对每个样本的噪声进行排序
    ids_shuffle = torch.argsort(noise, dim=-1)  # 升序：小的是保留，大的是移除
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]
# 从transformers.models.patchtst.modeling_patchtst.forecast_masking复制过来的函数
def forecast_masking(
    inputs: torch.Tensor,  # 输入张量，形状为（bs，num_channels，num_patch，patch_len）
    num_forecast_mask_patches: Union[list, int],  # 预测掩码的数量，可以是列表或整数
    unmasked_channel_indices: list = None,  # 未掩码通道的索引列表，默认为None
    mask_value: int = 0,  # 掩码值，默认为0
):
    """预测掩码，掩盖最后K个补丁，其中K来自num_forecast_mask_patches。
    如果num_forecast_mask_patches是列表，批次中的样本将随机被列表中定义的数字掩码。

    参数:
        inputs (`torch.Tensor`):
            输入张量，形状为`(bs，num_channels，num_patch，patch_len)`
        num_forecast_mask_patches (`list`):
            每个批次样本末尾要掩盖的补丁数量，例如：4或[3, 5]。
        unmasked_channel_indices (`list`, *optional*):
            未掩码的通道指数。
        mask_value (`int`, *optional*, 默认为0):
            掩盖补丁的数值将被填充为`mask_value`.

    返回:
        `tuple(torch.Tensor)`: inputs_mask, 掩盖的输入，和输入张量相同大小的张量和掩盖张量的形状`（bs，num_channels，num_patch）`或`（bs，tsg1，tsg2，num_channels，num_patch）`
    """

    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(
                f"num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches."
            )
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    t_list = sorted(t_list, key=lambda x: x[2])

    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    batch1 = 0
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patch x patch_len]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]
# 一个将时间序列序列划分为不同 patches 的类
class PatchTSMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类的构造函数
        super().__init__()

        # 获取配置中的序列长度和 patch 长度、步长
        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        # 如果序列长度小于等于 patch 长度，则会报错
        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # 计算 patch 的数量
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        # 获取输入序列的长度
        sequence_length = past_values.shape[-2]
        # 如果输入序列长度与配置不一致，则报错
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # 从输入序列中截取符合配置的部分
        output = past_values[:, self.sequence_start :, :]
        # 将序列划分为不同的 patches
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # 将 patches 的维度顺序调整为 [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


# 从 PatchTST 中拷贝的 PatchTSMixerMasking 类
class PatchTSMixerMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSMixerConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """
    # 该类继承自 PyTorch 的 nn.Module 基类，用于对输入的 patch 数据进行随机或预测性的遮蔽操作
        def __init__(self, config: PatchTSMixerConfig):
            # 调用父类的初始化方法
            super().__init__()
            # 设置随机遮蔽比例
            self.random_mask_ratio = config.random_mask_ratio
            # 设置是否保持通道一致的遮蔽
            self.channel_consistent_masking = config.channel_consistent_masking
            # 设置遮蔽类型
            self.mask_type = config.mask_type
            # 设置用于预测性遮蔽的 patch 数量
            self.num_forecast_mask_patches = config.num_forecast_mask_patches
            # 设置不被遮蔽的通道索引
            self.unmasked_channel_indices = config.unmasked_channel_indices
            # 设置遮蔽值
            self.mask_value = config.mask_value
            # 如果设置了不被遮蔽的通道索引，则对其进行排序
            if self.unmasked_channel_indices is not None:
                self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)
    
        def forward(self, patch_input: torch.Tensor):
            """
            对输入的 patch 数据进行遮蔽操作
    
            参数:
                patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *必需*):
                    输入的 patch 数据
    
            返回:
                masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                    遮蔽后的 patch 数据
                mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                    表示哪些位置被遮蔽的布尔张量
            """
            # 根据遮蔽类型进行不同的遮蔽操作
            if self.mask_type == "random":
                # 进行随机遮蔽
                masked_input, mask = random_masking(
                    inputs=patch_input,
                    mask_ratio=self.random_mask_ratio,
                    unmasked_channel_indices=self.unmasked_channel_indices,
                    channel_consistent_masking=self.channel_consistent_masking,
                    mask_value=self.mask_value,
                )
            elif self.mask_type == "forecast":
                # 进行预测性遮蔽
                masked_input, mask = forecast_masking(
                    inputs=patch_input,
                    num_forecast_mask_patches=self.num_forecast_mask_patches,
                    unmasked_channel_indices=self.unmasked_channel_indices,
                    mask_value=self.mask_value,
                )
            else:
                # 如果遮蔽类型无效，则抛出异常
                raise ValueError(f"Invalid mask type {self.mask_type}.")
    
            # 将 mask 张量转换为布尔张量
            mask = mask.bool()
            return masked_input, mask
# 导入必要的库
import torch
import torch.nn as nn
from typing import Tuple

# 定义 PatchTSMixerStdScaler 类，用于标准化特征
class PatchTSMixerStdScaler(nn.Module):
    """
    根据特征的均值和标准差进行标准化，即计算特征的均值并在第一维度上进行缩放，然后通过减去均值和除以标准差进行归一化。
    """

    def __init__(self, config: PatchTSMixerConfig):
        # 继承父类的初始化方法
        super().__init__()
        # 如果配置中有 scaling_dim，则将其赋值给 self.dim，否则设为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果配置中有 keepdim，则将其赋值给 self.keepdim，否则设为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果配置中有 minimum_scale，则将其赋值给 self.minimum_scale，否则设为 1e-5
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入：
            data（`torch.Tensor`，形状为 `(batch_size, sequence_length, num_input_channels)`）：
                用于计算批归一化的输入数据
            observed_indicator（`torch.BoolTensor`，形状为 `(batch_size, sequence_length, num_input_channels)`）：
                根据观测指标计算尺度。
        返回：
            元组，包含形状为
                (`(batch_size, sequence_length, num_input_channels)`，`(batch_size, 1, num_input_channels)`，
                `(batch_size, 1, num_input_channels)`) 的 `torch.Tensor`
        """
        # 根据观测指标在第一维度上求和，并保持维度不变
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 强制保持分母不小于 1.0
        denominator = denominator.clamp_min(1.0)
        # 计算均值，即对数据乘以观测指标并在第一维度上求和，再除以分母
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差，即对数据减去均值并乘以观测指标的平方，在第一维度上求和，再除以分母
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算尺度，即方差加上最小尺度的平方根
        scale = torch.sqrt(variance + self.minimum_scale)
        # 返回标准化后的数据、均值和尺度
        return (data - loc) / scale, loc, scale


# 定义 PatchTSMixerMeanScaler 类，用于计算特征的均值并进行缩放
class PatchTSMixerMeanScaler(nn.Module):
    """
    计算特征的加权平均绝��值作为缩放因子，并相应地缩放数据。
    """

    def __init__(self, config: PatchTSMixerConfig):
        # 继承父类的初始化方法
        super().__init__()
        # 如果配置中有 scaling_dim，则将其赋值给 self.dim，否则设为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果配置中有 keepdim，则将其赋值给 self.keepdim，否则设为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果配置中有 minimum_scale，则将其赋值给 self.minimum_scale，否则设为 1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 如果配置中有 default_scale，则将其赋值给 self.default_scale，否则设为 None
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    # 定义一个函数，用于计算 Batch Norm 的相关参数
    def calculate_batch_norm_parameters(
        data: torch.Tensor,
        observed_indicator: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation  输入用于Batch norm计算的张量
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.在观察指示器上计算尺度
    
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)` - normalized data,规范化的数据
                `(batch_size, 1, num_input_channels)` - mean per channel,每个通道的平均值
                `(batch_size, 1, num_input_channels)` - standard deviation per channel) 每个通道的标准差
        """
        # Calculate the sum of absolute values of the product of input data and observed indicator
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        # Calculate the number of observed values per channel
        num_observed = observed_indicator.sum(self.dim, keepdim=True)
    
        # Calculate the scale by dividing the sum of absolute values by the number of observed values
        scale = ts_sum / torch.clamp(num_observed, min=1)
    
        # If `default_scale` is provided, we use it, otherwise we use the scale of the batch
        if self.default_scale is None:
            # Calculate the sum of scale for each channel in the batch
            batch_sum = ts_sum.sum(dim=0)
            # Calculate the number of observed values for each channel in the batch
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            # Calculate the default scale by dividing the batch sum by the batch observations
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            # Use the provided default_scale
            default_scale = self.default_scale * torch.ones_like(scale)
    
        # Apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)
    
        # Ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        # Scale the input data
        scaled_data = data / scale
    
        # If `keepdim` is False, squeeze the dimensions of scale
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)
    
        # Return the scaled data, mean per channel, and standard deviation per channel
        return scaled_data, torch.zeros_like(scale), scale
# 从transformers.models.patchtst.modeling_patchtst.PatchTSTNOPScaler复制并修改为PatchTSMixerNOPScaler
class PatchTSMixerNOPScaler(nn.Module):
    """
    为输入数据的第一维度分配一个等于1的尺度因子，因此不对输入数据应用缩放。
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            data (`torch.Tensor`，形状为`(batch_size, sequence_length, num_input_channels)`):
                用于批量归一化计算的输入
        返回:
            返回类型为 `torch.Tensor` 的元组，形状为
                (`(batch_size, sequence_length, num_input_channels)`，`(batch_size, 1, num_input_channels)`，
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


@dataclass
class PatchTSMixerEncoderOutput(ModelOutput):
    """
    `PatchTSMixerEncoderOutput` 的基类，具有可能的隐藏状态。

    参数:
        last_hidden_state (`torch.FloatTensor`，形状为`(batch_size, num_channels, num_patches, d_model)`):
            模型最后一层的隐藏状态。
        hidden_states (`tuple(torch.FloatTensor)`，*可选*):
            模型在每层输出的隐藏状态。
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    PatchTSMixer的编码器，输入patched的时间序列，输出patched的嵌入。

    参数:
        config (`PatchTSMixerConfig`，*必需*):
            配置。
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = PatchTSMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = PatchTSMixerBlock(config=config)

        # 初始化权重并应用最终处理
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSMixerEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series. For a pretraining task, this denotes the input time series to
                predict the masked portion. For a forecasting task, this denotes the history/past time series values.
                Similarly, for classification or regression tasks, it denotes the appropriate context values of the
                time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.
                传入的时间序列的上下文值。对于预训练任务，这表示要预测的被屏蔽部分的输入时间序列。
                对于预测任务，这表示历史/过去的时间序列值。对于分类或回归任务，它表示时间序列的适当上下文值。
                对于单变量时间序列，`num_input_channels`维度应为1。对于多变量时间序列，它大于1。

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                是否返回所有层的隐藏状态。

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                是否返回一个[`~utils.ModelOutput`]而不是一个普通的元组。

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
            返回形状为`(batch_size, n_vars, num_patches, d_model)`的`torch.FloatTensor`。

        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # flatten [bs x num_patch x d_model]. common_channel/mix_channel: [bs x n_vars x num_patch x d_model]
        # 将 `[bs x num_patch x d_model]`扁平化。common_channel/mix_channel: `[bs x n_vars x num_patch x d_model]`
        patches = self.patcher(past_values)

        # add positional encoder
        # 增加位置编码器
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        # 获取最后一个隐藏状态和所有隐藏状态
        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        # 返回PatchTSMixerEncoderOutput对象
        return PatchTSMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)
# 使用 dataclass 装饰器定义 PatchTSMixerModelOutput 类，用于保存模型输出
@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
   Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
        Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
        Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
        Bool Tensor indicating True in masked patches and False otherwise.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
        Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
        enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
        Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
        enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


# 使用 add_start_docstrings 装饰器添加模型说明文档，引用 PATCHTSMIXER_START_DOCSTRING
@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)
# 定义 PatchTSMixerModel 类，继承自 PatchTSMixerPreTrainedModel
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    # 初始化方法
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化参数
        self.use_return_dict = config.use_return_dict
        self.encoder = PatchTSMixerEncoder(config)
        self.patching = PatchTSMixerPatchify(config)

        # 根据条件判断是否需要进行 masking
        if mask_input is True:
            self.masking = PatchTSMixerMasking(config)
        else:
            self.masking = None

        # 根据配置参数选择合适的缩放方法
        if config.scaling == "mean":
            self.scaler = PatchTSMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = PatchTSMixerStdScaler(config)
        else:
            self.scaler = PatchTSMixerNOPScaler(config)

        # 根据配置参数初始化权重并应用最终处理
        if config.post_init:
            self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器添加模型前向传播的说明文档
    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    # 使用 replace_return_docstrings 装饰器替换返回值的说明文档
    @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerModelOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
                
        Returns:
            PatchTSMixerModelOutput: Model output containing encoder outputs and additional information.

        """
        # Determine if return_dict is provided, otherwise use the class attribute
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # Initialize mask variable
        mask = None
        # If observed_mask is not provided, set it to a tensor of ones with the same shape as past_values
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        # Scale past_values based on observed_mask
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)

        # Patch the scaled_past_values
        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length]

        # Set encoder input to patched_x
        enc_input = patched_x
        # Apply masking if masking is not None
        if self.masking is not None:
            # Get masked input and mask
            enc_input, mask = self.masking(patched_x)
            # enc_input: [batch_size x num_input_channels x num_patch x patch_length]
            # mask: [batch_size x num_input_channels x num_patch]

        # Pass encoder input to the encoder
        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # If encoder_output is a tuple, convert it to PatchTSMixerEncoderOutput
        if isinstance(encoder_output, tuple):
            encoder_output = PatchTSMixerEncoderOutput(*encoder_output)

        # If return_dict is False, return a tuple of specific values
        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output.last_hidden_state,
                    encoder_output.hidden_states,
                    patched_x,
                    mask,
                    loc,
                    scale,
                ]
            )

        # If return_dict is True, return a PatchTSMixerModelOutput
        return PatchTSMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patch_input=patched_x,
            mask=mask,
            loc=loc,
            scale=scale,
        )
@dataclass
class PatchTSMixerForPreTrainingOutput(ModelOutput):
    """
    [`PatchTSMixerForPreTrainingOutput`]的输出类型。

    Args:
        prediction_outputs (`torch.FloatTensor`，形状为 `(batch_size, num_input_channels, num_patches, patch_length)`):
            预训练头部的预测输出。
        hidden_states (`tuple(torch.FloatTensor)`，*可选*):
            模型在每一层输出的隐藏状态。
        last_hidden_state (`torch.FloatTensor`，形状为 `(batch_size, num_input_channels, num_patches, d_model)`):
            通过头部之前的主干嵌入。
        loss (*可选*，当提供了 `y` 时返回，`torch.FloatTensor`，形状为 `()`):
            总损失。
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    r"""
    用于遮蔽预训练的 `PatchTSMixer`。

    Args:
        config (`PatchTSMixerConfig`，*必需*):
            配置。

    Returns:
        `None`。
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)  # 初始化PatchTSMixerModel
        self.head = PatchTSMixerPretrainHead(config=config)  # 初始化PatchTSMixerPretrainHead
        self.masked_loss = config.masked_loss  # 设置遮蔽损失
        self.use_return_dict = config.use_return_dict  # 设置是否使用返回字典

        # 初始化权重并应用最终处理
        if config.post_init:  # 如果需要进行后初始化
            self.post_init()  # 执行后初始化

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)  # 将文档字符串添加到模型的前向方法
    @replace_return_docstrings(output_type=PatchTSMixerForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerForPreTrainingOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:
            PatchTSMixerForPreTrainingOutput: An object containing pre-training outputs.
        """
        # Define whether to use the provided `return_dict` or use the class attribute `use_return_dict`.
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # Define the loss function based on whether masked loss is enabled or not.
        if self.masked_loss is True:
            loss = torch.nn.MSELoss(reduction="none")
        else:
            loss = torch.nn.MSELoss(reduction="mean")

        # Generate model output based on the provided inputs.
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x.last_hidden_state: [batch_size x nvars x num_patch x d_model]

        # Ensure that the model output is of type PatchTSMixerModelOutput.
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        # Generate predictions using the head of the model.
        x_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x nvars x num_patch x patch_length]

        # Calculate the loss value if `return_loss` is set to True.
        if return_loss is True:
            loss_val = loss(x_hat, model_output.patch_input)
        else:
            loss_val = None

        # Calculate masked loss if enabled and if loss value is not None.
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        # Return the outputs based on whether `return_dict` is enabled or not.
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    x_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )

        return PatchTSMixerForPreTrainingOutput(
            loss=loss_val,
            prediction_outputs=x_hat,  # tensor [batch_size x nvars x num_patch x patch_length]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import distributions
from torch.nn import Module



# 表示 PatchTSMixerForPredictionOutput 是一个数据类，用于存储预测输出
@dataclass
class PatchTSMixerForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForPredictionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_input_channels)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None



# 表示 SamplePatchTSMixerPredictionOutput 是一个数据类，用于存储时间序列模型的预测输出，包含从所选分布中抽样得到的值
@dataclass
class SamplePatchTSMixerPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None



# 表示 SamplePatchTSMixerRegressionOutput 是一个数据类，用于存储时间序列模型的预测输出，包含从所选分布中抽样得到的值
@dataclass
class SamplePatchTSMixerRegressionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, num_targets)`
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None



# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.nll 复制而来
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)



# 从 transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average 复制而来
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.
    """
```  
 #计算一个张量在指定维度上的加权平均值
    Args参数:
        input_tensor (`torch.FloatTensor`):
            输入张量，需要计算平均值的张量。
        weights (`torch.FloatTensor`, *optional*):
            权重张量，与`input_tensor`相同形状的张量。
        dim (`int`, *optional*):
            需要计算平均值的维度。

    Returns返回值:
        `torch.FloatTensor`: 在指定维度上取平均值后的张量。
    """
    if weights is not None:  # 如果传入权重张量，则进行加权平均
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))  # 根据权重和输入张量计算加权张量
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)  # 计算权重的总和，并且取1.0作为最小值
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights  # 返回加权张量在指定维度上的平均值
    else:  # 如果没有传入权重张量，则计算普通平均值
        return input_tensor.mean(dim=dim)  # 返回输入张量在指定维度上的平均值
class PatchTSMixerForPrediction(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类构造函数初始化对象
        super().__init__(config)
        # 初始化损失函数
        self.loss = config.loss
        # 是否使用返回字典的标志
        self.use_return_dict = config.use_return_dict
        # 预测通道索引列表
        self.prediction_channel_indices = config.prediction_channel_indices
        # 并行采样的数量
        self.num_parallel_samples = config.num_parallel_samples

        # 根据损失函数类型选择分布输出
        if config.loss == "mse":
            # 如果损失函数是均方误差，不需要设置分布输出
            self.distribution_output = None
        else:
            dim = config.prediction_length
            # 根据配置选择合适的分布输出类别
            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            output_class = distribution_output_map.get(config.distribution_output, None)
            if output_class is not None:
                # 如果找到了对应的分布输出类别，则实例化该类
                self.distribution_output = output_class(dim=dim)
            else:
                # 如果未找到对应的分布输出类别，则抛出异常
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 初始化模型和预测头部
        self.model = PatchTSMixerModel(config)
        self.head = PatchTSMixerForPredictionHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # 如果配置中有后初始化标志，则调用后初始化方法
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForPredictionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ):
        """
        根据给定的输入执行前向传播。

        Args:
            past_values (torch.Tensor):
                过去观测值的张量。
            observed_mask (Optional[torch.Tensor], optional):
                观测遮罩的张量，默认为 None。
            future_values (Optional[torch.Tensor], optional):
                未来观测值的张量，默认为 None。
            output_hidden_states (Optional[bool], optional):
                是否输出隐藏状态，默认为 False。
            return_loss (bool, optional):
                是否返回损失，默认为 True。
            return_dict (Optional[bool], optional):
                是否返回字典，默认为 None。

        Returns:
            根据情况返回字典或张量。
        """

    def generate(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    # 定义了一个方法，用于从具有概率分布头部的模型中生成样本预测序列
    # 参数:
    # past_values: 过去的时间序列值，用作预测未来的上下文
    # observed_mask: 一个布尔掩码，指示哪些past_values是观测到的，哪些是缺失的。掩码中的值选在[0, 1]之间，其中：
    #               - 1 表示观测到的值，
    #               - 0 表示缺失的值（即替换成0的NaN值）
    # 返回值:
    # [`SamplePatchTSMixerPredictionOutput`]，其中的`sequences`张量形状为`(batch_size, number of samples, prediction_length, num_input_channels)`
    def sample_patch_ts_mixer_prediction(self, past_values: torch.Tensor, observed_mask: Optional[torch.Tensor]




        # 获取并行采样次数
        num_parallel_samples = self.num_parallel_samples

        # 获取模型输出
        outputs = self(
            past_values=past_values,
            future_values=None,
            observed_mask=observed_mask,
            output_hidden_states=False,
        )

        # 获取分布
        distribution = self.distribution_output.distribution(
            outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
        )

        # 获取样本：`num_parallel_samples`次采样的列表
        samples = [distribution.sample() for _ in range(num_parallel_samples)]

        # 按维度`1`堆叠张量
        samples = torch.stack(samples, dim=1)  # [batch_size x num_samples x prediction_length x num_channels]
        # 返回`SamplePatchTSMixerPredictionOutput`对象，其中的sequences值为`samples`
        return SamplePatchTSMixerPredictionOutput(sequences=samples)
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from .modeling_patch_tsmixer import (
    PatchTSMixerPreTrainedModel,
    PatchTSMixerModel,
    PatchTSMixerLinearHead,
    InjectScalerStatistics4D,
    PatchTSMixerConfig,
)
from ..file_utils import add_start_docstrings_to_model_forward, replace_return_docstrings


@dataclass
class PatchTSMixerForTimeSeriesClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForTimeSeriesClassificationOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None  # 总损失，可选的浮点张量
    prediction_outputs: torch.FloatTensor = None  # 分类头的预测输出，torch.FloatTensor类型，形状为(batch_size, num_labels)
    last_hidden_state: torch.FloatTensor = None  # 通过头之前的骨干嵌入，torch.FloatTensor类型，形状为(batch_size, num_input_channels, num_patches, d_model)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型在每一层输出的隐藏状态，加上可选的初始嵌入输出，元组类型，包含torch.FloatTensor

class PatchTSMixerForTimeSeriesClassification(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)  # 使用给定配置创建 PatchTSMixerModel 对象
        self.head = PatchTSMixerLinearHead(  # 使用给定配置创建 PatchTSMixerLinearHead 对象
            config=config,
        )
        self.use_return_dict = config.use_return_dict  # 是否使用返回字典的配置
        if config.scaling in ["std", "mean", True]:
            # 如果配置中的缩放参数为标准差、均值或True，则使用InjectScalerStatistics4D进行缩放
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None  # 否则不进行缩放

        # 初始化权重并应用最终处理
        if config.post_init:
            self.post_init()  # 如果配置指定了post_init，则执行后初始化

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForTimeSeriesClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    # 定义前向传播函数
    def forward(
        self,
        past_values: torch.FloatTensor,
        future_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> PatchTSMixerForTimeSeriesClassificationOutput:
        # 该函数用于进行时间序列分类任务的前向传播
        # 输入:
        #   past_values: 输入时间序列数据, 形状为 (batch_size, num_input_channels, num_patches, d_model)
        #   future_values: 目标值, 可选, 形状取决于任务类型 (forecasting, regression, classification)
        #   output_hidden_states: 是否返回中间隐藏状态, 可选
        #   return_dict: 是否以字典形式返回输出, 可选
        # 输出:
        #   loss: 损失值, 当有 future_values 输入且 return_loss 为 True 时返回
        #   prediction_outputs: 预测输出, 形状为 (batch_size, n_labels)
        #   last_hidden_state: 最后一层隐藏状态, 形状为 (batch_size, num_input_channels, num_patches, d_model)
        #   hidden_states: 所有层的隐藏状态, 形状为 (num_layers, batch_size, num_input_channels, num_patches, d_model)
    
        # 定义交叉熵损失函数
        loss = torch.nn.CrossEntropyLoss()
    
        # 设置返回字典的使用方式
        return_dict = return_dict if return_dict is not None else self.use_return_dict
    
        # 通过模型得到输出
        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 如果模型输出是元组, 则转换为 PatchTSMixerModelOutput 对象
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)
    
        # 如果指定了注入缩放, 则应用缩放到最后一层隐藏状态
        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )
    
        # 通过分类头得到预测输出
        y_hat = self.head(model_output.last_hidden_state)
    
        # 如果有目标值 future_values 且需要计算损失, 则计算损失
        if future_values is not None and return_loss is True:
            loss_val = loss(y_hat, future_values)
        else:
            loss_val = None
    
        # 如果不需要以字典形式返回, 则以元组形式返回
        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    y_hat,
                    model_output.last_hidden_state,
                    model_output.hidden_states,
                ]
            )
    
        # 以字典形式返回输出
        return PatchTSMixerForTimeSeriesClassificationOutput(
            loss=loss_val,
            prediction_outputs=y_hat,
            last_hidden_state=model_output.last_hidden_state,
            hidden_states=model_output.hidden_states,
        )
@dataclass
class PatchTSMixerForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

# 定义一个用于回归输出的数据结构，包含了预测输出、最后的隐藏状态、隐藏层状态和损失

class InjectScalerStatistics4D(nn.Module):
    def __init__(self, d_model: int, num_patches: int, expansion: int = 2):
        super().__init__()

        # 初始化反向转换扩展模块，将维度扩展到expansion*d_model
        self.inverse_trans_expansion = nn.Linear(d_model + 2, expansion * d_model)
        # 初始化反向转换压缩模块，将维度压缩回d_model
        self.inverse_trans_compression = nn.Linear(expansion * d_model, d_model)
        # 初始化映射尺度扩展模块，将2维映射到2*expansion
        self.map_scale_expansion = nn.Linear(2, 2 * expansion)
        # 初始化映射尺度压缩模块，将2*expansion维映射压缩到2维
        self.map_scale_compression = nn.Linear(2 * expansion, 2)
        # 存储传入的num_patches参数
        self.num_patches = num_patches
    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`
        """

        # 在通道维度上翻转 loc 的矩阵，变成 [batch_size x n_channels x 1]
        mean = loc.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        # 在最后两个维度上插入维度为 1，变成 [batch_size x n_channels x 1 x 1]
        mean = mean.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        # 将 mean 沿着第一个维度重复 num_patch 次，即变成 [batch_size x n_channels x num_patch x 1]
        mean = mean.repeat(1, 1, self.num_patches, 1)  # [batch_size x n_channels x num_patch x 1]

        # 在通道维度上翻转 scale 的矩阵，变成 [batch_size x n_channels x 1 ]
        stdev = scale.transpose(-1, -2)  # [batch_size x n_channels x 1 ]
        # 在最后两个维度上插入维度为 1，变成 [batch_size x n_channels x 1 x 1]
        stdev = stdev.unsqueeze(-2)  # [batch_size x n_channels x 1 x 1]
        # 将 stdev 沿着第一个维度重复 num_patch 次，即变成 [batch_size x n_channels x num_patch x 1]
        stdev = stdev.repeat(1, 1, self.num_patches, 1)  # [batch_size x n_channels x num_patch x 1]

        # 拼接 mean 和 stdev，在最后一个维度上拼接，结果的形状是 [batch_size x n_channels x num_patch x 2]
        concat_stats = torch.cat([mean, stdev], dim=-1)  # [batch_size x n_channels x num_patch x 2]

        # 将拼接的统计信息输入到 map_scale_expansion 函数中，形状变为 [batch_size x n_channels x num_patch x (2*expansion)]
        concat_stats = self.map_scale_expansion(concat_stats)  # [batch_size x n_channels x num_patch x (2*expansion)]
        # 将上一步的结果输入到 map_scale_compression 函数中，形状变为 [batch_size x n_channels x num_patch x 2]
        concat_stats = self.map_scale_compression(concat_stats)  # [batch_size x n_channels x num_patch x 2]

        # 拼接输入的 features 和统计信息，最后一个维度上拼接，结果的形状是 [batch_size x channels x num_patch x (d_model+2)]
        inputs = torch.cat([inputs, concat_stats], dim=-1)  # [batch_size x channels x num_patch x d_model+2]
        # 将上一步的结果输入到 inverse_trans_expansion 函数中，形状变为 [batch_size x channels x num_patch x (expansion*d_model)]
        inputs = self.inverse_trans_expansion(inputs)  # [batch_size x channels x num_patch x (expansion*d_model)]
        # 将上一步的结果输入到 inverse_trans_compression 函数中，形状变为 [batch_size x channels x num_patch x d_model]
        inputs = self.inverse_trans_compression(inputs)  # [batch_size x channels x num_patch x d_model]

        # 返回输入的 features
        return inputs
class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 PatchTSMixerModel 实例
        self.model = PatchTSMixerModel(config)

        # 设置损失函数和输出分布
        self.loss = config.loss 
        self.distribution_output = config.distribution_output

        # 设置是否返回字典和并行采样的数量
        self.use_return_dict = config.use_return_dict
        self.num_parallel_samples = config.num_parallel_samples

        # 根据损失函数设置输出分布
        if config.loss == "mse":
            # 如果损失函数是均方误差，则输出分布设置为None
            self.distribution_output = None
        else:
            # 根据config.distribution_output选择输出分布类
            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            output_class = distribution_output_map.get(config.distribution_output)
            if output_class is not None:
                self.distribution_output = output_class(dim=config.num_targets)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 根据scaling参数选择是否使用InjectScalerStatistics4D
        if config.scaling in ["std", "mean", True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None

        # 创建 PatchTSMixerLinearHead 实例
        self.head = PatchTSMixerLinearHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # 初始化权重并应用最终处理
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    def generate(
        self,
        past_values: torch.Tensor,
    ) -> SamplePatchTSMixerRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

        Return:
            [`SamplePatchTSMixerRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_targets)`.
        """
        # 获取样本数量
        num_parallel_samples = self.num_parallel_samples

        # 获取模型输出
        outputs = self(
            past_values=past_values,
            future_values=None,
            output_hidden_states=False,
        )

        # 获取概率分布
        distribution = self.distribution_output.distribution(outputs.prediction_outputs)

        # 获取样本
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [batch_size x num_targets]
        # 堆叠张量
        samples = torch.stack(samples, dim=1)  # [batch_size x num_samples x num_targets]
        return SamplePatchTSMixerRegressionOutput(sequences=samples)
```