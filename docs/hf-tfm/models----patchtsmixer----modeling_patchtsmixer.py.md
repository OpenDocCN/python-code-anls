# `.\models\patchtsmixer\modeling_patchtsmixer.py`

```
# 引入 math 模块，用于数学计算函数
import math
# 引入 dataclass 模块，用于定义数据类
from dataclasses import dataclass
# 引入 Optional、Tuple、Union 类型，用于类型提示
from typing import Optional, Tuple, Union

# 引入 PyTorch 模块
import torch
# 引入 PyTorch 的神经网络模块
import torch.nn as nn

# 引入 Transformers 库中的预训练模型基类 PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
# 引入 Transformers 库中的模型输出类 ModelOutput
from transformers.utils import ModelOutput

# 引入日志记录工具、文档字符串添加函数、返回值替换函数等实用工具函数
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入 PatchTSMixer 的配置类
from .configuration_patchtsmixer import PatchTSMixerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# PatchTSMixer 预训练模型存档列表
PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtsmixer-etth1-pretrain",
    # 更多 PatchTSMixer 模型可在 https://huggingface.co/models?filter=patchtsmixer 查看
]

# PatchTSMixer 模型文档的起始字符串
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

# PatchTSMixer 模型输入参数文档字符串
PATCHTSMIXER_INPUTS_DOCSTRING = r"""
    # 定义函数的参数和类型注解，输入参数为过去时间序列的值
    # 对于预训练任务，这表示要预测掩码部分的输入时间序列；对于预测任务，这表示历史/过去的时间序列值；
    # 对于分类或回归任务，这表示时间序列的上下文值。
    # 对于单变量时间序列，num_input_channels 维度应为 1；对于多变量时间序列，它大于 1。
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        # Linear layer for computing attention weights
        self.attn_layer = nn.Linear(in_size, out_size)
        # Softmax activation to normalize attention weights across input dimensions
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        # Calculate attention weights using linear layer and apply softmax
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        # Apply gated attention mechanism to input data
        inputs = inputs * attn_weight
        return inputs


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTBatchNorm with PatchTST->PatchTSMixer
class PatchTSMixerBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # Batch normalization across the d_model dimension
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        # Transpose input tensor to match expected shape for batch normalization
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        # Apply batch normalization along the d_model dimension
        output = self.batchnorm(output)
        # Transpose output back to original shape
        return output.transpose(1, 2)


class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # positional encoding initialization based on config settings
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            # Initialize positional encoding as a parameter tensor filled with zeros
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: PatchTSMixerConfig) -> nn.Parameter:
        # Positional encoding initialization based on configuration
        # 根据配置初始化位置编码

        # If positional encoding type is 'random', initialize with random values
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
            # 使用随机值初始化位置编码张量

        # If positional encoding type is 'sincos', initialize with sine and cosine positional encodings
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(config.num_patches, config.d_model)
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
            # 使用sin和cos函数生成位置编码张量，并进行标准化处理

        else:
            # Raise an error if an unsupported positional encoding type is provided
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
            # 如果提供了不支持的位置编码类型，则引发错误

        return position_enc
        # 返回位置编码张量作为模型参数

    def forward(self, patch_input: torch.Tensor):
        # Calculate the hidden state by adding positional encoding to patch input
        # 计算隐藏状态，将位置编码添加到补丁输入中
        hidden_state = patch_input + self.position_enc
        return hidden_state
        # 返回隐藏状态张量作为前向传播的输出
class PatchTSMixerNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        # 根据配置选择合适的归一化层
        if "batch" in config.norm_mlp.lower():
            # 如果配置中包含"batch"，使用批量归一化层
            self.norm = PatchTSMixerBatchNorm(config)
        else:
            # 否则使用 Layer Normalization，设置 epsilon 为 config.norm_eps
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # 重塑数据形状为 [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )

            # 对重塑后的数据进行归一化处理
            inputs_reshaped = self.norm(inputs_reshaped)

            # 恢复数据到原始形状
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            # 使用选择的归一化层处理输入
            inputs = self.norm(inputs)

        return inputs


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
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
        # 第一层全连接 + GELU 激活 + dropout
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        # 第二层全连接 + dropout
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class PatchTSMixerChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """
    # 初始化函数，接受一个配置对象 `PatchTSMixerConfig` 作为参数
    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个 PatchTSMixerNormLayer 层，并将其赋值给 self.norm
        self.norm = PatchTSMixerNormLayer(config)
        
        # 将配置对象中的 gated_attn 属性赋值给 self.gated_attn
        self.gated_attn = config.gated_attn
        
        # 创建一个 PatchTSMixerMLP 实例，并将其赋值给 self.mlp
        self.mlp = PatchTSMixerMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        # 如果配置中的 gated_attn 为 True，则创建一个 PatchTSMixerGatedAttention 实例，并将其赋值给 self.gating_block
        if config.gated_attn:
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
        # 将输入的张量保存为 residual，用于后续的残差连接
        residual = inputs
        
        # 对输入进行归一化处理
        inputs = self.norm(inputs)

        # 将张量维度重新排列为 (batch_size, d_model, num_patches, num_channels)
        inputs = inputs.permute(0, 3, 2, 1)

        # 如果存在 gated_attn，对输入应用 gating_block 进行注意力机制操作
        if self.gated_attn:
            inputs = self.gating_block(inputs)

        # 通过 MLP 层处理输入张量
        inputs = self.mlp(inputs)

        # 将张量维度重新排列为原始顺序 (batch_size, num_channels, num_patches, d_model)
        inputs = inputs.permute(0, 3, 2, 1)

        # 将处理后的张量与残差张量相加，得到最终输出
        out = inputs + residual
        return out
# 从transformers.models.bart.modeling_bart.BartAttention复制到PatchTSMixerAttention并将Bart改为PatchTSMixer
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
        self.embed_dim = embed_dim  # 初始化注意力机制的输入维度
        self.num_heads = num_heads  # 注意力头的数量
        self.dropout = dropout  # dropout概率
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        self.config = config  # PatchTSMixer的配置对象

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于缩放点积注意力的输出
        self.is_decoder = is_decoder  # 是否用作解码器
        self.is_causal = is_causal  # 是否是因果注意力

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 线性变换，用于生成key
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 线性变换，用于生成value
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 线性变换，用于生成query
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 最终的输出线性变换

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 重塑张量形状以支持多头注意力计算，然后转置维度以便正确计算注意力

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 前向传播函数，实现注意力机制的计算



class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.norm = PatchTSMixerNormLayer(config)  # 规范化层，用于标准化输入数据

        self.self_attn = config.self_attn  # 是否使用自注意力机制
        self.gated_attn = config.gated_attn  # 是否使用门控注意力机制

        self.mlp = PatchTSMixerMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )
        # 多层感知机，用于处理补丁维度

        if config.gated_attn:
            self.gating_block = PatchTSMixerGatedAttention(in_size=config.num_patches, out_size=config.num_patches)
            # 如果使用门控注意力，初始化门控注意力模块

        if config.self_attn:
            self.self_attn_layer = PatchTSMixerAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = PatchTSMixerNormLayer(config)
            # 如果使用自注意力，初始化自注意力层和相应的规范化层
    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        # 保存输入张量作为残差连接的基准
        residual = hidden_state

        # 应用层归一化到输入张量
        hidden_state = self.norm(hidden_state)

        # 如果使用自注意力机制
        if self.self_attn:
            # 获取张量的形状信息
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            # 重塑张量以便进行自注意力操作
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            # 应用自注意力层，关闭注意力输出选项
            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            # 将输出张量重塑回原始形状
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # 将张量转置，使得 num_patches 成为最后一个维度
        hidden_state = hidden_state.transpose(2, 3)
        # 应用多层感知机（MLP）转换
        hidden_state = self.mlp(hidden_state)

        # 如果使用门控注意力机制
        if self.gated_attn:
            # 应用门控块
            hidden_state = self.gating_block(hidden_state)

        # 再次将张量转置回原始形状
        hidden_state = hidden_state.transpose(2, 3)

        # 如果使用自注意力机制，应用层归一化到注意力输出和输入张量的残差连接
        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        # 将残差连接的结果添加到变换后的张量上，作为最终输出
        out = hidden_state + residual
        return out
class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`PatchTSMixerConfig`, *required`):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 初始化层：使用给定的配置初始化规范化层
        self.norm = PatchTSMixerNormLayer(config)

        # 获取配置中的门控注意力标志
        self.gated_attn = config.gated_attn

        # 初始化层：使用给定的配置初始化多层感知机（MLP）层
        self.mlp = PatchTSMixerMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        # 如果配置中包含门控注意力，则初始化门控注意力块
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
        # 保存输入张量作为残差连接的一部分
        residual = hidden

        # 对输入张量进行规范化处理
        hidden = self.norm(hidden)

        # 通过多层感知机处理规范化后的张量
        hidden = self.mlp(hidden)

        # 如果启用了门控注意力，则使用门控注意力块处理张量
        if self.gated_attn:
            hidden = self.gating_block(hidden)

        # 将处理后的张量与残差连接起来作为输出
        out = hidden + residual
        return out


class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required`):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 初始化层：使用给定的配置初始化PatchMixerBlock
        self.patch_mixer = PatchMixerBlock(config=config)

        # 初始化层：使用给定的配置初始化FeatureMixerBlock
        self.feature_mixer = FeatureMixerBlock(config=config)

        # 获取配置中的模式信息
        self.mode = config.mode

        # 如果模式是"mix_channel"，则初始化通道特征混合块
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
        # 如果模式是"mix_channel"，则使用通道特征混合块处理输入张量
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        # 使用PatchMixerBlock处理输入张量
        hidden = self.patch_mixer(hidden)

        # 使用FeatureMixerBlock处理输入张量
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class PatchTSMixerBlock(nn.Module):
    """The main computing framework of the `PatchTSMixer` model.

    Args:
        config (`PatchTSMixerConfig`, *required`):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        # 获取层数
        num_layers = config.num_layers

        # 使用循环初始化PatchTSMixerLayer模块列表
        self.mixers = nn.ModuleList([PatchTSMixerLayer(config=config) for _ in range(num_layers)])
    # 定义一个方法 `forward`，用于前向传播计算
    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): 输入张量。
            output_hidden_states (`bool`, *optional*, 默认为 False):
                是否输出所有隐藏状态。

        Returns:
            `torch.Tensor`: 嵌入结果。 `list`: 如果 `output_hidden_states` 设置为 `True`，则返回所有隐藏状态的列表。
        """
        # 初始化一个空列表，用于存储所有的隐藏状态
        all_hidden_states = []

        # 初始嵌入为输入的隐藏状态张量
        embedding = hidden_state

        # 遍历所有的混合模块
        for mod in self.mixers:
            # 将当前嵌入张量通过混合模块进行处理
            embedding = mod(embedding)
            # 如果设置要输出隐藏状态，则将当前处理后的嵌入张量加入列表中
            if output_hidden_states:
                all_hidden_states.append(embedding)

        # 如果设置要输出隐藏状态，则返回最终的嵌入张量和所有隐藏状态列表
        if output_hidden_states:
            return embedding, all_hidden_states
        # 否则，只返回最终的嵌入张量和空值
        else:
            return embedding, None
class PatchTSMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`PatchTSMixerConfig`, *required*): Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()

        self.prediction_channel_indices = config.prediction_channel_indices  # 获取预测通道的索引列表

        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()  # 如果索引列表不为空，则排序索引

        self.dropout_layer = nn.Dropout(config.head_dropout)  # 创建一个Dropout层，用于随机失活
        if distribution_output is None:
            self.base_forecast_block = nn.Linear((config.num_patches * config.d_model), config.prediction_length)
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(
                config.num_patches * config.d_model
            )  # 根据分布输出类型选择线性层或其他投影层

        self.flatten = nn.Flatten(start_dim=-2)  # 创建一个展平层，用于展平输入张量

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """

        hidden_features = self.flatten(hidden_features)  # 将输入张量展平成 `(batch_size x n_vars x num_patch * d_model)`
        hidden_features = self.dropout_layer(hidden_features)  # 对展平后的张量进行Dropout操作
        forecast = self.base_forecast_block(hidden_features)  # 使用预测头线性层进行预测
        if isinstance(forecast, tuple):
            forecast = tuple(z.transpose(-1, -2) for z in forecast)  # 如果预测结果是元组，则对每个元素进行维度转置
        else:
            forecast = forecast.transpose(-1, -2)  # 否则，对预测张量进行维度转置为 `(batch_size x prediction_length x n_vars)`

        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple(z[..., self.prediction_channel_indices] for z in forecast)  # 如果有预测通道索引，仅保留指定通道的预测结果
            else:
                forecast = forecast[..., self.prediction_channel_indices]  # 对预测结果张量仅保留指定通道的预测结果

        return forecast


class PatchTSMixerLinearHead(nn.Module):
    """Linear head for Classification and Regression.

    Args:
        config (`PatchTSMixerConfig`, *required*): Configuration.
    """
    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()  # 调用父类的初始化方法

        self.head_aggregation = config.head_aggregation  # 设置头部聚合方式
        self.output_range = config.output_range  # 设置输出范围

        if config.head_aggregation is None:
            mul_factor = config.num_patches  # 如果头部聚合方式为None，则设置乘数因子为patch数量
        else:
            mul_factor = 1  # 否则设置乘数因子为1
        self.distribution_output = distribution_output  # 设置分布输出
        if distribution_output is None:
            self.projection = nn.Linear(
                config.d_model * config.num_input_channels * mul_factor,
                config.num_targets,
            )  # 如果分布输出为None，则设置线性投影层
        else:
            self.projection = distribution_output.get_parameter_projection(
                config.d_model * config.num_input_channels * mul_factor
            )  # 否则根据分布输出获取参数投影

        if config.head_aggregation is None:
            self.flatten = nn.Flatten(start_dim=-3)  # 如果头部聚合方式为None，则设置展平层
        else:
            self.flatten = nn.Flatten(start_dim=-2)  # 否则设置展平层

        self.dropout = nn.Dropout(config.head_dropout)  # 设置dropout层

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x num_targets)`.
        """

        # 调整hidden_features的维度顺序，将最后两个维度进行交换
        hidden_features = hidden_features.transpose(-1, -2)

        if self.head_aggregation == "use_last":
            # 如果头部聚合方式为"use_last"，选择最后一个位置的特征
            hidden_features = hidden_features[..., -1]
        elif self.head_aggregation == "max_pool":
            # 如果头部聚合方式为"max_pool"，对最后一个维度进行最大池化操作
            hidden_features = hidden_features.max(dim=-1).values
        elif self.head_aggregation == "avg_pool":
            # 如果头部聚合方式为"avg_pool"，对最后一个维度进行平均池化操作
            hidden_features = hidden_features.mean(dim=-1)

        if self.flatten:
            hidden_features = self.flatten(hidden_features)  # 如果需要展平，则进行展平操作
        hidden_features = self.dropout(hidden_features)  # 对特征进行dropout处理
        hidden_features = self.projection(hidden_features)  # 使用投影层进行特征投影

        if (self.distribution_output is None) and (self.output_range is not None):
            # 如果分布输出为None且输出范围不为None，则对输出进行sigmoid归一化处理
            hidden_features = (
                torch.sigmoid(hidden_features) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
            )

        return hidden_features  # 返回处理后的特征
class PatchTSMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = PatchTSMixerConfig  # 设置配置类为 PatchTSMixerConfig
    base_model_prefix = "model"  # 基础模型前缀设为 "model"
    main_input_name = "past_values"  # 主输入名称设为 "past_values"
    supports_gradient_checkpointing = False  # 不支持梯度检查点

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, PatchTSMixerPositionalEncoding):
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()  # 将偏置项初始化为零
            module.weight.data.fill_(1.0)  # 将权重初始化为全1
        elif isinstance(module, PatchTSMixerBatchNorm):
            module.batchnorm.bias.data.zero_()  # 将批归一化层的偏置项初始化为零
            module.batchnorm.weight.data.fill_(1.0)  # 将批归一化层的权重初始化为全1
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)  # 使用正态分布初始化权重
            if module.bias is not None:
                module.bias.data.zero_()  # 如果存在偏置项，则初始化为零

class PatchTSMixerPretrainHead(nn.Module):
    """Pretraining head.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.dropout_layer = nn.Dropout(config.head_dropout)  # 使用给定的 dropout 概率创建 dropout 层
        self.base_pt_block = nn.Linear(config.d_model, config.patch_length)  # 创建线性层

    def forward(self, hidden_features):
        """
        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size x num_patch x d_model)` in `flatten` mode
                or `(batch_size x n_vars x num_patch x d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size x n_vars x num_patch x patch_length)`.
        """

        hidden_features = self.dropout_layer(hidden_features)  # 应用 dropout 层到输入特征上
        forecast = self.base_pt_block(hidden_features)  # 使用线性层进行预测
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
    
    Args:
        inputs (torch.Tensor): Input tensor to be masked.
        mask_ratio (float): Ratio of elements to be masked.
        unmasked_channel_indices (list, optional): List of unmasked channel indices.
        channel_consistent_masking (bool, optional): Whether to mask consistently across channels.
        mask_value (int, optional): Value to fill in for masked elements.

    Returns:
        torch.Tensor: Masked input tensor.
    """
    # 检查掩码比例是否在有效范围内
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    # 获取输入张量的形状信息
    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    # 计算不被掩码的数据长度
    len_keep = int(sequence_length * (1 - mask_ratio))

    # 根据channel_consistent_masking的设置生成随机噪声张量
    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # noise in [0, 1], bs x 1 x L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)  # noise in [0, 1], bs x num_channels x L

    # 创建掩码张量，初始化为全1
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)

    # 将前len_keep个位置置为0，即进行掩码操作
    mask[:, :, :len_keep] = 0

    # 对噪声进行排序，以便后续恢复掩码位置
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    # 根据排序后的索引恢复掩码的顺序
    mask = torch.gather(mask, dim=-1, index=ids_restore)

    # 将掩码张量的形状调整为与输入张量相同，且每个掩码值重复num_features次
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]

    # 如果有指定不被掩码的通道，将这些通道的掩码值置为0
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    # 使用掩码值进行输入张量的掩码操作，将掩码后的结果作为inputs_mask返回
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)

    # 返回掩码后的输入张量和掩码张量的第一个维度
    return inputs_mask, mask[..., 0]
# Copied from transformers.models.patchtst.modeling_patchtst.forecast_masking
def forecast_masking(
    inputs: torch.Tensor,
    num_forecast_mask_patches: Union[list, int],
    unmasked_channel_indices: list = None,
    mask_value: int = 0,
):
    """Forecast masking that masks the last K patches where K is from the num_forecast_mask_patches.
    If num_forecast_mask_patches is a list, samples in the batch will be randomly masked by numbers defined in the list.

    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_length)`
        num_forecast_mask_patches (`list`):
            Number of patches to be masked at the end of each batch sample. e.g. 4 or [3, 5].
        unmasked_channel_indices (`list`, *optional*):
            Indices of channels that are not masked.
        mask_value (`int`, *optional*, defaults to 0):
            Values in the masked patches will be filled by `mask_value`.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as inputs Tensor and Mask tensor of shape `(bs,
        num_channels , num_patch)` or `(bs, tsg1, tsg2, num_channels, num_patch)`
    """

    # If num_forecast_mask_patches is an integer, convert it to a list for consistency
    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]

    # Initialize forecast_mask_ratios with a list of 1s for each num_forecast_mask_patches
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]

    # Extract dimensions from inputs tensor
    batch_size, num_channels, sequence_length, num_features = inputs.shape

    # Initialize mask tensor with zeros
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    # Initialize an empty list to store temporary computations
    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    # Iterate over num_forecast_mask_patches and forecast_mask_ratios to compute temporary lengths
    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        # Validate patch_length to ensure it is within valid range
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(
                f"num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches."
            )
        # Compute temporary length based on batch size and ratio
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    # Sort t_list based on the third element (temp_len)
    t_list = sorted(t_list, key=lambda x: x[2])

    # Adjust the last element in t_list to match batch size
    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    # Initialize batch indices
    batch1 = 0

    # Iterate over t_list to populate mask tensor
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    # Randomly permute the batch indices of mask tensor
    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    # Expand mask tensor dimensions to match inputs tensor
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)

    # If unmasked_channel_indices is provided, zero out corresponding channels in mask tensor
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    # Apply masking to inputs tensor using mask tensor and return masked inputs and mask tensor
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]
class PatchTSMixerPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()

        self.sequence_length = config.context_length  # 设置实例变量sequence_length为config中的context_length
        self.patch_length = config.patch_length  # 设置实例变量patch_length为config中的patch_length
        self.patch_stride = config.patch_stride  # 设置实例变量patch_stride为config中的patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # 计算patch的数量
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length  # 计算起始序列位置

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]  # 获取输入序列的长度
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start :, :]  # 截取序列的起始位置后的值
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)  # 使用unfold方法进行切片操作
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()  # 转置操作，调整维度顺序
        return output


# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTMasking with PatchTST->PatchTSMixer
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
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置对象中获取随机遮盖比例
        self.random_mask_ratio = config.random_mask_ratio
        # 从配置对象中获取通道一致性遮盖的标志
        self.channel_consistent_masking = config.channel_consistent_masking
        # 从配置对象中获取遮盖类型
        self.mask_type = config.mask_type
        # 从配置对象中获取预测遮盖的数量
        self.num_forecast_mask_patches = config.num_forecast_mask_patches
        # 从配置对象中获取未遮盖通道的索引列表
        self.unmasked_channel_indices = config.unmasked_channel_indices
        # 从配置对象中获取遮盖数值
        self.mask_value = config.mask_value
        # 如果存在未遮盖通道的索引列表，则对其进行排序
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input

        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points

        """
        # 根据遮盖类型选择不同的遮盖方法
        if self.mask_type == "random":
            # 调用随机遮盖函数，生成遮盖后的输入和遮盖掩码
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
            )
        elif self.mask_type == "forecast":
            # 调用预测遮盖函数，生成遮盖后的输入和遮盖掩码
            masked_input, mask = forecast_masking(
                inputs=patch_input,
                num_forecast_mask_patches=self.num_forecast_mask_patches,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value,
            )
        else:
            # 若遮盖类型无效，则抛出数值错误异常
            raise ValueError(f"Invalid mask type {self.mask_type}.")

        # 将遮盖掩码转换为布尔型张量
        mask = mask.bool()
        # 返回遮盖后的输入和遮盖掩码
        return masked_input, mask
# 从 transformers.models.patchtst.modeling_patchtst.PatchTSTStdScaler 复制的代码，将 PatchTST 替换为 PatchTSMixer
class PatchTSMixerStdScaler(nn.Module):
    """
    标准化特征，通过计算均值并沿第一个维度进行缩放，然后通过减去均值并除以标准差进行归一化。
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # 如果 config 中有 scaling_dim 属性，则使用其值作为 dim；否则默认为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果 config 中有 keepdim 属性，则使用其值作为 keepdim；否则默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果 config 中有 minimum_scale 属性，则使用其值作为 minimum_scale；否则默认为 1e-5
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播方法
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                输入数据用于批次归一化计算
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                计算观察指标上的缩放
        Returns:
            返回元组，包含三个 `torch.Tensor`：
                (`(batch_size, sequence_length, num_input_channels)`, 
                 `(batch_size, 1, num_input_channels)`,
                 `(batch_size, 1, num_input_channels)`)
        """
        # 计算分母，即观察指标的和
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将分母限制为最小值为 1.0
        denominator = denominator.clamp_min(1.0)
        # 计算均值，即数据乘以观察指标后的和除以分母
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差，即数据减去均值后乘以观察指标的平方和除以分母
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标准差，即方差加上最小缩放值后开平方
        scale = torch.sqrt(variance + self.minimum_scale)
        # 返回归一化后的数据，均值 loc，标准差 scale
        return (data - loc) / scale, loc, scale


# 从 transformers.models.patchtst.modeling_patchtst.PatchTSTMeanScaler 复制的代码，将 PatchTST 替换为 PatchTSMixer
class PatchTSMixerMeanScaler(nn.Module):
    """
    计算缩放因子作为第一个维度上的加权平均绝对值，并相应地缩放数据。
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # 如果 config 中有 scaling_dim 属性，则使用其值作为 dim；否则默认为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果 config 中有 keepdim 属性，则使用其值作为 keepdim；否则默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 如果 config 中有 minimum_scale 属性，则使用其值作为 minimum_scale；否则默认为 1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 如果 config 中有 default_scale 属性，则使用其值作为 default_scale；否则默认为 None
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
        # 这里 forward 方法未完成
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                输入用于批量归一化计算的数据
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                表示观察指标，用于计算缩放比例
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`, `(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算加权求和，得到每个通道的绝对值和
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        # 计算每个通道上的观察样本数量
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # 计算每个通道的缩放比例
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # 如果提供了 `default_scale`，则使用它；否则使用批量的缩放比例
        if self.default_scale is None:
            # 计算整个批次的加权绝对值和
            batch_sum = ts_sum.sum(dim=0)
            # 计算整个批次的观察样本数量，并至少为1
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            # 计算默认的缩放比例
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            # 使用给定的 `default_scale` 来初始化缩放比例
            default_scale = self.default_scale * torch.ones_like(scale)

        # 在没有观察到样本的位置应用默认的缩放比例
        scale = torch.where(num_observed > 0, scale, default_scale)

        # 确保缩放比例至少为 `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        # 对数据应用缩放
        scaled_data = data / scale

        # 如果不保持维度，则将缩放比例的维度压缩
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale
# Copied from transformers.models.patchtst.modeling_patchtst.PatchTSTNOPScaler with PatchTST->PatchTSMixer
class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        # 设置缩放维度为配置中的 scaling_dim，如果不存在则默认为 1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 设置是否保持维度的配置参数，默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算沿着指定维度 dim 的数据均值，得到缩放因子 scale
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 初始化位置信息为零向量，用于模型输出的位置参数
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回原始数据、位置信息 loc 和缩放因子 scale
        return data, loc, scale


@dataclass
class PatchTSMixerEncoderOutput(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.use_return_dict = config.use_return_dict

        # 线性层，用于将 patch 后的时间序列映射到 d_model 维度
        self.patcher = nn.Linear(config.patch_length, config.d_model)
        # 如果使用位置编码，则初始化位置编码器
        if config.use_positional_encoding:
            self.positional_encoder = PatchTSMixerPositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        # MLP-Mixer 编码器块
        self.mlp_mixer_encoder = PatchTSMixerBlock(config=config)

        # 如果设置了 post_init 标志，则调用后初始化方法
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

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """

        # Determine the final return format based on `return_dict` or `self.use_return_dict`
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # Flatten the input `past_values` into patches [bs x num_patch x d_model]
        # For multivariate time series, patches will be [bs x n_vars x num_patch x d_model]
        patches = self.patcher(past_values)

        # Add positional encoding to the patches if a positional encoder is provided
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        # Apply the MLP-Mixer encoder to obtain the last hidden state and potentially all hidden states
        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        # If `return_dict` is False, return the outputs as a tuple
        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        # If `return_dict` is True, return the outputs wrapped in PatchTSMixerEncoderOutput
        return PatchTSMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)
@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
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


@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        super().__init__(config)

        # 设置是否返回字典格式的输出
        self.use_return_dict = config.use_return_dict
        # 初始化编码器
        self.encoder = PatchTSMixerEncoder(config)
        # 初始化 patching 模块
        self.patching = PatchTSMixerPatchify(config)

        # 如果需要对输入进行掩码处理，则初始化 masking 模块；否则置为 None
        if mask_input is True:
            self.masking = PatchTSMixerMasking(config)
        else:
            self.masking = None

        # 根据配置选择标准化器（均值、标准差或无操作）
        if config.scaling == "mean":
            self.scaler = PatchTSMixerMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = PatchTSMixerStdScaler(config)
        else:
            self.scaler = PatchTSMixerNOPScaler(config)

        # 如果配置要求在初始化后进行进一步处理，则调用 post_init 方法
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
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
            `PatchTSMixerModelOutput`: An object containing encoder outputs and other processed inputs.

        """
        # Determine if the return_dict should be used or not
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # Initialize mask to None
        mask = None
        # If observed_mask is not provided, initialize it as a tensor of ones with the same shape as past_values
        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        
        # Scale the observed values using a scaler function, and get location and scale parameters
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)

        # Patch the scaled past values using a patching function
        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length]

        # Prepare encoder input; apply masking if masking function is defined
        enc_input = patched_x
        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
            # enc_input: [batch_size x num_input_channels x num_patch x patch_length]
            # mask: [batch_size x num_input_channels x num_patch]

        # Pass the encoder input to the encoder module
        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Ensure encoder_output is of type PatchTSMixerEncoderOutput if it is a tuple
        if isinstance(encoder_output, tuple):
            encoder_output = PatchTSMixerEncoderOutput(*encoder_output)

        # If return_dict is False, return a tuple of selected encoder outputs and inputs
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

        # If return_dict is True, return a PatchTSMixerModelOutput object with specified attributes
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
    Output type of [`PatchTSMixerForPreTrainingOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, patch_length)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_outputs: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerForPretraining(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        # 调用父类的构造函数初始化对象
        super().__init__(config)
        # 使用给定配置创建 PatchTSMixerModel 对象，设置 mask_input 为 True
        self.model = PatchTSMixerModel(config, mask_input=True)
        # 创建 PatchTSMixerPretrainHead 对象，使用给定配置
        self.head = PatchTSMixerPretrainHead(config=config)
        # 从配置中获取 masked_loss，并将其赋值给对象属性
        self.masked_loss = config.masked_loss
        # 从配置中获取 use_return_dict，并将其赋值给对象属性
        self.use_return_dict = config.use_return_dict

        # 如果配置中指定了 post_init 为 True，则调用对象的 post_init 方法
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
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
            PatchTSMixerForPreTrainingOutput: An instance of the output class containing various outputs based on the model's forward pass.
        """
        # Determine whether to use the provided return_dict or the default one from the class
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # Define the type of loss function based on whether masked loss is enabled
        if self.masked_loss is True:
            loss = torch.nn.MSELoss(reduction="none")
        else:
            loss = torch.nn.MSELoss(reduction="mean")

        # Perform forward pass through the model with specified arguments
        model_output = self.model(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x.last_hidden_state: [batch_size x nvars x num_patch x d_model]

        # Ensure model_output is of type PatchTSMixerModelOutput
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        # Generate predictions using the head module
        x_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x nvars x num_patch x patch_length]

        # Compute loss if return_loss flag is set to True
        if return_loss is True:
            loss_val = loss(x_hat, model_output.patch_input)
        else:
            loss_val = None

        # Calculate masked loss if enabled and loss_val is not None
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        # Return outputs based on whether return_dict is False
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

        # Return outputs wrapped in PatchTSMixerForPreTrainingOutput object if return_dict is True
        return PatchTSMixerForPreTrainingOutput(
            loss=loss_val,
            prediction_outputs=x_hat,  # tensor [batch_size x nvars x num_patch x patch_length]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )
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

    loss: Optional[torch.FloatTensor] = None  # 可选的损失值
    prediction_outputs: torch.FloatTensor = None  # 预测输出
    last_hidden_state: torch.FloatTensor = None  # 经过头部之前的背景嵌入
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型各层输出的隐藏状态
    loc: torch.FloatTensor = None  # 输入均值
    scale: torch.FloatTensor = None  # 输入标准差


@dataclass
class SamplePatchTSMixerPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length, number_channels)`):
            Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None  # 从选择的分布中抽样得到的序列值


@dataclass
class SamplePatchTSMixerRegressionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, num_targets)`
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None  # 从选择的分布中抽样得到的序列值


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)  # 计算输入分布相对于目标的负对数似然损失


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.
    """
    # 计算给定张量在给定维度上的加权平均值，遮蔽与零权重相关的值
    return torch.sum(input_tensor * weights, dim=dim, keepdim=True) / torch.sum(weights, dim=dim, keepdim=True)
    # 如果提供了权重，则计算加权平均值
    if weights is not None:
        # 计算加权后的张量，其中权重不为零的位置乘以输入张量对应位置的值，否则置为零
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        # 计算权重的总和，对给定维度进行求和，如果没有指定维度，则对整个张量进行求和
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        # 返回加权平均值，对给定维度进行求和并除以总权重，如果没有指定维度，则对整个张量进行操作
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        # 如果未提供权重，则计算输入张量沿指定维度的平均值
        return input_tensor.mean(dim=dim)
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
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 从配置中获取损失函数类型
        self.loss = config.loss
        # 从配置中获取是否返回字典类型结果的选项
        self.use_return_dict = config.use_return_dict
        # 从配置中获取预测通道索引列表
        self.prediction_channel_indices = config.prediction_channel_indices
        # 从配置中获取并行采样数量
        self.num_parallel_samples = config.num_parallel_samples

        # 根据配置中的损失函数类型选择分布输出类型
        if config.loss == "mse":
            self.distribution_output = None
        else:
            dim = config.prediction_length
            distribution_output_map = {
                "student_t": StudentTOutput,
                "normal": NormalOutput,
                "negative_binomial": NegativeBinomialOutput,
            }
            # 根据配置中的分布输出类型选择相应的输出类
            output_class = distribution_output_map.get(config.distribution_output, None)
            if output_class is not None:
                self.distribution_output = output_class(dim=dim)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 创建 PatchTSMixerModel 模型对象
        self.model = PatchTSMixerModel(config)
        # 创建 PatchTSMixerForPredictionHead 头部对象
        self.head = PatchTSMixerForPredictionHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # 如果配置指定了后初始化操作，则执行后初始化
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
        # 前向传播函数，接受多个输入参数并返回预测结果
        ...

    def generate(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        ...
    ) -> SamplePatchTSMixerPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSMixerPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_input_channels)`.
        """
        # 获取并行采样数量
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

        # 获取样本：列表，每个元素为 [batch_size x prediction_length x num_channels]
        samples = [distribution.sample() for _ in range(num_parallel_samples)]

        # 堆叠张量
        samples = torch.stack(samples, dim=1)  # [batch_size x num_samples x prediction_length x num_channels]
        return SamplePatchTSMixerPredictionOutput(sequences=samples)
@dataclass
class PatchTSMixerForTimeSeriesClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForTimeSeriesClassificationOutput`].

    Args:
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Prediction output from the classification head.
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

        # Initialize the main model backbone
        self.model = PatchTSMixerModel(config)

        # Initialize the classification head
        self.head = PatchTSMixerLinearHead(
            config=config,
        )

        # Determine if statistical scaling should be applied
        self.use_return_dict = config.use_return_dict
        if config.scaling in ["std", "mean", True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None

        # Apply post-initialization steps if specified in the configuration
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=PatchTSMixerForTimeSeriesClassificationOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ):
        """
        Perform forward pass of the PatchTSMixerForTimeSeriesClassification model.

        Args:
            past_values (`torch.Tensor`):
                Tensor of past input values.
            target_values (`torch.Tensor`, *optional*):
                Tensor of target values for training.
            output_hidden_states (`bool`, *optional*):
                Whether to output hidden states.
            return_loss (`bool`, *optional*):
                Whether to return the loss.
            return_dict (`bool`, *optional*):
                Whether to return a dictionary of outputs.

        Returns:
            Depending on the `return_dict` setting, returns either a dictionary of outputs or directly the outputs.
        """
        # Forward pass through the main model backbone
        # and the classification head
        pass  # Actual computation details are omitted for brevity
        r"""
        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, num_input_channels)` for forecasting,
            `(batch_size, num_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target
            values of the time series, that serve as labels for the model. The `target_values` is what the
            Transformer needs during training to learn to output, given the `past_values`. Note that, this is NOT
            required for a pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, num_input_channels)`. Even if we want
            to forecast only specific channels by setting the indices in `prediction_channel_indices` parameter,
            pass the target data with all channels, as channel Filtering for both prediction and target will be
            manually applied before the loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, num_targets)`.
        return_loss (`bool`, *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        # 定义交叉熵损失函数
        loss = torch.nn.CrossEntropyLoss()

        # 确定是否使用预定义的返回字典，如果未定义则使用类属性中的默认设置
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # 将输入数据传递给模型进行前向推理
        model_output = self.model(
            past_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # x: [batch_size x nvars x num_patch x d_model]

        # 如果模型输出是元组，则转换为PatchTSMixerModelOutput对象
        if isinstance(model_output, tuple):
            model_output = PatchTSMixerModelOutput(*model_output)

        # 如果定义了inject_scale方法，则将其应用于模型输出的最后隐藏状态
        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state,
                loc=model_output.loc,
                scale=model_output.scale,
            )  # x: [batch_size x nvars x num_patch x d_model]

        # 通过模型头部获取预测结果
        y_hat = self.head(model_output.last_hidden_state)  # tensor [batch_size x n_labels]

        # 如果提供了目标值并且需要计算损失，则计算交叉熵损失
        if target_values is not None and return_loss is True:
            loss_val = loss(y_hat, target_values)
        else:
            loss_val = None

        # 如果不要求返回字典形式的结果，则返回一个元组
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

        # 否则，返回PatchTSMixerForTimeSeriesClassificationOutput对象
        return PatchTSMixerForTimeSeriesClassificationOutput(
            loss=loss_val,
            prediction_outputs=y_hat,  # tensor [batch_size x n_labels]
            last_hidden_state=model_output.last_hidden_state,  # x: [batch_size x nvars x num_patch x d_model]
            hidden_states=model_output.hidden_states,
        )
@dataclass
class PatchTSMixerForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutput`].

    Args:
        regression_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    loss: Optional[torch.FloatTensor] = None  # 可选的总损失，如果提供了 `y`，则返回
    regression_outputs: torch.FloatTensor = None  # 回归头部的预测输出，形状为 `(batch_size, num_targets)`
    last_hidden_state: torch.FloatTensor = None  # 通过头部之前的主干嵌入，形状为 `(batch_size, num_input_channels, num_patches, d_model)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型各层输出的隐藏状态，以及可选的初始嵌入输出的元组

class InjectScalerStatistics4D(nn.Module):
    def __init__(self, d_model: int, num_patches: int, expansion: int = 2):
        super().__init__()

        self.inverse_trans_expansion = nn.Linear(d_model + 2, expansion * d_model)
        # 反向转换扩展线性层，输入维度为 `d_model + 2`，输出维度为 `expansion * d_model`
        self.inverse_trans_compression = nn.Linear(expansion * d_model, d_model)
        # 反向转换压缩线性层，输入维度为 `expansion * d_model`，输出维度为 `d_model`
        self.map_scale_expansion = nn.Linear(2, 2 * expansion)
        # 映射尺度扩展线性层，输入维度为 `2`，输出维度为 `2 * expansion`
        self.map_scale_compression = nn.Linear(2 * expansion, 2)
        # 映射尺度压缩线性层，输入维度为 `2 * expansion`，输出维度为 `2`
        self.num_patches = num_patches
        # 存储传入的补丁数
    def forward(self, inputs: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`)
            loc (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
            scale (`torch.Tensor` of shape `(batch_size, 1, num_input_channels)`)
        Returns:
            `torch.Tensor` of shape `(batch_size, num_input_channels, num_patch, d_model)`
        """

        mean = loc.transpose(-1, -2)  # 将 loc 的最后两个维度交换位置，变为 `[batch_size x n_channels x 1]`
        mean = mean.unsqueeze(-2)  # 在倒数第二个位置添加一个维度，变为 `[batch_size x n_channels x 1 x 1]`
        mean = mean.repeat(1, 1, self.num_patches, 1)  # 沿着指定维度重复张量，扩展为 `[batch_size x n_channels x num_patch x 1]`

        stdev = scale.transpose(-1, -2)  # 将 scale 的最后两个维度交换位置，变为 `[batch_size x n_channels x 1]`
        stdev = stdev.unsqueeze(-2)  # 在倒数第二个位置添加一个维度，变为 `[batch_size x n_channels x 1 x 1]`
        stdev = stdev.repeat(1, 1, self.num_patches, 1)  # 沿着指定维度重复张量，扩展为 `[batch_size x n_channels x num_patch x 1]`

        concat_stats = torch.cat([mean, stdev], dim=-1)  # 沿着最后一个维度连接张量，得到 `[batch_size x n_channels x num_patch x 2]`

        concat_stats = self.map_scale_expansion(concat_stats)  # 使用模型的 `map_scale_expansion` 方法处理张量，输出 `[batch_size x n_channels x num_patch x (2*expansion)]`
        concat_stats = self.map_scale_compression(concat_stats)  # 使用模型的 `map_scale_compression` 方法处理张量，输出 `[batch_size x n_channels x num_patch x 2]`

        inputs = torch.cat([inputs, concat_stats], dim=-1)  # 沿着最后一个维度连接张量，得到 `[batch_size x channels x num_patch x d_model+2]`
        inputs = self.inverse_trans_expansion(inputs)  # 使用模型的 `inverse_trans_expansion` 方法处理张量，输出 `[batch_size x channels x num_patch x (expansion*d_model)]`
        inputs = self.inverse_trans_compression(inputs)  # 使用模型的 `inverse_trans_compression` 方法处理张量，输出 `[batch_size x channels x num_patch x d_model]`

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
        super().__init__(config)

        # 初始化 PatchTSMixerModel 模型
        self.model = PatchTSMixerModel(config)

        # 设置损失函数和输出分布
        self.loss = config.loss
        self.distribution_output = config.distribution_output

        # 是否返回字典形式的输出
        self.use_return_dict = config.use_return_dict
        # 并行采样的数量
        self.num_parallel_samples = config.num_parallel_samples

        # 根据损失函数选择相应的输出分布类别
        if config.loss == "mse":
            self.distribution_output = None
        else:
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

        # 根据 scaling 参数选择是否注入尺度统计信息
        if config.scaling in ["std", "mean", True]:
            self.inject_scale = InjectScalerStatistics4D(d_model=config.d_model, num_patches=config.num_patches)
        else:
            self.inject_scale = None

        # 初始化线性头部
        self.head = PatchTSMixerLinearHead(
            config=config,
            distribution_output=self.distribution_output,
        )

        # 如果需要，在初始化后执行后处理操作
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ):
        """
        实现模型的前向传播。

        Args:
            past_values (torch.Tensor):
                过去的值，作为模型的输入。
            target_values (torch.Tensor, optional):
                目标值，用于计算损失。默认为 None。
            output_hidden_states (bool, optional):
                是否输出隐藏状态。默认为 False。
            return_loss (bool):
                是否返回损失值。默认为 True。
            return_dict (bool, optional):
                是否返回字典形式的输出。默认为 None。

        Returns:
            根据 return_dict 参数决定的输出形式。
        """
        # 实现模型的具体逻辑，这里可以包含调用各个模块的过程
        ...

    def generate(
        self,
        past_values: torch.Tensor,
        ...
    ):
        # 生成方法的具体实现
        ...
    ) -> SamplePatchTSMixerRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the target values.

        Return:
            [`SamplePatchTSMixerRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_targets)`.
        """
        # 获取并行采样的数量
        num_parallel_samples = self.num_parallel_samples

        # 获得模型输出
        outputs = self(
            past_values=past_values,
            target_values=None,
            output_hidden_states=False,
        )

        # 获取输出分布
        distribution = self.distribution_output.distribution(outputs.regression_outputs)

        # 生成样本
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [batch_size x num_targets]
        # 堆叠张量
        # [batch_size x num_samples x num_targets]
        samples = torch.stack(samples, dim=1).view(-1, num_parallel_samples, self.config.num_targets)
        return SamplePatchTSMixerRegressionOutput(sequences=samples)
```