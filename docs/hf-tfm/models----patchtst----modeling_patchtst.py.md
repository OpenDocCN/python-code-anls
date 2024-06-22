# `.\transformers\models\patchtst\modeling_patchtst.py`

```py
# 设置文件编码为 utf-8
# 版权声明，声明文件版权，版权所有，IBM & Hugging Face 保留所有权利
# 根据 Apache 许可证版本 2.0 授权使用该文件
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非根据相关法律要求或书面同意，否则按原样提供的软件
# 分发在"原样"基础上提供，没有任何明示或暗示的担保或条件
# 请参阅许可证，了解特定语言的权限和限制
""" PyTorch PatchTST model."""

# 导入必要的模块和类
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

# 导入自定义模块
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "PatchTSTConfig"

# 预训练模型归档列表
PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtst-etth1-pretrain",
    # 查看全部 PatchTST 模型 https://huggingface.co/models?filter=patchtst
]

# 类定义：PatchTSTAttention，继承自 nn.Module
class PatchTSTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PatchTSTConfig] = None,
    ):
        super().__init__()
        # 初始化注意力机制的各项参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查参数是否合法
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

    # 辅助函数：重塑张量形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义一个前向传播函数，用于Transformer的一个层级
    def forward(
        self,
        # 输入的隐藏状态张量，通常是上一层的输出
        hidden_states: torch.Tensor,
        # 键值状态张量，可选参数，默认为None
        key_value_states: Optional[torch.Tensor] = None,
        # 过去的键值对状态元组，可选参数，默认为None
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 注意力掩码张量，可选参数，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 层级头部掩码张量，可选参数，默认为None
        layer_head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重张量的标志，默认为False
        output_attentions: bool = False,
class PatchTSTBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 初始化批归一化层，对输入的维度为config.d_model的维度进行归一化，设置eps为config.norm_eps
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        # 调整输入张量的维度，将序列长度维度与特征维度维度互换，即(output: (batch_size, d_model, sequence_length))
        output = inputs.transpose(1, 2)  
        # 对调整维度后的张量进行批归一化操作
        output = self.batchnorm(output)
        # 再次调整张量维度，恢复原始顺序
        return output.transpose(1, 2)


def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value: int = 0,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Masking ratio applied to mask the input data during random pretraining. It is the number between 0 and 1.
        unmasked_channel_indices (list, *optional*):
            Indices of channels that will not be masked.
        channel_consistent_masking (bool, *optional*, defaults to `False`):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels.
        mask_value (int, *optional*, defaults to 0):
            Define the value of masked patches for pretraining.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    # 如果mask_ratio小于0或大于等于1，则抛出值错误
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    # 获取输入张量的形状信息
    batch_size, num_channels, sequence_length, num_features = inputs.shape
    # 获取输入张量所在设备信息
    device = inputs.device

    # 计算不被遮盖的片段长度
    len_keep = int(sequence_length * (1 - mask_ratio))

    # 如果channel_consistent_masking为True，表示在所有通道上的遮盖行为相同
    if channel_consistent_masking:
        # 生成随机噪声，形状为[batch_size, 1, sequence_length]，取值范围为[0, 1]
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  
        # 将噪声在通道维度上复制，形状变为[batch_size, num_channels, sequence_length]
        noise = noise.repeat(1, num_channels, 1)  
    else:
        # 在每个通道上生成随机噪声，形状为[batch_size, num_channels, sequence_length]，取值范围为[0, 1]
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # 初始化遮盖张量，形状为[batch_size, num_channels, sequence_length]，初始值全为1
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    # 根据len_keep的值，将mask张量对应位置置为0，即遮盖掉一部分输入
    mask[:, :, :len_keep] = 0

    # 对每个样本的噪声进行排序，得到对应的索引，形状为[batch_size, num_channels, sequence_length]
    ids_shuffle = torch.argsort(noise, dim=-1)  # 升序排序，小的表示保留，大的表示移除
    # 恢复噪声排序前的顺序，得到恢复索引，形状同上
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [batch_size, num_channels, sequence_length]
    # 使用torch.gather函数，按索引ids_restore在最后一个维度上聚合mask张量
    mask = torch.gather(mask, dim=-1, index=ids_restore)
    # 在最后一个维度上增加一个维度，并重复num_features次，生成一个与mask形状相同的张量
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    # 如果提供了未屏蔽的通道索引，则将这些通道对应位置的mask设为0
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    # 使用inputs.masked_fill函数，将inputs张量中mask为True的位置用mask_value填充
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    # 返回经过mask处理后的inputs_mask张量以及mask张量的第一个维度的切片
    return inputs_mask, mask[..., 0]
# 定义 forecast_masking 函数，用于对时间序列数据进行预测蒙版处理
def forecast_masking(
    inputs: torch.Tensor,  # 输入时间序列数据，shape 为 (bs, num_channels, num_patch, patch_len)
    num_forecast_mask_patches: Union[list, int],  # 预测蒙版的 patch 数量，可以是整数或整数列表
    unmasked_channel_indices: list = None,  # 未被蒙版的通道索引列表，默认为 None
    mask_value: int = 0,  # 蒙版值，默认为 0
):
    """Forecast masking that masks the last K patches where K is from the num_forecast_mask_patches.
    If num_forecast_mask_patches is a list, samples in the batch will be randomly masked by numbers defined in the list.

    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_len)`
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

    # 如果 num_forecast_mask_patches 是整数，则转换为列表
    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]

    # 获取输入数据形状信息
    batch_size, num_channels, sequence_length, num_features = inputs.shape
    # 创建一个全零的蒙版
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    # 遍历 patch_length 和 ratio 并计算蒙版数量
    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(
                f"num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches."
            )
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    # 对 temp_len 进行排序
    t_list = sorted(t_list, key=lambda x: x[2])

    # 调整蒙版数量，使其与批处理大小一致
    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    batch1 = 0
    # 对不同 patch 长度的蒙版进行填充
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    # 对蒙版进行随机排列
    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    # 将蒙版维度扩展，与输入数据相匹配，若存在未被蒙版的通道则处理
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patch x patch_len]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    # 使用蒙版值填充被蒙版的输入数据
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


# 定义 PatchTSTPatchify 类，用于将时间序列数据分割成不同的 patch
class PatchTSTPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    def __init__(self, config: PatchTSTConfig):
        # 调用父类构造函数初始化对象
        super().__init__()

        # 设置对象属性：序列长度等于配置文件中的上下文长度
        self.sequence_length = config.context_length
        # 设置对象属性：补丁长度等于配置文件中的补丁长度
        self.patch_length = config.patch_length
        # 设置对象属性：补丁步长等于配置文件中的补丁步长
        self.patch_stride = config.patch_stride

        # 如果序列长度小于等于补丁长度，抛出 ValueError 异常
        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # 计算补丁数量
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        # 计算新的序列长度
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        # 计算序列开始位置
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        # 获取输入张量的序列长度
        sequence_length = past_values.shape[-2]
        # 如果输入序列长度与模型配置的序列长度不匹配，抛出 ValueError 异常
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # 输出: [bs x new_sequence_length x num_channels]
        # 从输入张量中切片获取输出
        output = past_values[:, self.sequence_start :, :]
        # 输出: [bs x num_patches x num_input_channels x patch_length]
        # 在第二维度上对输出进行展开，切分为长度为补丁长度的子张量
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # 输出: [bs x num_input_channels x num_patches x patch_length]
        # 转置输出张量的倒数第二维和倒数第三维，并确保连续性
        output = output.transpose(-2, -3).contiguous()
        # 返回输出张量
        return output
class PatchTSTMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        config (`PatchTSTConfig`): model config
    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
            Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points
    """

    def __init__(self, config: PatchTSTConfig):
        # 初始化函数，接受配置参数，并设置相关属性
        super().__init__()
        self.random_mask_ratio = config.random_mask_ratio
        self.channel_consistent_masking = config.channel_consistent_masking
        self.mask_type = config.mask_type
        self.num_forecast_mask_patches = config.num_forecast_mask_patches
        self.unmasked_channel_indices = config.unmasked_channel_indices
        self.mask_value = config.mask_value
        if self.unmasked_channel_indices is not None:
            # 如果存在未遮罩的通道索引，则对其进行排序
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
        if self.mask_type == "random":
            # 如果使用随机遮罩，调用random_masking函数进行遮罩操作
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
            )
        elif self.mask_type == "forecast":
            # 如果使用预测遮罩，调用forecast_masking函数进行遮罩操作
            masked_input, mask = forecast_masking(
                inputs=patch_input,
                num_forecast_mask_patches=self.num_forecast_mask_patches,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value,
            )
        else:
            # 抛出异常，提示无效的遮罩类型
            raise ValueError(f"Invalid mask type {self.mask_type}.")

        # mask: [bs x num_input_channels x num_patch]
        # 将mask转换成布尔类型
        mask = mask.bool()
        return masked_input, mask


class PatchTSTEncoderLayer(nn.Module):
    """
    PatchTST encoder layer
    """
    # 初始化函数，接受一个 PatchTSTConfig 类型的配置对象作为参数
    def __init__(self, config: PatchTSTConfig):
        # 调用父类的初始化函数
        super().__init__()

        # 是否使用通道注意力机制的标志
        self.channel_attention = config.channel_attention
        # 多头注意力机制
        self.self_attn = PatchTSTAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )

        # 子层 1 的 Add & Norm 操作
        # 如果路径 dropout 大于 0，则使用 Dropout，否则使用恒等映射
        self.dropout_path1 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        # 根据配置选择规范化层的类型
        if config.norm_type == "batchnorm":
            self.norm_sublayer1 = PatchTSTBatchNorm(config)
        elif config.norm_type == "layernorm":
            self.norm_sublayer1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # 子层 2 的 Add & Norm 操作，仅当使用通道注意力时才执行
        if self.channel_attention:
            # 如果路径 dropout 大于 0，则使用 Dropout，否则使用恒等映射
            self.dropout_path2 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
            # 根据配置选择规范化层的类型
            if config.norm_type == "batchnorm":
                self.norm_sublayer2 = PatchTSTBatchNorm(config)
            elif config.norm_type == "layernorm":
                self.norm_sublayer2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # 位置编码前馈网络
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),  # 使用激活函数
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),  # 如果前馈 dropout 大于 0，则使用 Dropout，否则使用恒等映射
            nn.Linear(config.ffn_dim, config.d_model, bias=config.bias),
        )

        # 子层 3 的 Add & Norm 操作
        # 如果路径 dropout 大于 0，则使用 Dropout，否则使用恒等映射
        self.dropout_path3 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        # 根据配置选择规范化层的类型
        if config.norm_type == "batchnorm":
            self.norm_sublayer3 = PatchTSTBatchNorm(config)
        elif config.norm_type == "layernorm":
            self.norm_sublayer3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # 是否在规范化前应用规范化操作的标志
        self.pre_norm = config.pre_norm
class PatchTSTPreTrainedModel(PreTrainedModel):
    # 设置配置类为PatchTSTConfig
    config_class = PatchTSTConfig
    # 设置基础模型前缀为"model"
    base_model_prefix = "model"
    # 设置主要输入名称为"past_values"
    main_input_name = "past_values"
    # 设置是否支持梯度检查点为False
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """
        初始化权重
        """
        if isinstance(module, PatchTSTPositionalEncoding):
            # 初始化cls_token
            if self.config.use_cls_token:
                nn.init.normal_(module.cls_token, std=0.02)
            # 初始化位置编码
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重项初始化为1.0
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSTBatchNorm):
            # 将batchnorm的偏置项初始化为零
            module.batchnorm.bias.data.zero_()
            # 将batchnorm的权重项初始化为1.0
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # 将权重项初始化为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                # 如果存在偏置项，将其初始化为零
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSTEncoder)):
            # 设置梯度检查点
            module.gradient_checkpointing = value


class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 设置输入通道数为配置中的num_input_channels
        self.num_input_channels = config.num_input_channels
        # 设置是否共享嵌入层
        self.share_embedding = config.share_embedding
        # 输入编码：将特征向量投影到d维向量空间
        if self.share_embedding:
            # 如果共享嵌入层，则初始化为线性层
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)
        else:
            # 否则，初始化为模块列表
            self.input_embedding = nn.ModuleList()
            for _ in range(config.num_input_channels):
                # 为每个输入通道添加线性层
                self.input_embedding.append(nn.Linear(config.patch_length, config.d_model))
    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        # Input encoding
        # 获取输入张量中的通道数
        num_input_channels = patch_input.shape[1]
        # 如果输入通道数与配置中定义的不同，则引发 ValueError
        if num_input_channels != self.num_input_channels:
            raise ValueError(
                f"The defined number of input channels ({self.num_input_channels}) in the config "
                f"has to be the same as the number of channels in the batch input ({num_input_channels})"
            )
        # 如果共享嵌入，则使用单一嵌入层对所有通道进行嵌入
        if self.share_embedding:
            # 对输入张量进行嵌入
            embeddings = self.input_embedding(patch_input)  # x: [bs x num_channels  x num_patches x d_model]
        else:
            # 如果不共享嵌入，则对每个通道单独进行嵌入
            # 将每个通道的输入张量分别传递给相应的嵌入层，并存储嵌入结果
            embeddings = [self.input_embedding[i](patch_input[:, i, :, :]) for i in range(num_input_channels)]
            # 将每个通道的嵌入结果沿着通道维度进行堆叠，以形成最终的嵌入张量
            embeddings = torch.stack(embeddings, dim=1)
        # 返回嵌入结果张量
        return embeddings
# 用于位置编码的类
class PatchTSTPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__()
        # 是否使用 cls_token
        self.use_cls_token = config.use_cls_token
        self.num_input_channels = config.num_input_channels
        if config.use_cls_token:
            # 如果使用 cls_token，创建一个可学习的 cls_token 参数，形状为 [1 x num_input_channels x 1 x d_model]
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
            # 加上 cls_token 之后，patch 数量加 1
            num_patches += 1
        # 初始化位置编码，形状为 [num_patches x d_model]
        self.position_enc = self._init_pe(config, num_patches)
        # 添加位置编码的dropout
        self.positional_dropout = (
            nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()
        )

    @staticmethod
    def _init_pe(config: PatchTSTConfig, num_patches: int) -> nn.Parameter:
        # 初始化位置编码
        if config.positional_encoding_type == "random":
            # 使用随机初始化的位置编码
            position_enc = nn.Parameter(torch.randn(num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            # 使用正弦余弦的位置编码
            position_enc = torch.zeros(num_patches, config.d_model)
            position = torch.arange(0, num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        if self.use_cls_token:
            # patch_input: [bs x num_channels x num_patches x d_model]
            patch_input = self.positional_dropout(patch_input + self.position_enc[1:, :])
            # 添加 cls_token，形状为 [1 x num_channels x 1 x d_model]
            cls_token = self.cls_token + self.position_enc[:1, :]
            # 复制 cls_token 到 batch 中所有样本, 形状为 [bs x num_channels x 1 x d_model]
            cls_tokens = cls_token.expand(patch_input.shape[0], self.num_input_channels, -1, -1)
            # 将 cls_token 和 patch_input 拼接, 形状为 [bs x num_channels x (num_patches+1) x d_model]
            hidden_state = torch.cat((cls_tokens, patch_input), dim=2)
        else:
            # hidden_state: [bs x num_channels x num_patches x d_model]
            hidden_state = self.positional_dropout(patch_input + self.position_enc)
        return hidden_state


# PatchTST 编码器
class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """
    # 该类是 PatchTST (Patch Time Series Transformer) 模型的实现
    # 它继承自 PyTorch 的 nn.Module，是一个基于时间序列的 Transformer 模型
    def __init__(self, config: PatchTSTConfig, num_patches: int):
        # 调用父类的初始化方法
        super().__init__(config)
        # 禁用梯度检查点
        self.gradient_checkpointing = False
    
        # 输入嵌入层: 将特征向量投影到 d 维向量空间
        self.embedder = PatchTSTEmbedding(config)
        # 位置编码层
        self.positional_encoder = PatchTSTPositionalEncoding(config, num_patches)
        # 编码器层列表
        self.layers = nn.ModuleList([PatchTSTEncoderLayer(config) for i in range(config.num_hidden_layers)])
    
        # 初始化权重并执行最终处理
        self.post_init()
    
    # 前向传播方法
    def forward(
        self,
        patch_input: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BaseModelOutput:
        """
        参数:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                时间序列的过去值
            output_hidden_states (bool, optional): 是否输出隐藏状态
            output_attentions (bool, optional): 是否输出注意力权重
    
        返回:
            `BaseModelOutput`
        """
        # 设置是否输出隐藏状态和注意力权重的标志
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
    
        # 输入嵌入
        patch_input = self.embedder(patch_input)
        # 位置编码
        hidden_state = self.positional_encoder(patch_input)
    
        # 初始化编码器状态和注意力权重
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
    
        # 遍历编码器层
        for encoder_layer in self.layers:
            # 如果需要输出隐藏状态, 则将当前隐藏状态添加到 encoder_states 中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_state,)
    
            # 执行当前编码器层的前向传播, 获得隐藏状态和注意力权重
            layer_outputs = encoder_layer(hidden_state=hidden_state, output_attentions=output_attentions)
            hidden_state = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
    
        # 返回最终的隐藏状态, 所有隐藏状态, 和所有注意力权重
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=encoder_states, attentions=all_attentions)
PATCHTST_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# PatchTST_START_DOCSTRING 定义了包含此模型文档字符串的原始文本。
# 这个模型继承自 PreTrainModel，检查超类文档以获取库实现的通用方法的信息，例如下载或保存模型，修改输入嵌入大小，修剪头等等。
# 这个模型也是 PyTorch 的 torch.nn.Module 子类。将其视为常规的 PyTorch 模块，并参考 PyTorch 文档以了解与一般使用和行为有关的所有内容。
# 参数部分包括 config 参数，是模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只有配置。查看 PreTrainedModel.from_pretrained 方法以加载模型权重。

@dataclass
class PatchTSTModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Parameters:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`, *optional*)
            Bool masked tensor indicating which patches are masked
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input to the Transformer
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None
    patch_input: torch.FloatTensor = None
# PatchTSTModelOutput 是模型输出的基类，具有潜在的隐藏状态。包括参数说明和每个参数的数据类型、形状和可选性。

@dataclass
class PatchTSTForPretrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPretraining`].
"""
# PatchTSTForPretrainingOutput 是 PatchTSTForPretraining 的输出类型。
        Parameters:
            loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
                MSE loss.
            prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
                Prediction outputs of the time series modeling heads.
            hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
                Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
                shape `(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
                Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
                sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.
        """

        # 定义可选的损失值，当提供`labels`时返回，类型为`torch.FloatTensor`，形状为`(1,)`
        loss: Optional[torch.FloatTensor] = None
        # 定义预测输出，类型为`torch.FloatTensor`，形状为`(batch_size, sequence_length, config.vocab_size)`
        prediction_output: torch.FloatTensor = None
        # 定义可选的隐藏状态元组，当`output_hidden_states=True`传递或配置为`config.output_hidden_states=True`时返回
        # 包含`torch.FloatTensor`类型的元组（一个用于嵌入的输出+一个用于每个层的输出）形状为`(batch_size, sequence_length, hidden_size)`
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        # 定义可选的注意力元组，当`output_attentions=True`传递或配置为`config.output_attentions=True`时返回
        # 包含`torch.FloatTensor`类型的元组（每个层一个）形状为`(batch_size, num_heads, sequence_length, sequence_length)`
        attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储 `PatchTSTForRegression` 模型的输出
@dataclass
class PatchTSTForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForRegression`].

    Parameters:
        # 如果提供了 `labels`，则返回 MSE 损失
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        # 时间序列建模头的回归输出
        regression_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Regression outputs of the time series modeling heads.
        # 如果传递了 `output_hidden_states=True` 或 `config.output_hidden_states=True`，则返回隐藏状态
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        # 如果传递了 `output_attentions=True` 或 `config.output_attentions=True`，则返回注意力权重
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # MSE 损失，可选项
    loss: Optional[torch.FloatTensor] = None
    # 时间序列建模头的回归输出
    regression_outputs: torch.FloatTensor = None
    # 隐藏状态，可选项
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，可选项
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储 `PatchTSTForPrediction` 模型的输出
@dataclass
class PatchTSTForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPrediction`].
    """
```  
    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            # 参数：损失（*可选*，在提供`labels`时返回），表示均方误差损失
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, -1)`):
            # 预测输出：时间序列建模头的预测输出，形状为`(batch_size, prediction_length, -1)`
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 隐藏状态：模型在每一层输出的隐藏状态组成的元组，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            # 元组包含了嵌入层的输出以及每一层的输出，形状为`(batch_size, sequence_length, hidden_size)`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 注意力值：注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值，并在传递`output_attentions=True`或`config.output_attentions=True`时返回
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            # 注意力权重值，每一层的注意力权重，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            # loc：输入数据（batch_size, sequence_length, num_channels）在sequence_length上的均值，形状为`(batch_size, 1, num_channels)`
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            # scale：输入数据（batch_size, sequence_length, num_channels）在sequence_length上的标准差，形状为`(batch_size, 1, num_channels)`
    """

    loss: Optional[torch.FloatTensor] = None
    # 损失：可选的torch.FloatTensor类型，初始值为None
    prediction_outputs: torch.FloatTensor = None
    # 预测输出：torch.FloatTensor类型，初始值为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态：可选的元组类型，其中每个元素是torch.FloatTensor类型，初始值为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力值：可选的元组类型，其中每个元素是torch.FloatTensor类型，初始值为None
    loc: torch.FloatTensor = None
    # loc：torch.FloatTensor类型，初始值为None
    scale: torch.FloatTensor = None
    # scale：torch.FloatTensor类型，初始值为None
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import distributions
from transformers.modeling_outputs import ModelOutput

@dataclass
class PatchTSTForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSTForClassification`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Prediction scores of the PatchTST modeling head (scores before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SamplePatchTSTOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences `(batch_size, num_samples, prediction_length, num_targets)`):
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.
    """
```py  
    # 定义平均函数，接受输入张量、权重张量以及计算维度作为参数
    def weighted_average(input_tensor, weights=None, dim=None):
        # 如果提供了权重张量
        if weights is not None:
            # 根据权重值计算加权张量，权重为 0 的位置用 0 填充
            weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
            # 计算非零权重的和，并确保至少为 1.0
            sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
            # 返回加权张量沿指定维度求和后除以权重和的结果
            return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
        # 如果未提供权重张量
        else:
            # 直接沿指定维度计算输入张量的平均值
            return input_tensor.mean(dim=dim)
# 对输入数据进行标准化处理的模块
class PatchTSTStdScaler(nn.Module):
    """
    标准化输入特征的类。通过计算均值和标准差来进行标准化操作。
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 指定标准化维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保留原有维度，默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小标准差阈值，防止除零
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            data (`torch.Tensor` 形状为 `(batch_size, sequence_length, num_input_channels)`):
                输入数据，用于计算标准化参数
            observed_indicator (`torch.BoolTensor` 形状为 `(batch_size, sequence_length, num_input_channels)`):
                用于指示哪些数据是有效的，以计算正确的标准化参数
        返回:
            tuple of `torch.Tensor` 形状为
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
                分别为: 标准化后的数据、计算得到的均值、计算得到的标准差
        """
        # 根据observed_indicator计算各通道的有效数据个数
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 确保分母不为0
        denominator = denominator.clamp_min(1.0)
        # 计算各通道的均值
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算各通道的方差
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算各通道的标准差
        scale = torch.sqrt(variance + self.minimum_scale)
        # 返回标准化后的数据、均值、标准差
        return (data - loc) / scale, loc, scale


# 对输入数据进行均值归一化处理的模块
class PatchTSTMeanScaler(nn.Module):
    """
    计算各通道的加权平均绝对值作为缩放因子，并相应地缩放数据。
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 指定缩放维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保留原有维度，默认为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小缩放因子阈值，防止除零
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 默认缩放因子，可选配置
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            data (`torch.Tensor` 形状为 `(batch_size, sequence_length, num_input_channels)`):
                输入数据，用于计算缩放因子
            observed_indicator (`torch.BoolTensor` 形状为 `(batch_size, sequence_length, num_input_channels)`):
                用于指示哪些数据是有效的，以计算正确的缩放因子
        返回:
            tuple of `torch.Tensor` 形状为
                (`(batch_size, sequence_length, num_input_channels)`, `(batch_size, 1, num_input_channels)`)
                分别为: 缩放后的数据、计算得到的缩放因子
        """
        # 计算各通道的有效数据绝对值的加权平均值作为缩放因子
        scale = torch.abs(data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / (
            observed_indicator.sum(self.dim, keepdim=self.keepdim).clamp_min(1.0)
        )
        # 确保缩放因子不小于最小阈值
        scale = scale.clamp_min(self.minimum_scale)
        # 如果配置了默认缩放因子，则使用默认值
        if self.default_scale is not None:
            scale = scale.fill_(self.default_scale)
        # 根据计算得到的缩放因子对输入数据进行缩放
        return data / scale, scale
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                输入数据，形状为(batch_size, sequence_length, num_input_channels)
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                观察指示器，用于计算标准差的观察标记。
        Returns:
            返回值为元组，包含三个 `torch.Tensor`，形状分别为
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算数据乘以观察指示器的绝对值之和
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        # 计算观察指示器在指定维度上的总数
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # 计算标准差
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # 如果提供了`default_scale`，则使用它，否则使用批次的标准差
        if self.default_scale is None:
            # 计算批次标准差的总和
            batch_sum = ts_sum.sum(dim=0)
            # 计算批次观察总数
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            # 计算默认标准差
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            # 使用指定的默认标准差
            default_scale = self.default_scale * torch.ones_like(scale)

        # 在没有观察到数据的地方应用默认标准差
        scale = torch.where(num_observed > 0, scale, default_scale)

        # 确保标准差至少为`self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        # 数据除以标准差
        scaled_data = data / scale

        # 如果不保持维度，压缩标准差维度
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale
# 定义名为 PatchTSTNOPScaler 的类，继承自 nn.Module
# 该类用于给输入数据的第一维度分配一个等于1的缩放因子，因此对输入数据不进行缩放
class PatchTSTNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    # 初始化方法，接受一个PatchTSTConfig对象作为参数
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 如果配置对象有scaling_dim属性，则将其赋值给dim，否则设置为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 如果配置对象有keepdim属性，则将其赋值给keepdim，否则设置为True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    # 前向传播方法，接受torch.Tensor类型的数据和观测指示器作为参数，返回三个torch.Tensor类型的元组
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
        # 计算data的均值并设置为scale
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 设置loc为data的零均值
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回data, loc, scale的元组
        return data, loc, scale


# 定义名为 PatchTSTScaler 的类，继承自 nn.Module
class PatchTSTScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 根据配置对象的scaling属性选择不同的Scaler
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSTMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = PatchTSTStdScaler(config)
        else:
            self.scaler = PatchTSTNOPScaler(config)

    # 前向传播方法，接受torch.Tensor类型的数据和观测指示器作为参数，返回三个torch.Tensor类型的元组
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Input for scaler calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, um_input_channels)`)
        """
        # 调用相应的Scaler的前向传播方法
        data, loc, scale = self.scaler(data, observed_indicator)
        # 返回数据、均值和缩放因子的元组
        return data, loc, scale


# 以 PatchTSTPreTrainedModel 为基类，定义名为 PatchTSTModel 的类
# 该类用于输出原始的隐藏状态，不附带特定的头部
@add_start_docstrings(
    "The bare PatchTST Model outputting raw hidden-states without any specific head.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTModel(PatchTSTPreTrainedModel):
    # 初始化方法，接受一个 PatchTSTConfig 类型的参数，并调用父类的初始化方法
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # 初始化 PatchTSTScaler 对象
        self.scaler = PatchTSTScaler(config)
        # 初始化 PatchTSTPatchify 对象
        self.patchifier = PatchTSTPatchify(config)
        # 从配置中获取是否需要对输入进行掩码处理的信息
        self.do_mask_input = config.do_mask_input
        # 从 PatchTSTPatchify 对象中获取 num_patches 信息
        num_patches = self.patchifier.num_patches

        # 根据是否需要进行输入掩码处理，初始化 Masking 对象
        if self.do_mask_input:
            self.masking = PatchTSTMasking(config)
        else:
            # 如果不需要进行输入掩码处理，则使用 nn.Identity()，即不进行任何处理
            self.masking = nn.Identity()
        # 初始化 PatchTSTEncoder 对象，并传入配置和 num_patches 参数
        self.encoder = PatchTSTEncoder(config, num_patches=num_patches)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 用于掩码预训练的头部
class PatchTSTMaskPretrainHead(nn.Module):
    """
    预训练头部用于掩码建模
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 使用 dropout 层
        self.dropout = nn.Dropout(config.dropout)
        # 使用线性层将模型输出映射到 patch 长度
        self.linear = nn.Linear(config.d_model, config.patch_length)
        # 是否使用 cls token
        self.use_cls_token = config.use_cls_token

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        参数:
            embedding (torch.Tensor, 形状为 (bs, num_channels, num_patches, d_model) 或者
                    (bs, num_channels, num_patches+1, d_model), 如果设置 cls_token 为 True):
                来自模型的嵌入
        返回:
            torch.Tensor, 形状为 (bs, num_channels, num_patches, d_model) 或者
                        (bs, num_channels, num_patches+1, d_model), 如果 cls_token 设置为 True
        """
        # 对输入应用 dropout 后通过线性层
        embedding = self.linear(self.dropout(embedding))
        # 如果使用 cls token, 则移除第一个 cls token
        if self.use_cls_token:
            embedding = embedding[:, :, 1:, :]
        return embedding


# 用于预训练的 PatchTST 模型
@add_start_docstrings(
    "The PatchTST for pretrain model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # 设置掩码输入为 True
        config.do_mask_input = True
        # 创建 PatchTSTModel 实例
        self.model = PatchTSTModel(config=config)
        # 创建掩码预训练头部
        self.head = PatchTSTMaskPretrainHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
# 用于分类任务的头部
class PatchTSTClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 是否使用 cls token
        self.use_cls_token = config.use_cls_token
        # 池化类型
        self.pooling_type = config.pooling_type
        # 使用 Flatten 层
        self.flatten = nn.Flatten(start_dim=1)
        # 使用 Dropout 层
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        # 使用线性层进行分类
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_targets)
    # 定义前向传播函数，接受嵌入向量作为输入
    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_targets)`

        """
        # 如果使用了 cls_token
        if self.use_cls_token:
            # 使用第一个输出的 token，pooled_embedding: bs x num_channels x d_model
            pooled_embedding = embedding[:, :, 0, :]
        # 如果使用平均池化
        elif self.pooling_type == "mean":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        # 如果使用最大池化
        elif self.pooling_type == "max":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2)
        # 如果都不是，则抛出数值错误
        else:
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        # 将池化后的嵌入向量展开成一维向量
        pooled_embedding = self.flatten(pooled_embedding)
        # output: bs x n_classes
        output = self.linear(self.dropout(pooled_embedding))
        # 返回输出
        return output
# 添加类和函数的文档字符串
@add_start_docstrings(
    "The PatchTST for classification model.",  # 类 PatchTSTForClassification 的描述性文档字符串
    PATCHTST_START_DOCSTRING,  # 附加的文档字符串常量
)
# 定义用于分类的 PatchTST 类，继承自 PatchTSTPreTrainedModel
class PatchTSTForClassification(PatchTSTPreTrainedModel):
    # 初始化方法，接受配置参数
    def __init__(self, config: PatchTSTConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中启用了输入遮罩，则警告并将其关闭
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")  # 记录一个警告信息
            config.do_mask_input = False  # 将配置中的输入遮罩设置为 False

        self.model = PatchTSTModel(config)  # 创建 PatchTST 模型实例
        self.head = PatchTSTClassificationHead(config)  # 创建用于分类的头部模型实例

        # 初始化权重并应用最终处理
        self.post_init()  # 初始化模型的权重和执行最终的处理

    # 前向方法，接受不同参数并处理
    def forward(
        self,
        past_values: torch.Tensor,  # 过去的值，作为输入张量
        target_values: torch.Tensor = None,  # 目标值，默认为 None
        past_observed_mask: Optional[bool] = None,  # 过去的观察遮罩，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
    ):
        pass  # 占位符，用于示例的前向方法


# 添加类和函数的文档字符串
@add_start_docstrings(
    "The PatchTST for regression Model.",  # 类 PatchTSTPredictionHead 的描述性文档字符串
    PATCHTST_START_DOCSTRING,  # 附加的文档字符串常量
)
# 定义用于回归的 PatchTST 预测头部类，继承自 nn.Module
class PatchTSTPredictionHead(nn.Module):
    # 初始化方法，接受配置、补丁数量和可选的分布输出参数
    def __init__(self, config: PatchTSTConfig, num_patches, distribution_output=None):
        # 调用父类的初始化方法
        super().__init__()

        self.share_projection = config.share_projection  # 是否共享投影
        self.num_input_channels = config.num_input_channels  # 输入通道的数量
        self.use_cls_token = config.use_cls_token  # 是否使用分类标记
        self.pooling_type = config.pooling_type  # 池化类型
        if self.pooling_type or self.use_cls_token:  # 根据池化类型或分类标记确定头部维度
            head_dim = config.d_model  # 使用配置中的模型维度
        else:
            head_dim = config.d_model * num_patches  # 使用配置中的模型维度乘以补丁数量

        if not self.share_projection:  # 如果不共享投影
            # 定义多通道的独立投影和其他组件
            self.projections = nn.ModuleList()  # 投影模块列表
            self.dropouts = nn.ModuleList()  # dropout 模块列表
            self.flattens = nn.ModuleList()  # flatten 模块列表
            # 为每个输入通道创建相应的模块
            for i in range(self.num_input_channels):
                self.flattens.append(nn.Flatten(start_dim=2))  # 添加从第二维开始的 flatten 模块
                if distribution_output is None:  # 如果没有分布输出
                    # 使用线性头
                    self.projections.append(nn.Linear(head_dim, config.prediction_length))  # 线性投影
                else:
                    # 使用分布头
                    self.projections.append(distribution_output.get_parameter_projection(head_dim))  # 分布输出的投影
                # 添加 dropout 或 Identity 模块
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:  # 如果共享投影
            self.flatten = nn.Flatten(start_dim=2)  # 添加 flatten 模块
            if distribution_output is None:  # 如果没有分布输出
                # 使用线性头
                self.projection = nn.Linear(head_dim, config.prediction_length)  # 线性投影
            else:
                # 使用分布头
                self.projection = distribution_output.get_parameter_projection(head_dim)  # 分布输出的投影
            # 添加 dropout 或 Identity 模块
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

        """
        # 如果使用了CLS Token
        if self.use_cls_token:
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        else:
            # 如果使用的是均值池化
            if self.pooling_type == "mean":
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.mean(dim=2)
            # 如果使用的是最大值池化
            elif self.pooling_type == "max":
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.max(dim=2)
            else:
                # pooled_embedding: [bs x num_channels x num_patches x d_model]
                pooled_embedding = embedding

        # 如果不共享投影层
        if not self.share_projection:
            output = []
            # 遍历输入通道数量
            for i in range(self.num_input_channels):
                # pooled_embedding: [bs x (d_model * num_patches)] or [bs x d_model)]
                pooled_embedding = self.flattens[i](pooled_embedding[:, i, :])
                pooled_embedding = self.dropouts[i](pooled_embedding)
                # pooled_embedding: [bs x forecast_len]
                #  or tuple ([bs x forecast_len], [bs x forecast_len]) if using distribution head
                pooled_embedding = self.projections[i](pooled_embedding)
                output.append(pooled_embedding)
            # output: [bs x num_channels x forecast_len]
            output = torch.stack(output, dim=1)
        else:
            # pooled_embedding: [bs x num_channels x (d_model * num_patches)] or [bs x num_channels x d_model)]
            pooled_embedding = self.flatten(pooled_embedding)
            pooled_embedding = self.dropout(pooled_embedding)
            # output: [bs x num_channels x forecast_len] or
            # tuple ([bs x num_channels x forecast_len], [bs x num_channels x forecast_len]) if using distribution head
            output = self.projection(pooled_embedding)

        # 如果输出是一个元组
        if isinstance(output, tuple):
            # output: ([bs x forecast_len x num_channels], [bs x forecast_len x num_channels])
            output = tuple(z.transpose(2, 1) for z in output)
        else:
            output = output.transpose(2, 1)  # [bs x forecast_len x num_channels]
        return output
# 添加文档字符串描述 PatchTST 用于预测模型
# 继承自 PatchTSTPreTrainedModel
@add_start_docstrings(
    "The PatchTST for prediction model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    # 初始化方法，接收配置对象作为参数
    def __init__(self, config: PatchTSTConfig):
        # 调用父类初始化方法
        super().__init__(config)

        # 关闭掩码
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        # 创建 PatchTSTModel 模型
        self.model = PatchTSTModel(config)

        # 根据配置选择损失函数
        if config.loss == "mse":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.prediction_length)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.prediction_length)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 创建 PatchTSTPredictionHead 头部
        self.head = PatchTSTPredictionHead(
            config, self.model.patchifier.num_patches, distribution_output=self.distribution_output
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        # 定义函数，返回值类型为 SamplePatchTSTOutput
        def forward(
            ) -> SamplePatchTSTOutput:
            """
            生成具有概率分布头的模型的样本预测序列。

            参数:
                past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                    时间序列的过去值，作为上下文用于预测未来。
                past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                    布尔蒙版，指示哪些`past_values`是已观察的，哪些是缺失的。蒙版值选择在 `[0, 1]` 范围内:

                    - 1 表示 **已观察** 的值，
                    - 0 表示 **缺失** 的值（即已被零替换的 NaN）。

            返回:
                [`SamplePatchTSTOutput`]，其中输出 `sequences` 张量的形状为 `(batch_size, number of samples, prediction_length, 1)` 或 `(batch_size, number of samples, prediction_length, num_input_channels)`
                用于多变量预测。
            """
            # 获取样本数量
            num_parallel_samples = self.config.num_parallel_samples

            # 获取模型输出
            outputs = self(
                past_values=past_values,
                future_values=None,
                past_observed_mask=past_observed_mask,
                output_hidden_states=False,
            )
            if self.distribution_output:
                # 获取分布
                distribution = self.distribution_output.distribution(
                    outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
                )
                # 获取样本: 列表中包含 `[bs x forecast_len x num_channels]`
                samples = [distribution.sample() for _ in range(num_parallel_samples)]
                # samples: [bs x num_samples x forecast_len x num_channels]
                samples = torch.stack(samples, dim=1)
            else:
                samples = outputs.prediction_outputs.unsqueeze(1)

            return SamplePatchTSTOutput(sequences=samples)
class PatchTSTRegressionHead(nn.Module):
    """
    Regression head
    """

    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()
        self.y_range = config.output_range  # 设置输出范围
        self.use_cls_token = config.use_cls_token  # 是否使用CLS token
        self.pooling_type = config.pooling_type  # 池化类型
        self.distribution_output = distribution_output  # 分布输出

        head_dim = config.num_input_channels * config.d_model  # 计算头部维度

        self.flatten = nn.Flatten(start_dim=1)  # 展平操作
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()  # dropout操作

        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.num_targets)  # 线性投影
        else:
            self.projection = distribution_output.get_parameter_projection(head_dim)  # 获取分布输出的参数投影

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                    `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, output_dim)`

        """
        if self.use_cls_token:
            # 使用第一个输出 token，pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # 池化平均值，pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # 池化最大值，pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2)
        else:
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")  # 未实现的池化类型错误
        # 展平输入
        # pooled_embedding: bs x (num_channels * d_model)
        pooled_embedding = self.dropout(self.flatten(pooled_embedding))
        # 投影
        # output: bs x output_dim 或一个这个形状的元组，用于分布头部
        output = self.projection(pooled_embedding)
        # 如果需要，应用sigmoid来限制输出范围
        if (self.distribution_output is None) & (self.y_range is not None):  # 线性头部
            output = torch.sigmoid(output) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        return output


@add_start_docstrings(
    "The PatchTST for regression model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForRegression(PatchTSTPreTrainedModel):
    # 初始化 PatchTSTModel 类
        def __init__(self, config: PatchTSTConfig):
            # 调用父类（可能是nn.Module或其他基类）的初始化方法
            super().__init__(config)
    
            # 如果配置中要求进行输入掩码，则关闭该功能并打印警告信息
            if config.do_mask_input:
                logger.warning("Setting `do_mask_input` parameter to False.")
                config.do_mask_input = False
    
            # 创建 PatchTSTModel 对象
            self.model = PatchTSTModel(config)
    
            # 根据损失函数的设置，创建不同的分布输出层
            if config.loss == "mse":
                self.distribution_output = None
            else:
                if config.distribution_output == "student_t":
                    self.distribution_output = StudentTOutput(dim=config.prediction_length * config.num_targets)
                elif config.distribution_output == "normal":
                    self.distribution_output = NormalOutput(dim=config.prediction_length * config.num_targets)
                elif config.distribution_output == "negative_binomial":
                    self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length * config.num_targets)
                else:
                    raise ValueError(f"Unknown distribution output {config.distribution_output}")
    
            # 创建 PatchTSTRegressionHead 对象
            self.head = PatchTSTRegressionHead(config, self.distribution_output)
    
            # 调用 post_init 方法，初始化权重并进行后处理
            self.post_init()
    
        # 前向传播方法
        def forward(
            self,
            past_values: torch.Tensor,
            target_values: torch.Tensor = None,
            past_observed_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # 在此处实现前向传播逻辑
    
        # 生成方法
        def generate(
            self,
            past_values: torch.Tensor,
            past_observed_mask: Optional[torch.Tensor] = None,
        ):
            # 在此处实现生成逻辑
    # 定义函数，接受输入参数并返回SamplePatchTSTOutput类型的输出
        """
        从具有概率分布头的模型生成样本预测序列。

        参数：
            past_values（形状为`(batch_size, sequence_length, num_input_channels)`的`torch.FloatTensor`）：
                时间序列的过去值，作为上下文用于预测未来。
            past_observed_mask（形状为`(batch_size, sequence_length, num_input_channels)`的`torch.BoolTensor`，*可选*）：
                用于指示`past_values`中哪些值被观察到，哪些值是缺失的布尔蒙版。在`[0, 1]`范围内选择遮罩值：

                - 对于**观察到**的值，设置为1，
                - 对于**缺失**的值，设置为0（即被替换为零的NaN值）。

        返回：
            [`SamplePatchTSTOutput`]，其中输出`sequences`张量的形状为`(batch_size, number of samples, num_targets)`。
        """
        # 获取样本数量
        num_parallel_samples = self.config.num_parallel_samples

        # 获取模型输出
        outputs = self(
            past_values=past_values,
            target_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # 获取分布
        distribution = self.distribution_output.distribution(outputs.regression_outputs)
        # 获取样本：列表形式的[bs x num_targets]
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        # 样本：[bs x num_samples x num_targets]
        samples = torch.stack(samples, dim=1)
        返回SamplePatchTSTOutput类型的输出，其中sequences为samples
```  
```