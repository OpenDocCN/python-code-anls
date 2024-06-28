# `.\models\patchtst\modeling_patchtst.py`

```
# coding=utf-8
# Copyright 2023 IBM & Hugging Face. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch PatchTST model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSTConfig"

PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtst-etth1-pretrain",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtst
]


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PatchTST
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
        # 初始化注意力机制模块
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads  # 计算每个头的维度
        self.config = config

        # 检查embed_dim必须能被num_heads整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性变换层，用于计算查询、键、值、输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重塑成适合多头注意力计算的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义 Transformer 模型的前向传播函数
    def forward(
        self,
        # 输入隐藏状态张量，通常是上一层的输出
        hidden_states: torch.Tensor,
        # 键-值状态张量，用于自注意力机制，可选参数
        key_value_states: Optional[torch.Tensor] = None,
        # 上一步的键-值状态元组，用于复用键-值状态，可选参数
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 注意力掩码张量，控制哪些位置被注意力机制处理，可选参数
        attention_mask: Optional[torch.Tensor] = None,
        # 层级头掩码张量，用于控制层级注意力，可选参数
        layer_head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，布尔值，默认为 False
        output_attentions: bool = False,
class PatchTSTBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 初始化 BatchNorm1d，设置输入维度为 config.d_model，epsilon 参数为 config.norm_eps
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        # 将输入张量转置，调整维度顺序为 (batch_size, d_model, sequence_length)
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        # 应用 BatchNorm1d 进行批量归一化
        output = self.batchnorm(output)
        # 再次转置，将维度顺序恢复为 (batch_size, sequence_length, d_model)，并返回结果
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
    # 检查 mask_ratio 是否在有效范围内
    if mask_ratio < 0 or mask_ratio >= 1:
        raise ValueError(f"Mask ratio {mask_ratio} has to be between 0 and 1.")

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    # 计算不被遮盖的长度
    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        # 生成随机噪声，形状为 bs x 1 x L
        noise = torch.rand(batch_size, 1, sequence_length, device=device)
        # 将噪声在通道维度上复制，形状变为 bs x num_channels x L
        noise = noise.repeat(1, num_channels, 1)
    else:
        # 生成随机噪声，形状为 bs x num_channels x L
        noise = torch.rand(batch_size, num_channels, sequence_length, device=device)

    # 创建遮罩张量，形状为 bs x num_channels x L，并初始化为全 1
    mask = torch.ones(batch_size, num_channels, sequence_length, device=device)
    # 将部分位置置为 0，以实现遮盖操作
    mask[:, :, :len_keep] = 0

    # 对噪声进行排序，得到排序后的索引，用于确定要保留的位置
    ids_shuffle = torch.argsort(noise, dim=-1)
    # 创建恢复索引，将排序后的索引恢复到原始顺序
    ids_restore = torch.argsort(ids_shuffle, dim=-1)
    # 使用给定的索引ids_restore从mask张量中按列收集数据，形成新的mask张量
    mask = torch.gather(mask, dim=-1, index=ids_restore)
    # 在最后一个维度上增加一个维度，并将其复制多次，扩展为指定形状
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    # 如果unmasked_channel_indices不为None，则将指定通道的mask值置为0
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    # 使用bool类型的mask张量，在inputs中将对应位置的值填充为指定的mask_value
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    # 返回处理后的inputs_mask和mask张量的第一个通道的数据
    return inputs_mask, mask[..., 0]
# 定义一个预测掩码函数，用于在输入的时间序列数据中掩盖预测期末的部分补丁。如果 num_forecast_mask_patches 是一个列表，批次中的样本将随机掩盖列表中定义的补丁数。
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

    # 如果 num_forecast_mask_patches 是整数，则转换为列表形式方便处理
    if isinstance(num_forecast_mask_patches, int):
        num_forecast_mask_patches = [num_forecast_mask_patches]
    
    # 初始化每个预测掩码比例为 1
    forecast_mask_ratios = [1 for _ in num_forecast_mask_patches]

    # 获取输入的形状信息
    batch_size, num_channels, sequence_length, num_features = inputs.shape

    # 创建一个全零的掩码张量，形状与输入数据相同
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    # 初始化用于存储各个补丁长度、比例和临时长度的列表
    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    # 遍历每个预测掩码长度和比例，并根据比例分配临时长度
    for patch_length, ratio in zip(num_forecast_mask_patches, forecast_mask_ratios):
        # 检查补丁长度是否合理
        if patch_length <= 0 or patch_length >= sequence_length:
            raise ValueError(
                f"num_forecast_mask_patches {patch_length} should be greater than 0 and less than total patches."
            )
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    # 按临时长度排序 t_list
    t_list = sorted(t_list, key=lambda x: x[2])

    # 如果总临时长度小于批次大小，调整第一个补丁的临时长度
    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    # 如果总临时长度大于批次大小，调整最后一个补丁的临时长度
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    # 初始化变量用于迭代赋值掩码
    batch1 = 0
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        # 在掩码的最后 patch_len 长度处进行赋值为 1，表示需要掩盖的部分
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    # 随机打乱掩码的顺序
    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    # 将掩码扩展维度以匹配输入数据的形状
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patch x patch_len]

    # 如果提供了未掩盖的通道索引，将这些通道的掩码值设为 0
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    # 根据掩码值将输入数据进行掩码处理
    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)

    # 返回掩码后的输入数据和掩码张量的第一个通道
    return inputs_mask, mask[..., 0]
    # 初始化方法，接受一个 PatchTSTConfig 类型的配置对象作为参数
    def __init__(self, config: PatchTSTConfig):
        # 调用父类的初始化方法
        super().__init__()

        # 设置对象的序列长度、补丁长度和补丁步幅
        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        # 如果序列长度小于等于补丁长度，则抛出数值错误异常
        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # 计算补丁的数量
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        # 计算新的序列长度
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        # 计算序列的起始位置
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        # 检查输入的序列长度是否与模型配置的序列长度相匹配
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )

        # 输出形状: [batch_size x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start :, :]
        # 按照补丁步幅展开序列
        # 输出形状: [batch_size x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # 转置输出，调整维度顺序
        # 输出形状: [batch_size x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
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
        super().__init__()
        self.random_mask_ratio = config.random_mask_ratio  # 设置随机遮蔽比例
        self.channel_consistent_masking = config.channel_consistent_masking  # 是否进行通道一致的遮蔽
        self.mask_type = config.mask_type  # 遮蔽类型，随机或预测
        self.num_forecast_mask_patches = config.num_forecast_mask_patches  # 预测遮蔽时的遮蔽补丁数量
        self.unmasked_channel_indices = config.unmasked_channel_indices  # 未遮蔽的通道索引列表
        self.mask_value = config.mask_value  # 遮蔽值的设置
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices = sorted(self.unmasked_channel_indices)  # 如果有未遮蔽的通道索引，进行排序

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
            # 执行随机遮蔽
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
            )
        elif self.mask_type == "forecast":
            # 执行预测遮蔽
            masked_input, mask = forecast_masking(
                inputs=patch_input,
                num_forecast_mask_patches=self.num_forecast_mask_patches,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value,
            )
        else:
            # 抛出无效的遮蔽类型错误
            raise ValueError(f"Invalid mask type {self.mask_type}.")

        # 将遮蔽张量转换为布尔类型
        mask = mask.bool()
        return masked_input, mask


class PatchTSTEncoderLayer(nn.Module):
    """
    PatchTST encoder layer
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.channel_attention = config.channel_attention
        # Multi-Head attention
        self.self_attn = PatchTSTAttention(
            embed_dim=config.d_model,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )

        # Add & Norm of the sublayer 1
        self.dropout_path1 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        # 根据配置选择不同的规范化层（批标准化或层标准化）
        if config.norm_type == "batchnorm":
            self.norm_sublayer1 = PatchTSTBatchNorm(config)
        elif config.norm_type == "layernorm":
            self.norm_sublayer1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # Add & Norm of the sublayer 2, conditionally based on self.channel_attention
        if self.channel_attention:
            self.dropout_path2 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
            # 根据配置选择不同的规范化层（批标准化或层标准化）
            if config.norm_type == "batchnorm":
                self.norm_sublayer2 = PatchTSTBatchNorm(config)
            elif config.norm_type == "layernorm":
                self.norm_sublayer2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            else:
                raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),  # 使用配置中的激活函数类别激活线性层输出
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 3
        self.dropout_path3 = nn.Dropout(config.path_dropout) if config.path_dropout > 0 else nn.Identity()
        # 根据配置选择不同的规范化层（批标准化或层标准化）
        if config.norm_type == "batchnorm":
            self.norm_sublayer3 = PatchTSTBatchNorm(config)
        elif config.norm_type == "layernorm":
            self.norm_sublayer3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")

        self.pre_norm = config.pre_norm
class PatchTSTPreTrainedModel(PreTrainedModel):
    # 设置配置类
    config_class = PatchTSTConfig
    # 基础模型前缀
    base_model_prefix = "model"
    # 主输入名称
    main_input_name = "past_values"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """
        初始化权重
        """
        if isinstance(module, PatchTSTPositionalEncoding):
            # 初始化 cls_token
            if self.config.use_cls_token:
                nn.init.normal_(module.cls_token, std=0.02)
            # 初始化位置编码
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1.0
            module.weight.data.fill_(1.0)
        elif isinstance(module, PatchTSTBatchNorm):
            # 将批归一化层的偏置项初始化为零
            module.batchnorm.bias.data.zero_()
            # 将批归一化层的权重初始化为1.0
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            # 将权重初始化为正态分布随机值
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            # 如果存在偏置项，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        # 如果是 PatchTSTEncoder 类型的模块，设置梯度检查点
        if isinstance(module, (PatchTSTEncoder)):
            module.gradient_checkpointing = value


class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_input_channels = config.num_input_channels
        self.share_embedding = config.share_embedding
        # 输入编码：将特征向量投影到 d 维向量空间
        if self.share_embedding:
            # 如果共享嵌入层，则使用线性映射
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)
        else:
            # 如果不共享嵌入层，则创建多个线性映射
            self.input_embedding = nn.ModuleList()
            for _ in range(config.num_input_channels):
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

        # 获取输入张量的通道数
        num_input_channels = patch_input.shape[1]

        # 检查输入通道数是否与配置中的要求一致
        if num_input_channels != self.num_input_channels:
            raise ValueError(
                f"The defined number of input channels ({self.num_input_channels}) in the config "
                f"has to be the same as the number of channels in the batch input ({num_input_channels})"
            )

        # 如果指定共享嵌入层，则使用单个嵌入层对所有通道进行嵌入
        if self.share_embedding:
            embeddings = self.input_embedding(patch_input)  # x: [bs x num_channels  x num_patches x d_model]
        else:
            # 否则，对每个通道分别使用对应的嵌入层进行嵌入
            embeddings = [self.input_embedding[i](patch_input[:, i, :, :]) for i in range(num_input_channels)]
            embeddings = torch.stack(embeddings, dim=1)

        # 返回嵌入后的张量
        return embeddings
class PatchTSTPositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__()
        self.use_cls_token = config.use_cls_token  # 是否使用类别令牌标志位
        self.num_input_channels = config.num_input_channels  # 输入通道数
        if config.use_cls_token:
            # cls_token: [1 x num_input_channels x 1 x d_model]
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))  # 类别令牌参数初始化
            num_patches += 1  # 如果使用类别令牌，增加补丁数量

        # postional encoding: [num_patches x d_model]
        self.position_enc = self._init_pe(config, num_patches)  # 初始化位置编码

        # Positional dropout
        self.positional_dropout = (
            nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()
        )  # 位置dropout，如果设置了dropout则使用，否则使用恒等映射

    @staticmethod
    def _init_pe(config: PatchTSTConfig, num_patches: int) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
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
        return position_enc  # 返回位置编码张量作为参数

    def forward(self, patch_input: torch.Tensor):
        if self.use_cls_token:
            # patch_input: [bs x num_channels x num_patches x d_model]
            patch_input = self.positional_dropout(patch_input + self.position_enc[1:, :])
            # append cls token where cls_token: [1 x num_channels x 1 x d_model]
            cls_token = self.cls_token + self.position_enc[:1, :]
            # get the same copy of cls_token for all the samples in batch: [bs x num_channels x 1 x d_model]
            cls_tokens = cls_token.expand(patch_input.shape[0], self.num_input_channels, -1, -1)
            # hidden_state: [bs x num_channels x (num_patches+1) x d_model]
            hidden_state = torch.cat((cls_tokens, patch_input), dim=2)
        else:
            # hidden_state: [bs x num_channels x num_patches x d_model]
            hidden_state = self.positional_dropout(patch_input + self.position_enc)
        return hidden_state


class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """
    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__(config)
        self.gradient_checkpointing = False

        # Input embedding: projection of feature vectors onto a d-dim vector space
        self.embedder = PatchTSTEmbedding(config)
        # Positional encoding
        self.positional_encoder = PatchTSTPositionalEncoding(config, num_patches)
        # Encoder
        self.layers = nn.ModuleList([PatchTSTEncoderLayer(config) for i in range(config.num_hidden_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        patch_input: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BaseModelOutput:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Past values of the time series
            output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
            output_attentions (bool, optional): Indicates if attentions should be outputted.

        return:
            `BaseModelOutput`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Input embedding
        patch_input = self.embedder(patch_input)
        # Positional encoding
        hidden_state = self.positional_encoder(patch_input)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                # Collect hidden states if requested
                encoder_states = encoder_states + (hidden_state,)

            # Process each encoder layer
            layer_outputs = encoder_layer(hidden_state=hidden_state, output_attentions=output_attentions)
            # Update hidden state to the output of the current layer
            hidden_state = layer_outputs[0]
            if output_attentions:
                # Collect attention matrices if requested
                all_attentions = all_attentions + (layer_outputs[1],)

        # Return model output including final hidden states and attentions
        return BaseModelOutput(last_hidden_state=hidden_state, hidden_states=encoder_states, attentions=all_attentions)
# 定义文档字符串，说明了这个模型继承自 `PreTrainedModel`，可以使用该类中的通用方法，如下载、保存、调整输入嵌入等。
# 这个模型也是一个 PyTorch 的 `torch.nn.Module` 子类，可以像普通的 PyTorch 模块一样使用，相关的使用和行为请参考 PyTorch 文档。

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


@dataclass
class PatchTSTForPretrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPretraining`].
    
    This class defines the output structure specifically for the `PatchTSTForPretraining` model, but does not contain any additional fields.
    It inherits from `ModelOutput`, which is a base class providing basic fields like `last_hidden_state`, `hidden_states`, etc.
    """
    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
            MSE（均方误差）损失值，仅在提供了`labels`时返回，类型为`torch.FloatTensor`，形状为`(1,)`。

        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction outputs of the time series modeling heads.
            时间序列建模头部的预测输出，类型为`torch.FloatTensor`，形状为`(batch_size, sequence_length, config.vocab_size)`。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型每一层输出的隐藏状态，以及初始嵌入输出的元组，类型为`tuple(torch.FloatTensor)`，形状为`(batch_size, sequence_length, hidden_size)`。
        
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，在经过注意力 softmax 后得到，用于计算自注意力头部的加权平均，类型为`tuple(torch.FloatTensor)`，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
# 使用 dataclass 装饰器定义 PatchTSTForRegressionOutput 类，表示回归模型的输出结果
@dataclass
class PatchTSTForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForRegression`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
            均方误差损失，仅在提供 `labels` 参数时返回，类型为 `torch.FloatTensor`，形状为 `(1,)`。
        regression_outputs (`torch.FloatTensor` of shape `(batch_size, num_targets)`):
            Regression outputs of the time series modeling heads.
            时间序列建模头部的回归输出，类型为 `torch.FloatTensor`，形状为 `(batch_size, num_targets)`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每层输出的隐藏状态，包括初始嵌入输出，类型为 `tuple(torch.FloatTensor)`，仅在传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重经过注意力 softmax 后的值，用于计算自注意力头中的加权平均，类型为 `tuple(torch.FloatTensor)`，仅在传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    """

    # 可选的属性：MSE 损失，类型为 torch.FloatTensor，形状为 `(1,)`
    loss: Optional[torch.FloatTensor] = None
    # 回归模型的输出结果，类型为 torch.FloatTensor，形状为 `(batch_size, num_targets)`
    regression_outputs: torch.FloatTensor = None
    # 可选的属性：模型各层的隐藏状态，类型为 tuple(torch.FloatTensor)，形状为 `(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选的属性：注意力权重，类型为 tuple(torch.FloatTensor)，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 PatchTSTForPredictionOutput 类，表示预测模型的输出结果
@dataclass
class PatchTSTForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPrediction`].
    """
    # 定义函数参数及其可选的类型和描述
    
    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
            MSE 损失（均方误差损失），当提供 `labels` 时返回，类型为 `torch.FloatTensor`，形状为 `(1,)`。
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, prediction_length, -1)`):
            Prediction outputs of the time series modeling heads.
            时间序列建模头的预测输出，类型为 `torch.FloatTensor`，形状为 `(batch_size, prediction_length, -1)`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
    
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每层输出的隐藏状态，以及初始嵌入输出的元组。返回条件包括传递 `output_hidden_states=True` 或 `config.output_hidden_states=True`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，经过注意力 softmax 后的结果，用于计算自注意力头部的加权平均值。返回条件包括传递 `output_attentions=True` 或 `config.output_attentions=True`。
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
            输入数据的均值（在序列长度上）。类型为 `torch.FloatTensor`，形状为 `(batch_size, 1, num_channels)`。
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`, *optional*)
            Std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
            输入数据的标准差（在序列长度上）。类型为 `torch.FloatTensor`，形状为 `(batch_size, 1, num_channels)`。
# 定义一个数据类，用于存储 PatchTST 模型用于分类的输出结果，继承自 ModelOutput。
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

    loss: Optional[torch.FloatTensor] = None  # 总损失，如果提供了 `labels` 参数，则返回
    prediction_logits: torch.FloatTensor = None  # PatchTST 模型头部的预测分数（SoftMax 前的分数）
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型每一层的隐藏状态和初始嵌入输出的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 自注意力头中注意力权重的元组


# 定义一个数据类，用于存储样本化的 PatchTST 模型输出结果，继承自 ModelOutput。
@dataclass
class SamplePatchTSTOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences `(batch_size, num_samples, prediction_length, num_targets)`):
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None  # 从选择的分布中抽样得到的值


# 从时间序列变换模型中引用的函数，计算给定分布对于目标的负对数似然损失。
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


# 从时间序列变换模型中引用的函数，计算给定张量在给定维度上的加权平均值，避免权重为零的部分置零而非 NaN。
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.
    """
    # 如果给定了权重张量 `weights`
    if weights is not None:
        # 计算加权后的张量，其中权重不为零的位置进行乘法，其余位置置零
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        # 计算权重的总和，限制最小值为1.0，按指定的维度 `dim` 进行求和
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        # 返回加权张量沿指定维度 `dim` 的平均值
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        # 如果没有提供权重张量，则计算输入张量沿指定维度 `dim` 的平均值
        return input_tensor.mean(dim=dim)
# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeriesTransformer->PatchTST,TimeSeries->PatchTST
class PatchTSTStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 设置标准化的维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保持维度，True 表示保持，默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小标度，默认为1e-5
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # 计算标度的分母，根据 observed_indicator 在指定维度上的和
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将分母至少设为1.0，避免除以零
        denominator = denominator.clamp_min(1.0)
        # 计算均值 loc，根据 observed_indicator 对 data 加权平均
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算方差 variance，根据 observed_indicator 对 data 进行标准差计算
        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        # 计算标度 scale，将方差开根号并加上最小标度
        scale = torch.sqrt(variance + self.minimum_scale)
        # 返回标准化后的数据，均值 loc 和标度 scale
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeriesTransformer->PatchTST,TimeSeries->PatchTST
class PatchTSTMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 设置标准化的维度，默认为1
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 是否保持维度，True 表示保持，默认为 True
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        # 最小标度，默认为1e-10
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        # 默认标度，若为 None 则无默认标度
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`
                scaled data based on the computed scaling factor.
        """
        # 计算标度的分母，根据 observed_indicator 在指定维度上的和
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        # 将分母至少设为1.0，避免除以零
        denominator = denominator.clamp_min(1.0)
        # 计算均值 loc，根据 observed_indicator 对 data 加权平均
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        # 计算绝对值的加权平均数，作为标度 scale
        scale = torch.mean(torch.abs(data - loc), dim=self.dim, keepdim=self.keepdim)
        # 若存在默认标度，则应用默认标度
        if self.default_scale is not None:
            scale = torch.max(scale, self.default_scale)

        # 根据计算得到的标度对 data 进行缩放
        return data / scale
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        # Calculate the sum of absolute values of `data` multiplied by `observed_indicator`
        # along the specified dimension `self.dim`, maintaining the dimensionality.
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        
        # Count the number of observed elements (True values) in `observed_indicator`
        # along the specified dimension `self.dim`, maintaining the dimensionality.
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        # Compute the scale as the ratio of `ts_sum` to `num_observed`, clamping
        # `num_observed` to a minimum value of 1 to avoid division by zero.
        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is not provided, calculate it based on the sum of `ts_sum`
        # across the batch and the sum of `num_observed` across the batch, clamped to
        # ensure no division by zero.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            # Use the provided `default_scale` multiplied element-wise by a tensor of ones
            # with the same shape as `scale`.
            default_scale = self.default_scale * torch.ones_like(scale)

        # Apply `default_scale` where `num_observed` is greater than zero, otherwise use `scale`.
        scale = torch.where(num_observed > 0, scale, default_scale)

        # Ensure that `scale` is not less than `self.minimum_scale`.
        scale = torch.clamp(scale, min=self.minimum_scale)

        # Scale `data` by dividing each element by the corresponding element in `scale`.
        scaled_data = data / scale

        # If `self.keepdim` is False, squeeze `scale` along the specified dimension `self.dim`.
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        # Return the scaled data, a tensor of zeros with the same shape as `scale`,
        # and the computed `scale`.
        return scaled_data, torch.zeros_like(scale), scale
# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeriesTransformer->PatchTST,TimeSeries->PatchTST
# 定义一个模块 PatchTSTNOPScaler，用于数据缩放，不进行实际缩放，仅保持输入数据原样输出
class PatchTSTNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 初始化时设置缩放维度，默认为第一个维度（通常是 batch_size）
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        # 初始化时设置是否保持维度，默认为 True
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
        # 计算数据的均值，生成与输入数据相同形状的缩放因子
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 生成与输入数据相同形状的零向量，作为均值
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        # 返回原始输入数据、均值和缩放因子
        return data, loc, scale


# 定义一个模块 PatchTSTScaler，根据配置选择不同的缩放方式
class PatchTSTScaler(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # 根据配置选择不同的缩放方式
        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSTMeanScaler(config)
        elif config.scaling == "std":
            self.scaler = PatchTSTStdScaler(config)
        else:
            self.scaler = PatchTSTNOPScaler(config)

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
        # 调用所选的缩放器模块进行缩放操作
        data, loc, scale = self.scaler(data, observed_indicator)
        # 返回缩放后的数据、均值和缩放因子
        return data, loc, scale


# 添加文档字符串描述 PatchTSTModel 模型输出原始隐藏状态，不包含特定头部
@add_start_docstrings(
    "The bare PatchTST Model outputting raw hidden-states without any specific head.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTModel(PatchTSTPreTrainedModel):
    # 使用给定的配置对象初始化类，调用父类的初始化方法
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # 使用配置对象初始化 PatchTSTScaler 实例
        self.scaler = PatchTSTScaler(config)
        # 使用配置对象初始化 PatchTSTPatchify 实例
        self.patchifier = PatchTSTPatchify(config)
        # 从 PatchTSTPatchify 获取 num_patches 信息
        num_patches = self.patchifier.num_patches

        # 根据配置决定是否对输入进行屏蔽处理
        if self.do_mask_input:
            self.masking = PatchTSTMasking(config)
        else:
            # 如果不需要屏蔽输入，则使用恒等映射
            self.masking = nn.Identity()
        
        # 使用配置对象和 num_patches 初始化 PatchTSTEncoder 实例
        self.encoder = PatchTSTEncoder(config, num_patches=num_patches)

        # 初始化权重并进行最终处理
        self.post_init()

    # 定义前向传播方法，接受一些输入张量和可选参数，并返回预测结果
    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class PatchTSTMaskPretrainHead(nn.Module):
    """
    Pretraining head for mask modelling
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)  # 定义一个 dropout 层，根据配置决定丢弃概率
        self.linear = nn.Linear(config.d_model, config.patch_length)  # 定义一个全连接层，将输入维度映射到 patch_length
        self.use_cls_token = config.use_cls_token  # 是否使用类别标记（CLS token）

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                    `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                            `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True

        """
        embedding = self.linear(self.dropout(embedding))  # 使用线性层处理嵌入向量，形状变为 [bs x num_channels x num_patches x patch_length]
        if self.use_cls_token:
            embedding = embedding[:, :, 1:, :]  # 如果设置使用类别标记，去除第一个类别标记的部分
        return embedding


@add_start_docstrings(
    "The PatchTST for pretrain model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.do_mask_input = True  # 设置配置参数以掩蔽输入
        self.model = PatchTSTModel(config=config)  # 实例化 PatchTSTModel，并传入配置
        self.head = PatchTSTMaskPretrainHead(config)  # 实例化预训练头部模型 PatchTSTMaskPretrainHead

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters:
            past_values (`torch.Tensor`): Tensor containing past values.
            past_observed_mask (`Optional[torch.Tensor]`, optional): Mask tensor for observed values.
            output_hidden_states (`Optional[bool]`, optional): Whether to output hidden states.
            output_attentions (`Optional[bool]`, optional): Whether to output attention weights.
            return_dict (`Optional[bool]`, optional): Whether to return a dictionary.

        Returns:
            Dictionary containing output tensors depending on the model configuration.
        """
        # 省略部分代码...

class PatchTSTClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token  # 是否使用类别标记（CLS token）
        self.pooling_type = config.pooling_type  # 池化类型
        self.flatten = nn.Flatten(start_dim=1)  # 展开操作，从第一个维度开始展开
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()  # 如果设置了 dropout，则使用；否则使用恒等映射
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_targets)  # 全连接层，输入为 num_input_channels * d_model，输出为 num_targets
    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_targets)`

        """
        if self.use_cls_token:
            # 如果设置了使用CLS token，则使用第一个输出token作为池化的embedding: bs x num_channels x d_model
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # 如果使用均值池化，则对embedding在第2维（num_patches）上取均值: pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # 如果使用最大池化，则对embedding在第2维（num_patches）上取最大值: pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2).values
        else:
            # 如果指定的池化类型未实现，则抛出异常
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        
        # 将池化后的embedding展平，pooled_embedding: bs x num_channels * d_model
        pooled_embedding = self.flatten(pooled_embedding)
        
        # 经过线性层和dropout后得到最终输出，output: bs x n_classes
        output = self.linear(self.dropout(pooled_embedding))
        
        return output
@add_start_docstrings(
    "The PatchTST for classification model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForClassification(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # Turn off masking if specified in the configuration
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        # Initialize PatchTSTModel and PatchTSTClassificationHead
        self.model = PatchTSTModel(config)
        self.head = PatchTSTClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        past_observed_mask: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the PatchTSTForClassification model.

        Parameters:
        - past_values: Tensor of past input values.
        - target_values: Optional tensor of target values.
        - past_observed_mask: Optional boolean mask for observed values.
        - output_hidden_states: Optional boolean to output hidden states.
        - output_attentions: Optional boolean to output attentions.
        - return_dict: Optional boolean to return a dictionary.

        Returns:
        - Depending on configurations, returns classification predictions.
        """
        # Forward pass implementation details are defined elsewhere.
        pass


@add_start_docstrings(
    "The PatchTST for regression Model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTPredictionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, num_patches, distribution_output=None):
        super().__init__()

        self.share_projection = config.share_projection
        self.num_input_channels = config.num_input_channels
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type

        # Determine head dimension based on configuration
        if self.pooling_type or self.use_cls_token:
            head_dim = config.d_model
        else:
            head_dim = config.d_model * num_patches

        if not self.share_projection:
            # If each channel has its own head, initialize projections, dropouts, and flattens
            self.projections = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.flattens.append(nn.Flatten(start_dim=2))
                if distribution_output is None:
                    # Use linear head projection
                    self.projections.append(nn.Linear(head_dim, config.prediction_length))
                else:
                    # Use distribution head projection
                    self.projections.append(distribution_output.get_parameter_projection(head_dim))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:
            # All channels share the same head, initialize flatten, projection, and dropout
            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:
                # Use linear head projection
                self.projection = nn.Linear(head_dim, config.prediction_length)
            else:
                # Use distribution head projection
                self.projection = distribution_output.get_parameter_projection(head_dim)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

        # Additional initialization steps can be included here
    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

        """
        if self.use_cls_token:
            # 如果使用了 cls_token，则从 embedding 中选择第一个 patch 的 embedding
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        else:
            if self.pooling_type == "mean":
                # 如果使用平均池化，则对 embedding 在第二个维度（patch 维度）进行平均池化
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.mean(dim=2)
            elif self.pooling_type == "max":
                # 如果使用最大池化，则对 embedding 在第二个维度进行最大池化操作，取最大值
                # pooled_embedding: [bs x num_channels x d_model]
                pooled_embedding = embedding.max(dim=2).values
            else:
                # 如果没有指定池化方式，则直接使用 embedding
                # pooled_embedding: [bs x num_channels x num_patches x d_model]
                pooled_embedding = embedding

        if not self.share_projection:
            output = []
            for i in range(self.num_input_channels):
                # 对 pooled_embedding 进行展平操作，以便进行后续的线性变换
                # pooled_embedding: [bs x (d_model * num_patches)] or [bs x d_model)]
                pooled_embedding = self.flattens[i](pooled_embedding[:, i, :])
                pooled_embedding = self.dropouts[i](pooled_embedding)
                # 经过线性变换得到输出，可能返回一个或两个 tensor，视具体实现而定
                # pooled_embedding: [bs x forecast_len]
                #  or tuple ([bs x forecast_len], [bs x forecast_len]) if using distribution head
                pooled_embedding = self.projections[i](pooled_embedding)
                output.append(pooled_embedding)
            # 将每个通道的输出堆叠起来
            # output: [bs x num_channels x forecast_len]
            output = torch.stack(output, dim=1)
        else:
            # 如果共享投影层，则对 pooled_embedding 进行统一的展平操作
            # pooled_embedding: [bs x num_channels x (d_model * num_patches)] or [bs x num_channels x d_model)]
            pooled_embedding = self.flatten(pooled_embedding)
            pooled_embedding = self.dropout(pooled_embedding)
            # 经过线性变换得到输出，可能返回一个或两个 tensor，视具体实现而定
            # output: [bs x num_channels x forecast_len] or
            # tuple ([bs x num_channels x forecast_len], [bs x num_channels x forecast_len]) if using distribution head
            output = self.projection(pooled_embedding)

        if isinstance(output, tuple):
            # 如果输出是一个 tuple，则交换第二个和第三个维度
            # output: ([bs x forecast_len x num_channels], [bs x forecast_len x num_channels])
            output = tuple(z.transpose(2, 1) for z in output)
        else:
            # 否则，交换第二个和第三个维度
            output = output.transpose(2, 1)  # [bs x forecast_len x num_channels]
        return output
@add_start_docstrings(
    "The PatchTST for prediction model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # Turn off masking if specified in the configuration
        if config.do_mask_input:
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        # Instantiate the PatchTSTModel with the provided configuration
        self.model = PatchTSTModel(config)

        # Determine the type of distribution output based on the configuration
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

        # Initialize PatchTSTPredictionHead with necessary configurations and distribution output
        self.head = PatchTSTPredictionHead(
            config, self.model.patchifier.num_patches, distribution_output=self.distribution_output
        )

        # Initialize weights and apply final processing for the model
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Forward pass method for the model, computes predictions based on input
        ...

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        ...
    ) -> SamplePatchTSTOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number of
            samples, prediction_length, 1)` or `(batch_size, number of samples, prediction_length, num_input_channels)`
            for multivariate predictions.
        """
        # 获取并行采样的数量
        num_parallel_samples = self.config.num_parallel_samples

        # 获取模型的输出
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        if self.distribution_output:
            # 获取分布对象
            distribution = self.distribution_output.distribution(
                outputs.prediction_outputs, loc=outputs.loc, scale=outputs.scale
            )
            # 生成样本：列表形式的 [bs x forecast_len x num_channels]
            samples = [distribution.sample() for _ in range(num_parallel_samples)]
            # 将样本堆叠起来：[bs x num_samples x forecast_len x num_channels]
            samples = torch.stack(samples, dim=1)
        else:
            # 如果没有指定分布输出，直接使用预测输出，并在样本维度上增加一个维度
            samples = outputs.prediction_outputs.unsqueeze(1)

        # 返回包含样本预测序列的 SamplePatchTSTOutput 对象
        return SamplePatchTSTOutput(sequences=samples)
class PatchTSTRegressionHead(nn.Module):
    """
    Regression head
    """

    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()
        # 设置输出范围
        self.y_range = config.output_range
        # 是否使用类别标记
        self.use_cls_token = config.use_cls_token
        # 池化类型
        self.pooling_type = config.pooling_type
        # 分布输出
        self.distribution_output = distribution_output

        # 计算头部维度
        head_dim = config.num_input_channels * config.d_model

        # 展平层，将输入展平
        self.flatten = nn.Flatten(start_dim=1)
        # dropout层，如果配置了dropout，则应用dropout；否则使用恒等映射
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

        # 如果未提供分布输出，使用线性层进行投影
        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.num_targets)
        else:
            # 否则，使用分布输出对象提供的投影
            self.projection = distribution_output.get_parameter_projection(head_dim)

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
            # 如果使用类别标记，选择第一个输出标记，池化后的嵌入：[bs x num_channels x d_model]
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # 使用均值池化，池化后的嵌入：[bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # 使用最大池化，池化后的嵌入：[bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2).values
        else:
            # 抛出错误，指定的池化类型尚未实现
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        
        # 展平输入
        # pooled_embedding: bs x (num_channels * d_model)
        pooled_embedding = self.dropout(self.flatten(pooled_embedding))
        
        # 投影操作
        # output: bs x output_dim 或者是一个这样形状的元组，用于分布头部
        output = self.projection(pooled_embedding)
        
        # 如果需要，应用sigmoid函数来限制输出范围
        if (self.distribution_output is None) & (self.y_range is not None):  # 线性头部
            output = torch.sigmoid(output) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        
        return output


@add_start_docstrings(
    "The PatchTST for regression model.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTForRegression(PatchTSTPreTrainedModel):
    """
    PatchTST for regression model.
    
    Inherits from PatchTSTPreTrainedModel.
    """
    def __init__(self, config: PatchTSTConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 关闭输入数据的掩码处理
        if config.do_mask_input:
            # 如果需要掩码输入，发出警告并设置参数为 False
            logger.warning("Setting `do_mask_input` parameter to False.")
            config.do_mask_input = False

        # 使用配置对象初始化 PatchTSTModel 模型
        self.model = PatchTSTModel(config)

        # 根据损失函数类型确定输出分布
        if config.loss == "mse":
            # 如果损失函数是均方误差，则不需要特定的分布输出
            self.distribution_output = None
        else:
            # 根据配置中的分布输出类型选择对应的输出对象
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.num_targets)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.num_targets)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.num_targets)
            else:
                # 如果配置中指定了未知的分布输出类型，抛出数值错误
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        # 使用 PatchTSTRegressionHead 初始化模型的头部
        self.head = PatchTSTRegressionHead(config, self.distribution_output)

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 正向传播函数，根据输入参数进行模型推断和预测

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        ) -> SamplePatchTSTOutput:
        """
        从具有概率分布输出头的模型生成样本预测序列。

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                时间序列的过去值，用作上下文以预测未来。
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                布尔掩码，指示哪些 `past_values` 是观察到的，哪些是缺失的。掩码的取值范围为 `[0, 1]`:

                - 1 表示 **观察到** 的值，
                - 0 表示 **缺失** 的值（即被 NaN 替换为零）。

        Return:
            [`SamplePatchTSTOutput`]，输出的 `sequences` 张量形状为 `(batch_size, number of samples, num_targets)`。
        """
        # 获取样本数
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
        # 获取样本: 列表 `[bs x num_targets]`
        samples = [distribution.sample() for _ in range(num_parallel_samples)]
        # samples: `[bs x num_samples x num_targets]`
        samples = torch.stack(samples, dim=1).view(-1, num_parallel_samples, self.config.num_targets)
        return SamplePatchTSTOutput(sequences=samples)
```