# `so-vits-svc\vencoder\dphubert\components.py`

```py
"""Building blocks for speech SSL models supporting pruning.

Originally from:
https://github.com/pytorch/audio/blob/main/torchaudio/models/wav2vec2/components.py

"""

import math  # 导入数学库
from collections import defaultdict  # 导入默认字典
from typing import List, Optional, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 库
from torch import Tensor, nn  # 导入张量和神经网络模块
from torch.nn import Module  # 导入模块

from .hardconcrete import HardConcrete  # 从当前目录导入 HardConcrete 模块
from .pruning_utils import (  # 从 pruning_utils 模块导入以下函数
    prune_conv1d_layer,  # 剪枝一维卷积层
    prune_layer_norm,  # 剪枝层归一化
    prune_linear_layer,  # 剪枝线性层
)


def _init_transformer_params(module):
    """
    Initialize the weights of Transformer module in Wav2Vec2/HuBERT.

    If the module is ``nn.Linear``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``bias`` is set to ``True`` in the module, set ``bias`` to 0.

    If the module is ``nn.Embedding``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``padding_idx`` is not None, set the weight of padding to 0.

    Note:
        Ths method corresponds to
        `init_bert_params
        <https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_sentence_encoder.py#L21>`__
        in the original ``fairseq`` implementation.
    """
    # 初始化 Transformer 模块的权重
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):  # 如果模块是线性层
        normal_(module.weight.data)  # 标准化权重
        if module.bias is not None:  # 如果有偏置
            module.bias.data.zero_()  # 将偏置置零
    if isinstance(module, nn.Embedding):  # 如果模块是嵌入层
        normal_(module.weight.data)  # 标准化权重
        if module.padding_idx is not None:  # 如果有填充索引
            module.weight.data[module.padding_idx].zero_()  # 将填充的权重置零


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)  # 转置输入张量
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # 使用层归一化
        x = x.transpose(-2, -1)  # 再次转置张量
        return x  # 返回结果张量


class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""
    # 初始化函数，设置卷积层的参数
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
        prune_conv_channels: bool = False,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置步长
        self.stride = stride
        # 设置层归一化
        self.layer_norm = layer_norm
        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

        # 如果需要剪枝卷积通道
        if prune_conv_channels:
            # 创建一个 HardConcrete 对象
            self.hard_concrete = HardConcrete(n_in=out_channels, init_mean=0.01)
        else:
            # 否则设置为 None
            self.hard_concrete = None

    # 前向传播函数
    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        # 对输入进行卷积操作
        x = self.conv(x)
        # 如果有层归一化，进行层归一化操作
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        # 使用 GELU 激活函数
        x = nn.functional.gelu(x)

        # 如果有 HardConcrete 对象
        if self.hard_concrete is not None:
            # 获取 hard concrete mask
            channel_mask = self.hard_concrete()  # hard concrete mask, (out_channels,)
            # 对卷积结果进行通道剪枝
            x = x * channel_mask.unsqueeze(-1)

        # 如果输入长度不为空
        if length is not None:
            # 计算输出长度
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            # 当输入长度为0时，输出长度可能为负数，这里进行修正
            length = torch.max(torch.zeros_like(length), length)
        # 返回卷积结果和长度
        return x, length
    # 获取参数数量和输出通道数的方法，参数为输入通道数
    def get_num_params_and_out_channels(self, in_channels):
        # 如果存在硬混凝土对象，则输出通道数为硬混凝土对象的L0范数
        if self.hard_concrete is not None:
            out_channels = self.hard_concrete.l0_norm()
        # 否则，输出通道数为卷积层的输出通道数
        else:
            out_channels = self.conv.out_channels
        
        # 计算参数数量，等于输入通道数乘以输出通道数乘以卷积核大小
        num_params = in_channels * out_channels * self.kernel_size
        # 如果卷积层存在偏置，则参数数量加上输出通道数
        if self.conv.bias is not None:
            num_params += out_channels
        # 如果存在层归一化，则参数数量加上输出通道数乘以2
        if self.layer_norm is not None:
            num_params += out_channels * 2
        
        # 返回参数数量和输出通道数
        return num_params, out_channels
class FeatureExtractor(Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_layers = conv_layers

        # NOTE: a dummy weight used to save the soft mask of the last conv layer
        self.dummy_weight = nn.Parameter(
            torch.ones(conv_layers[-1].conv.out_channels, dtype=torch.float32),
            requires_grad=False
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError("Expected the input Tensor to be 2D (batch, time), " "but received {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        x = x * self.dummy_weight
        return x, length

    def get_num_params_and_final_out_channels(self):
        in_channels = 1
        num_params = 0
        for layer in self.conv_layers:
            layer_params, in_channels = layer.get_num_params_and_out_channels(in_channels)
            num_params += layer_params

        num_params += in_channels   # dummy weight
        
        return num_params, in_channels
    # 对卷积层和虚拟权重进行修剪，基于 hardconcrete 参数
    # 这是一个原地操作
    def prune(self):
        # 存储新的卷积层配置信息的列表，每个元素为 (输出通道数, 卷积核大小, 步长)
        new_config = []     
        # 遍历所有卷积层
        for idx, layer in enumerate(self.conv_layers):
            # 如果该层的 hardconcrete 参数不为空
            if layer.hard_concrete is not None:
                # 确保 hardconcrete 参数不处于训练状态
                assert not layer.hard_concrete.training
                # 获取 hardconcrete 参数生成的掩码
                mask = layer.hard_concrete()    
                # 获取掩码中非零元素的索引
                index = mask.nonzero().squeeze(-1)    
                # 确保索引的长度大于0，否则输出错误信息
                assert len(index) > 0, f"Conv channels pruned to zero at index {idx}"
                # 将新的卷积层配置信息添加到 new_config 列表中
                new_config.append(
                    (len(index), layer.kernel_size, layer.stride)
                )

                # 对当前层的卷积核进行修剪
                prune_conv1d_layer(layer.conv, index, "output")
                # 如果存在 layer_norm，则对其进行修剪
                if layer.layer_norm is not None:
                    prune_layer_norm(layer.layer_norm, index)

                # 对下一层的卷积核进行修剪
                if idx == len(self.conv_layers) - 1:
                    # 更新虚拟权重并修剪
                    self.dummy_weight.data *= mask
                    self.dummy_weight = nn.Parameter(
                        self.dummy_weight.index_select(0, index).clone().detach(), requires_grad=False
                    )
                else:
                    # 更新下一层的卷积核并修剪
                    self.conv_layers[idx+1].conv.weight.data *= mask.unsqueeze(-1)
                    prune_conv1d_layer(self.conv_layers[idx+1].conv, index, dim="input")

                # 将当前层的 hardconcrete 参数设为 None，表示已经使用过
                layer.hard_concrete = None
            else:
                # 如果该层的 hardconcrete 参数为空，则将当前卷积层的配置信息添加到 new_config 列表中
                new_config.append(
                    (layer.conv.out_channels, layer.kernel_size, layer.stride)
                )
                # 生成索引，包含当前卷积层的所有输出通道
                index = torch.arange(layer.conv.out_channels, dtype=torch.long)

        # 返回新的卷积层配置信息和索引
        return new_config, index
        super().__init__()
        # 初始化层归一化对象，输入特征维度为 in_features
        self.layer_norm = nn.LayerNorm(in_features)
        # 初始化线性投影层，输入特征维度为 in_features，输出特征维度为 out_features
        self.projection = nn.Linear(
            in_features,
            out_features,
        )
        # 初始化丢弃层，丢弃概率为 dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        # 对输入特征进行层归一化
        x = self.layer_norm(x)
        # 对层归一化后的特征进行线性投影
        x = self.projection(x)
        # 对线性投影后的特征进行丢弃
        x = self.dropout(x)
        # 返回处理后的特征
        return x
    
    def get_num_params(self, in_features):
        # 返回参数数量，计算公式为：输入特征维度 * 2 + (输入特征维度 + 1) * 输出特征维度
        return in_features * 2 + (in_features + 1) * self.projection.out_features


class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    # 初始化函数，继承父类的初始化方法
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int = 1,
    ):
        super().__init__()
        # 初始化嵌入维度和卷积核大小
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        # 对卷积层进行权重归一化处理
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
        # 根据卷积核大小确定需要移除的数量
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    # 准备可脚本化的方法
    def __prepare_scriptable__(self):
        # 遍历卷积层的前向钩子
        for hook in self.conv._forward_pre_hooks.values():
            # 需要移除的钩子是 WeightNorm 类的实例，但由于阴影效应，无法直接访问该类，因此直接检查模块名称和类名
            if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                # 移除权重归一化
                torch.nn.utils.remove_weight_norm(self.conv)
        # 返回当前对象
        return self

    # 前向传播方法
    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        # 调整输入张量的维度顺序
        x = x.transpose(-2, -1)
        # 经过卷积层处理
        x = self.conv(x)
        # 如果需要移除数据，则进行裁剪
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        # 使用 GELU 激活函数处理数据
        x = torch.nn.functional.gelu(x)
        # 再次调整张量的维度顺序
        x = x.transpose(-2, -1)
        # 返回处理后的张量
        return x
class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        prune_heads: bool = False,  # whether to prune attention heads
        prune_layer: bool = False,  # whether to prune entire attention layers
    ):
        super().__init__()

        self.embed_dim = embed_dim  # 设置模型的总维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.head_dim = head_dim  # 设置每个头的维度
        self.dropout = torch.nn.Dropout(dropout)  # 设置dropout层，用于attn_output_weights

        self.scaling = self.head_dim**-0.5  # 设置缩放因子

        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)  # 创建线性层，用于计算key
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)  # 创建线性层，用于计算value
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)  # 创建线性层，用于计算query
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)  # 创建线性层，用于输出

        if prune_heads:
            self.hard_concrete_for_heads = HardConcrete(n_in=num_heads, init_mean=0.01)  # 如果需要修剪注意力头，则创建HardConcrete对象
        else:
            self.hard_concrete_for_heads = None

        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)  # 如果需要修剪整个注意力层，则创建HardConcrete对象
        else:
            self.hard_concrete_for_layer = None

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    # 获取模型参数的数量
    def get_num_params(self):
        # 如果存在头部的硬具体值，则获取头部的数量
        if self.hard_concrete_for_heads is not None:
            num_heads = self.hard_concrete_for_heads.l0_norm()
        else:
            num_heads = self.num_heads
        # 计算参数数量
        num_params = (self.embed_dim + 1) * num_heads * self.head_dim * 3 \
            + (num_heads * self.head_dim + 1) * self.embed_dim

        # 如果存在层的硬具体值，则根据其值调整参数数量
        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        
        return num_params

    # 剪枝操作
    def prune(self):
        # 新的配置字典
        new_config = {
            "use_attention": True,
            "num_heads": self.num_heads,
        }
        # 如果存在层的硬具体值，则进行相应的剪枝操作
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer() # (1,)
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config["use_attention"] = False
            self.hard_concrete_for_layer = None

        # 如果存在头部的硬具体值，则进行相应的剪枝操作
        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            new_config["num_heads"] = len(head_mask.nonzero())
            if new_config["num_heads"] == 0:
                new_config["use_attention"] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, "input")
            self.hard_concrete_for_heads = None

        return new_config
class WavLMSelfAttention(SelfAttention):
    """Multi-headed self-attention for WavLM model :cite:`chen2022wavlm`.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional): Dropout probability on attn_output_weights. (Default: to ``0.0``)
        bias (bool, optional): If ``True``, add bias to input / output projection layers. (Default: ``True``)
        has_relative_attention_bias (bool, optional): If ``True``, apply relative position embedding.
            Necessary in the first encoder layer, but not in the subsequent ones. (Default: ``False``)
        num_buckets (int, optional): Number of buckets for relative position embedding. (Default: ``32``)
        max_distance (int, optional): Naximum distance for relative position embedding. (Default: ``128``)
        gru_rel_pos (bool, optional): If ``True``, apply gated relative position embedding. (Default: ``False``)
    """

    def __init__(
        self,
        embed_dim: int,
        total_num_heads: int,
        remaining_heads: Optional[List[int]] = None,
        dropout: float = 0.0,
        bias: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = True,
        prune_heads: bool = False,
        prune_layer: bool = False,
        ):
        # 初始化总头数
        self.total_num_heads = total_num_heads
        # 如果剩余头部为空，则初始化剩余头部为总头部的索引列表
        if remaining_heads is None:
            self.remaining_heads = list(range(total_num_heads))
        else:
            self.remaining_heads = remaining_heads  # list of indices  # 否则使用给定的剩余头部索引列表

        # 计算每个头部的维度
        self.head_dim = embed_dim // total_num_heads

        # 调用父类的初始化方法，传入相关参数
        super().__init__(embed_dim, len(self.remaining_heads), self.head_dim, dropout, prune_heads, prune_layer)

        # 设置是否具有相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # 如果具有相对注意力偏置，则创建相对注意力嵌入层
        if has_relative_attention_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, total_num_heads)
        else:
            self.rel_attn_embed = None

        # 重写线性层以自定义偏置
        self.k_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(len(self.remaining_heads) * self.head_dim, embed_dim, bias=bias)

        # 设置是否使用相对位置编码
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            # 如果使用相对位置编码，则初始化相关参数
            self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, total_num_heads, 1, 1))
        # 设置是否具有位置偏置
        self.has_position_bias = True
    # 计算相对位置嵌入，用于 WavLM 模型
    def compute_bias(self, query_length: int, key_length: int) -> Tensor:
        """Compute relative position embeddings for WavLM model.
        Args:
            query_length (int): Query position can take values between 0 and ``query_length - 1``.
            key_length (int): Key position can take values between 0 and ``key_length - 1``.
        Returns:
            Tensor of shape `(num_heads, query_length, key_length)`, relative positions embeddings
        """
        # 创建一个包含从 0 到 query_length-1 的整数的张量，用于表示查询位置
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        # 创建一个包含从 0 到 key_length-1 的整数的张量，用于表示键位置
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        # 计算相对位置，形状为 (query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置映射到桶中，使用双向映射
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        # 将相对位置桶转换到与 rel_attn_embed 权重相同的设备上
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        # 使用 rel_attn_embed 权重计算相对位置嵌入，形状为 (query_length, key_length, num_heads)
        values = self.rel_attn_embed(relative_position_bucket)
        # 调整张量的维度顺序，形状变为 (num_heads, query_length, key_length)
        values = values.permute([2, 0, 1])
        # 返回相对位置嵌入
        return values
    def _relative_positions_bucket(self, relative_positions: Tensor, bidirectional: bool = True):
        """Compute relative position buckets for WavLM model. Computation similar to formula (5) in WavLM
           paper :cite:`chen2022wavlm`.
        Args:
            relative_positions (Tensor): Relative offsets between query and key positions,
                of shape ``(query_length, key_length)``.
            bidirectional (bool): If ``True``, values will be filled both above and below the diagonal in the resulting
                matrix. If ``False``, the elements above the diagonal (i.e. with negative relative offsets) will be set
                to zero. (Default ``True``)
        Returns:
            Tensor of shape ``(query_length, key_length)`` filled bucketed values of with relative positions.
        """
        num_buckets = self.num_buckets  # 获取相对位置桶的数量
        max_distance = self.max_distance  # 获取最大距离

        # Shape (query_length, key_length)
        relative_buckets = torch.zeros_like(relative_positions, dtype=torch.long)  # 创建与relative_positions相同形状的全零张量

        if bidirectional:  # 如果是双向的
            num_buckets = num_buckets // 2  # 更新桶的数量
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets  # 根据相对位置的正负情况，更新相对桶的值
            relative_positions = torch.abs(relative_positions)  # 取相对位置的绝对值
        else:  # 如果是单向的
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))  # 将相对位置中的负值设为0

        max_exact = num_buckets // 2  # 计算最大精确值
        is_small = relative_positions < max_exact  # 判断相对位置是否小于最大精确值

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)  # 根据公式计算大于最大精确值的相对位置对应的桶值
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )  # 将计算得到的桶值限制在桶的范围内

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)  # 根据相对位置大小，选择填充相对桶的值
        return relative_buckets  # 返回填充后的相对桶
    # 定义一个前向传播函数，接受查询张量和可选的注意力掩码、位置偏置、键填充掩码作为输入
    def forward(
        self,
        query: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    # 定义一个修剪函数
    def prune(self):
        # 创建一个新的配置字典，包含使用注意力和剩余头数
        new_config = {
            "use_attention": True,
            "remaining_heads": self.remaining_heads,
        }
        # 如果存在层级硬混凝土掩码
        if self.hard_concrete_for_layer is not None:
            # 确保层级硬混凝土掩码不处于训练状态
            assert not self.hard_concrete_for_layer.training
            # 获取层级掩码
            layer_mask = self.hard_concrete_for_layer() # (1,)
            # 将输出投影层的权重和偏置乘以层级掩码
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            # 如果层级掩码为0，则更新配置字典中的使用注意力为False
            if layer_mask == 0:
                new_config["use_attention"] = False
            # 将层级硬混凝土掩码设置为None
            self.hard_concrete_for_layer = None

        # 如果存在头部硬混凝土掩码
        if self.hard_concrete_for_heads is not None:
            # 确保头部硬混凝土掩码不处于训练状态
            assert not self.hard_concrete_for_heads.training
            # 获取头部掩码
            head_mask = self.hard_concrete_for_heads()  # (num_heads,)
            # 更新配置字典中的剩余头数，将头部掩码中非零元素的索引添加到剩余头数列表中
            new_config["remaining_heads"] = head_mask.nonzero().squeeze(-1).tolist()
            # 如果剩余头数为0，则更新配置字典中的使用注意力为False
            if len(new_config["remaining_heads"]) == 0:
                new_config["use_attention"] = False
            else:
                # 将头部掩码重复扩展到与头维度相同的长度
                full_mask = head_mask.repeat_interleave(self.head_dim)
                # 获取全掩码中非零元素的索引
                full_index = full_mask.nonzero().squeeze(-1)  # 1D

                # 对k_proj、v_proj、q_proj进行线性层修剪
                prune_linear_layer(self.k_proj, full_index, "output")
                prune_linear_layer(self.v_proj, full_index, "output")
                prune_linear_layer(self.q_proj, full_index, "output")

                # 将输出投影层的权重乘以全掩码
                self.out_proj.weight.data *= full_mask
                # 对输出投影层进行线性层修剪
                prune_linear_layer(self.out_proj, full_index, "input")
            # 将头部硬混凝土掩码设置为None
            self.hard_concrete_for_heads = None

        # 返回新的配置字典
        return new_config
class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
        prune_intermediate: bool = False,
        prune_layer: bool = False,
    ):
        # 调用父类的构造函数
        super().__init__()
        # 创建一个全连接层，输入特征数为io_features，输出特征数为intermediate_features
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        # 创建一个dropout层，用于中间层的输出
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        # 创建一个全连接层，输入特征数为intermediate_features，输出特征数为io_features
        self.output_dense = nn.Linear(intermediate_features, io_features)
        # 创建一个dropout层，用于输出层的输出
        self.output_dropout = nn.Dropout(output_dropout)

        # 如果需要对中间层进行剪枝
        if prune_intermediate:
            # 创建一个HardConcrete对象，用于中间层的剪枝
            self.hard_concrete_for_intermediate = HardConcrete(
                n_in=intermediate_features, init_mean=0.5
            )
        else:
            self.hard_concrete_for_intermediate = None
        
        # 如果需要对整个层进行剪枝
        if prune_layer:
            # 创建一个HardConcrete对象，用于整个层的剪枝
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        # 中间层的全连接操作
        x = self.intermediate_dense(x)
        # 使用GELU激活函数
        x = torch.nn.functional.gelu(x)
        # 中间层的dropout操作
        x = self.intermediate_dropout(x)

        # 如果需要对中间层进行剪枝
        if self.hard_concrete_for_intermediate is not None:
            # 生成中间层的掩码
            intermediate_mask = self.hard_concrete_for_intermediate()   # (intermediate_features,)
            # 对中间层的输出应用掩码
            x = x * intermediate_mask

        # 输出层的全连接操作
        x = self.output_dense(x)
        # 输出层的dropout操作
        x = self.output_dropout(x)

        # 如果需要对整个层进行剪枝
        if self.hard_concrete_for_layer is not None:
            # 生成整个层的掩码
            layer_mask = self.hard_concrete_for_layer()     # (1,)
            # 对整个层的输出应用掩码
            x = x * layer_mask

        return x
    # 获取模型参数的数量
    def get_num_params(self):
        # 获取输入层的特征数量
        io_features = self.intermediate_dense.in_features
        # 如果存在用于中间层的 hard concrete mask，则获取中间层的特征数量
        if self.hard_concrete_for_intermediate is not None:
            intermediate_features = self.hard_concrete_for_intermediate.l0_norm()
        else:
            intermediate_features = self.intermediate_dense.out_features
        # 计算参数数量
        num_params = (io_features + 1) * intermediate_features + (intermediate_features + 1) * io_features

        # 如果存在用于层的 hard concrete mask，则根据其 l0 范数调整参数数量
        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        
        return num_params
    
    # 剪枝操作
    def prune(self):
        # 创建新的配置字典
        new_config = {
            "use_feed_forward": True,
            "ff_interm_features": self.intermediate_dense.out_features
        }
        # 如果存在用于层的 hard concrete mask
        if self.hard_concrete_for_layer is not None:
            # 确保 hard concrete mask 不处于训练状态
            assert not self.hard_concrete_for_layer.training
            # 获取层的 mask
            layer_mask = self.hard_concrete_for_layer()
            # 根据 mask 剪枝输出层的权重和偏置
            self.output_dense.weight.data *= layer_mask
            self.output_dense.bias.data *= layer_mask
            # 如果 mask 为 0，则更新配置字典
            if layer_mask == 0:
                new_config["use_feed_forward"] = False
            # 将用于层的 hard concrete mask 置为 None
            self.hard_concrete_for_layer = None

        # 如果存在用于中间层的 hard concrete mask
        if self.hard_concrete_for_intermediate is not None:
            # 确保 hard concrete mask 不处于训练状态
            assert not self.hard_concrete_for_intermediate.training
            # 获取中间层的 mask
            interm_mask = self.hard_concrete_for_intermediate()
            # 获取非零元素的索引并更新配置字典中的特征数量
            interm_index = interm_mask.nonzero().squeeze(-1)    # NOTE: must specify dim=-1
            new_config["ff_interm_features"] = len(interm_index)
            # 如果特征数量为 0，则更新配置字典
            if new_config["ff_interm_features"] == 0:
                new_config["use_feed_forward"] = False
            else:
                # 根据 mask 剪枝中间层和输出层的权重
                prune_linear_layer(self.intermediate_dense, interm_index, "output")
                self.output_dense.weight.data *= interm_mask
                prune_linear_layer(self.output_dense, interm_index, "input")
            # 将用于中间层的 hard concrete mask 置为 None
            self.hard_concrete_for_intermediate = None

        return new_config
class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: Optional[Module],    # 可选的注意力模块，如果整个层被修剪掉则为None
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Optional[Module], # 可选的前馈模块，如果整个层被修剪掉则为None
        embed_dim: int,
    ):
        super().__init__()
        self.attention = attention  # 设置注意力模块
        self.dropout = nn.Dropout(dropout)  # 设置dropout层
        self.layer_norm = nn.LayerNorm(embed_dim)  # 设置层归一化
        self.layer_norm_first = layer_norm_first  # 设置是否先进行层归一化
        self.feed_forward = feed_forward  # 设置前馈模块
        self.final_layer_norm = nn.LayerNorm(embed_dim)  # 设置最终层归一化
        self.embed_dim = embed_dim  # 设置嵌入维度

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,  # 可选的注意力掩码
        position_bias: Optional[Tensor] = None,  # 可选的位置偏置
        key_padding_mask: Optional[Tensor] = None,  # 可选的键填充掩码
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): 输入张量，形状为 ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): 注意力掩码，形状为 ``(batch, 1, sequence_length, sequence_length)``。 (默认值: ``None``)
            position_bias (Tensor or ``None``, optional): 位置偏置，形状为 ``(batch_size * num_heads, src_len, src_len)``。
                仅对 WavLM 模型必要，否则为 ``None``。 (默认值: ``None``)
            key_padding_mask (Tensor or ``None``, optional): 键填充掩码，形状为 ``(batch_size, src_len)``。
                仅对 WavLM 模型使用，否则忽略。 (默认值: ``None``)
        Returns:
            (x, position_bias): 返回值的形状与输入相同。位置偏置仅对 WaLM 模型相关，否则为 ``None``。
        """
        if self.attention is not None:
            residual = x

            if self.layer_norm_first:
                x = self.layer_norm(x)

            x, position_bias = self.attention(
                x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
            )

            x = self.dropout(x)
            x = residual + x

        if self.layer_norm_first:
            if self.feed_forward is not None:
                x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            # 注意：对于后置层归一化，即使层被修剪，层归一化也应始终应用。
            x = self.layer_norm(x)
            if self.feed_forward is not None:
                x = x + self.feed_forward(x)
            x = self.final_layer_norm(x)
        return x, position_bias
    # 获取模型参数的数量
    def get_num_params(self):
        # 计算嵌入维度乘以2再乘以2的结果，表示两个层归一化的参数数量
        num_params = self.embed_dim * 2 * 2
        # 如果存在注意力机制模块，则累加其参数数量
        if self.attention is not None:
            num_params += self.attention.get_num_params()
        # 如果存在前馈神经网络模块，则累加其参数数量
        if self.feed_forward is not None:
            num_params += self.feed_forward.get_num_params()
        # 返回总的参数数量
        return num_params
# 定义一个名为 Transformer 的类，继承自 Module 类
class Transformer(Module):
    # 初始化方法，接受位置卷积嵌入、dropout、层、是否先进行层归一化、层丢弃率等参数
    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将位置卷积嵌入赋值给实例变量 pos_conv_embed
        self.pos_conv_embed = pos_conv_embed
        # 创建一个具有与位置卷积嵌入相同维度的 LayerNorm 层，并赋值给实例变量 layer_norm
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        # 将是否先进行层归一化的标志赋值给实例变量 layer_norm_first
        self.layer_norm_first = layer_norm_first
        # 将层丢弃率赋值给实例变量 layer_drop
        self.layer_drop = layer_drop
        # 创建一个具有指定丢弃率的 Dropout 层，并赋值给实例变量 dropout
        self.dropout = nn.Dropout(dropout)
        # 将层列表赋值给实例变量 layers
        self.layers = layers

    # 定义一个名为 _preprocess 的方法，接受输入张量 x
    def _preprocess(self, x: Tensor):
        # 将输入张量与位置卷积嵌入的结果相加
        x = x + self.pos_conv_embed(x)

        # 如果设置了先进行层归一化
        if self.layer_norm_first:
            # 对输入张量进行层归一化
            x = self.layer_norm(x)

        # 对输入张量进行丢弃操作
        x = self.dropout(x)
        # 返回处理后的张量
        return x

    # 定义前向传播方法，接受输入张量 x、注意力掩码、位置偏置等参数
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        # 对输入张量进行预处理
        x = self._preprocess(x)
        # 遍历层列表
        for layer in self.layers:
            # 如果不处于训练状态或者随机数小于等于层丢弃率
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                # 调用层的前向传播方法，更新输入张量和位置偏置
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)

        # 如果没有设置先进行层归一化
        if not self.layer_norm_first:
            # 对输入张量进行层归一化
            x = self.layer_norm(x)
        # 返回处理后的张量
        return x

    # 定义获取中间输出的方法，接受输入张量 x、注意力掩码、层数、位置偏置等参数
    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
        position_bias: Optional[Tensor] = None,
    ) -> List[Tensor]:
        # 如果设置了层数
        if num_layers is not None:
            # 如果层数不在有效范围内，抛出数值错误
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        # 创建一个空列表用于存储中间输出
        ret: List[Tensor] = []
        # 对输入张量进行预处理
        x = self._preprocess(x)
        # 遍历层列表
        for layer in self.layers:
            # 调用层的前向传播方法，更新输入张量和位置偏置
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            # 将中间输出添加到列表中
            ret.append(x)
            # 如果设置了层数并且列表长度大于等于层数，返回列表
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        # 返回列表
        return ret
    # 获取模型参数的数量
    def get_num_params(self):
        # 计算位置卷积嵌入和层归一化的参数数量
        num_params = sum(p.numel() for p in self.pos_conv_embed.parameters()) + self.pos_conv_embed.embed_dim * 2
        # 遍历每个层，获取参数数量并累加
        for layer in self.layers:
            num_params += layer.get_num_params()
        # 返回总参数数量
        return num_params
    
    # 对模型进行剪枝
    def prune(self):
        # 创建一个新的配置字典
        new_config = defaultdict(list)
        # 遍历每个层
        for layer in self.layers:
            # 对注意力层进行剪枝，并更新配置字典
            attention_config = layer.attention.prune()
            new_config["use_attention"].append(attention_config["use_attention"])
            if "remaining_heads" in attention_config:
                new_config["remaining_heads"].append(attention_config["remaining_heads"])
            else:
                new_config["num_heads"].append(attention_config["num_heads"])

            # 如果不使用注意力机制，则将注意力层置为None
            if not attention_config["use_attention"]:
                layer.attention = None
            
            # 对前馈层进行剪枝，并更新配置字典
            ff_config = layer.feed_forward.prune()
            new_config["use_feed_forward"].append(ff_config["use_feed_forward"])
            new_config["ff_interm_features"].append(ff_config["ff_interm_features"])
            # 如果不使用前馈层，则将前馈层置为None
            if not ff_config["use_feed_forward"]:
                layer.feed_forward = None
        
        # 返回新的配置字典
        return new_config
# 定义一个编码器类，继承自 Module 类
class Encoder(Module):
    # 初始化方法，接受特征投影和变换器两个模块作为参数
    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ):
        super().__init__()
        # 将特征投影模块和变换器模块保存为类的属性
        self.feature_projection = feature_projection
        self.transformer = transformer

    # 定义一个内部方法，用于预处理输入特征
    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # 使用特征投影模块处理输入特征
        x = self.feature_projection(features)

        # 初始化 mask 为 None
        mask: Optional[Tensor] = None
        # 如果存在长度信息
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # 创建用于填充元素的掩码，并将它们置零
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # 将掩码扩展到注意力形状，并设置权重
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    # 前向传播方法，接受输入特征和长度信息，返回处理后的特征
    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        # 调用内部预处理方法
        x, mask = self._preprocess(features, lengths)
        # 使用变换器模块处理特征并返回结果
        x = self.transformer(x, attention_mask=mask)
        return x

    # 提取特征方法，接受输入特征和长度信息，返回多层特征的列表
    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        # 调用内部预处理方法
        x, masks = self._preprocess(features, lengths)
        # 获取变换器模块的中间输出并返回结果
        interm = self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)
        return [x] + interm
    
    # 获取模型参数数量的方法，接受输入特征维度，返回模型参数数量
    def get_num_params(self, in_features):
        """Calculate the current model size."""
        # 获取特征投影模块的参数数量
        feature_projection_size = self.feature_projection.get_num_params(in_features)
        # 获取变换器模块的参数数量
        transformer_size = self.transformer.get_num_params()
        # 返回特征投影模块和变换器模块参数数量之和
        return feature_projection_size + transformer_size
    # 对子模块进行原地修剪
    def prune(self, conv_out_index):
        """In-place pruning of submodules."""
        # 对特征投影层的 LayerNorm 进行修剪
        prune_layer_norm(self.feature_projection.layer_norm, conv_out_index)
        # 对特征投影层的投影层进行修剪
        prune_linear_layer(self.feature_projection.projection, conv_out_index, "input")
        # 调用transformer的修剪方法，并返回transformer的配置
        transformer_config = self.transformer.prune()
        # 返回transformer的配置
        return transformer_config
################################################################################
# 定义一个私有函数，用于获取特征提取器
def _get_feature_extractor(
    norm_mode: str,  # 规范化模式，可以是"group_norm"或"layer_norm"
    shapes: List[Tuple[int, int, int]],  # 卷积层的配置，包括输出通道数、卷积核大小和步长
    bias: bool,  # 是否在每个卷积操作中包含偏置项
    prune_conv_channels: bool = False,  # 是否修剪卷积通道，默认为False
) -> FeatureExtractor:  # 返回类型为FeatureExtractor类的对象
    """
    Args:
        norm_mode (str):
            Either "group_norm" or "layer_norm".
            If "group_norm", then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.
            This option corresponds to "extractor_mode" from fairseq.
            Expected values are "group_norm" for Base arch, and
            "layer_norm" for Large arch.
        shapes (list of tuple of int):
            Configuration of convolution layers. List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``
            This option corresponds to "conv_feature_layers" from fairseq.
            Expected values are
            ``[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2``
            for all the architectures.
        bias (bool):
            Whether to include bias term to each convolution operation.
            This option corresponds to "conv_bias" from fairseq.
            Expected values are False for Base arch, and True for Large arch.
    """
    # 参考链接，指向原始实现和配置文件中的相关部分
    # 如果 norm_mode 不是 "group_norm" 或 "layer_norm"，则抛出数值错误
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    # 初始化一个空的列表用于存储块
    blocks = []
    # 初始化输入通道数为 1
    in_channels = 1
    # 遍历 shapes 列表，获取索引 i 和元组 (out_channels, kernel_size, stride)
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        # 初始化 normalization 变量
        normalization = None
        # 如果 norm_mode 为 "group_norm" 并且索引 i 为 0
        if norm_mode == "group_norm" and i == 0:
            # 创建 GroupNorm 归一化层对象
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        # 如果 norm_mode 为 "layer_norm"
        elif norm_mode == "layer_norm":
            # 创建 LayerNorm 归一化层对象
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        # 将 ConvLayerBlock 对象添加到 blocks 列表中
        blocks.append(
            ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
                prune_conv_channels=prune_conv_channels,
            )
        )
        # 更新 in_channels 为当前 out_channels
        in_channels = out_channels
    # 返回 FeatureExtractor 对象，其中包含 ConvLayerBlock 对象列表
    return FeatureExtractor(nn.ModuleList(blocks))
# 定义一个函数，用于获取编码器对象
def _get_encoder(
    # 输入特征的维度
    in_features: int,
    # 嵌入维度
    embed_dim: int,
    # 输入层的丢弃率
    dropout_input: float,
    # 位置卷积的卷积核大小
    pos_conv_kernel: int,
    # 位置卷积的分组数
    pos_conv_groups: int,
    # 编码器的层数
    num_layers: int,
    # 是否使用注意力机制的列表
    use_attention: List[bool],
    # 是否使用前馈网络的列表
    use_feed_forward: List[bool],
    # 注意力头的数量的列表
    num_heads: List[int],
    # 注意力头的维度
    head_dim: int,
    # 注意力机制的丢弃率
    attention_dropout: float,
    # 前馈网络中间特征的列表
    ff_interm_features: List[int],
    # 前馈网络中间层的丢弃率
    ff_interm_dropout: float,
    # 整体的丢弃率
    dropout: float,
    # 是否在层归一化之前应用层归一化
    layer_norm_first: bool,
    # 层的丢弃率
    layer_drop: float,
    # 是否修剪注意力头
    prune_attention_heads: bool = False,
    # 是否修剪注意力层
    prune_attention_layer: bool = False,
    # 是否修剪前馈网络中间特征
    prune_feed_forward_intermediate: bool = False,
    # 是否修剪前馈网络层
    prune_feed_forward_layer: bool = False,
) -> Encoder:
    """
    获取编码器对象的函数
    """
    # 特征投影层
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    # 位置卷积层
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # 原始实现
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    # 编码器层列表
    encoder_layers = nn.ModuleList()
    # 遍历层数范围内的索引
    for idx in range(num_layers):
        # 如果使用注意力机制
        if use_attention[idx]:
            # 创建自注意力层对象
            attention = SelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads[idx],
                head_dim=head_dim,
                dropout=attention_dropout,
                prune_heads=prune_attention_heads,
                prune_layer=prune_attention_layer,
            )
        else:
            # 如果不使用注意力机制，则设为None
            attention = None
        # 如果使用前馈神经网络
        if use_feed_forward[idx]:
            # 创建前馈神经网络对象
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[idx],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
                prune_intermediate=prune_feed_forward_intermediate,
                prune_layer=prune_feed_forward_layer,
            )
        else:
            # 如果不使用前馈神经网络，则设为None
            feed_forward = None
        # 将创建的注意力和前馈神经网络对象添加到编码器层列表中
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    # 创建变压器对象，包括位置卷积嵌入、丢弃率、编码器层列表、是否先进行层归一化、层丢弃率
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    # 返回编码器对象
    return Encoder(feature_projection, transformer)
# 构建 WavLM 模型的编码器
# 参数包括输入特征数、嵌入维度、输入丢弃率、位置卷积核大小、位置卷积分组数、层数、是否使用注意力、是否使用前馈网络、总注意力头数、剩余头数、桶数、最大距离、注意力丢弃率、前馈网络中间特征数、前馈网络中间丢弃率、总体丢弃率、是否先进行层归一化、层丢弃率、是否修剪注意力头、是否修剪注意力层、是否修剪前馈网络中间、是否修剪前馈网络层
def _get_wavlm_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    use_attention: List[bool],
    use_feed_forward: List[bool],
    total_num_heads: List[int],
    remaining_heads: List[List[int]],
    num_buckets: int,
    max_distance: int,
    attention_dropout: float,
    ff_interm_features: List[int],
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
    prune_attention_heads: bool = False,
    prune_attention_layer: bool = False,
    prune_feed_forward_intermediate: bool = False,
    prune_feed_forward_layer: bool = False,
) -> Encoder:
    """
    构建 WavLM 模型的编码器。编码器的结构和大部分参数与 get_encoder 相同，因此请参考那里的文档。与 Wav2Vec2 编码器的唯一区别是使用 WavLMSelfAttention 而不是 SelfAttention，并且有两个额外的参数：num_buckets 和 max_distance。
    """
    # 定义函数参数，描述每个参数的含义和用途
    Args:
        in_features (int): 用于获取编码器的输入特征
        embed_dim (int): 用于获取编码器的嵌入维度
        dropout_input (float): 用于获取编码器的输入丢弃率
        pos_conv_kernel (int): 用于获取编码器的位置卷积核大小
        pos_conv_groups (int): 用于获取编码器的位置卷积分组数
        num_layers (int): 用于获取编码器的层数
        num_heads (int): 用于获取编码器的注意力头数
        num_buckets (int): 相对位置嵌入的桶数
        max_distance (int): 相对位置嵌入的最大距离
        attention_dropout (float): 用于获取编码器的注意力丢弃率
        ff_interm_features (int): 用于获取编码器的前馈中间特征数
        ff_interm_dropout (float): 用于获取编码器的前馈中间丢弃率
        dropout (float): 用于获取编码器的丢弃率
        layer_norm_first (bool): 用于获取编码器的层归一化顺序
        layer_drop (float): 用于获取编码器的层丢弃率

    """
    # 创建特征投影对象，用于将输入特征投影到嵌入维度
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    # 创建卷积位置嵌入对象，用于处理位置信息
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # 原始实现
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    # 创建一个空的神经网络模块列表，用于存储编码器的各个层
    encoder_layers = nn.ModuleList()
    # 遍历编码器层的数量
    for i in range(num_layers):
        # 如果使用注意力机制
        if use_attention[i]:
            # 创建 WavLMSelfAttention 对象
            attention = WavLMSelfAttention(
                embed_dim=embed_dim,
                total_num_heads=total_num_heads[i],
                remaining_heads=remaining_heads[i],
                dropout=attention_dropout,
                has_relative_attention_bias=(i == 0),  # 仅在第一层需要位置嵌入
                num_buckets=num_buckets,
                max_distance=max_distance,
                prune_heads=prune_attention_heads,
                prune_layer=prune_attention_layer,
            )
        else:
            # 如果不使用注意力机制，则设为 None
            attention = None
        # 如果使用前馈网络
        if use_feed_forward[i]:
            # 创建 FeedForward 对象
            feed_forward = FeedForward(
                io_features=embed_dim,
                intermediate_features=ff_interm_features[i],
                intermediate_dropout=ff_interm_dropout,
                output_dropout=dropout,
                prune_intermediate=prune_feed_forward_intermediate,
                prune_layer=prune_feed_forward_layer,
            )
        else:
            # 如果不使用前馈网络，则设为 None
            feed_forward = None
        # 将注意力机制和前馈网络添加到编码器层列表中
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
                embed_dim=embed_dim,
            )
        )
    # 创建 Transformer 对象
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    # 返回编码器对象
    return Encoder(feature_projection, transformer)
# 生成填充掩码，给定填充输入和长度张量
def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    # 获取输入的批大小、最大长度和频率
    batch_size, max_len, _ = input.shape
    # 生成填充掩码，根据长度张量和最大长度
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    # 返回掩码
    return mask


# 定义一个自定义的 PyTorch 自动求导函数
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        # 保存缩放因子到上下文中
        ctx.scale = scale
        # 创建一个新的张量作为结果
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        # 返回梯度乘以缩放因子，以及 None（因为输入是两个参数）
        return grad * ctx.scale, None
```