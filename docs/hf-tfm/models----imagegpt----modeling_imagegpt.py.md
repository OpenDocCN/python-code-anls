# `.\models\imagegpt\modeling_imagegpt.py`

```
# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and HuggingFace Inc. team.
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
"""PyTorch OpenAI ImageGPT model."""

import math  # 导入数学函数库
import os  # 导入操作系统功能
import warnings  # 导入警告处理模块
from typing import Any, Optional, Tuple, Union  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint模块
from torch import nn  # 导入PyTorch的神经网络模块
from torch.cuda.amp import autocast  # 导入PyTorch的混合精度训练模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入PyTorch的损失函数模块

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import (  # 导入模型输出相关模块
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer  # 导入PyTorch工具函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入工具函数和日志模块
from .configuration_imagegpt import ImageGPTConfig  # 导入ImageGPT模型的配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "openai/imagegpt-small"  # ImageGPT模型的checkpoint位置，用于文档说明
_CONFIG_FOR_DOC = "ImageGPTConfig"  # ImageGPT模型的配置类，用于文档说明

IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练的ImageGPT模型列表
    "openai/imagegpt-small",
    "openai/imagegpt-medium",
    "openai/imagegpt-large",
    # See all Image GPT models at https://huggingface.co/models?filter=imagegpt
]


def load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path):
    """
    Load tf checkpoints in a pytorch model
    """
    try:
        import re  # 导入正则表达式模块
        import tensorflow as tf  # 导入TensorFlow库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(imagegpt_checkpoint_path)  # 获取TensorFlow模型的绝对路径
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))  # 记录日志，指示正在转换TensorFlow的checkpoint
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)  # 获取TensorFlow模型中的变量列表
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))  # 记录日志，指示正在加载TensorFlow模型的权重和形状
        array = tf.train.load_variable(tf_path, name)  # 加载TensorFlow模型中的变量
        names.append(name)
        arrays.append(array.squeeze())  # 将加载的变量压缩并添加到数组中

    return model  # 返回加载了TensorFlow权重的PyTorch模型


class ImageGPTLayerNorm(nn.Module):
    def __init__(self, hidden_size: Tuple[int], eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 设置层标准化的epsilon值
        self.weight = nn.Parameter(torch.Tensor(hidden_size))  # 初始化标准化的权重参数
    # 定义一个方法 `forward`，接受一个 torch.Tensor 类型的参数 `tensor`，返回一个元组
    def forward(self, tensor: torch.Tensor) -> tuple:
        # input is not mean centered
        # 返回值是输入张量 `tensor` 除以标准差，然后乘以权重数据 `self.weight.data[..., :]`，以实现标准化处理
        return (
            tensor
            / torch.sqrt(torch.mean(torch.square(tensor), axis=-1, keepdim=True) + self.eps)
            * self.weight.data[..., :]
        )
# 定义一个名为 ImageGPTAttention 的类，继承自 nn.Module
class ImageGPTAttention(nn.Module):
    # 初始化函数，接受配置参数 config 和两个可选参数 is_cross_attention 和 layer_idx
    def __init__(self, config, is_cross_attention: Optional[bool] = False, layer_idx: Optional[int] = None):
        # 调用父类的初始化函数
        super().__init__()

        # 获取最大位置嵌入数
        max_positions = config.max_position_embeddings
        # 注册一个缓冲区 "bias"，包含一个下三角形状的张量，用于自注意力机制
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册一个缓冲区 "masked_bias"，包含一个很大的负数，用于掩码注意力
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        # 获取嵌入维度
        self.embed_dim = config.hidden_size
        # 获取注意力头数和每个头的维度
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        # 检查 embed_dim 必须能够被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 是否对注意力权重进行缩放
        self.scale_attn_weights = config.scale_attn_weights
        # 是否是跨注意力机制
        self.is_cross_attention = is_cross_attention

        # 层级注意力缩放、重排序和上投
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果是跨注意力机制，定义两个卷积层
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            # 如果不是跨注意力机制，定义一个卷积层
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        # 定义一个卷积层 c_proj
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # 注意力和残差的 Dropout 层
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 初始化被修剪的注意力头集合
        self.pruned_heads = set()

    # 函数 prune_heads，用于修剪不需要的注意力头
    def prune_heads(self, heads):
        # 如果 heads 集合为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可修剪的头部和索引
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        # 构造索引张量
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # 在卷积层上进行修剪
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # 更新超参数
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    # 定义注意力函数，接受查询(query)、键(key)、值(value)以及注意力掩码和头掩码作为输入
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 计算注意力权重，使用query与key的矩阵乘积
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # 如果开启了注意力权重的缩放
        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        # 如果开启了逐层注意力权重的缩放
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # 如果不是交叉注意力模式
        if not self.is_cross_attention:
            # 如果只有“正常”注意力层实现了因果遮罩
            query_length, key_length = query.size(-2), key.size(-2)
            # 创建因果遮罩，遮罩值设置为极小值以确保被遮罩区域的权重为负无穷大
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # 将遮罩值转换为与attn_weights相同的数据类型，并放置在相同的设备上
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            # 使用遮罩更新注意力权重，被遮罩区域用mask_value填充
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # 如果有注意力掩码，则应用该掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行softmax归一化
        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # 将注意力权重降回到值(value)的数据类型（如果在混合精度下使用），否则无操作
        attn_weights = attn_weights.type(value.dtype)

        # 对注意力权重应用dropout操作
        attn_weights = self.attn_dropout(attn_weights)

        # 如果指定了头掩码，则应用头掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出，使用softmax归一化后的注意力权重与值(value)的矩阵乘积
        attn_output = torch.matmul(attn_weights, value)

        # 返回注意力输出和注意力权重
        return attn_output, attn_weights
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)

        # Extract batch size, number of heads, query sequence length, and dk (last dimension of query)
        bsz, num_heads, q_seq_len, dk = query.size()

        # Extract key sequence length
        _, _, k_seq_len, _ = key.size()

        # Preallocate attention weights tensor for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute scale factor based on configuration settings
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (scale K by 1 / root(dk))
        with autocast(enabled=False):
            # Reshape query and key tensors for matrix multiplication
            q = query.reshape(-1, q_seq_len, dk)
            k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            
            # Perform batched matrix multiplication with `baddbmm`, scaling with alpha and adding to attn_weights
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            
            # Reshape attn_weights back to original shape (batch size, num heads, query seq length, key seq length)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # Apply causal mask for "normal" attention layer
            query_length, key_length = query.size(-2), key.size(-2)
            
            # Generate causal mask tensor from self.bias for current query and key lengths
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            
            # Set mask value to minimum float value of attn_weights' dtype
            mask_value = torch.finfo(attn_weights.dtype).min
            
            # Create mask_value tensor of attn_weights' dtype and device
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            
            # Apply causal_mask to attn_weights, replacing values where causal_mask is False with mask_value
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply attention_mask to attn_weights
            attn_weights = attn_weights + attention_mask

        # Apply softmax along the last dimension of attn_weights
        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # Ensure attn_weights is of type torch.float32
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        
        # Convert attn_weights to type value.dtype
        attn_weights = attn_weights.type(value.dtype)
        
        # Apply attention dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Apply head_mask if provided
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # Compute attention output by matrix multiplication of attn_weights and value tensors
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # Permute dimensions to merge attn_head_size and num_heads dimensions
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # Calculate new shape by merging num_heads and attn_head_size into the last dimension
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tuple:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                # Raise an error if `q_attn` weights are not defined for cross-attention usage
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `ImageGPTAttention(..., is_cross_attention=True)`."
                )

            # Compute query using self.q_attn module
            query = self.q_attn(hidden_states)
            # Compute key and value from encoder_hidden_states using self.c_attn module and split them
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            # Use encoder_attention_mask for attention masking
            attention_mask = encoder_attention_mask
        else:
            # Compute query, key, and value from hidden_states using self.c_attn module and split them
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # Split query, key, and value tensors into multiple heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Concatenate past_key with key and past_value with value if layer_past is not None
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # Store key and value tensors in present if use_cache is True
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # Perform attention calculation based on self.reorder_and_upcast_attn flag
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Merge heads back into the hidden_size dimension
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # Project merged heads using self.c_proj module
        attn_output = self.c_proj(attn_output)
        # Apply residual dropout
        attn_output = self.resid_dropout(attn_output)

        # Prepare outputs tuple including attn_output and present
        outputs = (attn_output, present)
        # Include attention weights in outputs if output_attentions is True
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
# 定义一个基于图像的 GPT 模型的单个 MLP 层
class ImageGPTMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # 定义一个一维卷积层，输入维度是 hidden_size，输出维度是 intermediate_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 定义一个一维卷积层，输入维度是 intermediate_size，输出维度是 hidden_size
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 激活函数根据配置选择，将激活函数映射到 ACT2FN 中对应的函数
        self.act = ACT2FN[config.activation_function]
        # Dropout 层，以 config 中配置的概率进行随机丢弃
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 先通过 c_fc 卷积层处理隐藏状态
        hidden_states = self.c_fc(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 再通过 c_proj 卷积层处理输出
        hidden_states = self.c_proj(hidden_states)
        # 应用 Dropout 进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个基于图像的 GPT 模型的单个 Block
class ImageGPTBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        # 如果没有配置内部维度，则使用默认的 4 倍 hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # 图像 GPT 特有的 LayerNorm 层，用于归一化隐藏状态
        self.ln_1 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 图像 GPT 特有的 Attention 层
        self.attn = ImageGPTAttention(config, layer_idx=layer_idx)
        # 再次应用 LayerNorm 层
        self.ln_2 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 如果配置中包含交叉注意力，添加交叉注意力层和 LayerNorm 层
        if config.add_cross_attention:
            self.crossattention = ImageGPTAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # MLP 层，用于处理内部维度
        self.mlp = ImageGPTMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tuple:
        residual = hidden_states  # 保存输入的隐藏状态作为残差连接的一部分
        hidden_states = self.ln_1(hidden_states)  # 应用 LayerNorm 到隐藏状态
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # 提取注意力机制的输出
        outputs = attn_outputs[1:]  # 提取其他输出，如 present, (attentions)
        # 残差连接
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # 为交叉注意力添加一个自注意力块
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states  # 保存交叉注意力前的隐藏状态作为残差连接的一部分
            hidden_states = self.ln_cross_attn(hidden_states)  # 应用 LayerNorm 到隐藏状态
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]  # 提取交叉注意力机制的输出
            # 残差连接
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # 如果输出注意力权重，则添加交叉注意力权重

        residual = hidden_states  # 保存输入的隐藏状态作为残差连接的一部分
        hidden_states = self.ln_2(hidden_states)  # 应用 LayerNorm 到隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states)  # 应用 MLP 层到隐藏状态
        # 残差连接
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + (outputs if use_cache else outputs[1:])  # 构建输出元组，包括隐藏状态和其他输出

        return outputs  # 返回隐藏状态、present、(attentions, cross_attentions)
class ImageGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ImageGPTConfig  # 指定配置类为 ImageGPTConfig，用于模型配置参数
    load_tf_weights = load_tf_weights_in_imagegpt  # 指定加载 TensorFlow 权重的函数
    base_model_prefix = "transformer"  # 基础模型前缀，用于命名模型的主要部分
    main_input_name = "input_ids"  # 主要输入名称，通常用于输入模型的主要特征
    supports_gradient_checkpointing = True  # 支持梯度检查点，用于加速训练和减少内存消耗

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)  # 调用父类 PreTrainedModel 的初始化方法

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # 初始化线性层和一维卷积层的权重
            # 与 TensorFlow 版本稍有不同，后者使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ImageGPTLayerNorm):
            # 初始化自定义的 ImageGPTLayerNorm 层的权重
            module.weight.data.fill_(1.0)

        # 重新初始化选定的权重，遵循 OpenAI GPT-2 论文的方案：
        #   > 通过修改的初始化方式，考虑到模型深度的残差路径累积。在初始化时通过因子 1/√N 缩放残差层的权重，其中 N 是残差层的数量。
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # 参考（Megatron-LM）：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # 特殊的缩放初始化 --> 每个 Transformer 块有 2 个 Layer Norms
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


IMAGEGPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# 定义一个文档字符串常量，用于描述ImageGPT模型的输入格式和功能
IMAGEGPT_INPUTS_DOCSTRING = r"""
"""


# 使用装饰器为类添加文档字符串，在ImageGPT模型上方输出原始隐藏状态，没有特定的输出头
@add_start_docstrings(
    "The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.",
    IMAGEGPT_START_DOCSTRING,
)
class ImageGPTModel(ImageGPTPreTrainedModel):
    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # 输入词嵌入和位置嵌入
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        # 多层ImageGPT块的堆叠
        self.h = nn.ModuleList([ImageGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        # 最后一层的LayerNorm
        self.ln_f = ImageGPTLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 模型并行设置
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # 使用装饰器添加文档字符串，在模型前向传播时输出ImageGPT模型的输入格式和功能
    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
@add_start_docstrings(
    """
    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    IMAGEGPT_START_DOCSTRING,
)
class ImageGPTForCausalImageModeling(ImageGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)
        # 使用给定的配置初始化父类（ImageGPTConfig），调用其构造函数
        self.transformer = ImageGPTModel(config)
        # 根据配置创建图像GPT模型的转换器部分
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
        # 创建一个线性层作为语言模型头部，设置输入和输出维度

        # Model parallel
        self.model_parallel = False
        # 设定模型并行计算为假，表示未启用模型的并行计算
        self.device_map = None
        # 设定设备映射为None，表示设备映射未定义

        # Initialize weights and apply final processing
        self.post_init()
        # 调用自定义方法post_init()，用于初始化权重并进行最终处理

    def get_output_embeddings(self):
        return self.lm_head
        # 返回语言模型头部（lm_head）作为输出的嵌入层

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        # 设置新的嵌入层作为语言模型头部（lm_head）

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[bool] = None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # 从kwargs中获取token_type_ids参数，如果不存在则设为None

        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            # 获取past_key_values的长度信息

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1
            # 根据past_key_values来决定保留的输入ID的长度

            input_ids = input_ids[:, remove_prefix_length:]
            # 根据remove_prefix_length截取输入ID

            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]
            # 如果token_type_ids存在，则截取相应长度的token_type_ids

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 根据注意力掩码在批处理生成时动态创建position_ids
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 根据注意力掩码填充position_ids
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
            # 如果存在past_key_values，则截取相应长度的position_ids
        else:
            position_ids = None
        # 否则设置position_ids为None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        # 返回准备好的输入字典，包含input_ids、past_key_values、use_cache、position_ids、attention_mask和token_type_ids
    # 定义模型的前向传播方法，接受多个可选的输入参数和配置选项
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的词索引序列
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 存储循环生成过程中的键值对
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指定哪些位置需要进行注意力计算
        token_type_ids: Optional[torch.Tensor] = None,  # 区分不同句子或段落的类型信息
        position_ids: Optional[torch.Tensor] = None,  # 指定输入序列中每个词的位置信息
        head_mask: Optional[torch.Tensor] = None,  # 控制多头注意力中每个头的掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 直接提供的嵌入输入
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，用于注意力机制
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码
        labels: Optional[torch.Tensor] = None,  # 用于监督学习的标签
        use_cache: Optional[bool] = None,  # 是否使用缓存机制
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果
        **kwargs: Any,  # 其他未指定的关键字参数
    ):
        pass  # 此处为方法定义的占位符，实际实现未提供

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]],  # 存储着多层循环生成过程中的键值对
        beam_idx: torch.Tensor  # 当前束搜索的索引，用于重新排序缓存
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        用于在调用 [`~PreTrainedModel.beam_search`] 或 [`~PreTrainedModel.beam_sample`] 时重新排序
        `past_key_values` 缓存的函数。这是为了在每个生成步骤中将 `past_key_values` 与正确的 `beam_idx` 匹配。
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 给 ImageGPT 模型添加一个图片分类的头部，使用线性层进行分类
# 使用 average-pooling 对隐藏状态进行处理以进行分类
@add_start_docstrings(
    """
    The ImageGPT Model transformer with an image classification head on top (linear layer).
    [`ImageGPTForImageClassification`] average-pools the hidden states in order to do the classification.
    """,
    IMAGEGPT_START_DOCSTRING,
)
class ImageGPTForImageClassification(ImageGPTPreTrainedModel):
    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = ImageGPTModel(config)
        # 创建一个线性层，输入维度为 config.n_embd，输出维度为 num_labels，无偏置项
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
```