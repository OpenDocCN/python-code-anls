# `.\models\pix2struct\modeling_pix2struct.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. & Google team. All rights reserved.
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
""" Pix2Struct modeling file"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# Importing specific modules from the HuggingFace library
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)

from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig

# Initialize logger for logging purposes in this module
logger = logging.get_logger(__name__)

# General docstring describing the configuration used in this module
_CONFIG_FOR_DOC = "Pix2StructConfig"

# List of pretrained model archive paths specific to Pix2Struct
PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/pix2struct-textcaps-base",
    "google/pix2struct-textcaps-large",
    "google/pix2struct-base",
    "google/pix2struct-large",
    "google/pix2struct-ai2d-base",
    "google/pix2struct-ai2d-large",
    "google/pix2struct-widget-captioning-base",
    "google/pix2struct-widget-captioning-large",
    "google/pix2struct-screen2words-base",
    "google/pix2struct-screen2words-large",
    "google/pix2struct-docvqa-base",
    "google/pix2struct-docvqa-large",
    "google/pix2struct-ocrvqa-base",
    "google/pix2struct-ocrvqa-large",
    "google/pix2struct-chartqa-base",
    "google/pix2struct-inforgraphics-vqa-base",
    "google/pix2struct-inforgraphics-vqa-large",
    # See all Pix2StructVision models at https://huggingface.co/models?filter=pix2struct
]

# Adapted layer normalization module from T5 to Pix2Struct style
class Pix2StructLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        # Initializing weight parameter for layer normalization
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # Epsilon value for numerical stability in variance calculation
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        # T5模型使用一种仅进行缩放而不进行偏移的层归一化，即均方根层归一化
        # 参考文献：https://arxiv.org/abs/1910.07467
        # 因此，方差是在没有均值的情况下计算的，且没有偏置。此外，我们希望确保对半精度输入的累积在fp32中完成

        # 计算隐藏状态的方差，将隐藏状态转换为torch.float32类型后求平方，然后沿着最后一个维度求平均
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化，使用倒数平方根进行缩放
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是半精度（torch.float16或torch.bfloat16），则将隐藏状态转换为相同的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回加权后的隐藏状态
        return self.weight * hidden_states
try:
    from apex.normalization import FusedRMSNorm

    # 使用 apex 库中的 FusedRMSNorm 替代 Pix2StructLayerNorm
    Pix2StructLayerNorm = FusedRMSNorm  # noqa

    # 日志记录：发现了 apex.normalization.FusedRMSNorm，将使用它代替 Pix2StructLayerNorm
    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of Pix2StructLayerNorm")
except ImportError:
    # 如果导入失败，继续使用 Pix2StructLayerNorm
    # 使用普通的 Pix2StructLayerNorm
    pass
except Exception:
    # 异常处理：apex 库加载失败，回退到使用 Pix2StructLayerNorm
    logger.warning("Discovered apex but it failed to load, falling back to Pix2StructLayerNorm")
    pass

# 将 Pix2StructLayerNorm 添加到全局列表 ALL_LAYERNORM_LAYERS 中
ALL_LAYERNORM_LAYERS.append(Pix2StructLayerNorm)


class Pix2StructVisionEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        # 线性层，用于将 patch 的隐藏表示映射到 hidden_size
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        # 行索引的嵌入层，将 seq_len 映射到 hidden_size
        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        
        # 列索引的嵌入层，将 seq_len 映射到 hidden_size
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)

        # dropout 层，使用指定的 dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        # 从 flattened_patches 中获取行索引和列索引
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()

        # 提取除索引外的数据部分
        flattened_patches = flattened_patches[:, :, 2:]

        # 将 patch 投影到指定的 hidden_size
        embeddings = self.patch_projection(flattened_patches)
        # 获取行嵌入向量
        row_embeddings = self.row_embedder(row_indices)
        # 获取列嵌入向量
        col_embeddings = self.column_embedder(col_indices)

        # 将三部分嵌入向量相加
        embeddings = embeddings + row_embeddings + col_embeddings

        # 应用 dropout
        embeddings = self.dropout(embeddings)

        return embeddings


class Pix2StructVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow 初始化，避免 softmax 前的缩放
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Self-attention block
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        def to_projection_shape(states):
            """将输入状态调整为投影形状"""
            return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # 获取查询状态
        # (batch_size, n_heads, seq_length, dim_per_head)
        query_states = to_projection_shape(self.query(hidden_states))

        # 获取键/值状态
        key_states = to_projection_shape(self.key(hidden_states))
        value_states = to_projection_shape(self.value(hidden_states))

        # 计算注意力分数
        # 相当于 torch.einsum("bnqd,bnkd->bnqk", query_states, key_states)，与 onnx 操作兼容
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            position_bias = torch.zeros(
                (1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True

            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length), device=scores.device, dtype=scores.dtype)

            if attention_mask.dim() == 2:
                position_bias = position_bias + attention_mask[:, None, None, :].to(position_bias.device)
            else:
                # (batch_size, n_heads, seq_length, key_length)
                position_bias = position_bias + attention_mask.to(position_bias.device)
            position_bias = 1 - position_bias

        position_bias_masked = position_bias.masked_fill(position_bias == 1, torch.finfo(scores.dtype).min)
        scores += position_bias_masked
        scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min))

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 如果需要，掩盖注意力头
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        # (batch_size, seq_length, dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        attn_output = self.output(attn_output)

        outputs = (attn_output,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
# 从 transformers.models.t5.modeling_t5.T5DenseGatedActDense 复制的类，现在命名为 Pix2StructVisionMlp。
# T5DenseGatedActDense 被改名为 Pix2StructVisionMlp，T5Config 改为 Pix2StructVisionConfig，
# config.d_model 改为 config.hidden_size，dropout_rate 改为 dropout_rate。

class Pix2StructVisionMlp(nn.Module):
    def __init__(self, config: Pix2StructVisionConfig):
        super().__init__()
        # 创建一个线性层 wi_0，输入维度为 config.hidden_size，输出维度为 config.d_ff，没有偏置项。
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 创建一个线性层 wi_1，输入维度为 config.hidden_size，输出维度为 config.d_ff，没有偏置项。
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 创建一个线性层 wo，输入维度为 config.d_ff，输出维度为 config.hidden_size，没有偏置项。
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        # 创建一个以 config.dropout_rate 概率随机将输入张量置零的 Dropout 层。
        self.dropout = nn.Dropout(config.dropout_rate)
        # 选择激活函数，根据 config.dense_act_fn 选择 ACT2FN 字典中对应的激活函数。
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 使用 wi_0 对隐藏状态进行线性变换，然后应用激活函数，得到 hidden_gelu。
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 使用 wi_1 对隐藏状态进行线性变换，得到 hidden_linear。
        hidden_linear = self.wi_1(hidden_states)
        # 将 hidden_gelu 和 hidden_linear 的按元素乘积作为新的隐藏状态。
        hidden_states = hidden_gelu * hidden_linear
        # 对新的隐藏状态应用 Dropout。
        hidden_states = self.dropout(hidden_states)

        # 为了使得 Google Flan 的 8 位量化工作，保持 self.wo 在 float32 类型。
        # 参考：https://github.com/huggingface/transformers/issues/20287
        # 如果 self.wo.weight 是 torch.Tensor 类型，并且 hidden_states 的数据类型与 self.wo.weight 的数据类型不同，
        # 且 self.wo.weight 的数据类型不是 torch.int8，则将 hidden_states 转换为 self.wo.weight 的数据类型。
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 使用 wo 对最终的隐藏状态进行线性变换，得到输出。
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 定义了一个 Pix2StructVisionLayer 类，用于组成 Pix2Struct 模型的一个层。
class Pix2StructVisionLayer(nn.Module):
    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        # 设置 feed forward 操作的块大小。
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度。
        self.seq_len_dim = 1
        # 初始化注意力层，使用 Pix2StructVisionAttention 类。
        self.attention = Pix2StructVisionAttention(config)
        # 初始化 MLP 层，使用 Pix2StructVisionMlp 类。
        self.mlp = Pix2StructVisionMlp(config)
        # 初始化前 MLP 层归一化，使用 Pix2StructLayerNorm 类。
        self.pre_mlp_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前注意力层归一化，使用 Pix2StructLayerNorm 类。
        self.pre_attention_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # 函数签名：接受 hidden_states 张量作为输入，可选的 attention_mask 张量，可选的 head_mask 张量，
        # 是否输出注意力权重的标志 output_attentions。
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 定义函数签名，指定输入输出类型为包含两个张量的元组或包含一个张量的元组

        residual = hidden_states
        # 保存输入隐藏状态作为残差连接的基础

        # 在 Pix2StructVision 中，进行自注意力之前先应用层归一化
        hidden_states = self.pre_attention_layer_norm(hidden_states)
        # 使用预自注意力层归一化对隐藏状态进行处理

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
        )
        # 调用注意力机制模块进行自注意力计算，返回自注意力输出和其他相关信息
        attention_output = self_attention_outputs[0]
        # 从自注意力输出中提取注意力输出结果
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力输出

        # 第一个残差连接
        hidden_states = attention_output + residual
        # 将自注意力输出与残差相加，作为隐藏状态的更新结果

        # 在 Pix2StructVision 中，自注意力之后同样应用层归一化
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        # 使用预多层感知机层归一化对隐藏状态进行处理
        layer_output = self.mlp(layer_output) + hidden_states  # 第二个残差连接
        # 经过多层感知机处理后的层输出与原始隐藏状态进行残差连接

        outputs = (layer_output,) + outputs
        # 将最终的层输出添加到输出元组中

        return outputs
        # 返回所有输出，包括层输出和可能的注意力信息
class Pix2StructVisionEncoder(nn.Module):
    # Pix2StructVisionEncoder 类，继承自 nn.Module
    def __init__(self, config: Pix2StructConfig) -> None:
        # 初始化方法，接收一个 Pix2StructConfig 类型的参数 config
        super().__init__()
        self.config = config
        # 创建一个 nn.ModuleList，包含 config.num_hidden_layers 个 Pix2StructVisionLayer 的实例
        self.layer = nn.ModuleList([Pix2StructVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # 是否使用梯度检查点的标志，默认为 False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # forward 方法定义了模型的前向传播逻辑
        # 如果输出隐藏状态，则初始化 all_hidden_states 为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化 all_self_attentions 为空元组，否则为 None
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的 Pix2StructVisionLayer
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点并且处于训练模式，则使用梯度检查点函数进行调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的 __call__ 方法（即 forward 方法）
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回非 None 的元组值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回 BaseModelOutput 类的实例，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Pix2StructPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # Pix2StructPreTrainedModel 类，继承自 PreTrainedModel
    config_class = Pix2StructConfig  # 类属性，指定配置类为 Pix2StructConfig

    @property
    def dummy_inputs(self):
        # dummy_inputs 属性，返回一个字典，包含输入和注意力掩码的示例数据
        input_ids = torch.tensor(DUMMY_INPUTS)  # 输入的张量示例
        input_mask = torch.tensor(DUMMY_MASK)   # 输入的掩码示例
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    # 以下代码段来自于 transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right，已改为 Pix2Struct
    # 定义一个私有方法 `_shift_right`，用于将输入的标识符序列向右移动一位
    def _shift_right(self, input_ids):
        # 获取解码器起始标记的 ID
        decoder_start_token_id = self.config.decoder_start_token_id
        # 获取填充标记的 ID
        pad_token_id = self.config.pad_token_id

        # 如果解码器起始标记 ID 未定义，则抛出数值错误异常
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id. "
                "See Pix2Struct docs for more information."
            )

        # 将输入向右移动一位
        # 如果使用 Torch FX 代理对象
        if is_torch_fx_proxy(input_ids):
            # 对于代理对象，不支持原生的项目赋值操作
            # 创建一个形状与 input_ids 除最后一维外相同的张量，填充为 decoder_start_token_id
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            # 在最后一维上连接 shifted_input_ids 和 input_ids 的前 n-1 列
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 对于普通的张量
            # 创建一个与 input_ids 形状相同的全零张量
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # 将 input_ids 的前 n-1 列赋值给 shifted_input_ids 的后 n-1 列
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            # 将 decoder_start_token_id 赋值给 shifted_input_ids 的第一列
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果 pad_token_id 未定义，则抛出数值错误异常
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        
        # 将 shifted_input_ids 中可能的 -100 值替换为 pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 返回向右移位后的输入标识符序列
        return shifted_input_ids
@add_start_docstrings(
    "The bare Pix2StructVision Model transformer outputting raw hidden-states without any specific head on top.",
    PIX2STRUCT_VISION_START_DOCSTRING,
)
class Pix2StructVisionModel(Pix2StructPreTrainedModel):
    # 设置配置类，用于模型参数配置
    config_class = Pix2StructVisionConfig
    # 主要输入名称为 "flattened_patches"
    main_input_name = "flattened_patches"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不拆分的模块列表
    _no_split_modules = ["Pix2StructVisionLayer"]

    def __init__(self, config: Pix2StructConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 设置模型配置
        self.config = config

        # 初始化嵌入层
        self.embeddings = Pix2StructVisionEmbeddings(config)
        # 初始化编码器
        self.encoder = Pix2StructVisionEncoder(config)

        # 初始化层归一化
        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()
    # 返回当前模型的输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    # 剪枝模型中指定层的注意力头
    # heads_to_prune: 要剪枝的注意力头的字典，格式为 {层号: 要在该层剪枝的头列表}
    # 参见基类 PreTrainedModel
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        for layer, heads in heads_to_prune.items():
            # 获取指定层的注意力模块并进行剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 重写的前向传播函数，应用于 PIX2STRUCT_VISION_INPUTS_DOCSTRING 的输入文档字符串
    # 以及返回值替换为 BaseModelOutputWithPooling 的文档字符串，使用 _CONFIG_FOR_DOC 配置类
    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_attentions 不为 None，则使用其值；否则使用配置中的 output_attentions 值

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_hidden_states 不为 None，则使用其值；否则使用配置中的 output_hidden_states 值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用其值；否则使用配置中的 use_return_dict 值

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")
        # 如果 flattened_patches 为 None，则抛出 ValueError 异常，提示必须指定 flattened_patches

        if attention_mask is None:
            # 检查 flattened_patches 中哪些部分不为 0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()
        # 如果 attention_mask 为 None，则根据 flattened_patches 的和不为 0 的位置创建注意力掩码

        # 准备头部掩码（如果需要）
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 使用 get_head_mask 方法生成头部掩码，参数为 head_mask 和 num_hidden_layers

        embedding_output = self.embeddings(flattened_patches)
        # 将 flattened_patches 输入到 embeddings 中，得到嵌入输出

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 将嵌入输出传递给编码器（encoder），并传递相关参数

        sequence_output = encoder_outputs[0]
        # 从编码器输出中获取序列输出

        sequence_output = self.layernorm(sequence_output)
        # 应用层归一化到序列输出

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        # 如果 return_dict 为 False，则返回头部输出和编码器输出中的其他部分

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        # 如果 return_dict 为 True，则返回包含最后隐藏状态、隐藏状态和注意力的 BaseModelOutput 对象
# 从transformers.models.t5.modeling_t5.T5DenseGatedActDense复制过来，将类名T5DenseGatedActDense改为Pix2StructTextDenseGatedActDense，将d_model改为hidden_size
class Pix2StructTextDenseGatedActDense(nn.Module):
    def __init__(self, config: Pix2StructTextConfig):
        super().__init__()
        # 初始化线性层wi_0，输入维度为config.hidden_size，输出维度为config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 初始化线性层wi_1，输入维度为config.hidden_size，输出维度为config.d_ff，无偏置
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 初始化线性层wo，输入维度为config.d_ff，输出维度为config.hidden_size，无偏置
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        # 初始化Dropout层，使用config.dropout_rate作为丢弃率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 选择激活函数，根据config.dense_act_fn从ACT2FN字典中获取对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    # 定义前向传播函数，接受hidden_states作为输入
    def forward(self, hidden_states):
        # 将hidden_states输入wi_0线性层并经过激活函数处理，得到hidden_gelu
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将hidden_states输入wi_1线性层，得到hidden_linear
        hidden_linear = self.wi_1(hidden_states)
        # 计算element-wise乘积，得到新的hidden_states
        hidden_states = hidden_gelu * hidden_linear
        # 对hidden_states应用dropout操作
        hidden_states = self.dropout(hidden_states)

        # 为了使得8位量化在google/flan-t5-xxl上工作，保持self.wo为float32类型
        # 参考：https://github.com/huggingface/transformers/issues/20287
        # 同时确保权重不是`int8`类型，以防用户将`_keep_in_fp32_modules`强制设为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states转换为self.wo.weight相同的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 将hidden_states输入wo线性层，得到最终的输出hidden_states
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 定义Pix2StructTextLayerFF类
class Pix2StructTextLayerFF(nn.Module):
    def __init__(self, config: Pix2StructTextConfig):
        super().__init__()
        # 初始化DenseReluDense层，使用Pix2StructTextDenseGatedActDense类，传入config参数
        self.DenseReluDense = Pix2StructTextDenseGatedActDense(config)

        # 初始化LayerNorm层，输入维度为config.hidden_size，epsilon为config.layer_norm_epsilon
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，使用config.dropout_rate作为丢弃率
        self.dropout = nn.Dropout(config.dropout_rate)

    # 定义前向传播函数，接受hidden_states作为输入
    def forward(self, hidden_states):
        # 对输入hidden_states进行LayerNorm归一化处理
        forwarded_states = self.layer_norm(hidden_states)
        # 将归一化后的hidden_states输入DenseReluDense层进行前向传播
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将原始hidden_states与经Dropout后的forwarded_states相加，得到最终输出的hidden_states
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
    def __init__(self, config: Pix2StructTextConfig, has_relative_attention_bias=False):
        super().__init__()
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        # 初始化查询、键、值和输出线性层，用于注意力机制
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.has_relative_attention_bias:
            # 如果使用相对注意力偏置，创建相对注意力偏置的嵌入层
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    # 从transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket复制而来
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - the relative position between memory and query
            bidirectional: a boolean - whether the attention is bidirectional or not
            num_buckets: an integer - number of buckets to categorize relative positions into
            max_distance: an integer - maximum distance to consider for bucketing

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        # Initialize relative_buckets to 0
        relative_buckets = 0

        # Adjust num_buckets if bidirectional is True
        if bidirectional:
            num_buckets //= 2
            # Calculate relative_buckets based on whether relative_position is positive
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # Take absolute value of relative_position for further processing
            relative_position = torch.abs(relative_position)
        else:
            # Convert relative_position to non-positive values for unidirectional attention
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        # now relative_position is in the range [0, inf)

        # Define max_exact as half of num_buckets for exact increments
        max_exact = num_buckets // 2

        # Determine if relative_position is small or large
        is_small = relative_position < max_exact

        # Compute relative_position_if_large for larger buckets using logarithmic scaling
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        # Ensure relative_position_if_large does not exceed the maximum bucket index
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # Combine small and large bucket calculations to get final relative_buckets
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)

        # Return the calculated relative_buckets tensor
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果没有指定设备，则使用 self.relative_attention_bias 的设备作为默认设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        # 创建一个形状为 (query_length, 1) 的长整型张量，表示查询的位置
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        
        # 创建一个形状为 (1, key_length) 的长整型张量，表示记忆的位置
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        
        # 计算相对位置，形状为 (query_length, key_length)，表示每个查询位置相对于每个记忆位置的偏移量
        relative_position = memory_position - context_position
        
        # 对相对位置进行分桶，使用 _relative_position_bucket 方法
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # 形状为 (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        
        # 根据分桶后的相对位置获取相对位置偏置值，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        
        # 将维度重新排列为 (1, num_heads, query_length, key_length)，并在最前面添加一个维度
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        # 返回计算得到的相对位置偏置值
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# 从transformers.models.t5.modeling_t5.T5LayerSelfAttention复制代码，并将T5LayerNorm重命名为Pix2StructLayerNorm，T5Attention重命名为Pix2StructTextAttention，self.SelfAttention重命名为self.attention，config.d_model重命名为config.hidden_size
class Pix2StructTextLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层，使用Pix2StructTextAttention作为注意力机制
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化层，使用Pix2StructLayerNorm，并设定eps为config.layer_norm_epsilon
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，使用config.dropout_rate作为dropout率
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 对隐藏状态进行归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用注意力层进行注意力计算
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 原始隐藏状态与注意力输出的dropout结果相加
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 如果需要输出注意力，将其包含在输出中
        outputs = (hidden_states,) + attention_output[1:]  # 如果输出注意力，则添加它们
        return outputs


# 从transformers.models.t5.modeling_t5.T5LayerCrossAttention复制代码，并将T5LayerNorm重命名为Pix2StructLayerNorm，T5Attention重命名为Pix2StructTextAttention，self.EncDecAttention重命名为self.attention，config.d_model重命名为config.hidden_size
class Pix2StructTextLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化跨注意力层，使用Pix2StructTextAttention作为注意力机制
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=False)
        # 初始化层归一化层，使用Pix2StructLayerNorm，并设定eps为config.layer_norm_epsilon
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，使用config.dropout_rate作为dropout率
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对隐藏状态进行归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用注意力层进行注意力计算
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 原始隐藏状态与注意力输出的dropout结果相加
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 如果需要输出注意力，将其包含在输出中
        outputs = (layer_output,) + attention_output[1:]  # 如果输出注意力，则添加它们
        return outputs
class Pix2StructTextBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()

        # 初始化自注意力层，用于Pix2StructTextBlock模块
        self.self_attention = Pix2StructTextLayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )

        # 初始化编码器-解码器注意力层，用于Pix2StructTextBlock模块
        self.encoder_decoder_attention = Pix2StructTextLayerCrossAttention(config)

        # 初始化前馈网络层，用于Pix2StructTextBlock模块
        self.mlp = Pix2StructTextLayerFF(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        # 此方法定义了Pix2StructTextBlock模块的前向传播逻辑
        pass


PIX2STRUCT_START_DOCSTRING = r"""

    The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language
    Understanding](https://arxiv.org/abs/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu,
    Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. It's an encoder decoder
    transformer pre-trained in a image-to-text setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (Union[`Pix2StructConfig`, `Pix2StructTextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PIX2STRUCT_TEXT_INPUTS_DOCSTRING = r"""
"""

PIX2STRUCT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The standalone text decoder of Pix2Struct",
    PIX2STRUCT_START_DOCSTRING,
)
class Pix2StructTextModel(Pix2StructPreTrainedModel):
    config_class = Pix2StructTextConfig
    _no_split_modules = ["Pix2StructTextBlock"]
    _tied_weights_keys = ["lm_head.weight"]
    supports_gradient_checkpointing = True
    def __init__(self, config):
        # 调用父类构造函数初始化模型配置
        super().__init__(config)
        # 初始化词嵌入层，将词汇表大小映射到隐藏层大小
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 创建模型层列表，每层为一个Pix2StructTextBlock对象，根据配置决定是否使用相对注意力偏置
        self.layer = nn.ModuleList(
            [Pix2StructTextBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 初始化最终的层归一化模块，使用给定的隐藏层大小和层归一化系数
        self.final_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，使用给定的丢弃率
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化语言模型的输出层线性变换，将隐藏层的输出映射到词汇表大小的输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()
        # 关闭渐变检查点功能
        self.gradient_checkpointing = False

    # 从transformers库中的T5PreTrainedModel._reorder_cache方法复制而来
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果过去的键值对未包含在输出中，则禁用速度解码并无需重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 重新排序解码器过去的状态
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 根据beam_idx重新排序每层的过去状态
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 使用beam_idx在设备上选择正确的过去状态
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            # 检查重新排序后的状态形状是否与原始状态一致
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            # 检查重新排序后的状态列表长度是否与原始状态列表一致
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            # 将重新排序后的层过去状态添加到重新排序的解码器过去状态元组中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def get_input_embeddings(self):
        # 返回输入词嵌入层
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        # 设置新的输入词嵌入层
        self.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        # 返回输出层线性变换
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出层线性变换
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(PIX2STRUCT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播方法，用于模型的前向推理过程
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为长整型张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，类型为浮点数张量，可选
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态，类型为浮点数张量，可选
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器注意力遮罩，类型为浮点数张量，可选
        inputs_embeds: Optional[torch.LongTensor] = None,  # 输入嵌入表示，类型为长整型张量，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩，类型为浮点数张量，可选
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部遮罩，类型为张量，可选
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，类型为元组中嵌套的浮点数张量元组，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为布尔值，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力，类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为布尔值，可选
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为长整型张量，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，类型为布尔值，可选
        **kwargs,  # 其余关键字参数
# 添加文档字符串到模型类，描述了其作为条件生成模型和语言建模头部的功能
@add_start_docstrings(
    "A conditional generation model with a language modeling head. Can be used for sequence generation tasks.",
    PIX2STRUCT_START_DOCSTRING,
)
class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel):
    # 指定配置类
    config_class = Pix2StructConfig
    # 主要输入名称为"flattened_patches"
    main_input_name = "flattened_patches"
    # 需要共享权重的键列表
    _tied_weights_keys = ["decoder.lm_head.weight"]

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)

        # 初始化编码器和解码器
        self.encoder = Pix2StructVisionModel(config.vision_config)
        self.decoder = Pix2StructTextModel(config.text_config)

        # 是否为视觉问答模型的标志
        self.is_vqa = config.is_vqa

        # 执行后续的初始化步骤
        self.post_init()

    def get_input_embeddings(self):
        # 获取解码器的输入嵌入层
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        # 设置解码器的输入嵌入层
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        # 获取解码器的输出嵌入层
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # 设置解码器的输出嵌入层
        self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        # 调整令牌嵌入层的大小
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)

        # 更新词汇表大小配置
        self.config.text_config.vocab_size = new_num_tokens

        return model_embeds

    def get_decoder(self):
        # 获取解码器模块
        return self.decoder

    def get_encoder(self):
        # 获取编码器模块
        return self.encoder

    @add_start_docstrings_to_model_forward(PIX2STRUCT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成器准备输入数据
    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果未提供decoder_attention_mask，则创建一个全为1的张量，与input_ids的设备匹配
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        # 如果使用了past_key_values，则调整input_ids以去除前缀
        if past_key_values is not None:
            # 获取past_key_values的长度
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法可能已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认情况下保留仅最终的ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 调整input_ids，仅保留后缀部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含各种生成器输入的字典
        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
```