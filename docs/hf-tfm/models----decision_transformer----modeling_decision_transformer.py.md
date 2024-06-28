# `.\models\decision_transformer\modeling_decision_transformer.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch DecisionTransformer model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast

# 导入激活函数映射表
from ...activations import ACT2FN
# 导入模型输出的基类，包含过去注意力和交叉注意力
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
# 导入预训练模型的基类
from ...modeling_utils import PreTrainedModel
# 导入与PyTorch相关的实用工具
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
# 导入通用的模型输出类型
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入决策Transformer的配置文件类
from .configuration_decision_transformer import DecisionTransformerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "edbeeching/decision-transformer-gym-hopper-medium"
# 用于文档的配置文件名称
_CONFIG_FOR_DOC = "DecisionTransformerConfig"

# 决策Transformer预训练模型的存档列表
DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "edbeeching/decision-transformer-gym-hopper-medium",
    # 可以查看所有决策Transformer模型的列表
    # https://huggingface.co/models?filter=decision_transformer
]


# 从transformers.models.gpt2.modeling_gpt2.load_tf_weights_in_gpt2中复制而来
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取TF检查点的绝对路径
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从TF模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    # 对于每个名字和数组的组合，执行以下操作
    for name, array in zip(names, arrays):
        # 去掉名字中的"model/"前缀
        name = name[6:]  # skip "model/"
        # 使用斜杠分割名字
        name = name.split("/")
        # 初始化指针为模型本身
        pointer = model
        # 遍历名字中的每个部分
        for m_name in name:
            # 如果名字匹配字母+数字的模式
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                # 按数字分割名字
                scope_names = re.split(r"(\d+)", m_name)
            else:
                # 否则将整个名字作为列表中的一个元素
                scope_names = [m_name]
            # 根据第一个部分选择不同的属性
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                # 处理"wpe"或"wte"的情况
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                # 根据名字的第一个部分选择属性
                pointer = getattr(pointer, scope_names[0])
            # 如果名字有第二个部分，则选择对应索引的元素
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            # 检查指针的形状是否与数组的形状匹配
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            # 如果形状不匹配，将详细信息添加到异常中并重新抛出
            e.args += (pointer.shape, array.shape)
            raise
        # 记录初始化操作信息
        logger.info(f"Initialize PyTorch weight {name}")
        # 将数组转换为PyTorch张量，并赋值给指针的数据属性
        pointer.data = torch.from_numpy(array)
    # 返回处理后的模型
    return model
# Copied from transformers.models.gpt2.modeling_gpt2.GPT2Attention with GPT2->DecisionTransformerGPT2
class DecisionTransformerGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        # 初始化注意事项
        max_positions = config.max_position_embeddings
        # 注册缓冲区并生成一个下三角形状的布尔张量作为注意力偏置
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册缓冲区并设置掩码偏置
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size  # 嵌入维度大小
        self.num_heads = config.num_attention_heads  # 注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads  # 每个注意力头的维度
        self.split_size = self.embed_dim  # 分割后的大小
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights  # 注意力权重缩放
        self.is_cross_attention = is_cross_attention  # 是否是交叉注意力

        # 层级注意力权重缩放、重排序和向上转换
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            # 如果是交叉注意力，创建交叉注意力层和查询注意力层
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            # 如果不是交叉注意力，创建常规注意力层
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)  # 创建投影层

        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 注意力丢弃率
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 残差丢弃率

        self.pruned_heads = set()  # 初始化被修剪的注意力头集合

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # 对 conv1d 层进行修剪
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # 更新超参数
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    # 计算注意力权重，通过查询和键的矩阵乘法得到
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # 如果设置了缩放注意力权重标志，则对注意力权重进行缩放
    if self.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # 如果设置了按逆层索引缩放注意力权重，则对注意力权重进行额外缩放
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    # 如果不是交叉注意力，实现因果屏蔽
    if not self.is_cross_attention:
        # 获取查询和键的长度
        query_length, key_length = query.size(-2), key.size(-2)
        # 生成因果屏蔽掩码
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        # 设定掩码的值为极小值，用于屏蔽不需要的位置
        mask_value = torch.finfo(attn_weights.dtype).min
        # 创建与注意力权重相同类型和设备的掩码张量
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        # 将因果屏蔽应用于注意力权重
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    # 如果存在注意力掩码，则将其应用于注意力权重
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 对注意力权重进行 softmax 归一化
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # 将注意力权重的数据类型降回到 value 张量的数据类型（如果使用了混合精度）
    attn_weights = attn_weights.type(value.dtype)

    # 应用注意力 dropout
    attn_weights = self.attn_dropout(attn_weights)

    # 如果需要，对注意力权重应用头部掩码
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # 计算最终的注意力输出
    attn_output = torch.matmul(attn_weights, value)

    # 返回注意力输出和注意力权重
    return attn_output, attn_weights
    # 将 query, key, value 和 attention_mask（如果存在）按照指定的方式进行上转型和重新排序，并计算注意力权重
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 获取 query 的维度信息：batch size, num_heads, query sequence length, key dimension
        bsz, num_heads, q_seq_len, dk = query.size()
        # 获取 key 的维度信息：batch size, num_heads, key sequence length, key dimension
        _, _, k_seq_len, _ = key.size()

        # 为 `baddbmm` 预先分配注意力权重张量
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # 计算注意力权重的缩放因子
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # 关闭自动混合精度并上转型和重新排序 (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            # 将 query 转换为形状为 (-1, q_seq_len, dk) 的张量
            q = query.reshape(-1, q_seq_len, dk)
            # 将 key 转置并重塑为形状为 (-1, dk, k_seq_len) 的张量
            k = key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            # 使用 `torch.baddbmm` 计算加权和，注意力权重使用缩放因子进行缩放
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            # 将注意力权重张量重塑为形状为 (bsz, num_heads, q_seq_len, k_seq_len)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        # 如果不是跨注意力（cross-attention），实现因果掩码
        if not self.is_cross_attention:
            # 获取 query 和 key 的长度
            query_length, key_length = query.size(-2), key.size(-2)
            # 创建因果掩码，限制只能看到过去的信息
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            # 计算用于掩码的最小值，确保张量的类型和设备一致
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            # 根据因果掩码应用掩码操作
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # 如果存在注意力掩码，则应用该掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 对注意力权重张量进行 softmax 操作，以获得归一化的注意力分布
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 将注意力权重张量转换回 value 张量的数据类型（如果需要）
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)

        # 应用注意力 dropout 操作
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对注意力权重应用头部掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算最终的注意力输出，通过注意力权重与 value 的乘积得到
        attn_output = torch.matmul(attn_weights, value)

        # 返回注意力输出和注意力权重张量
        return attn_output, attn_weights

    # 将张量按照给定的方式进行分割为多个头部
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        # 计算新的张量形状，将 hidden_size 维度分割为 num_heads 和 attn_head_size
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        # 重新调整张量形状，并交换维度以符合注意力头部的分割要求
        tensor = tensor.view(new_shape)
        tensor = tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        return tensor
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # 交换张量的维度顺序，将注意力头和头数的维度合并到隐藏层维度中
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # 计算新的形状，将注意力头和头数维度合并成新的隐藏层维度
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        # 重新视图张量以适应新形状
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                # 如果作为跨注意力使用，则必须定义权重 `q_attn`，否则引发错误
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `DecisionTransformerGPT2Attention(..., is_cross_attention=True)`."
                )

            # 使用 self.q_attn 处理隐藏状态以生成查询张量
            query = self.q_attn(hidden_states)
            # 使用 self.c_attn 处理编码器隐藏状态以生成键和值张量，并按指定维度分割
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            # 使用编码器的注意力掩码
            attention_mask = encoder_attention_mask
        else:
            # 使用 self.c_attn 处理隐藏状态以生成查询、键和值张量，并按指定维度分割
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # 将查询张量按头数和头维度分割
        query = self._split_heads(query, self.num_heads, self.head_dim)
        # 将键张量按头数和头维度分割
        key = self._split_heads(key, self.num_heads, self.head_dim)
        # 将值张量按头数和头维度分割
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果存在过去的层状态，将过去的键和值与当前的键和值拼接在一起
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果使用缓存，保存当前的键和值
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 如果需要重新排序和向上转型的注意力机制
        if self.reorder_and_upcast_attn:
            # 使用特定方法处理注意力机制，得到注意力输出和注意力权重
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # 使用默认的注意力方法处理注意力机制，得到注意力输出和注意力权重
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 将注意力输出按头数和头维度合并成隐藏层维度
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 使用投影层处理注意力输出
        attn_output = self.c_proj(attn_output)
        # 应用残差连接和dropout到注意力输出
        attn_output = self.resid_dropout(attn_output)

        # 输出包括注意力输出和可能的 present
        outputs = (attn_output, present)
        # 如果需要输出注意力权重，也将其加入到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出
        return outputs  # a, present, (attentions)
# 从transformers.models.gpt2.modeling_gpt2.GPT2MLP复制代码，将GPT2改为DecisionTransformerGPT2
class DecisionTransformerGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # 定义一个一维卷积层，输入维度为embed_dim，输出维度为intermediate_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 定义另一个一维卷积层，输入维度为intermediate_size，输出维度为embed_dim
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 设置激活函数为配置中指定的激活函数类型对应的函数
        self.act = ACT2FN[config.activation_function]
        # 设置dropout层，丢弃概率为config.resid_pdrop
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 应用第一个卷积层
        hidden_states = self.c_fc(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 应用第二个卷积层
        hidden_states = self.c_proj(hidden_states)
        # 应用dropout层
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从transformers.models.gpt2.modeling_gpt2.GPT2Block复制代码，将GPT2改为DecisionTransformerGPT2
class DecisionTransformerGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        # 初始化LayerNorm层，输入维度为hidden_size，eps为config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 初始化DecisionTransformerGPT2Attention层
        self.attn = DecisionTransformerGPT2Attention(config, layer_idx=layer_idx)
        # 初始化LayerNorm层，输入维度为hidden_size，eps为config.layer_norm_epsilon
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            # 如果配置中指定添加跨注意力层，则初始化DecisionTransformerGPT2Attention层作为跨注意力层
            self.crossattention = DecisionTransformerGPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            # 初始化LayerNorm层，输入维度为hidden_size，eps为config.layer_norm_epsilon
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 初始化DecisionTransformerGPT2MLP层
        self.mlp = DecisionTransformerGPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 定义函数的返回类型，可以返回包含 torch.Tensor 的元组或者包含可选元组的 Union
        residual = hidden_states
        # 应用 LayerNormalization，归一化隐藏状态
        hidden_states = self.ln_1(hidden_states)
        # 使用 self.attn 处理注意力机制
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 提取注意力输出的第一个元素，即注意力的输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 提取除了第一个元素外的所有输出，作为其他输出
        outputs = attn_outputs[1:]
        # 残差连接，将注意力输出与原始隐藏状态相加
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # 如果传入了 encoder_hidden_states，则进行交叉注意力处理
            if not hasattr(self, "crossattention"):
                # 如果模型未配置交叉注意力层，则引发错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            # 应用交叉注意力层前的 LayerNormalization
            hidden_states = self.ln_cross_attn(hidden_states)
            # 使用 self.crossattention 进行交叉注意力计算
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 提取交叉注意力输出的第一个元素
            attn_output = cross_attn_outputs[0]
            # 残差连接，将交叉注意力输出与之前的隐藏状态相加
            hidden_states = residual + attn_output
            # 将交叉注意力输出的其他部分添加到已有的 outputs 中，如果输出了注意力权重
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        # 应用 LayerNormalization
        hidden_states = self.ln_2(hidden_states)
        # 应用 MLP（Feed Forward）层
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 残差连接，将 MLP 层的输出与原始隐藏状态相加
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            # 如果需要缓存，则将隐藏状态和其他输出组成一个元组返回
            outputs = (hidden_states,) + outputs
        else:
            # 否则，只返回隐藏状态和除第一个元素外的其他输出
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # 返回隐藏状态、present、(attentions, cross_attentions)
class DecisionTransformerGPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 DecisionTransformerConfig 作为配置类
    config_class = DecisionTransformerConfig
    # 使用 load_tf_weights_in_gpt2 函数加载 TensorFlow 权重
    load_tf_weights = load_tf_weights_in_gpt2
    # 基础模型前缀
    base_model_prefix = "transformer"
    # 可并行化处理
    is_parallelizable = True
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # 初始化线性层和一维卷积层的权重
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的偏置和权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 根据 OpenAI GPT-2 论文的方案重新初始化选定的权重：
        #   > 修改的初始化方法考虑到了模型深度中残差路径的累积。在初始化时，通过因子 1/√N 缩放残差层的权重，
        #   > 其中 N 是残差层数量。
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # 参考 (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # 特殊的缩放初始化 --> 每个 Transformer 块中有 2 个 Layer Norm
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


class DecisionTransformerGPT2Model(DecisionTransformerGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # 词嵌入层和位置编码层的初始化
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # Dropout 层的初始化
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer 块的初始化
        self.h = nn.ModuleList(
            [DecisionTransformerGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # 最终的 LayerNorm 层的初始化
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 模型并行
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入的词嵌入（词向量）矩阵
    def get_input_embeddings(self):
        return self.wte

    # 设置输入的词嵌入（词向量）矩阵为新的嵌入矩阵
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 从transformers库中GPT2Model类的forward方法复制而来
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 为决策变换器模型输出定义一个数据类，继承自模型输出基类
@dataclass
class DecisionTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
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
    
    # 环境状态预测
    state_preds: torch.FloatTensor = None
    # 模型动作预测
    action_preds: torch.FloatTensor = None
    # 对每个状态的预测返回
    return_preds: torch.FloatTensor = None
    # 模型隐藏状态
    hidden_states: torch.FloatTensor = None
    # 注意力权重
    attentions: torch.FloatTensor = None
    # 最后一层隐藏状态
    last_hidden_state: torch.FloatTensor = None


# 决策变换器预训练模型的抽象类，处理权重初始化、预训练模型下载和加载的简单接口
class DecisionTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 决策变换器配置类
    config_class = DecisionTransformerConfig
    # 基础模型前缀
    base_model_prefix = "decision_transformer"
    # 主输入名称
    main_input_name = "states"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = False
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，将填充索引位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
# 决策变换器模型的文档字符串，描述了这是一个 PyTorch 的子类模块，可作为常规的 PyTorch 模块使用。建议参考 PyTorch 文档以获取有关通用用法和行为的详细信息。
DECISION_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~DecisionTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 决策变换器模型的输入文档字符串，描述了模型的输入参数及其形状。
DECISION_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        states (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`):
            The states for each step in the trajectory
        actions (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`):
            The actions taken by the "expert" policy for the current state, these are masked for auto regressive
            prediction
        rewards (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The rewards for each state, action
        returns_to_go (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The returns for each state in the trajectory
        timesteps (`torch.LongTensor` of shape `(batch_size, episode_length)`):
            The timestep for each step in the trajectory
        attention_mask (`torch.FloatTensor` of shape `(batch_size, episode_length)`):
            Masking, used to mask the actions when performing autoregressive prediction
"""

# 通过装饰器 @add_start_docstrings 将决策变换器模型的文档字符串和起始描述串联接起来，用以说明决策变换器模型的作用和功能。
@add_start_docstrings("The Decision Transformer Model", DECISION_TRANSFORMER_START_DOCSTRING)
class DecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """
    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 将配置对象保存在实例中
        self.config = config
        # 设置隐藏层大小为配置对象中指定的隐藏层大小
        self.hidden_size = config.hidden_size

        # 创建一个 DecisionTransformerGPT2Model 实例作为编码器
        # 注意：与 Huggingface 默认版本唯一的区别是移除了位置嵌入（因为我们将自己添加）
        self.encoder = DecisionTransformerGPT2Model(config)

        # 创建嵌入层，用于不同类型的输入
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        # LayerNorm 层，用于标准化隐藏层表示
        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # 不预测状态或回报值（根据论文设定）
        
        # 线性层，用于预测状态
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        # 序列模块，用于预测动作
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )
        # 线性层，用于预测回报值
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数并返回一个输出
    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```