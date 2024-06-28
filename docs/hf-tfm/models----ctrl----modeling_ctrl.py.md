# `.\models\ctrl\modeling_ctrl.py`

```
# coding=utf-8
# 设置编码格式为 UTF-8

# Copyright 2018 Salesforce and HuggingFace Inc. team.
# Copyright 2018 年 Salesforce 和 HuggingFace Inc. 团队的版权声明
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权所有 (c) 2018 年 NVIDIA 公司。保留所有权利。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 依据 Apache 许可证，版本 2.0 (下称“许可证”)
# you may not use this file except in compliance with the License.
# 除非符合许可证，否则不得使用此文件。
# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非适用法律要求或书面同意，否则本许可下的软件均为“按原样”提供，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或默示的担保或条件。
# See the License for the specific language governing permissions and
# 请参阅许可证以了解特定语言的权限和
# limitations under the License.
# 许可下的限制。

""" PyTorch CTRL model."""
# PyTorch CTRL 模型

from typing import Optional, Tuple, Union
# 导入类型提示，包括 Optional（可选值）、Tuple（元组）、Union（联合类型）

import numpy as np
# 导入 NumPy 库，用于处理数组和矩阵的数学计算

import torch
# 导入 PyTorch 库

from torch import nn
# 从 PyTorch 中导入 nn 模块

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 从 PyTorch 的 nn 模块中导入不同类型的损失函数

from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutput
# 导入模型输出相关的类，来自 modeling_outputs 模块

from ...modeling_utils import PreTrainedModel
# 导入预训练模型的工具类，来自 modeling_utils 模块

from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_linear_layer
# 导入 PyTorch 相关的工具类和函数，来自 pytorch_utils 模块

from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入添加文档字符串的函数和工具，以及日志记录和替换返回文档字符串的工具

from .configuration_ctrl import CTRLConfig
# 从当前目录中的 configuration_ctrl 模块中导入 CTRLConfig 类

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

_CONFIG_FOR_DOC = "CTRLConfig"
# 用于文档的配置信息，指定为 "CTRLConfig"

CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/ctrl"
    # CTRL 预训练模型的存档列表，包含一个预训练模型
    # 详见 https://huggingface.co/models?filter=ctrl 查看所有 CTRL 模型
]


def angle_defn(pos, i, d_model_size):
    # 定义角度函数，用于位置编码中计算角度率
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates
    # 返回位置和角度率的乘积


def positional_encoding(position, d_model_size, dtype):
    # 创建位置编码的正弦模式
    angle_rads = angle_defn(
        torch.arange(position, dtype=torch.int64).to(dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=torch.int64).to(dtype).unsqueeze(0),
        d_model_size,
    )
    # 计算角度弧度

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    # 组合正弦和余弦的编码结果
    return pos_encoding
    # 返回位置编码向量


def scaled_dot_product_attention(q, k, v, mask, attention_mask=None, head_mask=None):
    # 缩放点积注意力机制

    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))
    # 计算 Q 和 K 的转置的矩阵乘积

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    # 对矩阵乘积进行缩放，按照 K 的维度进行开方缩放

    if mask is not None:
        nd, ns = scaled_attention_logits.size(-2), scaled_attention_logits.size(-1)
        scaled_attention_logits += mask[ns - nd : ns, :ns] * -1e4
    # 如果有掩码，则应用掩码

    if attention_mask is not None:
        # 应用注意力掩码
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    # 计算注意力权重，使用 softmax 归一化

    if head_mask is not None:
        attention_weights = attention_weights * head_mask
    # 如果有头部掩码，则应用头部掩码

    output = torch.matmul(attention_weights, v)
    # 计算加权和，得到输出

    return output, attention_weights
    # 返回输出和注意力权重
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)  # 计算每个注意力头的深度

        self.Wq = nn.Linear(d_model_size, d_model_size)  # Query 线性变换层
        self.Wk = nn.Linear(d_model_size, d_model_size)  # Key 线性变换层
        self.Wv = nn.Linear(d_model_size, d_model_size)  # Value 线性变换层

        self.dense = nn.Linear(d_model_size, d_model_size)  # 最终输出的线性变换层
        self.pruned_heads = set()  # 初始化被剪枝的注意力头集合

    def prune_heads(self, heads):
        attention_head_size = self.d_model_size // self.num_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, attention_head_size, self.pruned_heads)

        # 剪枝线性层
        self.Wq = prune_linear_layer(self.Wq, index)
        self.Wk = prune_linear_layer(self.Wk, index)
        self.Wv = prune_linear_layer(self.Wv, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # 更新超参数
        self.num_heads = self.num_heads - len(heads)
        self.d_model_size = attention_head_size * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # 将输入张量分割成多个注意力头
        return x.permute([0, 2, 1, 3])  # 调整张量维度顺序以便并行处理

    def forward(
        self,
        v,
        k,
        q,
        mask,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        batch_size = q.shape[0]

        q = self.Wq(q)  # 查询向量线性变换
        k = self.Wk(k)  # 键向量线性变换
        v = self.Wv(v)  # 值向量线性变换

        q = self.split_into_heads(q, batch_size)  # 将查询向量分割成多个头
        k = self.split_into_heads(k, batch_size)  # 将键向量分割成多个头
        v = self.split_into_heads(v, batch_size)  # 将值向量分割成多个头
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            k = torch.cat((past_key, k), dim=-2)  # 连接过去的键向量和当前的键向量
            v = torch.cat((past_value, v), dim=-2)  # 连接过去的值向量和当前的值向量

        if use_cache is True:
            present = torch.stack((k, v))  # 存储当前的键和值向量
        else:
            present = (None,)

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask, head_mask)  # 执行缩放点积注意力
        scaled_attention = output[0].permute([0, 2, 1, 3])  # 调整输出注意力张量的维度顺序
        attn = output[1]  # 获取注意力权重
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        output = self.dense(original_size_attention)  # 最终输出的线性变换

        outputs = (output, present)
        if output_attentions:
            outputs = outputs + (attn,)  # 如果需要输出注意力权重，则添加到输出中
        return outputs


def point_wise_feed_forward_network(d_model_size, dff):
    return nn.Sequential(nn.Linear(d_model_size, dff), nn.ReLU(), nn.Linear(dff, d_model_size))


class EncoderLayer(nn.Module):
    # 初始化函数，定义了 TransformerEncoderLayer 类的构造方法
    def __init__(self, d_model_size, num_heads, dff, rate=0.1):
        super().__init__()  # 调用父类构造方法

        # 创建多头注意力机制对象
        self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads)
        # 创建前馈神经网络对象
        self.ffn = point_wise_feed_forward_network(d_model_size, dff)

        # 创建 Layer Normalization 层，用于注意力输出
        self.layernorm1 = nn.LayerNorm(d_model_size, eps=1e-6)
        # 创建 Layer Normalization 层，用于前馈网络输出
        self.layernorm2 = nn.LayerNorm(d_model_size, eps=1e-6)

        # 创建 Dropout 层，用于注意力输出
        self.dropout1 = nn.Dropout(rate)
        # 创建 Dropout 层，用于前馈网络输出
        self.dropout2 = nn.Dropout(rate)

    # 前向传播函数，定义了 TransformerEncoderLayer 类的前向计算过程
    def forward(
        self, x, mask, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False
    ):
        # 应用 Layer Normalization 到输入张量 x
        normed = self.layernorm1(x)
        # 使用多头注意力机制处理 Layer Normalization 后的张量
        attn_outputs = self.multi_head_attention(
            normed,
            normed,
            normed,
            mask,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 从多头注意力输出中取第一个元素作为注意力输出张量
        attn_output = attn_outputs[0]
        # 对注意力输出张量应用 Dropout
        attn_output = self.dropout1(attn_output)
        # 将原始输入张量 x 与处理后的注意力输出张量相加，得到部分前向传播输出 out1
        out1 = x + attn_output

        # 应用 Layer Normalization 到部分前向传播输出 out1
        out2 = self.layernorm2(out1)
        # 使用前馈神经网络处理 Layer Normalization 后的张量
        ffn_output = self.ffn(out2)
        # 对前馈网络输出张量应用 Dropout
        ffn_output = self.dropout2(ffn_output)
        # 将部分前向传播输出 out1 与处理后的前馈网络输出相加，得到最终前向传播输出 out2
        out2 = out1 + ffn_output

        # 构造最终输出元组，包含最终前向传播输出 out2 和可能的注意力输出附加信息
        outputs = (out2,) + attn_outputs[1:]
        # 返回最终输出元组
        return outputs
@add_start_docstrings(
    "The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.",
    CTRL_START_DOCSTRING,
)
class CTRLModel(CTRLPreTrainedModel):
    """CTRL 模型类，继承自 CTRLPreTrainedModel。用于生成原始隐藏状态，没有特定的输出头部。"""

    def __init__(self, config):
        """CTRL 模型的初始化函数。

        Args:
            config (`CTRLConfig`): 包含模型所有参数的配置类对象。
                通过配置文件初始化模型时不会加载与模型关联的权重，只加载配置。
                可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
        """
        super().__init__(config)

        # 设定模型的维度大小和层数
        self.d_model_size = config.n_embd
        self.num_layers = config.n_layer

        # 初始化位置编码
        self.pos_encoding = positional_encoding(config.n_positions, self.d_model_size, torch.float)

        # 设定词嵌入层
        self.w = nn.Embedding(config.vocab_size, config.n_embd)

        # 设定 dropout 层
        self.dropout = nn.Dropout(config.embd_pdrop)

        # 设定 Transformer 编码层列表
        self.h = nn.ModuleList(
            [EncoderLayer(config.n_embd, config.n_head, config.dff, config.resid_pdrop) for _ in range(config.n_layer)]
        )

        # 设定 Layer Normalization 层
        self.layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        """返回输入词嵌入层 `w`。"""
        return self.w
    # 设置新的输入嵌入（embeddings）到模型中
    def set_input_embeddings(self, new_embeddings):
        self.w = new_embeddings

    # 剪枝模型中的注意力头（heads）
    # heads_to_prune: 需要在每个层剪枝的头部字典 {层号: 需要剪枝的头部列表}
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            # 调用多头注意力（multi_head_attention）的剪枝方法
            self.h[layer].multi_head_attention.prune_heads(heads)

    # 重写模型的前向传播方法，添加文档字符串和输出类型的注释
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    CTRL_START_DOCSTRING,
)
"""
定义了一个带有语言建模头部的CTRL模型变换器。语言建模头部是一个线性层，其权重与输入的嵌入层相绑定。
"""

class CTRLLMHeadModel(CTRLPreTrainedModel):
    """
    CTRL语言模型的头部模型，继承自CTRL预训练模型。
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        初始化函数，接受一个配置参数config，并调用父类的初始化函数。
        创建了CTRL模型的transformer部分和语言建模头部的线性层。
        """
        super().__init__(config)
        self.transformer = CTRLModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        返回语言建模头部的嵌入层。
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        设置语言建模头部的新嵌入层。
        """
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        """
        根据输入准备生成过程中的输入。
        如果past_key_values不为None，则只保留输入ids的最后一个token。
        返回一个包含输入信息的字典。
        """
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache}

    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播函数，接受多种输入参数，执行CTRL模型的前向计算。
        返回一个CausalLMOutputWithPast对象，其中包含模型的输出和过去的关键值。
        """
        pass  # Placeholder for forward function

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        静态方法，用于重新排序past_key_values缓存，以匹配每个生成步骤的正确beam_idx。
        返回重新排序后的past_key_values。
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

@add_start_docstrings(
    """
    The CTRL Model transformer with a sequence classification head on top (linear layer).
    [`CTRLForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last
    token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in
    each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last
    value in each row of the batch).
    """
    CTRL_START_DOCSTRING,
)
# 导入需要的类和函数
class CTRLForSequenceClassification(CTRLPreTrainedModel):
    # 初始化方法，继承自父类 CTRLPreTrainedModel
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels
        # 初始化 CTRLModel 模型
        self.transformer = CTRLModel(config)
        # 设置分类器，线性层的输入维度为 config.n_embd，输出维度为类别数，不使用偏置
        self.classifier = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法，添加了模型输入的文档字符串和返回值文档字符串的修饰器
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```