# `.\models\xlm\modeling_xlm.py`

```py
# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
"""
 PyTorch XLM model.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import gelu
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary, SQuADHead
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_xlm import XLMConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "FacebookAI/xlm-mlm-en-2048"
_CONFIG_FOR_DOC = "XLMConfig"

XLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FacebookAI/xlm-mlm-en-2048",
    "FacebookAI/xlm-mlm-ende-1024",
    "FacebookAI/xlm-mlm-enfr-1024",
    "FacebookAI/xlm-mlm-enro-1024",
    "FacebookAI/xlm-mlm-tlm-xnli15-1024",
    "FacebookAI/xlm-mlm-xnli15-1024",
    "FacebookAI/xlm-clm-enfr-1024",
    "FacebookAI/xlm-clm-ende-1024",
    "FacebookAI/xlm-mlm-17-1280",
    "FacebookAI/xlm-mlm-100-1280",
    # See all XLM models at https://huggingface.co/models?filter=xlm
]


def create_sinusoidal_embeddings(n_pos, dim, out):
    """
    Create sinusoidal positional embeddings.
    
    Args:
    - n_pos (int): Number of positions.
    - dim (int): Dimension of embeddings.
    - out (Tensor): Output tensor to store the embeddings.
    
    This function computes sinusoidal embeddings based on position and dimension,
    storing them in the provided output tensor.
    """
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate masks for hidden states and optionally an attention mask.
    
    Args:
    - slen (int): Sequence length.
    - lengths (Tensor): Lengths of each sequence in a batch.
    - causal (bool): If True, generate a causal (triangular) attention mask.
    - padding_mask (Tensor, optional): Mask indicating padded elements.
    
    Returns:
    - Tensor: Mask for hidden states.
    
    This function generates a mask to hide elements beyond the actual length
    of each sequence, and optionally a causal attention mask if specified.
    """
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = lengths.size(0)
    # 如果 causal 变量为真，创建一个注意力掩码，基于 alen 的长度重复创建一个 (bs, slen, slen) 的张量，
    # 并检查每个位置上的长度是否小于等于相应的 alen 值。
    # 如果 causal 变量为假，则直接使用 mask 作为注意力掩码。

    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # 执行一些基本的健全性检查，确保 mask 的形状为 (bs, slen)
    assert mask.size() == (bs, slen)
    # 如果 causal 为真，则检查 attn_mask 的形状为 (bs, slen, slen)，否则不需要此检查
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    # 返回最终的 mask 和 attn_mask
    return mask, attn_mask
# 定义多头注意力机制的类
class MultiHeadAttention(nn.Module):
    # 类变量，用于生成唯一的层 ID
    NEW_ID = itertools.count()

    # 初始化方法
    def __init__(self, n_heads, dim, config):
        super().__init__()
        # 分配新的层 ID 给当前实例
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim  # 注意力机制的维度
        self.n_heads = n_heads  # 头的数量
        self.dropout = config.attention_dropout  # 注意力机制的 dropout 概率
        assert self.dim % self.n_heads == 0  # 确保维度可以整除头的数量

        # 定义线性层，用于计算查询（Q）、键（K）、值（V）和输出
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        
        # 存储被修剪的注意力头的索引
        self.pruned_heads = set()

    # 方法：修剪不需要的注意力头
    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads  # 每个头的注意力大小
        if len(heads) == 0:
            return
        
        # 查找可修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        
        # 对线性层进行修剪
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        
        # 更新超参数：头的数量和注意力机制的维度
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        
        # 更新已修剪的头的集合
        self.pruned_heads = self.pruned_heads.union(heads)
    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()  # 获取输入张量的形状信息，bs为batch size，qlen为序列长度，dim为特征维度
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen  # 如果kv为None，计算klen为当前序列长度或加上缓存序列长度
        else:
            klen = kv.size(1)  # 如果kv不为None，计算klen为kv张量的第二维长度
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        n_heads = self.n_heads  # 获取注意力头的数量
        dim_per_head = self.dim // n_heads  # 计算每个注意力头的特征维度
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)  # 根据mask张量的维度，确定其重塑形状

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)  # 对输入张量x进行投影操作，变换其形状和维度顺序

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)  # 对输入张量x进行反投影操作，计算上下文信息

        q = shape(self.q_lin(input))  # 对输入input进行线性变换后，再进行投影操作，得到查询向量q
                                      # 形状为(bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # 对输入input进行线性变换后，再进行投影操作，得到键向量k
                                          # 形状为(bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # 对输入input进行线性变换后，再进行投影操作，得到值向量v
                                          # 形状为(bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # 对输入kv进行线性变换后，再进行投影操作，得到键向量k
                                      # 形状为(bs, n_heads, klen, dim_per_head)
            v = shape(self.v_lin(v))  # 对输入kv进行线性变换后，再进行投影操作，得到值向量v
                                      # 形状为(bs, n_heads, klen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # 将缓存中的键向量k_和当前计算得到的k拼接在一起
                                                  # 形状为(bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # 将缓存中的值向量v_和当前计算得到的v拼接在一起
                                                  # 形状为(bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]  # 直接从缓存中获取键向量k和值向量v

            cache[self.layer_id] = (k, v)  # 更新缓存中当前层的键值对

        q = q / math.sqrt(dim_per_head)  # 对查询向量q进行缩放操作，以确保在计算注意力分数时的数值稳定性
                                         # 形状为(bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # 计算查询向量q和键向量k的注意力分数
                                                    # 形状为(bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # 根据mask张量将无效位置的注意力分数置为极小值
                                                                 # 形状为(bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)  # 使用极小值填充无效位置的注意力分数
                                                                  # 形状为(bs, n_heads, qlen, klen)

        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # 计算注意力权重，对注意力分数进行softmax操作
                                                                                 # 形状为(bs, n_heads, qlen, klen)
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # 对注意力权重进行dropout操作，用于模型训练防止过拟合
                                                                                           # 形状为(bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask  # 如果指定了头部掩码，则对注意力权重进行头部掩码操作

        context = torch.matmul(weights, v)  # 使用注意力权重对值向量v进行加权求和，得到上下文张量
                                            # 形状为(bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # 将加权求和得到的上下文张量进行反投影操作，得到最终的上下文表示
                                    # 形状为(bs, qlen, dim)

        outputs = (self.out_lin(context),)  # 将上下文张量传入输出层进行线性变换，得到最终的输出
                                           # 形状为(bs, qlen, dim)
        if output_attentions:
            outputs = outputs + (weights,)  # 如果需要输出注意力权重，则将注意力权重作为额外输出

        return outputs  # 返回模型的输出结果
class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout  # 从配置中获取 dropout 率
        self.lin1 = nn.Linear(in_dim, dim_hidden)  # 创建一个线性层，输入维度为 in_dim，输出维度为 dim_hidden
        self.lin2 = nn.Linear(dim_hidden, out_dim)  # 创建另一个线性层，输入维度为 dim_hidden，输出维度为 out_dim
        self.act = gelu if config.gelu_activation else nn.functional.relu  # 根据配置选择激活函数为 GELU 或 ReLU
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 从配置中获取前向传播的分块大小
        self.seq_len_dim = 1  # 序列长度的维度设为 1

    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        x = self.lin1(input)  # 应用第一个线性层
        x = self.act(x)  # 应用激活函数
        x = self.lin2(x)  # 应用第二个线性层
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 应用 dropout
        return x


class XLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XLMConfig  # 设置配置类为 XLMConfig
    load_tf_weights = None  # 不使用 TensorFlow 权重加载
    base_model_prefix = "transformer"  # 基础模型前缀为 "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)  # 调用父类的构造方法

    @property
    def dummy_inputs(self):
        inputs_list = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])  # 创建模型的虚拟输入张量
        attns_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])  # 创建模型的虚拟注意力张量
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])  # 创建虚拟语言嵌入张量
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}  # 返回虚拟输入的字典形式

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):  # 如果是嵌入层
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)  # 使用正态分布初始化权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有 padding_idx，则将对应位置的权重置零
        if isinstance(module, nn.Linear):  # 如果是线性层
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)  # 使用正态分布初始化权重
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)  # 将偏置项初始化为常数 0
        if isinstance(module, nn.LayerNorm):  # 如果是 LayerNorm 层
            module.bias.data.zero_()  # 将偏置项置零
            module.weight.data.fill_(1.0)  # 将权重项填充为 1.0


@dataclass
class XLMForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a `SquadHead`.
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            分类损失，作为起始标记和结束标记分类损失的总和（如果提供了 `start_positions` 和 `end_positions`）。
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            开始标记可能性的对数概率，对应于前 `config.start_n_top` 个可能性（使用 Beam Search）。
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            开始标记可能性的索引，对应于前 `config.start_n_top` 个可能性（使用 Beam Search）。
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            结束标记可能性的对数概率，对应于前 `config.start_n_top * config.end_n_top` 个可能性（使用 Beam Search）。
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            结束标记可能性的索引，对应于前 `config.start_n_top * config.end_n_top` 个可能性（使用 Beam Search）。
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            答案是否不可能的标签的对数概率。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每层的隐藏状态，包括初始嵌入输出，形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力机制注意力权重，用于计算自注意力头中的加权平均值，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
"""
This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

Parameters:
    config ([`XLMConfig`]): Model configuration class with all the parameters of the model.
        Initializing with a config file does not load the weights associated with the model, only the
        configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XLM_INPUTS_DOCSTRING = r"""
"""

@add_start_docstrings(
    "The bare XLM Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_START_DOCSTRING,
)
class XLMModel(XLMPreTrainedModel):
    """
    XLM Model class inheriting from XLMPreTrainedModel.
    """

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the model.
        """
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings of the model to new_embeddings.
        """
        self.embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.
        
        Args:
            heads_to_prune (dict): Dictionary of {layer_num: list of heads to prune in this layer}.
                See base class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the XLM model.

        Args:
            input_ids (torch.Tensor, optional): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor, optional): Mask to avoid performing attention on padding token indices.
            langs (torch.Tensor, optional): Language IDs for multilingual models (not used here).
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            position_ids (torch.Tensor, optional): Indices of positions of each input sequence tokens in the position embeddings.
            lengths (torch.Tensor, optional): Lengths of each sequence to avoid masking beyond the sequence length.
            cache (Dict[str, torch.Tensor], optional): Dictionary with precomputed hidden-states.
            head_mask (torch.Tensor, optional): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.Tensor, optional): External embeddings for the input tokens.
            output_attentions (bool, optional): Whether to output the attentions weights.
            output_hidden_states (bool, optional): Whether to output the hidden states.
            return_dict (bool, optional): Whether to return a dictionary instead of a tuple of outputs.

        Returns:
            BaseModelOutput: Model output that contains various elements depending on the configuration.
        """
        # Implementation of the forward pass is omitted here as it's a part of the model's internal details.

class XLMPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """
    def __init__(self, config):
        super().__init__()
        self.asm = config.asm  # 从配置中获取是否使用自适应softmax的标志
        self.n_words = config.n_words  # 从配置中获取词汇表大小
        self.pad_index = config.pad_index  # 从配置中获取填充索引
        dim = config.emb_dim  # 从配置中获取词嵌入维度

        if config.asm is False:
            # 如果不使用自适应softmax，则创建一个线性投影层
            self.proj = nn.Linear(dim, config.n_words, bias=True)
        else:
            # 如果使用自适应softmax，则创建一个自适应softmax损失层
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # 默认为False，这里设置为True
            )

    def forward(self, x, y=None):
        """计算损失，并可选地计算分数。"""
        outputs = ()  # 初始化一个空的元组用于存储输出

        if self.asm is False:
            # 如果不使用自适应softmax，则计算投影层的分数
            scores = self.proj(x)
            outputs = (scores,) + outputs  # 将分数添加到输出元组中
            if y is not None:
                # 如果标签不为空，则计算交叉熵损失
                loss = nn.functional.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction="mean")
                outputs = (loss,) + outputs  # 将损失添加到输出元组中
        else:
            # 如果使用自适应softmax，则计算log_prob方法得到的分数
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs  # 将分数添加到输出元组中
            if y is not None:
                # 如果标签不为空，则调用自适应softmax的forward方法计算损失
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs  # 将损失添加到输出元组中

        return outputs
"""
The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
# 继承自预训练模型基类 XLMPreTrainedModel 的 XLM Model，增加了语言建模头部
class XLMWithLMHeadModel(XLMPreTrainedModel):
    # 定义需要共享权重的层
    _tied_weights_keys = ["pred_layer.proj.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 XLMModel 实例，用于进行主要的 Transformer 编码
        self.transformer = XLMModel(config)
        # 创建 XLMPredLayer 实例，用于语言模型头部预测
        self.pred_layer = XLMPredLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输出嵌入的方法
    def get_output_embeddings(self):
        return self.pred_layer.proj

    # 设置输出嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        self.pred_layer.proj = new_embeddings

    # 为生成准备输入的方法，处理输入数据和语言 ID
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = input_ids.shape[0]
        # 创建与输入形状相同的掩码张量，填充特殊标记 ID
        mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, mask_token], dim=1)
        # 如果存在语言 ID，则创建相同形状的语言 ID 张量；否则为 None
        if lang_id is not None:
            langs = torch.full_like(input_ids, lang_id)
        else:
            langs = None
        # 返回处理后的输入字典
        return {"input_ids": input_ids, "langs": langs}

    """
    Forward 方法的函数签名注释，描述了输入参数和输出的相关文档字符串。

    Parameters:
        input_ids (Optional[torch.Tensor]): 输入的 token IDs 张量，默认为 None。
        attention_mask (Optional[torch.Tensor]): 注意力掩码张量，默认为 None。
        langs (Optional[torch.Tensor]): 语言 ID 张量，默认为 None。
        token_type_ids (Optional[torch.Tensor]): token 类型 ID 张量，默认为 None。
        position_ids (Optional[torch.Tensor]): 位置 ID 张量，默认为 None。
        lengths (Optional[torch.Tensor]): 长度张量，默认为 None。
        cache (Optional[Dict[str, torch.Tensor]]): 缓存字典，默认为 None。
        head_mask (Optional[torch.Tensor]): 头部掩码张量，默认为 None。
        inputs_embeds (Optional[torch.Tensor]): 输入嵌入张量，默认为 None。
        labels (Optional[torch.Tensor]): 标签张量，默认为 None。
        output_attentions (Optional[bool]): 是否输出注意力，默认为 None。
        output_hidden_states (Optional[bool]): 是否输出隐藏状态，默认为 None。
        return_dict (Optional[bool]): 是否返回字典，默认为 None。
    """
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<special1>",
    )
    # 模型前向传播方法，接受多个输入参数，并按照预期的格式进行文档化
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 增加了多个文档字符串的装饰器，描述了该方法的使用情况和示例

        # 省略部分参数文档
        ...
        ):
        # 实际方法的具体实现在模型类的实际应用中完成，不在这里具体展示
        pass
        ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 不为 None，则使用传入的值，否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 模型处理输入数据
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 Transformer 模型的输出
        output = transformer_outputs[0]

        # 使用预测层处理输出和标签，返回结果为损失和对数概率或仅为对数概率，取决于是否提供了标签
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        # 如果 return_dict 为 False，则返回除了第一个元素（损失）外的所有元素
        if not return_dict:
            return outputs + transformer_outputs[1:]

        # 如果 return_dict 为 True，则返回 MaskedLMOutput 对象，包括损失、对数概率、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述这是一个 XLM 模型，用于序列分类/回归任务，例如 GLUE 任务
# 通过继承 XLMPreTrainedModel 类来定义 XLM 序列分类模型
class XLMForSequenceClassification(XLMPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型参数
        super().__init__(config)
        # 设置模型的类别数目
        self.num_labels = config.num_labels
        # 保存配置信息
        self.config = config

        # 初始化 XLM 模型和序列摘要处理器
        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        # 执行后期初始化，包括权重初始化和最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述该方法的输入
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未提供则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用transformer模型处理输入数据，获取transformer的输出结果
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从transformer的输出中获取主要的输出结果
        output = transformer_outputs[0]
        # 对transformer输出进行汇总处理，得到logits
        logits = self.sequence_summary(output)

        # 初始化损失值为None
        loss = None
        # 如果提供了标签数据
        if labels is not None:
            # 确定问题类型，如果未指定，则根据标签数据类型和标签数量设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数和计算损失值
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典形式的输出，重新组织输出格式
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回SequenceClassifierOutput对象，包含损失值、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
@add_start_docstrings(
    """
    XLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_START_DOCSTRING,
)


这段代码定义了一个XLM模型，该模型在其顶部具有一个用于抽取式问答任务（如SQuAD）的跨度分类头部，这个注释说明了模型的整体功能和用途。


class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):


定义了一个名为`XLMForQuestionAnsweringSimple`的类，它继承自`XLMPreTrainedModel`类，用于执行简单的问答任务。


def __init__(self, config):
    super().__init__(config)

    self.transformer = XLMModel(config)
    self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    # Initialize weights and apply final processing
    self.post_init()


初始化方法定义了模型的构造函数。它首先调用父类的构造函数来初始化模型配置。然后创建了一个`XLMModel`实例作为`transformer`，并创建了一个线性层`qa_outputs`，用于预测答案的开始和结束位置。最后调用`post_init()`方法来初始化权重并应用最终处理。


@add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
@add_code_sample_docstrings(
    checkpoint=_CHECKPOINT_FOR_DOC,
    output_type=QuestionAnsweringModelOutput,
    config_class=_CONFIG_FOR_DOC,
)


这些装饰器为`forward`方法添加了文档字符串，描述了模型前向传播的输入和输出格式，以及提供了示例代码和模型配置信息的链接。


def forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    langs: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    lengths: Optional[torch.Tensor] = None,
    cache: Optional[Dict[str, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    start_positions: Optional[torch.Tensor] = None,
    end_positions: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,


`forward`方法定义了模型的前向传播逻辑，接受多个输入参数，包括`input_ids`、`attention_mask`等，用于执行模型的计算和推理过程。
    # 此方法用于模型的前向传播，接受多个可选参数来控制输入和输出的细节

        input_ids: Optional[torch.Tensor] = None,
        # 输入的 token IDs，类型为 Torch 张量，默认为 None

        attention_mask: Optional[torch.Tensor] = None,
        # 注意力掩码，用于指示哪些位置是需要注意的，默认为 None

        langs: Optional[torch.Tensor] = None,
        # 输入序列的语言 ID，类型为 Torch 张量，默认为 None

        token_type_ids: Optional[torch.Tensor] = None,
        # 用于区分不同句子或序列的 token 类型 ID，默认为 None

        position_ids: Optional[torch.Tensor] = None,
        # 位置 ID，用于指示每个 token 在序列中的位置，默认为 None

        lengths: Optional[torch.Tensor] = None,
        # 输入序列的长度信息，类型为 Torch 张量，默认为 None

        cache: Optional[Dict[str, torch.Tensor]] = None,
        # 缓存字典，用于存储中间计算结果以加速后续计算，默认为 None

        head_mask: Optional[torch.Tensor] = None,
        # 多头注意力机制中的头部掩码，用于控制哪些注意力头部被屏蔽，默认为 None

        inputs_embeds: Optional[torch.Tensor] = None,
        # 输入的嵌入表示，类型为 Torch 张量，默认为 None

        start_positions: Optional[torch.Tensor] = None,
        # 开始位置的标签，用于答案抽取任务，默认为 None

        end_positions: Optional[torch.Tensor] = None,
        # 结束位置的标签，用于答案抽取任务，默认为 None

        is_impossible: Optional[torch.Tensor] = None,
        # 标记答案是否不可能存在的标签，默认为 None

        cls_index: Optional[torch.Tensor] = None,
        # CLS 标记的位置索引，默认为 None

        p_mask: Optional[torch.Tensor] = None,
        # 用于标记不需要参与损失计算的位置的掩码，默认为 None

        output_attentions: Optional[bool] = None,
        # 是否输出注意力权重，默认为 None

        output_hidden_states: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None

        return_dict: Optional[bool] = None,
        # 是否返回字典格式的输出，默认为 None
# 基于 XLM 模型，在其上面添加了一个用于标记分类（如命名实体识别）任务的线性层的模型定义
@add_start_docstrings(
    """
    XLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    XLM_START_DOCSTRING,
)
class XLMForTokenClassification(XLMPreTrainedModel):
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # XLM 模型的主体部分
        self.transformer = XLMModel(config)
        # Dropout 层
        self.dropout = nn.Dropout(config.dropout)
        # 标记分类器的线性层，输入大小为隐藏状态的大小，输出大小为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数的文档字符串
        **kwargs,
    ):
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否应该返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给transformer模型进行处理
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从transformer模型输出中提取序列输出
        sequence_output = outputs[0]

        # 应用dropout层到序列输出
        sequence_output = self.dropout(sequence_output)

        # 通过分类器获取logits（预测分数）
        logits = self.classifier(sequence_output)

        # 初始化损失值为None
        loss = None

        # 如果存在标签，则计算损失值
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典格式的输出，则返回元组格式的结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则使用TokenClassifierOutput封装结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义 XLM 多选分类模型，包含线性层和 softmax 在 transformer 的池化输出之上
@add_start_docstrings(
    """
    XLM Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_START_DOCSTRING,
)
class XLMForMultipleChoice(XLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 XLMModel，用于处理输入序列
        self.transformer = XLMModel(config)
        
        # 初始化 SequenceSummary，用于生成池化的输出
        self.sequence_summary = SequenceSummary(config)
        
        # 初始化 logits_proj 线性层，用于多选分类的最终输出
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 多选分类模型的前向传播方法
    @add_start_docstrings_to_model_forward(XLM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 如果 return_dict 不为 None，则使用传入的值，否则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择题的个数，即 input_ids 的第二维的大小，如果 input_ids 为 None 则为 inputs_embeds 的第二维大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 input_ids 重塑为二维张量的形式，如果 input_ids 为 None 则为 None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将 attention_mask 重塑为二维张量的形式，如果 attention_mask 为 None 则为 None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 token_type_ids 重塑为二维张量的形式，如果 token_type_ids 为 None 则为 None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 position_ids 重塑为二维张量的形式，如果 position_ids 为 None 则为 None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将 langs 重塑为二维张量的形式，如果 langs 为 None 则为 None
        langs = langs.view(-1, langs.size(-1)) if langs is not None else None
        # 将 inputs_embeds 重塑为三维张量的形式，如果 inputs_embeds 为 None 则为 None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 如果使用 lengths 参数，发出警告并将其设置为 None，XLM 多选模型不支持 lengths 参数
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        # 调用 Transformer 模型，传入各种参数进行计算
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从 Transformer 输出中获取最终的隐藏状态
        output = transformer_outputs[0]
        # 对输出进行序列摘要，得到 logits
        logits = self.sequence_summary(output)
        # 将 logits 投影到最终的结果空间
        logits = self.logits_proj(logits)
        # 将 logits 重塑为二维张量，形状为 (-1, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，则返回非字典格式的输出
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 MultipleChoiceModelOutput 对象，包括损失、重塑后的 logits，以及可能的额外信息
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```