# `.\models\flaubert\modeling_flaubert.py`

```
# coding=utf-8
# Copyright 2019-present CNRS, Facebook Inc. and the HuggingFace Inc. team.
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
""" PyTorch Flaubert model, based on XLM."""

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
from .configuration_flaubert import FlaubertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "flaubert/flaubert_base_cased"
_CONFIG_FOR_DOC = "FlaubertConfig"

FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "flaubert/flaubert_small_cased",
    "flaubert/flaubert_base_uncased",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    # See all Flaubert models at https://huggingface.co/models?filter=flaubert
]


# Copied from transformers.models.xlm.modeling_xlm.create_sinusoidal_embeddings
def create_sinusoidal_embeddings(n_pos, dim, out):
    # 创建正弦位置编码和余弦位置编码的嵌入
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


# Copied from transformers.models.xlm.modeling_xlm.get_masks
def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    if padding_mask is not None:
        # 如果提供了填充遮罩，则使用该遮罩
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        # 生成长度遮罩，确保每个位置不超过最大长度
        mask = alen < lengths[:, None]

    # attention mask 是遮罩本身，或是下三角形的自注意力（因果性）
    bs = lengths.size(0)
    # 如果 causal 参数为真，则创建自注意力掩码，使得每个位置只能注意到之前的位置
    if causal:
        # 创建自注意力掩码，通过比较每个位置的最大长度，生成一个布尔类型的掩码矩阵
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        # 如果 causal 参数为假，则直接使用给定的掩码
        attn_mask = mask

    # 进行一些基本的检查，确保掩码的尺寸符合预期
    assert mask.size() == (bs, slen)
    # 如果 causal 参数为真，则再次检查自注意力掩码的尺寸是否符合预期
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    # 返回计算后的掩码结果
    return mask, attn_mask
# 从 transformers.models.xlm.modeling_xlm.MultiHeadAttention 中复制的多头注意力机制类定义
class MultiHeadAttention(nn.Module):
    # 类变量，用于生成唯一的层标识符
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        # 调用父类的初始化方法
        super().__init__()
        # 为当前实例生成一个唯一的层标识符
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        # 注意力机制的维度
        self.dim = dim
        # 注意力头的数量
        self.n_heads = n_heads
        # 注意力机制的dropout概率，从配置中获取
        self.dropout = config.attention_dropout
        # 断言确保维度可以被头的数量整除
        assert self.dim % self.n_heads == 0

        # 以下是线性变换层的定义
        self.q_lin = nn.Linear(dim, dim)  # 查询线性层
        self.k_lin = nn.Linear(dim, dim)  # 键线性层
        self.v_lin = nn.Linear(dim, dim)  # 值线性层
        self.out_lin = nn.Linear(dim, dim)  # 输出线性层

        # 存储需要被剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 计算每个注意力头的尺寸
        attention_head_size = self.dim // self.n_heads
        # 如果没有需要剪枝的头，则直接返回
        if len(heads) == 0:
            return
        # 调用函数找到需要剪枝的头的索引
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        
        # 剪枝线性层
        self.q_lin = prune_linear_layer(self.q_lin, index)  # 剪枝查询线性层
        self.k_lin = prune_linear_layer(self.k_lin, index)  # 剪枝键线性层
        self.v_lin = prune_linear_layer(self.v_lin, index)  # 剪枝值线性层
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)  # 剪枝输出线性层

        # 更新超参数
        self.n_heads = self.n_heads - len(heads)  # 更新注意力头的数量
        self.dim = attention_head_size * self.n_heads  # 更新注意力机制的维度
        self.pruned_heads = self.pruned_heads.union(heads)  # 更新已剪枝头的集合
    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # 获取输入张量的维度信息
        bs, qlen, dim = input.size()
        # 根据条件确定键值对应的长度
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)
        
        # 计算注意力头的数量和每个头的维度
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        
        # 根据掩码张量的维度情况调整形状
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """对输入张量进行线性投影"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """计算上下文信息"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        # 对查询进行线性投影
        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        
        # 根据kv是否为空，选择相应的线性投影操作
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        # 如果存在缓存，则更新缓存中的键值对
        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        # 缩放查询向量以提高注意力分数的数值稳定性
        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, qlen, klen)
        # 根据掩码填充注意力分数张量
        mask = (mask == 0).view(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, torch.finfo(scores.dtype).min)  # (bs, n_heads, qlen, klen)

        # 使用 softmax 函数计算注意力权重
        weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        # 在训练时应用 dropout，以防止过拟合
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # 如果需要，对注意力头进行掩码操作
        if head_mask is not None:
            weights = weights * head_mask

        # 计算上下文向量
        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        # 应用输出层线性变换
        outputs = (self.out_lin(context),)
        # 如果需要输出注意力权重，则加入到输出中
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs
# Copied from transformers.models.xlm.modeling_xlm.TransformerFFN
class TransformerFFN(nn.Module):
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout  # 从配置中获取丢弃率
        self.lin1 = nn.Linear(in_dim, dim_hidden)  # 第一个线性层，输入维度为in_dim，输出维度为dim_hidden
        self.lin2 = nn.Linear(dim_hidden, out_dim)  # 第二个线性层，输入维度为dim_hidden，输出维度为out_dim
        self.act = gelu if config.gelu_activation else nn.functional.relu  # 激活函数选择为GELU或ReLU
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 前馈过程中的分块大小
        self.seq_len_dim = 1  # 序列长度的维度设为1

    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        x = self.lin1(input)  # 第一线性层的输出
        x = self.act(x)  # 激活函数的应用
        x = self.lin2(x)  # 第二线性层的输出
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)  # 使用丢弃法进行正则化
        return x


FLAUBERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FlaubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FLAUBERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.",
    FLAUBERT_START_DOCSTRING,
)
# Copied from transformers.models.xlm.modeling_xlm.XLMPredLayer with XLM->Flaubert
class FlaubertPredLayer(nn.Module):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super().__init__()
        self.asm = config.asm  # 是否使用自适应softmax
        self.n_words = config.n_words  # 词汇表中的词汇数
        self.pad_index = config.pad_index  # 填充索引
        dim = config.emb_dim  # 嵌入维度

        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias=True)  # 线性投影层，用于非自适应softmax情况
        else:
            self.proj = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # 默认为False
            )  # 自适应softmax层，用于自适应softmax情况
    # 定义前向传播函数，计算损失和（可选）分数
    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        # 初始化空的输出元组
        outputs = ()
        
        # 如果不使用自动微分机制（Autoregressive Sequence Modeling, ASM），则执行以下操作
        if self.asm is False:
            # 计算投影得分
            scores = self.proj(x)
            # 将得分添加到输出元组中
            outputs = (scores,) + outputs
            # 如果指定了目标标签 y，则计算交叉熵损失
            if y is not None:
                # 计算交叉熵损失，对预测得分进行视图重塑以匹配期望形状
                loss = nn.functional.cross_entropy(scores.view(-1, self.n_words), y.view(-1), reduction="mean")
                # 将损失添加到输出元组中
                outputs = (loss,) + outputs
        # 如果使用自动微分机制（ASM），则执行以下操作
        else:
            # 计算投影的对数概率得分
            scores = self.proj.log_prob(x)
            # 将得分添加到输出元组中
            outputs = (scores,) + outputs
            # 如果指定了目标标签 y，则计算投影和目标的损失
            if y is not None:
                # 使用投影模型和目标计算损失
                _, loss = self.proj(x, y)
                # 将损失添加到输出元组中
                outputs = (loss,) + outputs
        
        # 返回输出元组，包含得分和（可选）损失
        return outputs
# 从 transformers.models.xlm.modeling_xlm.XLMPreTrainedModel 复制并修改为支持 Flaubert 模型的基类
class FlaubertPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化、预训练模型下载和加载的抽象类。
    """

    # 配置类为 FlaubertConfig
    config_class = FlaubertConfig
    # 不加载 TensorFlow 的权重
    load_tf_weights = None
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @property
    def dummy_inputs(self):
        # 定义虚拟输入
        inputs_list = torch.tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        # 如果配置中使用语言嵌入且语言数量大于 1，则定义语言列表
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = torch.tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def _init_weights(self, module):
        """初始化模型的权重。"""
        if isinstance(module, nn.Embedding):
            # 如果是 Embedding 层，并且配置中指定了初始化标准差，则使用正态分布初始化权重
            if self.config is not None and self.config.embed_init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.embed_init_std)
            # 如果有填充索引，则将填充索引位置的权重置为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.Linear):
            # 如果是 Linear 层，并且配置中指定了初始化标准差，则使用正态分布初始化权重和常数初始化偏置
            if self.config is not None and self.config.init_std is not None:
                nn.init.normal_(module.weight, mean=0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        if isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，则将偏置置零，权重置为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class FlaubertModel(FlaubertPreTrainedModel):
    # 从 transformers.models.xlm.modeling_xlm.XLMModel.get_input_embeddings 复制
    def get_input_embeddings(self):
        # 返回 embeddings 属性作为输入嵌入
        return self.embeddings

    # 从 transformers.models.xlm.modeling_xlm.XLMModel.set_input_embeddings 复制
    def set_input_embeddings(self, new_embeddings):
        # 设置 embeddings 属性为新的嵌入
        self.embeddings = new_embeddings

    # 从 transformers.models.xlm.modeling_xlm.XLMModel._prune_heads 复制
    def _prune_heads(self, heads_to_prune):
        """
        对模型的注意力头进行修剪。
        heads_to_prune: {层号: 需要在该层中修剪的头列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        # 输入序列的token IDs，类型为长整型张量，可选
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，类型为单精度浮点张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,
        # 语言标识符张量，可选
        langs: Optional[torch.Tensor] = None,
        # token类型IDs张量，可选
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置IDs张量，可选
        position_ids: Optional[torch.LongTensor] = None,
        # 序列长度张量，可选
        lengths: Optional[torch.LongTensor] = None,
        # 缓存字典，键为字符串，值为单精度浮点张量，可选
        cache: Optional[Dict[str, torch.FloatTensor]] = None,
        # 头部掩码，类型为单精度浮点张量，可选
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入张量，类型为单精度浮点张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 是否输出注意力权重，布尔类型，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，布尔类型，可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出，布尔类型，可选
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    FLAUBERT_START_DOCSTRING,
)
# 通过继承FlaubertPreTrainedModel类定义了一个带有语言建模头部的Flaubert模型，头部线性层的权重与输入嵌入层相关联
class FlaubertWithLMHeadModel(FlaubertPreTrainedModel):
    _tied_weights_keys = ["pred_layer.proj.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化transformer部分，使用FlaubertModel
        self.transformer = FlaubertModel(config)
        # 初始化预测层，使用FlaubertPredLayer
        self.pred_layer = FlaubertPredLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回预测层的投影权重
        return self.pred_layer.proj

    def set_output_embeddings(self, new_embeddings):
        # 设置预测层的投影权重为新的嵌入
        self.pred_layer.proj = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 获取mask_token_id和lang_id
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        # 计算有效批次大小
        effective_batch_size = input_ids.shape[0]
        # 创建mask_token张量
        mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype=torch.long, device=input_ids.device)
        # 将mask_token连接到input_ids中
        input_ids = torch.cat([input_ids, mask_token], dim=1)
        # 如果存在lang_id，则创建相同维度的langs张量；否则设置为None
        if lang_id is not None:
            langs = torch.full_like(input_ids, lang_id)
        else:
            langs = None
        # 返回准备好的输入字典
        return {"input_ids": input_ids, "langs": langs}

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<special1>",
    )
    # 重写了forward方法，提供了输入和输出的详细文档字符串
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
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入数据，得到transformer的输出结果
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

        # 从transformer的输出中获取主要的输出（通常是logits）
        output = transformer_outputs[0]

        # 使用预测层（pred_layer）生成最终的输出
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        # 如果不是以字典形式返回结果，则将额外的输出（如hidden_states, attentions等）附加到outputs中返回
        if not return_dict:
            return outputs + transformer_outputs[1:]

        # 以MaskedLMOutput对象的形式返回结果，包括损失值（如果labels不为None）、logits、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 将模型描述文档添加到类定义上方，说明该类是基于 Flaubert 模型的序列分类/回归模型，
# 其顶部有一个线性层（线性层位于汇总输出之上），用于例如 GLUE 任务。
@add_start_docstrings(
    """
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从 transformers.models.xlm.modeling_xlm.XLMForSequenceClassification 复制而来，
# 将 XLM_INPUTS 替换为 FLAUBERT_INPUTS，将 XLM 替换为 Flaubert
class FlaubertForSequenceClassification(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数目
        self.config = config

        self.transformer = FlaubertModel(config)  # 初始化 Flaubert 模型
        self.sequence_summary = SequenceSummary(config)  # 初始化序列汇总器

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数定义，接受多种输入参数，返回序列分类器的输出
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
        # 如果 return_dict 不为 None，则使用指定的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 transformer 处理输入序列
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

        # 从 transformer 输出中获取主要输出
        output = transformer_outputs[0]
        
        # 对输出进行序列汇总
        logits = self.sequence_summary(output)

        # 初始化损失为 None
        loss = None

        # 如果提供了标签
        if labels is not None:
            # 根据问题类型确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果标签数为 1，则计算回归损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则计算多标签回归损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 如果是单标签分类问题，则计算交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 如果是多标签分类问题，则计算二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不使用 return_dict，则返回输出元组
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用 SequenceClassifierOutput 类返回结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
@add_start_docstrings(
    """
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FLAUBERT_START_DOCSTRING,
)
# 从transformers.models.xlm.modeling_xlm.XLMForTokenClassification复制而来，将XLM_INPUTS改为FLAUBERT_INPUTS，将XLM改为Flaubert
class FlaubertForTokenClassification(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化Flaubert模型
        self.transformer = FlaubertModel(config)
        # Dropout层，使用配置中的dropout率
        self.dropout = nn.Dropout(config.dropout)
        # 分类器，将隐藏状态输出映射到num_labels维度
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多种输入参数，返回TokenClassifierOutput类型的输出
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
    # 返回一个元组或 TokenClassifierOutput 对象
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        langs: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, Optional[torch.Tensor]]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确保 return_dict 不为 None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入传递给 transformer 层进行处理
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
    
        # 从 transformer 输出中获取序列输出
        sequence_output = outputs[0]
    
        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 使用分类器得到 logits
        logits = self.classifier(sequence_output)
    
        # 初始化损失为 None
        loss = None
        # 如果给定了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        # 如果 return_dict 为 True，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAUBERT_START_DOCSTRING,
)
# 定义一个新的类，用于执行简单的问答任务，基于 Flaubert 模型，添加了用于提取问题答案的分类头部
class FlaubertForQuestionAnsweringSimple(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Flaubert 模型
        self.transformer = FlaubertModel(config)
        # 线性层，用于生成 `span start logits` 和 `span end logits`
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接收多个输入参数，计算模型输出
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



@add_start_docstrings(
    """
    Flaubert Model with a beam-search span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    FLAUBERT_START_DOCSTRING,
)
# 数据类，用于定义 Flaubert 模型在问答输出时的结果结构，继承自 ModelOutput
@dataclass
class FlaubertForQuestionAnsweringOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a `SquadHead`.
    """



# Copied from transformers.models.xlm.modeling_xlm.XLMForQuestionAnsweringSimple with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
# 从 XLMForQuestionAnsweringSimple 模型复制而来，做了相应的替换以适应 Flaubert 模型
# 根据给定配置创建一个简单的问答模型
class FlaubertForQuestionAnsweringSimple(FlaubertPreTrainedModel):
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            分类损失，作为开始标记、结束标记（如果提供的话还包括is_impossible）分类损失的总和。
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            开始标记的前config.start_n_top个可能性的对数概率（beam-search）。
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            开始标记的前config.start_n_top个可能性的索引（beam-search）。
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            结束标记的前config.start_n_top * config.end_n_top个可能性的对数概率（beam-search）。
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            结束标记的前config.start_n_top * config.end_n_top个可能性的索引（beam-search）。
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            答案的`is_impossible`标签的对数概率。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态的元组，包括每层的输出和初始嵌入的输出，形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，每层一个，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，用于计算自注意力头中的加权平均值。
# Copied from transformer.models.xlm.modeling_xlm.XLMForQuestionAnswering with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForQuestionAnswering(FlaubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize FlaubertModel with provided configuration
        self.transformer = FlaubertModel(config)
        # Initialize SQuADHead for question answering tasks
        self.qa_outputs = SQuADHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FlaubertForQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
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
        is_impossible: Optional[torch.Tensor] = None,
        cls_index: Optional[torch.Tensor] = None,
        p_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for FlaubertForQuestionAnswering model.

        Args:
            input_ids (torch.Tensor, optional): Indices of input sequence tokens.
            attention_mask (torch.Tensor, optional): Mask to avoid performing attention on padding tokens.
            langs (torch.Tensor, optional): Language IDs for multi-lingual models like XLM.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            position_ids (torch.Tensor, optional): Indices of positions of each input sequence tokens in the position embeddings.
            lengths (torch.Tensor, optional): Lengths of each sequence to handle padded inputs.
            cache (Dict[str, torch.Tensor], optional): Dictionary with precomputed hidden states.
            head_mask (torch.Tensor, optional): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.Tensor, optional): Embedded representation of the inputs.
            start_positions (torch.Tensor, optional): Start position for the answer span.
            end_positions (torch.Tensor, optional): End position for the answer span.
            is_impossible (torch.Tensor, optional): Whether the question has no possible answer.
            cls_index (torch.Tensor, optional): Position of the classification token in the input sequence.
            p_mask (torch.Tensor, optional): Mask of tokens which can't be in the answer.
            output_attentions (bool, optional): Whether to output attentions weights.
            output_hidden_states (bool, optional): Whether to output all hidden-states.
            return_dict (bool, optional): Whether to return a single dictionary instead of a tuple of outputs.

        Returns:
            Union[FlaubertForQuestionAnsweringOutput, Tuple[torch.Tensor]]:
                Depending on `return_dict`, either a dictionary with main outputs or a tuple of outputs.
        """
        # Actual forward logic will be implemented here
        pass

# Copied from transformer.models.xlm.modeling_xlm.XLMForMultipleChoice with XLM_INPUTS->FLAUBERT_INPUTS,XLM->Flaubert
class FlaubertForMultipleChoice(FlaubertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # Initialize FlaubertModel with provided configuration
        self.transformer = FlaubertModel(config)
        # Sequence summarization layer
        self.sequence_summary = SequenceSummary(config)
        # Linear layer for projecting logits
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(
        FLAUBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
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
    ):
        """
        Forward pass for FlaubertForMultipleChoice model.

        Args:
            input_ids (torch.Tensor, optional): Indices of input sequence tokens.
            attention_mask (torch.Tensor, optional): Mask to avoid performing attention on padding tokens.
            langs (torch.Tensor, optional): Language IDs for multi-lingual models like XLM.
            token_type_ids (torch.Tensor, optional): Segment token indices to indicate first and second portions of the inputs.
            position_ids (torch.Tensor, optional): Indices of positions of each input sequence tokens in the position embeddings.
            lengths (torch.Tensor, optional): Lengths of each sequence to handle padded inputs.
            cache (Dict[str, torch.Tensor], optional): Dictionary with precomputed hidden states.
            head_mask (torch.Tensor, optional): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.Tensor, optional): Embedded representation of the inputs.
            labels (torch.Tensor, optional): Labels for computing the multiple choice classification loss.
            output_attentions (bool, optional): Whether to output attentions weights.
            output_hidden_states (bool, optional): Whether to output all hidden-states.
            return_dict (bool, optional): Whether to return a single dictionary instead of a tuple of outputs.

        Returns:
            Union[MultipleChoiceModelOutput, Tuple[torch.Tensor]]:
                Depending on `return_dict`, either a dictionary with main outputs or a tuple of outputs.
        """
        # Actual forward logic will be implemented here
        pass
    # 定义模型的前向传播函数，接受多个可选的输入参数，均为 torch.Tensor 类型
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选参数
        langs: Optional[torch.Tensor] = None,  # 语言标识符，可选参数
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选参数
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，可选参数
        lengths: Optional[torch.Tensor] = None,  # 序列长度信息，可选参数
        cache: Optional[Dict[str, torch.Tensor]] = None,  # 缓存，字典类型，可选参数
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选参数
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入表示，可选参数
        labels: Optional[torch.Tensor] = None,  # 标签，可选参数
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选参数
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数
        return_dict: Optional[bool] = None,  # 是否返回一个字典形式的结果，可选参数
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置返回字典的默认值为self.config.use_return_dict，如果return_dict不为None，则使用传入的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取第二维的大小，即选项数量，从input_ids获取，如果input_ids为None，则从inputs_embeds获取
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果input_ids不为None，则将其视图重新排列为(-1, input_ids.size(-1))，否则为None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果attention_mask不为None，则将其视图重新排列为(-1, attention_mask.size(-1))，否则为None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果token_type_ids不为None，则将其视图重新排列为(-1, token_type_ids.size(-1))，否则为None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果position_ids不为None，则将其视图重新排列为(-1, position_ids.size(-1))，否则为None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果langs不为None，则将其视图重新排列为(-1, langs.size(-1))，否则为None
        langs = langs.view(-1, langs.size(-1)) if langs is not None else None
        # 如果inputs_embeds不为None，则将其视图重新排列为(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))，否则为None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 如果lengths不为None，则发出警告并将其设为None，因为Flaubert多选模型不支持使用lengths参数
        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the Flaubert multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        # 将所有参数传递给transformer模型进行前向传播，获取transformer的输出
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
        # 获取transformer的输出中的第一个元素，通常是模型输出的主要部分
        output = transformer_outputs[0]
        # 使用sequence_summary方法对输出进行汇总
        logits = self.sequence_summary(output)
        # 使用logits_proj方法对logits进行处理，使其形状为(-1, num_choices)
        logits = self.logits_proj(logits)
        reshaped_logits = logits.view(-1, num_choices)

        # 如果labels不为None，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果return_dict为False，则返回元组形式的输出；否则返回MultipleChoiceModelOutput对象
        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回MultipleChoiceModelOutput对象，其中包含loss、logits以及transformer_outputs的hidden_states和attentions
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```