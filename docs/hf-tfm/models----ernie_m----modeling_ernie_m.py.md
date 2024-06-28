# `.\models\ernie_m\modeling_ernie_m.py`

```
# coding=utf-8
# 版权 2023 年 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang The HuggingFace Inc. team. 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件
# 没有任何形式的保证或条件，包括但不限于
# 特定用途的隐含保证或条件。
# 有关详细信息，请参阅许可证。

""" PyTorch ErnieM 模型。"""


import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "susnato/ernie-m-base_pytorch"
_CONFIG_FOR_DOC = "ErnieMConfig"
_TOKENIZER_FOR_DOC = "ErnieMTokenizer"

# ErnieM 预训练模型存档列表
ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "susnato/ernie-m-base_pytorch",
    "susnato/ernie-m-large_pytorch",
    # 查看所有 ErnieM 模型，请访问 https://huggingface.co/models?filter=ernie_m
]

# 从 paddlenlp.transformers.ernie_m.modeling.ErnieEmbeddings 改编而来
class ErnieMEmbeddings(nn.Module):
    """从词嵌入和位置嵌入构造嵌入。"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # 定义词嵌入层，将词汇表中的词映射到隐藏大小的向量空间
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，将位置索引映射到隐藏大小的向量空间
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
        )
        # LayerNorm 层，用于归一化隐藏层的输出
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # padding 的索引
        self.padding_idx = config.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果输入的嵌入向量为None，则使用模型的词嵌入层对输入的token IDs进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 如果位置ID为None，则计算序列的形状并生成位置ID
        if position_ids is None:
            input_shape = inputs_embeds.size()[:-1]  # 获取输入嵌入向量的形状（去掉最后一个维度，通常是序列长度）
            ones = torch.ones(input_shape, dtype=torch.int64, device=inputs_embeds.device)  # 创建全为1的张量，与inputs_embeds设备相同
            seq_length = torch.cumsum(ones, dim=1)  # 按行累积和，生成序列长度张量
            position_ids = seq_length - ones  # 生成位置ID，每个位置ID等于其位置在序列中的索引值减去1

            # 如果过去的键值长度大于0，则调整位置ID
            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length
        
        # 为了模仿paddlenlp的实现，在位置ID上增加一个偏移量2
        position_ids += 2
        
        # 使用位置ID获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        
        # 将输入嵌入向量和位置嵌入向量相加得到最终的嵌入向量表示
        embeddings = inputs_embeds + position_embeddings
        
        # 对嵌入向量进行Layer Norm归一化
        embeddings = self.layer_norm(embeddings)
        
        # 对归一化后的向量应用Dropout操作
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量表示
        return embeddings
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->ErnieM,self.value->self.v_proj,self.key->self.k_proj,self.query->self.q_proj
class ErnieMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，确保兼容性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义线性变换层，将隐藏状态映射到注意力头大小的维度空间
        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 dropout 层，用于在注意力计算时进行随机失活
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置编码，初始化距离编码的嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数计算所需的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，实现自注意力机制
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        ):
        pass


class ErnieMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 ErnieMAttention 的自注意力层
        self.self_attn = ErnieMSelfAttention(config, position_embedding_type=position_embedding_type)
        # 输出投影层，将隐藏状态映射回原始隐藏大小的空间
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个空集合，用于记录要修剪的注意力头
        self.pruned_heads = set()
    # 根据给定的头部列表来修剪自注意力机制中的头部
    def prune_heads(self, heads):
        # 如果头部列表为空，则直接返回，不执行修剪操作
        if len(heads) == 0:
            return
        
        # 调用函数找到可修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self_attn.num_attention_heads, self.self_attn.attention_head_size, self.pruned_heads
        )

        # 修剪自注意力机制中的线性层
        self.self_attn.q_proj = prune_linear_layer(self.self_attn.q_proj, index)
        self.self_attn.k_proj = prune_linear_layer(self.self_attn.k_proj, index)
        self.self_attn.v_proj = prune_linear_layer(self.self_attn.v_proj, index)
        self.out_proj = prune_linear_layer(self.out_proj, index, dim=1)

        # 更新超参数并存储已修剪的头部信息
        self.self_attn.num_attention_heads = self.self_attn.num_attention_heads - len(heads)
        self.self_attn.all_head_size = self.self_attn.attention_head_size * self.self_attn.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义模型的前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用自注意力机制处理输入的隐藏状态和其他可选参数
        self_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力机制的输出经过输出投影层处理
        attention_output = self.out_proj(self_outputs[0])
        # 如果需要输出注意力权重信息，则在输出中包含注意力权重
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中
        return outputs
class ErnieMEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 模仿 PaddleNLP 的实现，设置 dropout 为 0.1，如果配置中未指定隐藏层dropout，则使用默认值
        dropout = 0.1 if config.hidden_dropout_prob is None else config.hidden_dropout_prob
        # 如果配置中未指定激活层dropout，则使用隐藏层dropout值作为激活层dropout
        act_dropout = config.hidden_dropout_prob if config.act_dropout is None else config.act_dropout

        # 初始化自注意力层
        self.self_attn = ErnieMAttention(config)
        # 第一个线性变换层，将隐藏层大小转换为中间层大小
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活层dropout
        self.dropout = nn.Dropout(act_dropout)
        # 第二个线性变换层，将中间层大小转换回隐藏层大小
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # 第一个 LayerNorm 层，用于归一化隐藏层数据
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 第二个 LayerNorm 层，用于归一化线性变换后的数据
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 第一个 dropout 层，应用于第一个线性变换后的数据
        self.dropout1 = nn.Dropout(dropout)
        # 第二个 dropout 层，应用于第二个线性变换后的数据
        self.dropout2 = nn.Dropout(dropout)
        
        # 根据配置中的激活函数类型选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = True,
    ):
        # 保留残差连接
        residual = hidden_states
        # 如果需要输出注意力权重，则在自注意力层中返回注意力权重
        if output_attentions:
            hidden_states, attention_opt_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        else:
            # 否则，仅返回自注意力层的输出隐藏状态
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        
        # 添加第一个 dropout，并与残差连接
        hidden_states = residual + self.dropout1(hidden_states)
        # 第一个 LayerNorm 层，用于归一化第一次线性变换后的数据
        hidden_states = self.norm1(hidden_states)
        # 更新残差连接
        residual = hidden_states
        
        # 第二次线性变换，应用于归一化后的数据
        hidden_states = self.linear1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 第一个 dropout 层，应用于激活后的数据
        hidden_states = self.dropout(hidden_states)
        # 第二次线性变换
        hidden_states = self.linear2(hidden_states)
        # 添加第二个 dropout，并与残差连接
        hidden_states = residual + self.dropout2(hidden_states)
        # 第二个 LayerNorm 层，用于归一化第二次线性变换后的数据
        hidden_states = self.norm2(hidden_states)

        # 如果需要输出注意力权重，则返回注意力权重和隐藏状态
        if output_attentions:
            return hidden_states, attention_opt_weights
        else:
            # 否则，仅返回隐藏状态
            return hidden_states


class ErnieMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 存储配置
        self.config = config
        # 创建多个 ErnieMEncoderLayer 层，根据配置中的隐藏层数量
        self.layers = nn.ModuleList([ErnieMEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    # 定义前向传播函数，接收多个输入参数和可选的返回值设定
    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，则初始化一个空元组用于存储隐藏状态
        hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组用于存储注意力权重
        attentions = () if output_attentions else None

        # 初始化输出为输入的嵌入向量
        output = input_embeds
        # 如果需要输出隐藏状态，则将当前输出加入到隐藏状态元组中
        if output_hidden_states:
            hidden_states = hidden_states + (output,)

        # 遍历所有层进行前向传播
        for i, layer in enumerate(self.layers):
            # 获取当前层的头部掩码，如果未提供头部掩码则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对，如果未提供则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 调用当前层的前向传播函数，更新输出和可选的注意力权重
            output, opt_attn_weights = layer(
                hidden_states=output,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                past_key_value=past_key_value,
            )

            # 如果需要输出隐藏状态，则将当前输出加入到隐藏状态元组中
            if output_hidden_states:
                hidden_states = hidden_states + (output,)
            # 如果需要输出注意力权重，则将当前注意力权重加入到注意力元组中
            if output_attentions:
                attentions = attentions + (opt_attn_weights,)

        # 最终的隐藏状态为最后一层的输出
        last_hidden_state = output
        # 如果不需要返回字典形式的输出，则返回非空的元组
        if not return_dict:
            return tuple(v for v in [last_hidden_state, hidden_states, attentions] if v is not None)

        # 返回带有过去和交叉注意力的基础模型输出对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制过来，将Bert->ErnieM
class ErnieMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 初始化线性层，输入和输出维度都是config.hidden_size
        self.activation = nn.Tanh()  # Tanh激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取第一个标记对应的隐藏状态来“汇聚”模型
        first_token_tensor = hidden_states[:, 0]  # 取第一个标记对应的隐藏状态
        pooled_output = self.dense(first_token_tensor)  # 输入到线性层
        pooled_output = self.activation(pooled_output)  # 应用Tanh激活函数
        return pooled_output  # 返回汇聚输出


class ErnieMPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    config_class = ErnieMConfig  # 配置类为ErnieMConfig
    base_model_prefix = "ernie_m"  # 基础模型前缀为"ernie_m"

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与TF版本稍有不同，TF版本使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ERNIE_M_START_DOCSTRING = r"""

    此模型继承自[`PreTrainedModel`]。查看超类文档以获取库实现的所有模型的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。

    此模型是PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)的子类。将其用作常规PyTorch模块，并参考PyTorch文档，了解与一般使用和行为相关的所有内容。

    参数:
        config ([`ErnieMConfig`]): 包含模型所有参数的配置类。
            使用配置文件初始化不会加载与模型关联的权重，只加载配置。请查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

ERNIE_M_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`ErnieMTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.


注释：


# input_ids: 输入序列标记在词汇表中的索引
#   这些索引可以使用 ErnieMTokenizer 获取。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 以获取详细信息。
#   更多关于输入 ID 的信息请参考 glossary 中的 input-ids 页面。

# attention_mask: 注意力掩码，避免在填充的标记索引上执行注意力操作。掩码的取值范围为 [0, 1]：
#   - 1 表示不屏蔽的标记，
#   - 0 表示被屏蔽的标记。
#   更多关于注意力掩码的信息请参考 glossary 中的 attention-mask 页面。

# position_ids: 输入序列中每个标记的位置索引，在位置嵌入中使用。取值范围为 [0, config.max_position_embeddings - 1]。
#   更多关于位置 ID 的信息请参考 glossary 中的 position-ids 页面。

# head_mask: 自注意力模块中需要屏蔽的头部掩码。掩码的取值范围为 [0, 1]：
#   - 1 表示未屏蔽的头部，
#   - 0 表示屏蔽的头部。
#   更多关于头部掩码的信息请参考 glossary 中的 attention-mask 页面。

# inputs_embeds: 可选项，可以直接传入嵌入表示而不是传入 input_ids。如果希望更精确地控制如何将 input_ids 转换为关联向量，这非常有用。
#   这种方式比模型内部的嵌入查找矩阵更具控制性。
  
# output_attentions: 是否返回所有注意力层的注意力张量。请参见返回的张量中的 attentions 获取更多细节。

# output_hidden_states: 是否返回所有层的隐藏状态。请参见返回的张量中的 hidden_states 获取更多细节。

# return_dict: 是否返回 utils.ModelOutput 而不是普通的元组。
"""

@add_start_docstrings(
    "The bare ErnieM Model transformer outputting raw hidden-states without any specific head on top.",
    ERNIE_M_START_DOCSTRING,
)
# 定义 ErnieMModel 类，继承自 ErnieMPreTrainedModel
class ErnieMModel(ErnieMPreTrainedModel):
    # 初始化方法
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类初始化方法
        super(ErnieMModel, self).__init__(config)
        # 初始化变量 initializer_range
        self.initializer_range = config.initializer_range
        # 创建 ErnieMEmbeddings 对象
        self.embeddings = ErnieMEmbeddings(config)
        # 创建 ErnieMEncoder 对象
        self.encoder = ErnieMEncoder(config)
        # 如果 add_pooling_layer 为 True，则创建 ErnieMPooler 对象
        self.pooler = ErnieMPooler(config) if add_pooling_layer else None
        # 执行后续初始化
        self.post_init()

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].self_attn.prune_heads(heads)

    # 定义前向传播方法，用于模型推理
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[tensor] = None,
        position_ids: Optional[tensor] = None,
        attention_mask: Optional[tensor] = None,
        head_mask: Optional[tensor] = None,
        inputs_embeds: Optional[tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 详见模型前向传播的文档字符串
        pass


@add_start_docstrings(
    """ErnieM Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    ERNIE_M_START_DOCSTRING,
)
# 定义 ErnieMForSequenceClassification 类，继承自 ErnieMPreTrainedModel
class ErnieMForSequenceClassification(ErnieMPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 初始化变量 num_labels
        self.num_labels = config.num_labels
        # 将配置参数保存在 self.config 中
        self.config = config

        # 创建 ErnieMModel 对象
        self.ernie_m = ErnieMModel(config)
        # 设置分类器的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 Dropout 层对象
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性分类器层对象
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行后续处理
        self.post_init()

    # 定义前向传播方法，用于模型推理
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[tensor] = None,
        position_ids: Optional[tensor] = None,
        attention_mask: Optional[tensor] = None,
        head_mask: Optional[tensor] = None,
        inputs_embeds: Optional[tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 详见模型前向传播的文档字符串
        pass
    # 添加代码示例的文档字符串，用于自动文档生成
    @add_code_sample_docstrings(
        # 指定用于处理的处理器类别
        processor_class=_TOKENIZER_FOR_DOC,
        # 指定用于文档的检查点
        checkpoint=_CHECKPOINT_FOR_DOC,
        # 指定输出类型为序列分类器输出对象
        output_type=SequenceClassifierOutput,
        # 指定用于配置的配置类
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多个输入参数并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用给定的值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ERNIE 模型进行前向传播
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 将处理后的输出传入分类器，得到 logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果存在 labels，则计算损失
        if labels is not None:
            # 根据问题类型配置 self.config.problem_type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数并计算损失
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

        # 如果 return_dict 为 False，则返回一个元组，包含 logits 和可能的额外输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回一个 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加类的文档字符串，描述该类是基于ErnieM模型的多选分类模型，用于例如RocStories/SWAG任务
@add_start_docstrings(
    """ErnieM Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    ERNIE_M_START_DOCSTRING,
)
# 定义ErnieMForMultipleChoice类，继承自ErnieMPreTrainedModel类
class ErnieMForMultipleChoice(ErnieMPreTrainedModel):
    
    # 从transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__复制而来，修改了Bert为ErnieM，bert为ernie_m
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 创建ErnieMModel实例，用于提取特征
        self.ernie_m = ErnieMModel(config)
        
        # 根据配置设置分类器的dropout比率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个dropout层，应用于分类器
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 创建一个线性层，将隐藏状态的特征映射到1维输出（用于二元分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串到模型的前向传播方法，描述了输入的参数形状和用法
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 添加代码示例的文档字符串，指定了检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        # 输入参数详细描述如下：
        # input_ids: 输入的token IDs
        # attention_mask: 注意力掩码，指示模型注意力的计算范围
        # position_ids: 位置 IDs，指示输入token的位置信息
        # head_mask: 头部掩码，用于指定哪些注意力头部被屏蔽
        # inputs_embeds: 嵌入的输入特征，如果不是None，则忽略input_ids
        # labels: 模型的标签，用于训练时计算损失
        # output_attentions: 是否输出注意力权重
        # output_hidden_states: 是否输出隐藏状态
        # return_dict: 是否返回字典格式的输出
        

        # return_dict: 是否返回字典格式的输出
        ) -> Union[Tuple[torch.FloatTensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据函数签名，此函数接受输入并返回一个元组，包含浮点张量或多选模型输出对象
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定选择数目，根据输入的 `input_ids` 的第二维度或者 `inputs_embeds` 的第二维度
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果 `input_ids` 不为 `None`，重新视图化为二维张量，否则为 `None`
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果 `attention_mask` 不为 `None`，重新视图化为二维张量，否则为 `None`
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果 `position_ids` 不为 `None`，重新视图化为二维张量，否则为 `None`
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果 `inputs_embeds` 不为 `None`，重新视图化为三维张量，否则为 `None`
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 ERNIE 模型 (`self.ernie_m`) 进行前向传播
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取汇聚输出，通常是 ERNIE 模型的第二个输出
        pooled_output = outputs[1]

        # 应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器得出 logits
        logits = self.classifier(pooled_output)
        # 调整 logits 的形状，以便与标签匹配
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 `return_dict` 是 False，返回一个元组，包含重塑后的 logits 和可能的隐藏状态
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 `return_dict` 是 True，返回一个 `MultipleChoiceModelOutput` 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加文档字符串，描述了 ErnieM 模型在标记分类任务上的用途，例如命名实体识别（NER）任务
@add_start_docstrings(
    """ErnieM Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    ERNIE_M_START_DOCSTRING,
)
# 定义 ErnieMForTokenClassification 类，继承自 ErnieMPreTrainedModel 类
class ErnieMForTokenClassification(ErnieMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制而来，将 Bert 替换为 ErnieM，bert 替换为 ernie_m
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 设置类别数目
        self.num_labels = config.num_labels

        # 使用 ErnieMModel 构建 ErnieM 模型，关闭 pooling 层
        self.ernie_m = ErnieMModel(config, add_pooling_layer=False)
        
        # 根据配置决定分类器的 dropout，若未设置，则使用隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 Dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义线性分类器，输入大小为隐藏层大小，输出大小为类别数目
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器添加文档字符串到 forward 方法，描述了输入参数的含义和用法
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，描述了 processor_class、checkpoint、output_type 和 config_class 的信息
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 forward 方法，接收多个输入参数，返回模型的输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ErnieModel 对象进行前向传播
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 操作
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的结果输入分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回输出的元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 这里的 outputs[2:] 包含额外的隐藏状态
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,  # 返回所有隐藏状态
            attentions=outputs.attentions,        # 返回所有注意力权重
        )
# 在ErnieM模型基础上添加一个用于抽取式问答任务的分类头部，例如SQuAD任务（在隐藏状态输出之上的线性层，用于计算`span start logits`和`span end logits`）。
@add_start_docstrings(
    """ErnieM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForQuestionAnswering(ErnieMPreTrainedModel):
    # 从transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__中复制而来，将Bert->ErnieM, bert->ernie_m
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 设置分类任务的标签数目
        self.num_labels = config.num_labels

        # 初始化ErnieM模型，不添加池化层
        self.ernie_m = ErnieMModel(config, add_pooling_layer=False)
        # 线性层，用于生成分类任务的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # Decide whether to use return_dict based on input or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through the ERNIE model and retrieve outputs
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the sequence output from the model outputs
        sequence_output = outputs[0]

        # Compute logits for question answering from the sequence output
        logits = self.qa_outputs(sequence_output)
        
        # Split logits into start and end logits for the predicted spans
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # Calculate total loss if start_positions and end_positions are provided
        if start_positions is not None and end_positions is not None:
            # If inputs are on multi-GPU, adjust dimensions
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # Clamp positions to avoid errors when indices are out of range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define CrossEntropyLoss with ignored_index
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # Prepare output based on whether return_dict is False
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        # Return structured output using QuestionAnsweringModelOutput
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加类的文档字符串，描述了 ErnieMForInformationExtraction 类的作用和设计用途
@add_start_docstrings(
    """ErnieMForInformationExtraction is a Ernie-M Model with two linear layer on top of the hidden-states output to
    compute `start_prob` and `end_prob`, designed for Universal Information Extraction.""",
    ERNIE_M_START_DOCSTRING,
)
# 继承自 ErnieMPreTrainedModel 的 ErnieMForInformationExtraction 类，用于信息抽取任务
class ErnieMForInformationExtraction(ErnieMPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super(ErnieMForInformationExtraction, self).__init__(config)
        # 初始化 ErnieMModel 模型
        self.ernie_m = ErnieMModel(config)
        # 创建线性层，用于计算起始位置的概率
        self.linear_start = nn.Linear(config.hidden_size, 1)
        # 创建线性层，用于计算结束位置的概率
        self.linear_end = nn.Linear(config.hidden_size, 1)
        # 创建 sigmoid 激活函数，用于输出概率值
        self.sigmoid = nn.Sigmoid()
        # 执行后初始化操作
        self.post_init()

    # 为 forward 方法添加文档字符串，描述了输入参数及其含义
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for position (index) for computing the start_positions loss. Position outside of the sequence are
            not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) for computing the end_positions loss. Position outside of the sequence are not
            taken into account for computing the loss.
        """

        # 使用 ERNIE 模型处理输入数据，根据参数配置返回不同的输出格式
        result = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            # 如果 return_dict 为 True，则直接从 result 中获取最后一层隐藏状态
            sequence_output = result.last_hidden_state
        elif not return_dict:
            # 如果 return_dict 为 False，则从 result 的第一个元素获取最后一层隐藏状态
            sequence_output = result[0]

        # 经过线性层处理，获取起始位置的 logits，并进行维度压缩
        start_logits = self.linear_start(sequence_output)
        start_logits = start_logits.squeeze(-1)
        # 经过线性层处理，获取结束位置的 logits，并进行维度压缩
        end_logits = self.linear_end(sequence_output)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 的维度大于 1，进行维度压缩
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 对超出模型输入范围的 start/end positions 进行修正
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义二元交叉熵损失函数
            loss_fct = BCEWithLogitsLoss()
            # 计算起始位置和结束位置的损失值
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果 return_dict 为 False，返回一个包含非空结果的元组
            return tuple(
                i
                for i in [total_loss, start_logits, end_logits, result.hidden_states, result.attentions]
                if i is not None
            )

        # 如果 return_dict 为 True，返回一个 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
        )
```