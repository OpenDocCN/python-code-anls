# `.\models\markuplm\modeling_markuplm.py`

```
# coding=utf-8
# 版权 2022 年由 Microsoft Research Asia 和 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证要求或获得书面许可，否则您不能使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按“原样”分发，
# 没有任何形式的明示或暗示担保或条件。
# 请参阅许可证以获取特定语言的权限和限制。
""" PyTorch MarkupLM 模型。"""

import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从外部导入一些函数和类
from ...activations import ACT2FN
from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_markuplm import MarkupLMConfig

# 获取 logger 对象用于记录日志
logger = logging.get_logger(__name__)

# 以下是一些用于文档的变量
_CHECKPOINT_FOR_DOC = "microsoft/markuplm-base"
_CONFIG_FOR_DOC = "MarkupLMConfig"

# 预训练模型存档列表
MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/markuplm-base",
    "microsoft/markuplm-large",
]

# XPathEmbeddings 类定义，用于构建来自 xpath 标签和下标的嵌入
class XPathEmbeddings(nn.Module):
    """Construct the embeddings from xpath tags and subscripts.

    We drop tree-id in this version, as its info can be covered by xpath.
    """

    def __init__(self, config):
        super(XPathEmbeddings, self).__init__()
        # 最大深度设定
        self.max_depth = config.max_depth

        # 将 xpath 单元序列映射为隐藏尺寸
        self.xpath_unitseq2_embeddings = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)

        # dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 激活函数 ReLU
        self.activation = nn.ReLU()
        
        # 将 xpath 单元序列映射为内部尺寸的线性层
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        
        # 将内部尺寸映射为嵌入尺寸的线性层
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        # xpath 标签子嵌入的 Embedding 列表
        self.xpath_tag_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )

        # xpath 下标子嵌入的 Embedding 列表
        self.xpath_subs_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )
    python
        # 定义前向传播函数，接受两个可选参数：xpath_tags_seq 和 xpath_subs_seq
        def forward(self, xpath_tags_seq=None, xpath_subs_seq=None):
            # 初始化空列表，用于存储各层次标签路径的嵌入
            xpath_tags_embeddings = []
            # 初始化空列表，用于存储各层次子节点路径的嵌入
            xpath_subs_embeddings = []
    
            # 循环遍历每个层次的深度（最多为 self.max_depth）
            for i in range(self.max_depth):
                # 将当前层次的标签路径数据传入对应的嵌入层，并将结果添加到标签路径嵌入列表中
                xpath_tags_embeddings.append(self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i]))
                # 将当前层次的子节点路径数据传入对应的嵌入层，并将结果添加到子节点路径嵌入列表中
                xpath_subs_embeddings.append(self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i]))
    
            # 沿着最后一个维度（深度方向）连接所有标签路径的嵌入，形成完整的标签路径嵌入
            xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
            # 沿着最后一个维度（深度方向）连接所有子节点路径的嵌入，形成完整的子节点路径嵌入
            xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)
    
            # 将标签路径嵌入和子节点路径嵌入按元素相加，得到整体的路径嵌入
            xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings
    
            # 将整体路径嵌入传入内部层 self.xpath_unitseq2_inner 进行处理，并经过激活函数 activation 和 dropout 处理
            xpath_embeddings = self.inner2emb(self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))
    
            # 返回计算得到的最终路径嵌入
            return xpath_embeddings
# 从输入的 `input_ids` 中创建位置编码，替换非填充符号为它们的位置编号。位置编号从 `padding_idx+1` 开始，忽略填充符号。这个函数是从 fairseq 的 `utils.make_positions` 修改而来。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个 mask 张量，标记非填充符号为 1，其余为 0
    mask = input_ids.ne(padding_idx).int()
    # 对 mask 张量进行累积求和，并加上 `past_key_values_length`，再乘以 mask 本身，确保只有非填充位置会被计数
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 最后将得到的位置索引加上填充索引 `padding_idx`，并转换为长整型
    return incremental_indices.long() + padding_idx


class MarkupLMEmbeddings(nn.Module):
    """从词嵌入、位置嵌入和标记类型嵌入构建嵌入向量。"""

    def __init__(self, config):
        super(MarkupLMEmbeddings, self).__init__()
        self.config = config
        # 词嵌入层，将词汇表大小映射到隐藏大小，使用 `padding_idx` 指定填充符号
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，将最大位置编码数量映射到隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.max_depth = config.max_depth

        # XPath 嵌入层的初始化
        self.xpath_embeddings = XPathEmbeddings(config)

        # 标记类型嵌入层，将标记类型词汇表大小映射到隐藏大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 层，归一化隐藏状态向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃隐藏状态向量，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建一个持久化的缓冲区，存储位置编码，用于模型训练过程中的批处理
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # 填充符号索引
        self.padding_idx = config.pad_token_id
        # 重新定义位置嵌入层，指定填充符号的索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 从输入的嵌入张量 `inputs_embeds` 中创建位置编码
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        由于我们直接提供了嵌入向量，无法推断哪些是填充的，因此直接生成顺序的位置编号。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成一个从 `padding_idx + 1` 到 `sequence_length + padding_idx + 1` 的序列作为位置编码
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置编码扩展成与 `inputs_embeds` 相同的形状
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
        ):
            # 如果输入的 input_ids 不为空，则获取其形状作为 input_shape
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                # 否则，获取 inputs_embeds 的除最后一维外的所有维度作为 input_shape
                input_shape = inputs_embeds.size()[:-1]

            # 确定设备为 input_ids 的设备（如果 input_ids 不为空），否则为 inputs_embeds 的设备
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # 如果未提供 position_ids，则根据 input_ids 创建位置编码，保留任何填充的标记
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果未提供 token_type_ids，则创建与 input_shape 相同形状的全零张量作为 token_type_ids
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # 如果未提供 inputs_embeds，则使用 input_ids 获取对应的词嵌入
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            # 准备 xpath_tags_seq 序列，如果未提供，则创建全为 self.config.tag_pad_id 的张量
            if xpath_tags_seq is None:
                xpath_tags_seq = self.config.tag_pad_id * torch.ones(
                    tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
                )

            # 准备 xpath_subs_seq 序列，如果未提供，则创建全为 self.config.subs_pad_id 的张量
            if xpath_subs_seq is None:
                xpath_subs_seq = self.config.subs_pad_id * torch.ones(
                    tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
                )

            # 将词嵌入作为 words_embeddings
            words_embeddings = inputs_embeds

            # 根据 position_ids 获取位置编码作为 position_embeddings
            position_embeddings = self.position_embeddings(position_ids)

            # 根据 token_type_ids 获取 token 类型编码作为 token_type_embeddings
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            # 根据 xpath_tags_seq 和 xpath_subs_seq 获取 xpath 嵌入作为 xpath_embeddings
            xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)

            # 将所有嵌入进行加和作为最终的 embeddings
            embeddings = words_embeddings + position_embeddings + token_type_embeddings + xpath_embeddings

            # 对 embeddings 进行 LayerNorm 归一化处理
            embeddings = self.LayerNorm(embeddings)

            # 对 embeddings 进行 dropout 处理
            embeddings = self.dropout(embeddings)

            # 返回最终的嵌入向量 embeddings
            return embeddings
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制而来，将 Bert 替换为 MarkupLM
class MarkupLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout 随机断开神经元连接
        hidden_states = self.dropout(hidden_states)
        # Layer normalization，并添加输入张量进行残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制而来
class MarkupLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 中间激活函数，根据配置选择合适的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制而来，将 Bert 替换为 MarkupLM
class MarkupLMOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入维度为 config.intermediate_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # Layer normalization 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout 随机断开神经元连接
        hidden_states = self.dropout(hidden_states)
        # Layer normalization，并添加输入张量进行残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertPooler 复制而来
class MarkupLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Tanh 激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 池化操作，直接取第一个 token 对应的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 线性变换
        pooled_output = self.dense(first_token_tensor)
        # 应用 Tanh 激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从 transformers.models.bert.modeling_bert.BertPredictionHeadTransform 复制而来，将 Bert 替换为 MarkupLM
class MarkupLMPredictionHeadTransform(nn.Module):
    # 初始化函数，用于创建对象时的初始化操作，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置中的隐藏层激活函数，选择相应的激活函数或者使用预定义的激活函数映射表 ACT2FN
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化，设置归一化的维度和 epsilon 参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接收一个张量 hidden_states，返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层 self.dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的张量通过激活函数 self.transform_act_fn 进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 将经过激活函数变换后的张量通过 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的张量作为前向传播的结果
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLMPredictionHead 复制而来，将 Bert 替换为 MarkupLM
class MarkupLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MarkupLMPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记有一个仅输出的偏置项。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来确保偏置项在调整 `resize_token_embeddings` 时正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制而来，将 Bert 替换为 MarkupLM
class MarkupLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MarkupLMLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制而来，将 Bert 替换为 MarkupLM
class MarkupLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 查询、键、值的线性转换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # 如果使用相对位置嵌入，需要额外的距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
    # 将输入张量 x 进行形状变换，以便用于多头注意力机制
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    
    # 对变换后的张量进行维度置换，以便多头注意力机制能够操作
    return x.permute(0, 2, 1, 3)

# 实现自定义的前向传播方法，用于 Transformer 模型的每一层
def forward(
    self,
    hidden_states: torch.Tensor,  # 输入的隐藏状态张量
    attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码张量，可选
    head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码张量，可选
    encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态张量，可选
    encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器注意力掩码张量，可选
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，可选
    output_attentions: Optional[bool] = False,  # 是否输出注意力权重，可选
# 从transformers.models.bert.modeling_bert.BertAttention复制过来，将Bert改为MarkupLM
class MarkupLMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层，使用给定的配置和位置嵌入类型
        self.self = MarkupLMSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化自注意力输出层，使用给定的配置
        self.output = MarkupLMSelfOutput(config)
        # 初始化用于存储已修剪注意力头的集合
        self.pruned_heads = set()

    # 方法：修剪注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用辅助函数找到可修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
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
        # 使用自注意力层处理隐藏状态和其他可选参数
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用自注意力输出层处理自注意力层的输出和隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果输出注意力值，则添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 从transformers.models.bert.modeling_bert.BertLayer复制过来，将Bert改为MarkupLM
class MarkupLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化用于分块前馈传递的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 初始化自注意力层
        self.attention = MarkupLMAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # 如果添加交叉注意力但不是解码器，则抛出值错误异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置嵌入类型初始化交叉注意力层
            self.crossattention = MarkupLMAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = MarkupLMIntermediate(config)
        # 初始化输出层
        self.output = MarkupLMOutput(config)
    # 定义一个方法 forward，用于处理模型的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，可选参数
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力的掩码，可选参数
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态，可选参数
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器的注意力掩码，可选参数
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，可选参数
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，可选参数，默认为 False
    ) -> Tuple[torch.Tensor]:  # 返回类型为包含张量的元组
        # 如果有过去的键值对，则提取自注意力的过去键值对的缓存，位置在1和2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力层处理隐藏状态，返回自注意力的输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]  # 提取自注意力的输出

        # 如果当前模块是解码器，最后一个输出是自注意力的键值对缓存元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 提取除了最后一个元素外的所有输出
            present_key_value = self_attention_outputs[-1]  # 提取最后一个元素作为当前的键值对
        else:
            outputs = self_attention_outputs[1:]  # 如果不是解码器，添加自注意力权重到输出中

        cross_attn_present_key_value = None
        # 如果当前模块是解码器且有编码器的隐藏状态作为输入
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 提取跨注意力的过去键值对缓存，位置在过去键值对元组的倒数第二和最后一个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用跨注意力层处理自注意力的输出和编码器的隐藏状态，返回跨注意力的输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]  # 提取跨注意力的输出
            outputs = outputs + cross_attention_outputs[1:-1]  # 添加跨注意力的权重到输出中

            # 将跨注意力的现在的键值对添加到当前的键值对中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用到前向传播的分块处理中
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs  # 将处理后的输出和之前的输出合并成元组

        # 如果是解码器，将注意力的键值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs  # 返回处理后的输出元组
    # 定义一个方法用于执行神经网络的前向传播，处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用 self.intermediate 方法处理注意力输出，得到中间层输出
        intermediate_output = self.intermediate(attention_output)
        # 使用 self.output 方法处理中间层输出和注意力输出，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层的输出作为这个方法的结果
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码，并将Bert->MarkupLM
class MarkupLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个由多个MarkupLMLayer对象组成的层列表，层数由config.num_hidden_layers指定
        self.layer = nn.ModuleList([MarkupLMLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点设置为False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态为真，则初始化空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重为真，则初始化空元组，否则为None
        all_self_attentions = () if output_attentions else None
        # 如果输出交叉注意力为真且配置允许，则初始化空元组，否则为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用梯度检查点并且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果设置了使用缓存，则发出警告并设置use_cache为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不使用缓存，则初始化空元组，否则为None
        next_decoder_cache = () if use_cache else None
        # 遍历每一层的编码器层模块
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为真，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有头部掩码，则选择当前层的头部掩码，否则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果有过去的键值对，则选择当前层的过去键值对，否则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用梯度检查点并且处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的模块计算输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出的最后一个元素添加到next_decoder_cache中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力为真，则将当前层的输出的第二个元素添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置允许，则将当前层的输出的第三个元素添加到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态为真，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回非空值的元组
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则返回带有过去和交叉注意力的基础模型输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    """
    This model is a PyTorch `torch.nn.Module` sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MarkupLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中每个token在词汇表中的索引。

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

        xpath_tags_seq (`torch.LongTensor` of shape `({0}, config.max_depth)`, *optional*):
            # 输入序列中每个token对应的标签ID，填充到config.max_depth。

        xpath_subs_seq (`torch.LongTensor` of shape `({0}, config.max_depth)`, *optional*):
            # 输入序列中每个token对应的子脚本ID，填充到config.max_depth。

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖掩码，避免在填充的token索引上进行注意力计算。`1`表示未被遮盖的token，`0`表示被遮盖的token。

            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段落token索引，用于指示输入的第一部分和第二部分。`0`对应*句子A*的token，`1`对应*句子B*的token。

            [What are token type IDs?](../glossary#token-type-ids)

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列token在位置嵌入中的索引。选择范围为`[0, config.max_position_embeddings - 1]`。

            [What are position IDs?](../glossary#position-ids)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 自注意力模块中选择性屏蔽的头部的掩码。`1`表示头部未被屏蔽，`0`表示头部被屏蔽。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示而不是`input_ids`。如果想要对如何将*input_ids*索引转换为关联向量有更多控制，则很有用。

        output_attentions (`bool`, *optional*):
            # 如果设置为`True`，则返回所有注意力层的注意力张量。详见返回张量中的`attentions`获取更多详情。

        output_hidden_states (`bool`, *optional*):
            # 如果设置为`True`，则返回所有层的隐藏状态。详见返回张量中的`hidden_states`获取更多详情。

        return_dict (`bool`, *optional*):
            # 如果设置为`True`，模型将返回[`~file_utils.ModelOutput`]而不是简单的元组。
"""
@add_start_docstrings(
    "The bare MarkupLM Model transformer outputting raw hidden-states without any specific head on top.",
    MARKUPLM_START_DOCSTRING,
)
class MarkupLMModel(MarkupLMPreTrainedModel):
    # 从transformers.models.bert.modeling_bert.BertModel.__init__复制而来，将Bert改为MarkupLM
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化模型的嵌入层
        self.embeddings = MarkupLMEmbeddings(config)
        
        # 初始化模型的编码器层
        self.encoder = MarkupLMEncoder(config)

        # 如果指定要添加池化层，则初始化池化层；否则设为None
        self.pooler = MarkupLMPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.bert.modeling_bert.BertModel.forward复制而来
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        xpath_tags_seq: Optional[torch.LongTensor] = None,
        xpath_subs_seq: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Model forward pass. See BaseModelOutputWithPoolingAndCrossAttentions for specific outputs.
        """
        # 从transformers.models.bert.modeling_bert.BertModel.prepare_inputs_for_generation复制而来
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
    ):
        # 获取输入的形状
        input_shape = input_ids.shape
        # 如果没有提供注意力遮罩（mask），则创建全为1的注意力遮罩
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果存在过去的键值对（past_key_values），则根据它进行修剪输入的decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留旧行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 对输入进行修剪，仅保留后缀部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回一个包含重排后的缓存信息的字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 从transformers.models.bert.modeling_bert.BertModel._reorder_cache复制而来
    def _reorder_cache(self, past_key_values, beam_idx):
        # 重新排序过去的键值对
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 对每一层的过去状态根据beam_idx重新排序并添加到元组中
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
@add_start_docstrings(
    """
    MarkupLM Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MARKUPLM_START_DOCSTRING,
)
class MarkupLMForQuestionAnswering(MarkupLMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 复制而来，将 bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)  # 初始化 MarkupLMModel 模型
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 初始化用于 QA 输出的线性层

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """MarkupLM Model with a `token_classification` head on top.""",
    MARKUPLM_START_DOCSTRING
)
class MarkupLMForTokenClassification(MarkupLMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制而来，将 bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)  # 初始化 MarkupLMModel 模型
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 初始化 dropout 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 初始化用于分类的线性层

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        # 输入的词索引序列，可以是张量或者空值
        self,
        input_ids: Optional[torch.Tensor] = None,
        # XPath 标签序列，可以是张量或者空值
        xpath_tags_seq: Optional[torch.Tensor] = None,
        # XPath 子句序列，可以是张量或者空值
        xpath_subs_seq: Optional[torch.Tensor] = None,
        # 注意力掩码，可以是张量或者空值
        attention_mask: Optional[torch.Tensor] = None,
        # 标记类型 ID，可以是张量或者空值
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID，可以是张量或者空值
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码，可以是张量或者空值
        head_mask: Optional[torch.Tensor] = None,
        # 输入的嵌入表示，可以是张量或者空值
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签，可以是张量或者空值
        labels: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回输出，默认为 None
        return_dict: Optional[bool] = None,
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 参数不为 None，则使用该参数；否则使用 self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用 self.markuplm 进行模型推理，传入各种参数如 input_ids, xpath_tags_seq 等，根据 return_dict 返回不同的结果格式

        sequence_output = outputs[0]
        # 获取模型输出中的序列输出

        prediction_scores = self.classifier(sequence_output)  # (batch_size, seq_length, node_type_size)
        # 使用 self.classifier 对序列输出进行分类预测，得到预测分数，形状为 (batch_size, seq_length, node_type_size)

        loss = None
        if labels is not None:
            # 如果提供了标签信息
            loss_fct = CrossEntropyLoss()
            # 使用交叉熵损失函数
            loss = loss_fct(
                prediction_scores.view(-1, self.config.num_labels),
                labels.view(-1),
            )
            # 计算预测分数和标签之间的损失值

        if not return_dict:
            # 如果 return_dict 为 False
            output = (prediction_scores,) + outputs[2:]
            # 构建输出元组，包含预测分数和额外的输出内容
            return ((loss,) + output) if loss is not None else output
            # 如果有损失值，返回损失和输出内容；否则只返回输出内容

        return TokenClassifierOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 如果 return_dict 为 True，以 TokenClassifierOutput 形式返回结果，包括损失、预测分数、隐藏状态和注意力权重
"""
在标记LM模型的基础上增加一个顶部的序列分类/回归头部（即在池化输出之上的线性层），例如用于GLUE任务。
"""
@add_start_docstrings(
    """
    在标记LM模型的基础上增加一个顶部的序列分类/回归头部（即在池化输出之上的线性层），例如用于GLUE任务。
    """,
    MARKUPLM_START_DOCSTRING,
)
class MarkupLMForSequenceClassification(MarkupLMPreTrainedModel):
    """
    从transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__复制而来，将bert->markuplm, Bert->MarkupLM。
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量
        self.config = config  # 存储配置信息

        self.markuplm = MarkupLMModel(config)  # 初始化标记LM模型
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 定义dropout层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 定义线性分类器

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MARKUPLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        """
        将输入的多个张量传递给MarkupLM模型以进行前向传播，支持多种输入和输出设置。
        """
        **kwargs,
    ) -> SequenceClassifierOutput:
        pass  # 实际的前向传播逻辑在这里未展示，需要根据实际情况填充
```