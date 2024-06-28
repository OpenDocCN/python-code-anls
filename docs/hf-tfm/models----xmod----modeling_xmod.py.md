# `.\models\xmod\modeling_xmod.py`

```
# coding=utf-8
# 版权 2023 Meta AI Team 和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证版本 2.0（“许可证”）授权;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“现状”分发软件，
# 没有任何形式的明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。
"""PyTorch X-MOD 模型。"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xmod import XmodConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型存档列表
XMOD_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xmod-base",
    "facebook/xmod-large-prenorm",
    "facebook/xmod-base-13-125k",
    "facebook/xmod-base-30-125k",
    "facebook/xmod-base-30-195k",
    "facebook/xmod-base-60-125k",
    "facebook/xmod-base-60-265k",
    "facebook/xmod-base-75-125k",
    "facebook/xmod-base-75-269k",
    # 查看所有 X-MOD 模型，请访问 https://huggingface.co/models?filter=xmod
]

# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制并将 Roberta->Xmod
class XmodEmbeddings(nn.Module):
    """
    与 BertEmbeddings 相同，但对于位置嵌入索引进行了微小调整。
    """

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制
    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，用于将词汇索引映射到隐藏表示空间，大小为 vocab_size * hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，用于表示输入序列中每个位置的位置编码，大小为 max_position_embeddings * hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，用于区分不同类型的输入标记（如 segment A 和 segment B），大小为 type_vocab_size * hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建层归一化层，对隐藏表示进行归一化处理，大小为 hidden_size，epsilon 为 layer_norm_eps
        # 注：LayerNorm 命名未使用蛇形命名，以与 TensorFlow 模型变量名保持一致，便于加载 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，用于在训练过程中进行随机失活以防止过拟合，概率为 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区 position_ids，用于存储位置编码向量，大小为 1 * max_position_embeddings
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区 token_type_ids，用于存储标记类型向量，大小与 position_ids 相同，类型为 long
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置 padding_idx，用于指定词嵌入层中的填充标记索引
        self.padding_idx = config.pad_token_id
        # 重新定义位置嵌入层，指定填充标记索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    ):
        # 如果未提供位置 ids，则根据输入的 token ids 创建位置 ids。任何填充的 token 保持填充状态。
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 根据输入的 token ids 或者 embeddings 确定输入形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，通常情况下全为零，用于在不传递 token_type_ids 的情况下跟踪模型
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则使用输入的 input_ids 获取 embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的 embeddings
        embeddings = inputs_embeds + token_type_embeddings

        # 如果是绝对位置编码方式，添加位置 embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 应用 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        
        # 应用 dropout
        embeddings = self.dropout(embeddings)
        
        # 返回最终的 embeddings
        return embeddings

    # 从输入的 embeddings 直接生成位置 ids，因为无法推断哪些是填充的，因此生成连续的位置 ids
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成连续的位置 ids
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->Xmod
class XmodSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，若不能且未定义embedding_size则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        
        # 如果使用相对位置嵌入，初始化距离嵌入的Embedding层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器模式
        self.is_decoder = config.is_decoder

    # 将输入张量变形以便计算注意力分数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,



class XmodSelfOutput(nn.Module):
    # Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput.__init__
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # dropout层
        hidden_states = self.dropout(hidden_states)
        # 添加输入张量并返回结果
        hidden_states = hidden_states + input_tensor
        return hidden_states


class XmodAttention(nn.Module):
    # 初始化函数，用于初始化一个自注意力模块和一个自注意力输出模块
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建自注意力模块，根据配置和位置嵌入类型进行初始化
        self.self = XmodSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建自注意力输出模块，根据配置进行初始化
        self.output = XmodSelfOutput(config)
        # 存储需要剪枝的注意力头的索引集合
        self.pruned_heads = set()
        # 标记是否使用预层归一化
        self.pre_norm = config.pre_norm

    # 剪枝注意力头的方法，来自 transformers.models.roberta.modeling_roberta.RobertaAttention.prune_heads
    def prune_heads(self, heads):
        # 如果没有需要剪枝的头，直接返回
        if len(heads) == 0:
            return
        # 调用工具函数找到可以剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对自注意力模块的查询、键、值矩阵执行剪枝操作
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 对输出模块的线性层执行剪枝操作
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝过的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 正向传播函数，接受输入的隐藏状态和各种掩码，返回处理后的输出
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
        # 将原始输入作为残差连接的一部分
        residual = hidden_states
        # 如果使用预层归一化，则在自注意力输出之前对隐藏状态进行归一化处理
        if self.pre_norm:
            hidden_states = self.output.LayerNorm(hidden_states)
        # 调用自注意力模块进行计算，获取自注意力输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力输出和残差连接，送入自注意力输出模块
        attention_output = self.output(self_outputs[0], residual)
        # 如果不使用预层归一化，则在输出模块之后对注意力输出进行归一化处理
        if not self.pre_norm:
            attention_output = self.output.LayerNorm(attention_output)
        # 将注意力输出和可能的注意力权重返回作为结果
        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，还可以添加注意力权重
        return outputs
# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate

# 定义了一个名为XmodIntermediate的神经网络模块，继承自nn.Module
class XmodIntermediate(nn.Module):
    # 初始化函数，接收一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 使用线性层将输入特征从config.hidden_size转换为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据config中的hidden_act字段选择激活函数，并存储在self.intermediate_act_fn中
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接收一个torch.Tensor类型的hidden_states作为输入，返回一个torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入特征通过self.dense线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 使用存储在self.intermediate_act_fn中的激活函数处理转换后的特征
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义了一个名为XmodAdapter的神经网络模块，继承自nn.Module
class XmodAdapter(nn.Module):
    # 初始化函数，接收一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 计算适配器的瓶颈层大小，作为config.hidden_size除以config.adapter_reduction_factor的结果
        self.bottleneck_size = config.hidden_size // config.adapter_reduction_factor
        # 第一个线性层，将输入特征从config.hidden_size转换为self.bottleneck_size
        self.dense1 = nn.Linear(config.hidden_size, self.bottleneck_size)
        # 第二个线性层，将self.bottleneck_size大小的特征转换回config.hidden_size
        self.dense2 = nn.Linear(self.bottleneck_size, config.hidden_size)
        # 根据config中的hidden_act字段选择激活函数，并存储在self.adapter_act_fn中
        if isinstance(config.hidden_act, str):
            self.adapter_act_fn = ACT2FN[config.hidden_act]
        else:
            self.adapter_act_fn = config.hidden_act

    # 前向传播函数，接收一个torch.Tensor类型的hidden_states作为输入，返回一个torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入特征通过self.dense1线性层进行转换
        hidden_states = self.dense1(hidden_states)
        # 使用存储在self.adapter_act_fn中的激活函数处理转换后的特征
        hidden_states = self.adapter_act_fn(hidden_states)
        # 将特征再通过self.dense2线性层转换回config.hidden_size大小
        hidden_states = self.dense2(hidden_states)
        return hidden_states


# 定义了一个名为XmodOutput的神经网络模块，继承自nn.Module
class XmodOutput(nn.Module):
    # 初始化函数，接收一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 将intermediate_size大小的特征转换为config.hidden_size大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm层，对config.hidden_size大小的特征进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 是否在适配器之前应用LayerNorm
        self.ln_before_adapter = config.ln_before_adapter
        # Dropout层，根据config.hidden_dropout_prob概率随机丢弃部分特征
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 是否使用适配器层的LayerNorm
        if config.adapter_layer_norm:
            self.adapter_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.adapter_layer_norm = None
        # 是否重复使用适配器层的LayerNorm
        self.adapter_reuse_layer_norm = config.adapter_reuse_layer_norm
        # 为每种语言创建一个适配器模块，并存储在self.adapter_modules中
        self.adapter_modules = nn.ModuleDict({})
        for language in config.languages:
            self.adapter_modules[str(language)] = XmodAdapter(config)

    # 前向传播函数，接收torch.Tensor类型的hidden_states、input_tensor和lang_ids作为输入，返回一个torch.Tensor
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        # 将intermediate_size大小的特征转换为config.hidden_size大小
        hidden_states = self.dense(hidden_states)
        # 随机丢弃部分特征
        hidden_states = self.dropout(hidden_states)
        # 将丢弃的特征与输入特征相加
        hidden_states = hidden_states + input_tensor
        # 使用语言适配器层处理hidden_states
        hidden_states = self.lang_adapter(lang_ids, hidden_states)
        return hidden_states
    # 处理具有相同语言ID的后续样本，以并行方式处理
    lang_ids, lang_lengths = torch.unique_consecutive(lang_ids, return_counts=True)

    # 如果不在适配器之前进行LayerNorm，则将隐藏状态保留为残差连接的一部分
    if not self.ln_before_adapter:
        residual = hidden_states

    # 如果定义了适配器的LayerNorm，应用LayerNorm到隐藏状态
    if self.adapter_layer_norm is not None:
        hidden_states = self.adapter_layer_norm(hidden_states)
    # 否则，如果允许重用层归一化，则应用默认的LayerNorm到隐藏状态
    elif self.adapter_reuse_layer_norm:
        hidden_states = self.LayerNorm(hidden_states)

    # 如果在适配器之前进行LayerNorm，则将隐藏状态保留为残差连接的一部分
    if self.ln_before_adapter:
        residual = hidden_states

    # 将隐藏状态按照语言长度分割，以便按语言ID处理
    split_hidden_states = torch.split(hidden_states, lang_lengths.tolist(), 0)
    lang_wise_outputs = []
    # 遍历每个语言ID及其对应的分割隐藏状态
    for i, (lang_id, split_hidden_state) in enumerate(zip(lang_ids, split_hidden_states)):
        # 获取语言ID对应的适配器模块，并对分割的隐藏状态应用该适配器
        lang = list(self.adapter_modules.keys())[int(lang_id.item())]
        lang_wise_outputs.append(self.adapter_modules[lang](split_hidden_state))
    # 合并所有语言ID的适配器输出结果
    hidden_states = torch.cat(lang_wise_outputs, 0)

    # 应用dropout到合并后的隐藏状态
    hidden_states = self.dropout(hidden_states)
    # 将dropout后的结果与残差连接进行相加
    hidden_states += residual

    # 返回处理后的隐藏状态作为最终输出
    return hidden_states
# 定义一个名为 XmodLayer 的自定义神经网络层，继承自 nn.Module
class XmodLayer(nn.Module):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        
        # 设置当前层的 chunk_size_feed_forward 属性，从 config 中获取
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        
        # 设置序列长度维度为 1
        self.seq_len_dim = 1
        
        # 初始化一个 XmodAttention 对象，并赋值给 self.attention 属性
        self.attention = XmodAttention(config)
        
        # 根据配置设置是否作为解码器
        self.is_decoder = config.is_decoder
        
        # 根据配置设置是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            
            # 初始化一个带绝对位置嵌入的 XmodAttention 对象，并赋值给 self.crossattention 属性
            self.crossattention = XmodAttention(config, position_embedding_type="absolute")
        
        # 初始化一个 XmodIntermediate 对象，并赋值给 self.intermediate 属性
        self.intermediate = XmodIntermediate(config)
        
        # 初始化一个 XmodOutput 对象，并赋值给 self.output 属性
        self.output = XmodOutput(config)
        
        # 根据配置设置是否进行预正则化，并赋值给 self.pre_norm 属性
        self.pre_norm = config.pre_norm

    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 如果有过去的注意力键/值缓存数据，则仅保留解码器自注意力部分的键/值缓存数据
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # 使用自注意力层处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # 如果是解码器模式，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器模式，添加自注意力权重
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        
        # 如果是解码器且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 检查是否有交叉注意力层
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            
            # 如果有过去的注意力键/值缓存数据，则仅保留交叉注意力部分的键/值缓存数据
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # 使用交叉注意力层处理注意力输出、注意力掩码、头掩码、编码器隐藏状态及其掩码
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 添加交叉注意力权重
            
            # 将交叉注意力缓存添加到当前注意力键/值元组的第三和第四位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        residual = attention_output
        
        # 如果使用预层归一化，对注意力输出进行归一化处理
        if self.pre_norm:
            attention_output = self.output.LayerNorm(attention_output)
        
        # 对注意力输出应用前馈分块处理
        intermediate_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        
        # 使用输出层处理中间输出、残差连接和语言 ID
        layer_output = self.output(intermediate_output, residual, lang_ids)
        
        # 如果不使用预层归一化，对层输出进行归一化处理
        if not self.pre_norm:
            layer_output = self.output.LayerNorm(layer_output)
        
        outputs = (layer_output,) + outputs

        # 如果是解码器模式，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 对注意力输出进行前馈分块处理
        return self.intermediate(attention_output)
# 定义一个名为 XmodEncoder 的类，继承自 nn.Module
class XmodEncoder(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数保存在实例变量 self.config 中
        self.config = config
        # 创建一个由 XmodLayer 实例组成的 ModuleList，长度为 config.num_hidden_layers
        self.layer = nn.ModuleList([XmodLayer(config) for _ in range(config.num_hidden_layers)])
        # 根据配置决定是否进行预归一化
        self.is_pre_norm = config.pre_norm
        # 如果开启了预归一化，则创建一个 LayerNorm 层
        if self.is_pre_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 定义一个名为 XmodPooler 的类，继承自 nn.Module
class XmodPooler(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入大小为 config.hidden_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活函数实例
        self.activation = nn.Tanh()

    # 前向传播方法，接受一个输入参数 hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从 hidden_states 中取出第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将该隐藏状态作为输入，经过线性层
        pooled_output = self.dense(first_token_tensor)
        # 经过 Tanh 激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 定义一个名为 XmodPreTrainedModel 的类，继承自 PreTrainedModel
class XmodPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 类变量，指定该类的配置类为 XmodConfig
    config_class = XmodConfig
    # 基础模型前缀，设置为 "roberta"
    base_model_prefix = "roberta"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 定义一个内部方法 _init_weights，用于初始化模型的权重
    # 该方法来自于 transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了 padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1
            module.weight.data.fill_(1.0)
    # 设置模型的默认语言代码。当输入中未指定语言时使用。
    def set_default_language(self, language: str):
        # 如果指定的语言不在支持的语言列表中，则抛出数值错误异常。
        if language not in self.config.languages:
            raise ValueError(
                f"{self} does not have an adapter for {language}. Supported languages: {list(self.config.languages)}"
            )
        # 将默认语言设置为给定的语言代码。
        self.config.default_language = language

    # 冻结模型的嵌入和语言适配器。通常在模型在下游任务上进行微调之前应用此操作。
    def freeze_embeddings_and_language_adapters(self):
        # 输出信息日志，表明正在冻结嵌入层。
        logger.info("Freezing embeddings")
        # 遍历 RoBERTa 模型的嵌入层参数，并设置它们为不需要梯度。
        for parameter in self.roberta.embeddings.parameters():
            parameter.requires_grad = False
        # 输出信息日志，表明正在冻结适配器。
        logger.info("Freezing adapters")
        # 遍历 RoBERTa 模型的每个编码层。
        for layer in self.roberta.encoder.layer:
            # 如果编码层有输出适配器的层归一化，则将其参数设置为不需要梯度。
            if layer.output.adapter_layer_norm is not None:
                for parameter in layer.output.adapter_layer_norm.parameters():
                    parameter.requires_grad = False
            # 遍历编码层的输出适配器模块，并将其参数设置为不需要梯度。
            for parameter in layer.output.adapter_modules.parameters():
                parameter.requires_grad = False
# XMOD_START_DOCSTRING 是一个多行字符串常量，用于为该模型的文档提供详细说明和文档链接。
# 这个模型继承自 PreTrainedModel 类。查看超类文档以了解库实现的通用方法（如下载或保存模型、调整输入嵌入大小、修剪头等）。
# 同时，这个模型也是 PyTorch 的 torch.nn.Module 的子类，可以像常规的 PyTorch 模块一样使用。有关一般使用和行为的问题，请参考 PyTorch 文档。
# 参数：
#     config ([`XmodConfig`]): 该模型的配置类，包含模型的所有参数。使用配置文件进行初始化不会加载与模型相关的权重，只加载配置。查看 `~PreTrainedModel.from_pretrained` 方法以加载模型权重。

XMOD_INPUTS_DOCSTRING = r"""
    # 接受输入参数并进行处理的函数
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            输入序列标记在词汇表中的索引。
    
            可以使用 [`AutoTokenizer`] 获取这些索引。详情请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
    
            [什么是输入 ID？](../glossary#input-ids)
        lang_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            每个样本应该激活的语言适配器的索引。默认情况下为 `self.config.default_language` 对应的索引。
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            遮盖掩盖填充标记索引的注意力掩码。掩码值选取在 `[0, 1]` 之间：
    
            - 1 表示**未掩盖**的标记，
            - 0 表示**已掩盖**的标记。
    
            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            指示输入的第一部分和第二部分的段标记索引。索引选取在 `[0, 1]` 之间：
    
            - 0 对应*句子 A* 的标记，
            - 1 对应*句子 B* 的标记。
    
            [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。索引选取范围为 `[0, config.max_position_embeddings - 1]`。
    
            [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            自注意力模块中选择性屏蔽的头部掩码。掩码值选取在 `[0, 1]` 之间：
    
            - 1 表示**未掩盖**的头部，
            - 0 表示**已掩盖**的头部。
    
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            可选参数，而非传递 `input_ids`，你可以直接传递嵌入表示。如果你想要更多控制如何将 `input_ids` 索引转换为相关向量，而不是使用模型的内部嵌入查找矩阵时，这将非常有用。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。请参阅返回张量下的 `attentions` 以获取更多详情。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。请参阅返回张量下的 `hidden_states` 以获取更多详情。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
    """

    @add_start_docstrings(
        "The bare X-MOD Model transformer outputting raw hidden-states without any specific head on top.",
        XMOD_START_DOCSTRING,
    )
    class XmodModel(XmodPreTrainedModel):
        """

        The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
        cross-attention is added between the self-attention layers, following the architecture described in *Attention is
        all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
        Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
        to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
        `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

        .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

        """

        # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Xmod
        def __init__(self, config, add_pooling_layer=True):
            # 调用父类构造函数，初始化模型
            super().__init__(config)
            # 保存配置信息
            self.config = config

            # 初始化词嵌入层
            self.embeddings = XmodEmbeddings(config)
            # 初始化编码器层
            self.encoder = XmodEncoder(config)

            # 如果需要添加池化层，则初始化池化层；否则设为None
            self.pooler = XmodPooler(config) if add_pooling_layer else None

            # 执行初始化权重和最终处理
            self.post_init()

        # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.get_input_embeddings
        def get_input_embeddings(self):
            # 返回词嵌入层中的词嵌入矩阵
            return self.embeddings.word_embeddings

        # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.set_input_embeddings
        def set_input_embeddings(self, value):
            # 设置词嵌入层的词嵌入矩阵
            self.embeddings.word_embeddings = value

        # Copied from transformers.models.roberta.modeling_roberta.RobertaModel._prune_heads
        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            # 遍历需要修剪的层和头部，并调用注意力层的修剪方法
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        # 为模型的前向方法添加文档字符串注释
    # 前向传播函数，用于模型的前向推理过程
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可选的 Torch 张量
        lang_ids: Optional[torch.LongTensor] = None,  # 输入的语言 ID，可选的长整型 Torch 张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选的 Torch 张量
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选的 Torch 张量
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，可选的 Torch 张量
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，可选的 Torch 张量
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，可选的 Torch 张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，可选的 Torch 张量
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力遮罩，可选的 Torch 张量
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，可选的浮点数 Torch 张量列表
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式结果，可选的布尔值
# 为 XmodForCausalLM 类添加文档字符串，描述其作为 CLM 微调模型的功能
@add_start_docstrings(
    "X-MOD Model with a `language modeling` head on top for CLM fine-tuning.",
    XMOD_START_DOCSTRING,
)
class XmodForCausalLM(XmodPreTrainedModel):
    # 定义一个列表，包含 lm_head.decoder.weight 和 lm_head.decoder.bias，表示这些权重是被绑定的
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.__init__ 复制而来，用于初始化 XmodForCausalLM 类
    def __init__(self, config):
        super().__init__(config)

        # 如果 config 中 is_decoder 不为真，则记录警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `XmodLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 使用 XmodModel 类初始化 self.roberta，不添加池化层
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 使用 XmodLMHead 类初始化 self.lm_head
        self.lm_head = XmodLMHead(config)

        # 调用 self.post_init() 方法，用于初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        # 返回 lm_head.decoder，即输出的嵌入层权重
        return self.lm_head.decoder

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.set_output_embeddings 复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置 lm_head.decoder 为新的嵌入层权重 new_embeddings
        self.lm_head.decoder = new_embeddings

    # 使用 @add_start_docstrings_to_model_forward 装饰器，添加文档字符串描述 forward 方法的输入参数
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处应该继续添加后续的 forward 方法内容，但未提供完整的代码以供参考
        pass

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.prepare_inputs_for_generation 复制而来
    # 准备生成过程中的输入，根据给定的参数调整输入数据和注意力掩码
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入数据的形状信息
        input_shape = input_ids.shape

        # 如果未提供注意力掩码，则创建一个全为1的注意力掩码，与输入形状相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果提供了过去的键值对（past_key_values），则调整输入数据，仅保留未处理部分
        if past_key_values is not None:
            # 获取过去键值对中第一层的长度信息
            past_length = past_key_values[0][0].shape[2]

            # 如果输入数据的长度大于过去处理的长度，则移除前缀部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个输入ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 调整输入数据，仅保留需要生成的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含调整后输入数据、注意力掩码和过去键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM._reorder_cache 复制而来
    # 重新排序过去的键值对，以便与beam搜索索引对应
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排序后的过去键值对
        reordered_past = ()
        
        # 遍历每一层的过去键值对，根据beam搜索索引重新排序
        for layer_past in past_key_values:
            reordered_past += (
                # 对于每一个过去状态，根据beam搜索索引重新选择对应的位置，并将结果添加到元组中
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        
        # 返回重新排序后的过去键值对
        return reordered_past
# 定义 XmodForMaskedLM 类，用于 X-MOD 模型并在顶部添加语言建模头部
@add_start_docstrings(
    """X-MOD Model with a `language modeling` head on top.""",
    XMOD_START_DOCSTRING,
)
class XmodForMaskedLM(XmodPreTrainedModel):
    # 定义共享权重的键列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.__init__ 复制而来，将 Roberta 替换为 Xmod
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)

        # 如果配置为解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `XmodForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 XmodModel 实例，并禁用添加池化层选项
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 初始化 XmodLMHead 实例
        self.lm_head = XmodLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        # 返回语言建模头部的解码器权重
        return self.lm_head.decoder

    # 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.set_output_embeddings 复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置语言建模头部的解码器权重为新的嵌入
        self.lm_head.decoder = new_embeddings

    # 添加注释到模型的前向方法，使用 XMOD_INPUTS_DOCSTRING 格式化的输入文档字符串
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # Determine whether to use return_dict based on input or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to the Roberta model for processing
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Extract the sequence output from Roberta model outputs
        sequence_output = outputs[0]
        
        # Generate prediction scores using the language model head
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # Calculate masked language modeling loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # If return_dict is False, prepare output tuple including prediction scores and additional outputs
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # If return_dict is True, return MaskedLMOutput with detailed outputs
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead
class XmodLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入维度为 config.hidden_size 的特征映射到相同维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层，用于归一化输入特征
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化一个全连接层，将特征映射到词汇表大小的维度
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 初始化一个偏置参数，用于 decoder 层
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 将特征传入全连接层进行线性变换
        x = self.dense(features)
        # 使用 GELU 激活函数
        x = gelu(x)
        # 将结果归一化
        x = self.layer_norm(x)

        # 将归一化后的结果映射回词汇表维度，加上偏置
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果两个权重被断开（在 TPU 上或者偏置被重新调整大小时），将它们绑定在一起
        # 用于加速兼容性，避免破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    X-MOD Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    XMOD_START_DOCSTRING,
)
class XmodForSequenceClassification(XmodPreTrainedModel):
    # Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.__init__ with Roberta->Xmod
    def __init__(self, config):
        super().__init__(config)
        # 初始化分类任务的模型，包括 XMOD 模型和分类器
        self.num_labels = config.num_labels
        self.config = config

        # 初始化 XMOD 模型，不包括池化层
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 初始化分类头部
        self.classifier = XmodClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 模型进行前向传播，获取输出
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 使用分类器模型进行 logits 计算
        logits = self.classifier(sequence_output)

        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据 num_labels 类型进行定义
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
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

        # 如果 return_dict 为 False，则返回输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回 SequenceClassifierOutput 对象，包含损失、logits、隐藏状态和注意力
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
X-MOD Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""
# 使用 XMOD_START_DOCSTRING 和 add_start_docstrings 装饰器为类添加文档字符串
@add_start_docstrings(
    """
    X-MOD Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    XMOD_START_DOCSTRING,
)
# 定义 XmodForMultipleChoice 类，继承自 XmodPreTrainedModel
class XmodForMultipleChoice(XmodPreTrainedModel):
    # 从 transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice.__init__ 复制过来并修改了 Roberta 为 Xmod
    def __init__(self, config):
        # 调用父类 XmodPreTrainedModel 的构造函数
        super().__init__(config)

        # 初始化 self.roberta 属性为 XmodModel(config)
        self.roberta = XmodModel(config)
        # 初始化 self.dropout 属性为 Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化 self.classifier 属性为 Linear(config.hidden_size, 1)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器为 forward 方法添加文档字符串
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 定义 forward 方法，接收多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 继续声明其他输入参数
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 return_dict 参数确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算 num_choices，即选择数量，为 input_ids 的第二个维度大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 input_ids 展平为二维张量，以便处理
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果提供了 lang_ids，则将其重复以匹配 input_ids 的扁平化后形状
        flat_lang_ids = lang_ids.repeat(input_ids.size(0) * input_ids.size(1)) if lang_ids is not None else None
        # 将 position_ids 展平为二维张量，以便处理
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将 token_type_ids 展平为二维张量，以便处理
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 attention_mask 展平为二维张量，以便处理
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 inputs_embeds 展平为三维张量，以便处理
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用 RoBERTa 模型进行前向传播，传入扁平化后的各种输入和参数
        outputs = self.roberta(
            flat_input_ids,
            lang_ids=flat_lang_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取汇聚的输出向量
        pooled_output = outputs[1]

        # 对汇聚的输出向量应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类预测
        logits = self.classifier(pooled_output)
        # 将 logits 重塑为二维张量，以适应多选项任务的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了 labels，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，则返回未打包的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回打包为 MultipleChoiceModelOutput 的输出对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
X-MOD Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 引入自定义的文档字符串，描述了这个类是一个基于 X-MOD 模型的标记分类模型，通常用于命名实体识别（NER）等任务。

@add_start_docstrings(
    """
    XMOD_START_DOCSTRING,
    """
    # 使用装饰器 @add_start_docstrings，引入了一个文档字符串，可能是为了扩展其他模型的基础文档字符串。
)
class XmodForTokenClassification(XmodPreTrainedModel):
    # 从 transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.__init__ 复制而来，只是将 Roberta 替换为 Xmod
    def __init__(self, config):
        super().__init__(config)
        # 调用父类构造函数初始化模型配置

        self.num_labels = config.num_labels
        # 初始化标签数量，从配置中获取

        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 使用 XmodModel 初始化一个 XMOD 模型，关闭添加池化层的选项

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 如果配置中提供了分类器的 dropout，就使用它，否则使用隐藏层 dropout 的设置
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化一个 dropout 层，用于模型训练中的随机失活

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 初始化一个线性层，将隐藏状态映射到标签数量上

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器 @add_start_docstrings_to_model_forward，引入了一个文档字符串，可能是为了扩展模型前向传播的输入说明。
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 定义模型的前向传播方法，接受多个输入参数和可选的控制参数
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 不为 None，则使用给定的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 模型进行前向传播，获取输出结果
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出结果中获取序列输出
        sequence_output = outputs[0]

        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 将经过 dropout 的序列输出输入分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 将 logits 和 labels 展平并计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回 logits 和额外的 hidden states
        if not return_dict:
            output = (logits,) + outputs[2:]  # 包括 logits 和额外的 hidden states
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.roberta.modeling_roberta.RobertaClassificationHead 复制而来的 Xmod 分类头部模块，用于句子级别的分类任务。
class XmodClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 使用线性层将输入特征的大小从 config.hidden_size 转换为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 确定分类器的 dropout 概率，如果未指定，则使用 config.hidden_dropout_prob 的值
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 使用线性层将输入特征的大小从 config.hidden_size 转换为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 从 features 中选择第一个 token 的隐藏状态（相当于 [CLS] 标记）
        x = features[:, 0, :]
        # 对选定的特征进行 dropout 处理
        x = self.dropout(x)
        # 通过全连接层 dense 进行线性变换
        x = self.dense(x)
        # 对变换后的输出应用双曲正切激活函数
        x = torch.tanh(x)
        # 再次对输出进行 dropout 处理
        x = self.dropout(x)
        # 通过输出投影层 out_proj 进行最终的线性变换，得到分类任务的输出
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    X-MOD Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XMOD_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.__init__ 复制而来，用于支持抽取式问答任务的 X-MOD 模型
class XmodForQuestionAnswering(XmodPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        # 初始化模型的标签数量
        self.num_labels = config.num_labels

        # 使用 XmodModel 初始化 self.roberta，不添加池化层
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 使用线性层初始化 self.qa_outputs，将隐藏状态的大小转换为标签数量
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # XMOD 模型的前向传播函数，接受多种输入参数，并输出相应的结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
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
        # Determine whether to use return_dict based on self.config or provided argument
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Pass input tensors and optional arguments to the Roberta model
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # Get the sequence output from the Roberta model's outputs
        sequence_output = outputs[0]
    
        # Feed the sequence output into the QA outputs layer to get logits
        logits = self.qa_outputs(sequence_output)
    
        # Split logits into start and end logits and process them
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If start_positions or end_positions have more than one dimension, squeeze them
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
    
            # Define ignored_index and clamp positions to ignore out-of-sequence positions
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
    
            # Compute CrossEntropyLoss for start and end logits
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
    
            # Calculate total loss as the average of start_loss and end_loss
            total_loss = (start_loss + end_loss) / 2
    
        # If return_dict is False, return output tuple without loss
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        # If return_dict is True, return structured output using QuestionAnsweringModelOutput
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的input_ids中创建位置编码，替换非填充符号为它们的位置编号。位置编号从padding_idx+1开始，忽略填充符号。这段代码修改自fairseq的`utils.make_positions`。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: torch.Tensor 输入的张量，包含需要进行位置编码的元素
        padding_idx: int 填充符号的索引，非填充符号将被替换为其位置编号
        past_key_values_length: int 过去键值对的长度，用于计算增量索引

    Returns:
        torch.Tensor 返回与输入张量相同形状的张量，其中非填充符号被替换为它们的位置编号
    """
    # 创建一个掩码张量，指示输入张量中非填充符号的位置
    mask = input_ids.ne(padding_idx).int()
    # 计算每个非填充符号的位置编号，累积求和后加上past_key_values_length，并乘以掩码以保留填充符号位置的值
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将增量索引转换为长整型，并加上padding_idx以得到最终位置编号
    return incremental_indices.long() + padding_idx
```