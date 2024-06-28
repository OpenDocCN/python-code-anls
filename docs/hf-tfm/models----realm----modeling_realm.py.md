# `.\models\realm\modeling_realm.py`

```py
# 导入必要的库和模块
import math  # 导入数学库
import os    # 导入操作系统相关的库
from dataclasses import dataclass  # 导入dataclass用于定义数据类
from typing import Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

# 导入自定义的模块和类
from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
)  # 导入模型输出相关的类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型相关的工具函数和类
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer  # 导入PyTorch工具函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入辅助函数和日志模块
from .configuration_realm import RealmConfig  # 导入REALM模型的配置类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)
# 以下是用于文档的预训练模型的名称和路径
_EMBEDDER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-embedder"
_ENCODER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-encoder"
_SCORER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-scorer"
_CONFIG_FOR_DOC = "RealmConfig"

# REALM的预训练模型的存档列表
REALM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/realm-cc-news-pretrained-embedder",
    "google/realm-cc-news-pretrained-encoder",
    "google/realm-cc-news-pretrained-scorer",
    "google/realm-cc-news-pretrained-openqa",
    "google/realm-orqa-nq-openqa",
    "google/realm-orqa-nq-reader",
    "google/realm-orqa-wq-openqa",
    "google/realm-orqa-wq-reader",
    # 查看所有REALM模型的完整列表：https://huggingface.co/models?filter=realm
]


def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入NumPy库
        import tensorflow as tf  # 导入TensorFlow库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取TensorFlow检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从TF模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 获取TF模型中的变量列表
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)  # 加载TF模型中的变量
        names.append(name)
        arrays.append(array)
    # 返回函数中的变量 `model`
    return model
# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert->Realm
# 定义 RealmEmbeddings 类，用于构建包含单词、位置和标记类型嵌入的总体嵌入。
class RealmEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 初始化单词嵌入层，将词汇表大小、隐藏层大小和填充标记 ID 作为参数
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，将最大位置嵌入大小和隐藏层大小作为参数
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化标记类型嵌入层，将类型词汇表大小和隐藏层大小作为参数
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 没有使用蛇形命名以保持与 TensorFlow 模型变量名一致，可以加载任何 TensorFlow 检查点文件
        # 初始化 LayerNorm 层，将隐藏层大小和层归一化 epsilon 作为参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，将隐藏层 dropout 概率作为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 在序列化时是连续的内存，并在导出时被导出
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，用于存储位置 IDs，扩展为 (1, max_position_embeddings)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，用于存储标记类型 IDs，初始化为与位置 IDs 相同形状的零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        # 这里定义了 forward 方法的输入参数和默认值
    # 定义函数的输入类型和返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 如果输入的 input_ids 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状，但是不包括最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，这里假设 input_shape 的第二个维度为序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为空，则从 self.position_ids 中切片获取对应位置的位置 ids
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置 token_type_ids 为构造函数中注册的缓冲区，通常为全零，用于在模型追踪时帮助用户
        # 如果 token_type_ids 为空，则检查模型是否具有 "token_type_ids" 属性
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 获取已注册的缓冲区的 token_type_ids，展开以匹配输入形状的第二个维度长度
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，创建全零的 token_type_ids 张量，与输入形状相同
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为空，则使用 word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据 token_type_ids 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将 inputs_embeds 和 token_type_embeddings 相加作为嵌入向量
        embeddings = inputs_embeds + token_type_embeddings

        # 如果 position_embedding_type 是 "absolute"，则添加位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 应用 LayerNorm 规范化嵌入向量
        embeddings = self.LayerNorm(embeddings)

        # 对嵌入向量应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量作为输出
        return embeddings
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制并将 Bert 替换为 Realm 的自注意力机制
class RealmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不是注意力头数的倍数且没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 用于生成查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 转置张量以适应注意力分数的计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # RealmSelfAttention 的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        # 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并将 Bert 替换为 Realm 的自输出层
class RealmSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，用于变换隐藏状态
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # RealmSelfOutput 的前向传播
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # 层归一化并添加输入张量，然后返回
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertAttention 复制并将 Bert 替换为 Realm 的注意力模块
# RealmAttention 类，继承自 nn.Module
class RealmAttention(nn.Module):
    # 初始化方法，接受 config 和 position_embedding_type 两个参数
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建 RealmSelfAttention 对象并赋值给 self.self 属性
        self.self = RealmSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 RealmSelfOutput 对象并赋值给 self.output 属性
        self.output = RealmSelfOutput(config)
        # 创建一个空集合用于存储被剪枝的注意力头的索引
        self.pruned_heads = set()

    # 剪枝注意力头的方法
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 方法，获取可剪枝头的索引和具体头的信息
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝 self 属性中的 query、key、value 线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 剪枝 self.output 属性中的 dense 线性层
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新 self 属性中的注意力头数量和总大小
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        # 将剪枝过的头索引添加到 self.pruned_heads 中
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
        # 调用 self.self 的前向传播方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将 self_outputs 的第一个元素作为输入，调用 self.output 的前向传播方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构建输出元组，包括 attention_output 和 self_outputs 的其余部分
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出 attentions，则添加它们
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 RealmIntermediate 类
class RealmIntermediate(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入特征维度为 config.hidden_size 转换为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串类型，则使用 ACT2FN 字典中对应的激活函数；否则直接使用 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过全连接层 dense
        hidden_states = self.dense(hidden_states)
        # 将全连接层的输出经过激活函数 intermediate_act_fn
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 RealmOutput 类
class RealmOutput(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为配置对象中的 intermediate_size，输出大小为 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入大小为 hidden_size，设置 epsilon 参数为配置对象中的 layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，设置丢弃概率为配置对象中的 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受两个张量作为输入，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入张量经过全连接层得到新的隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 对新的隐藏状态张量进行 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的隐藏状态张量与输入张量相加，然后经过 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制并修改为RealmLayer，用于Realm模型中的一个层
class RealmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化层的配置参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度，默认为1，用于指定输入张量中表示序列长度的维度
        self.seq_len_dim = 1
        # 创建RealmAttention对象，处理注意力机制
        self.attention = RealmAttention(config)
        # 是否作为解码器使用的标志
        self.is_decoder = config.is_decoder
        # 是否添加跨层注意力机制的标志
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨层注意力机制，必须作为解码器使用
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建具有绝对位置嵌入的RealmAttention对象，用于跨层注意力机制
            self.crossattention = RealmAttention(config, position_embedding_type="absolute")
        # Realm模型的中间层，负责前向传播中的中间处理
        self.intermediate = RealmIntermediate(config)
        # Realm模型的输出层，负责生成最终输出
        self.output = RealmOutput(config)

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
        # 定义函数签名，返回类型为元组，包含一个 torch.Tensor 类型的对象

        # 如果有过去的注意力头/值缓存，则从中提取解码器单向自注意力的缓存键/值对，位置在1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        
        # 使用 self.attention 方法进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 排除最后一个元素，因为它是自注意力的缓存
            present_key_value = self_attention_outputs[-1]  # 获取当前注意力的键/值对
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加自注意力
            # outputs 现在包含所有的输出元素，除了第一个元素，即自注意力输出

        # 初始化交叉注意力的键/值对为 None
        cross_attn_present_key_value = None
        
        # 如果是解码器并且提供了编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有 crossattention 属性，抛出 ValueError
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果有过去的注意力头/值缓存，则从中提取交叉注意力的缓存键/值对，位置在3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            # 使用 self.crossattention 方法进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            
            # 获取交叉注意力计算的输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 添加交叉注意力的输出
            
            # 将交叉注意力的键/值对添加到当前的注意力键/值对中的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向传播的分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 对注意力输出应用 feed forward 网络的一部分，并返回处理后的层输出
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制并修改为 RealmEncoder
class RealmEncoder(nn.Module):
    # RealmEncoder 类的初始化方法
    def __init__(self, config):
        super().__init__()
        # 将传入的配置信息保存到实例变量中
        self.config = config
        # 创建一个由多个 RealmLayer 组成的层列表，列表长度等于配置中指定的隐藏层数
        self.layer = nn.ModuleList([RealmLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认不启用梯度检查点
        self.gradient_checkpointing = False

    # RealmEncoder 类的前向传播方法
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
    # 返回的类型注解和输出类型，指示此函数返回一个元组，元素为torch.Tensor或BaseModelOutputWithPastAndCrossAttentions对象
    -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不输出隐藏状态，则将all_hidden_states设为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则将all_self_attentions设为空元组
        all_self_attentions = () if output_attentions else None
        # 如果不输出跨层注意力权重或模型配置不支持，则将all_cross_attentions设为空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果use_cache为True，则给出警告并将其设置为False，因为梯度检查点和缓存不兼容
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不使用缓存，则初始化next_decoder_cache为空元组
        next_decoder_cache = () if use_cache else None
        # 遍历每个Transformer层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在头部掩码，则将其从head_mask中取出
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果存在先前的键值对，则从past_key_values中取出
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用_gradient_checkpointing_func函数来执行梯度检查点，减少内存使用
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
                # 否则直接调用当前层的模块，计算输出
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
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置支持添加跨层注意力，则将当前层的跨层注意力权重添加到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则将各个结果组成元组返回，过滤掉为None的部分
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
        # 否则返回BaseModelOutputWithPastAndCrossAttentions对象，包含最终的输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 定义一个名为 RealmPooler 的类，继承自 nn.Module
class RealmPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    # 前向传播函数，接收隐藏状态 hidden_states，并返回池化后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 选择隐藏状态中的第一个标记对应的张量
        first_token_tensor = hidden_states[:, 0]
        # 将选择的张量通过全连接层进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 将线性变换后的结果通过激活函数处理
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


# 定义一个名为 RealmEmbedderOutput 的数据类，继承自 ModelOutput
@dataclass
class RealmEmbedderOutput(ModelOutput):
    """
    RealmEmbedder 模型的输出。

    Args:
        projected_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):
            投影分数。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            一个元组，包含 `torch.FloatTensor`（一个用于嵌入输出 + 一个用于每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            一个元组，包含 `torch.FloatTensor`（每层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    projected_score: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个名为 RealmScorerOutput 的数据类，继承自 ModelOutput
@dataclass
class RealmScorerOutput(ModelOutput):
    """
    RealmScorer 模型的输出。

    Args:
        relevance_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`):
            文件候选的相关性分数（softmax 之前）。
        query_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):
            源自查询嵌入的查询分数。
        candidate_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`):
            源自嵌入器的候选分数。
    """

    relevance_score: torch.FloatTensor = None
    query_score: torch.FloatTensor = None
    candidate_score: torch.FloatTensor = None


# 定义一个名为 RealmReaderOutput 的数据类，继承自 ModelOutput
@dataclass
class RealmReaderOutput(ModelOutput):
    """
    RealmReader 模型的输出。
    
    这里没有特定的参数和注释，仅作为占位使用。
    """
    pass
    # 定义函数的参数和返回值的类型注解，使用了 torch 库中的数据类型
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            总损失。
        retriever_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            检索器损失。
        reader_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            阅读器损失。
        retriever_correct (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*):
            检索器是否正确检测到包含答案的证据块。
        reader_correct (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*):
            阅读器是否正确检测到包含答案的文本片段候选。
        block_idx (`torch.LongTensor` of shape `()`):
            预测答案最有可能出现的检索到的证据块的索引。
        candidate (`torch.LongTensor` of shape `()`):
            预测答案最有可能出现的检索到的文本片段候选的索引。
        start_pos (`torch.IntTensor` of shape `()`):
            预测答案在 *RealmReader* 输入中起始位置的索引。
        end_pos (`torch.IntTensor` of shape `()`):
            预测答案在 *RealmReader* 输入中结束位置的索引。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态，包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            self-attention 头部的注意力权重，用于计算注意力头部的加权平均值。

    # 初始化所有变量为 None，这些变量用于存储模型输出的各种损失、正确性和位置信息等。
    loss: torch.FloatTensor = None
    retriever_loss: torch.FloatTensor = None
    reader_loss: torch.FloatTensor = None
    retriever_correct: torch.BoolTensor = None
    reader_correct: torch.BoolTensor = None
    block_idx: torch.LongTensor = None
    candidate: torch.LongTensor = None
    start_pos: torch.int32 = None
    end_pos: torch.int32 = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class RealmForOpenQAOutput(ModelOutput):
    """

    Outputs of [`RealmForOpenQA`] models.

    Args:
        reader_output (`dict`):
            Reader output.
        predicted_answer_ids (`torch.LongTensor` of shape `(answer_sequence_length)`):
            Predicted answer ids.
    """

    # 定义了一个数据类，用于封装 RealmForOpenQA 模型的输出结果
    reader_output: dict = None  # 用于存储阅读器模型的输出，是一个字典类型
    predicted_answer_ids: torch.LongTensor = None  # 存储预测的答案 id，是一个长整型张量


class RealmPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]  # 根据配置选择激活函数
        else:
            self.transform_act_fn = config.hidden_act  # 使用给定的激活函数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建一个 LayerNorm 层

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 全连接层的前向传播
        hidden_states = self.transform_act_fn(hidden_states)  # 应用激活函数
        hidden_states = self.LayerNorm(hidden_states)  # 应用 LayerNorm
        return hidden_states


class RealmLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RealmPredictionHeadTransform(config)  # 创建一个预测头变换模块

        # 输出权重与输入嵌入相同，但每个标记有一个只输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 创建一个线性层，用于预测词汇表中的词

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个偏置参数

        # 需要一个链接，以便偏置在调整标记嵌入大小时正确调整大小
        self.decoder.bias = self.bias  # 将创建的偏置赋给 decoder 层的偏置

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # 应用预测头变换
        hidden_states = self.decoder(hidden_states)  # 应用线性层进行预测
        return hidden_states


class RealmOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)  # 创建一个仅包含 MLM 预测头的模块

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)  # 使用预测头进行序列输出的预测
        return prediction_scores


class RealmScorerProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)  # 创建一个用于打分投影的预测头模块
        self.dense = nn.Linear(config.hidden_size, config.retriever_proj_size)  # 创建一个全连接层
        self.LayerNorm = nn.LayerNorm(config.retriever_proj_size, eps=config.layer_norm_eps)  # 创建一个 LayerNorm 层

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)  # 应用全连接层
        hidden_states = self.LayerNorm(hidden_states)  # 应用 LayerNorm
        return hidden_states


class RealmReaderProjection(nn.Module):
    # 此处添加 RealmReaderProjection 类的定义和实现
    pass
    # 初始化方法，接受一个配置对象并设置模型的各种参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存在实例变量中
        self.config = config
        # 创建一个线性层，输入大小为配置中的隐藏大小，输出大小为span_hidden_size * 2
        self.dense_intermediate = nn.Linear(config.hidden_size, config.span_hidden_size * 2)
        # 创建一个线性层，输入大小为span_hidden_size，输出大小为1
        self.dense_output = nn.Linear(config.span_hidden_size, 1)
        # 创建一个LayerNorm层，标准化大小为span_hidden_size的向量，epsilon值为config中的设定值
        self.layer_normalization = nn.LayerNorm(config.span_hidden_size, eps=config.reader_layer_norm_eps)
        # 创建ReLU激活函数实例
        self.relu = nn.ReLU()

    # 前向传播方法，接受隐藏状态和块掩码作为输入，输出阅读器的逻辑概率、候选开始位置和结束位置
    def forward(self, hidden_states, block_mask):
        # 内部函数，生成跨度候选
        def span_candidates(masks):
            """
            Generate span candidates.

            Args:
                masks: <bool> [num_retrievals, max_sequence_len]

            Returns:
                starts: <int32> [num_spans] ends: <int32> [num_spans] span_masks: <int32> [num_retrievals, num_spans]
                whether spans locate in evidence block.
            """
            # 获取掩码的形状信息
            _, max_sequence_len = masks.shape

            # 内部函数，根据宽度生成跨度
            def _spans_given_width(width):
                current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
                current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
                return current_starts, current_ends

            # 生成不同宽度下的起始点和结束点列表
            starts, ends = zip(*(_spans_given_width(w + 1) for w in range(self.config.max_span_width)))

            # 将列表合并成一个张量 [num_spans]
            starts = torch.cat(starts, 0)
            ends = torch.cat(ends, 0)

            # 根据开始和结束位置索引掩码张量 [num_retrievals, num_spans]
            start_masks = torch.index_select(masks, dim=-1, index=starts)
            end_masks = torch.index_select(masks, dim=-1, index=ends)
            span_masks = start_masks * end_masks

            return starts, ends, span_masks

        # 将掩码转换为得分，用于屏蔽无效候选
        def mask_to_score(mask, dtype=torch.float32):
            return (1.0 - mask.type(dtype)) * torch.finfo(dtype).min

        # 使用线性层处理隐藏状态 [reader_beam_size, max_sequence_len, span_hidden_size * 2]
        hidden_states = self.dense_intermediate(hidden_states)
        # 将处理后的隐藏状态分成开始和结束投影 [reader_beam_size, max_sequence_len, span_hidden_size]
        start_projection, end_projection = hidden_states.chunk(2, dim=-1)

        # 生成跨度候选及其对应的掩码 [reader_beam_size, num_candidates, span_hidden_size]
        candidate_starts, candidate_ends, candidate_mask = span_candidates(block_mask)

        # 根据候选开始和结束索引获取对应的投影向量 [reader_beam_size, num_candidates, span_hidden_size]
        candidate_start_projections = torch.index_select(start_projection, dim=1, index=candidate_starts)
        candidate_end_projections = torch.index_select(end_projection, dim=1, index=candidate_ends)
        candidate_hidden = candidate_start_projections + candidate_end_projections

        # 应用ReLU激活函数 [reader_beam_size, num_candidates, span_hidden_size]
        candidate_hidden = self.relu(candidate_hidden)
        # 应用LayerNorm进行标准化 [reader_beam_size, num_candidates, span_hidden_size]
        candidate_hidden = self.layer_normalization(candidate_hidden)
        # 使用线性层计算阅读器的逻辑概率，然后压缩维度 [reader_beam_size, num_candidates]
        reader_logits = self.dense_output(candidate_hidden).squeeze(-1)
        # 添加掩码转换为得分的结果到阅读器的逻辑概率中 [reader_beam_size, num_candidates]
        reader_logits += mask_to_score(candidate_mask, dtype=reader_logits.dtype)

        return reader_logits, candidate_starts, candidate_ends
# 定义一个多行文档字符串，描述了该模型是一个 PyTorch 的 torch.nn.Module 子类，用法与一般的 PyTorch 模块相同，
# 并建议查阅 PyTorch 文档以获取有关一般用法和行为的所有信息。
REALM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    
    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个空的文档字符串，将用于记录函数的输入说明
REALM_INPUTS_DOCSTRING = r"""
    
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取这些索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # 
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于在填充的标记索引上避免执行注意力操作。
            # 遮罩值选择在 `[0, 1]` 之间：
            # - 1 表示**未遮罩**的标记，
            # - 0 表示**遮罩**的标记。
            # 
            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 之间：
            # - 0 对应于 *句子 A* 的标记，
            # - 1 对应于 *句子 B* 的标记。
            # 
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选取范围为 `[0, config.max_position_embeddings - 1]`。
            # 
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选定头部的遮罩。遮罩值选择在 `[0, 1]` 之间：
            # - 1 表示头部**未遮罩**，
            # - 0 表示头部**遮罩**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，而不是传递 `input_ids`，您可以选择直接传递嵌入表示。
            # 如果您希望对如何将 *input_ids* 索引转换为关联向量有更多控制权，则这很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回的张量下的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回的张量下的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # RealmPreTrainedModel 类，用于处理权重初始化和预训练模型下载加载的抽象类

    config_class = RealmConfig
    # 类属性 config_class 指定为 RealmConfig，用于配置模型的配置类

    load_tf_weights = load_tf_weights_in_realm
    # 类属性 load_tf_weights 指定为 load_tf_weights_in_realm，用于加载 TF 格式的权重到 Realm 模型中

    base_model_prefix = "realm"
    # 类属性 base_model_prefix 指定为 "realm"，作为基础模型的前缀名称

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化模型的权重

        if isinstance(module, nn.Linear):
            # 如果模块是线性层
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化权重为正态分布，均值为 0，标准差为 self.config.initializer_range
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果有偏置项，则将偏置项初始化为 0
        elif isinstance(module, nn.Embedding):
            # 如果模块是嵌入层
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化权重为正态分布，均值为 0，标准差为 self.config.initializer_range
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                # 如果设置了 padding_idx，则将对应位置的权重初始化为 0
        elif isinstance(module, nn.LayerNorm):
            # 如果模块是 LayerNorm 层
            module.bias.data.zero_()
            # 将偏置项初始化为 0
            module.weight.data.fill_(1.0)
            # 将权重初始化为全 1

    def _flatten_inputs(self, *inputs):
        """Flatten inputs' shape to (-1, input_shape[-1])"""
        # 将输入张量的形状展平为 (-1, input_shape[-1])

        flattened_inputs = []
        # 初始化空列表，用于存储展平后的输入张量

        for tensor in inputs:
            # 遍历输入张量列表
            if tensor is None:
                flattened_inputs.append(None)
                # 如果张量为 None，则直接添加 None 到展平后的输入列表
            else:
                input_shape = tensor.shape
                # 获取张量的形状
                if len(input_shape) > 2:
                    tensor = tensor.view((-1, input_shape[-1]))
                    # 如果张量维度大于 2，则将其展平为 (-1, input_shape[-1])
                flattened_inputs.append(tensor)
                # 将展平后的张量添加到展平后的输入列表中

        return flattened_inputs
        # 返回展平后的输入列表


class RealmBertModel(RealmPreTrainedModel):
    """
    Same as the original BertModel but remove docstrings.
    """
    
    # RealmBertModel 类，继承自 RealmPreTrainedModel，与原始的 BertModel 类似，但删除了文档字符串

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # 调用父类 RealmPreTrainedModel 的初始化方法，传入配置参数 config

        self.config = config
        # 设置实例属性 config 为传入的配置参数 config

        self.embeddings = RealmEmbeddings(config)
        # 初始化 embeddings 层，使用 RealmEmbeddings 类，并传入配置参数 config
        self.encoder = RealmEncoder(config)
        # 初始化 encoder 层，使用 RealmEncoder 类，并传入配置参数 config

        self.pooler = RealmPooler(config) if add_pooling_layer else None
        # 如果 add_pooling_layer 为 True，则初始化 pooler 层为 RealmPooler 类，传入配置参数 config；否则设为 None

        # Weights initialization is mostly managed by other Realm models,
        # but we also have them initialized here to keep a consistency.
        # 权重初始化大部分由其他 Realm 模型管理，
        # 但我们在这里也进行初始化以保持一致性。
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
        # 返回 embeddings 层的 word_embeddings 属性作为输入嵌入层

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        # 设置 embeddings 层的 word_embeddings 属性为指定的 value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的注意力头进行剪枝

        for layer, heads in heads_to_prune.items():
            # 遍历要剪枝的层及其对应的头部列表
            self.encoder.layer[layer].attention.prune_heads(heads)
            # 调用 encoder 的指定层的 attention 对象的 prune_heads 方法，对指定头部列表进行剪枝
    # 定义模型的前向传播方法，处理模型的输入和返回输出
    def forward(
        self,
        input_ids=None,                    # 输入的token IDs
        attention_mask=None,               # 注意力掩码，指定哪些token需被注意
        token_type_ids=None,               # token类型IDs，用于区分句子A和句子B
        position_ids=None,                 # 位置IDs，指定每个token的位置信息
        head_mask=None,                    # 头部掩码，指定每个注意力头是否可用
        inputs_embeds=None,                # 输入的嵌入表示
        encoder_hidden_states=None,        # 编码器的隐藏状态
        encoder_attention_mask=None,       # 编码器的注意力掩码
        past_key_values=None,              # 用于存储循环计算的键值对
        use_cache=None,                    # 是否使用缓存
        output_attentions=None,            # 是否输出注意力权重
        output_hidden_states=None,         # 是否输出隐藏状态
        return_dict=None,                  # 是否返回一个字典作为输出
# 添加起始文档字符串和相关信息到 RealmEmbedder 类
@add_start_docstrings(
    "The embedder of REALM outputting projected score that will be used to calculate relevance score.",
    REALM_START_DOCSTRING,
)
class RealmEmbedder(RealmPreTrainedModel):
    # 定义共享权重的键列表
    _tied_weights_keys = ["cls.predictions.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 实例化 RealmBertModel 并传入配置
        self.realm = RealmBertModel(self.config)
        # 实例化 RealmScorerProjection 并传入配置
        self.cls = RealmScorerProjection(self.config)
        # 执行额外的初始化操作
        self.post_init()

    def get_input_embeddings(self):
        # 返回 RealmEmbedder 使用的输入嵌入
        return self.realm.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置 RealmEmbedder 使用的输入嵌入
        self.realm.embeddings.word_embeddings = value

    # 向模型前向方法添加起始文档字符串和输入说明
    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换前向方法的返回值文档字符串
    @replace_return_docstrings(output_type=RealmEmbedderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RealmEmbedderOutput]:
        """
        RealmEmbedder 的前向传播方法。

        Returns:
            如果 return_dict 为 False，则返回元组 (projected_score, hidden_states, attentions)。
            如果 return_dict 为 True，则返回 RealmEmbedderOutput 对象，其中包含 projected_score、hidden_states 和 attentions。
        """

        # 根据 return_dict 的值确定是否使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RealmBertModel 的前向传播
        realm_outputs = self.realm(
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

        # 获取池化后的输出，维度为 [batch_size, hidden_size]
        pooler_output = realm_outputs[1]
        # 通过 RealmScorerProjection 计算投影分数，维度为 [batch_size, retriever_proj_size]
        projected_score = self.cls(pooler_output)

        # 根据 return_dict 的值决定返回的结果类型
        if not return_dict:
            return (projected_score,) + realm_outputs[2:4]
        else:
            return RealmEmbedderOutput(
                projected_score=projected_score,
                hidden_states=realm_outputs.hidden_states,
                attentions=realm_outputs.attentions,
            )
    # 描述了 REALM 输出的评分器，生成候选文档的相关性分数（softmax 之前的分数）。
    # REALM_START_DOCSTRING 是一个可能是常量或字符串的变量或符号，可能用于文档字符串的起始。
    "The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).",
    REALM_START_DOCSTRING,
# RealmScorer 类的定义，继承自 RealmPreTrainedModel
class RealmScorer(RealmPreTrainedModel):
    r"""
    Args:
        query_embedder ([`RealmEmbedder`]):
            Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.
    """

    # 初始化方法，接受 config 和可选的 query_embedder 参数
    def __init__(self, config, query_embedder=None):
        super().__init__(config)

        # 创建 RealmEmbedder 对象并赋值给 self.embedder
        self.embedder = RealmEmbedder(self.config)

        # 如果 query_embedder 参数不为 None，则使用该参数作为 query_embedder；否则使用 self.embedder
        self.query_embedder = query_embedder if query_embedder is not None else self.embedder

        # 调用 post_init 方法，用于进一步初始化
        self.post_init()

    # 前向传播方法，使用 add_start_docstrings_to_model_forward 和 replace_return_docstrings 进行文档字符串的处理
    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=RealmScorerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        candidate_input_ids: Optional[torch.LongTensor] = None,
        candidate_attention_mask: Optional[torch.FloatTensor] = None,
        candidate_token_type_ids: Optional[torch.LongTensor] = None,
        candidate_inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# RealmKnowledgeAugEncoder 类的定义，继承自 RealmPreTrainedModel
@add_start_docstrings(
    "The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood"
    " loss.",
    REALM_START_DOCSTRING,
)
class RealmKnowledgeAugEncoder(RealmPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化方法，接受 config 参数
    def __init__(self, config):
        super().__init__(config)
        
        # 创建 RealmBertModel 对象并赋值给 self.realm
        self.realm = RealmBertModel(self.config)
        
        # 创建 RealmOnlyMLMHead 对象并赋值给 self.cls
        self.cls = RealmOnlyMLMHead(self.config)
        
        # 调用 post_init 方法，用于进一步初始化
        self.post_init()

    # 获取输入嵌入层的方法，返回 self.realm.embeddings.word_embeddings
    def get_input_embeddings(self):
        return self.realm.embeddings.word_embeddings

    # 设置输入嵌入层的方法，将 value 赋给 self.realm.embeddings.word_embeddings
    def set_input_embeddings(self, value):
        self.realm.embeddings.word_embeddings = value

    # 获取输出嵌入层的方法，返回 self.cls.predictions.decoder
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层的方法，将 new_embeddings 赋给 self.cls.predictions.decoder
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，使用 add_start_docstrings_to_model_forward 和 replace_return_docstrings 进行文档字符串的处理
    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("batch_size, num_candidates, sequence_length")
    )
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 forward，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，类型为可选的浮点张量
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs，类型为可选的长整型张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，类型为可选的长整型张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，类型为可选的浮点张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入，类型为可选的浮点张量
        relevance_score: Optional[torch.FloatTensor] = None,  # 相关性分数，类型为可选的浮点张量
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量
        mlm_mask: Optional[torch.LongTensor] = None,  # MLM 掩码，类型为可选的长整型张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态信息，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，类型为可选的布尔值
# 使用装饰器添加文档字符串到 RealmReader 类，描述其作用为 REALM 的阅读器。
@add_start_docstrings("The reader of REALM.", REALM_START_DOCSTRING)
class RealmReader(RealmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化函数，继承自 RealmPreTrainedModel，设置类的标签数量
        self.num_labels = config.num_labels

        # 创建 REALM 的 BERT 模型
        self.realm = RealmBertModel(config)
        # 创建仅包含 MLM 头部的模型
        self.cls = RealmOnlyMLMHead(config)
        # 创建用于 Realm 阅读器的投影层
        self.qa_outputs = RealmReaderProjection(config)

        # 执行后续初始化
        self.post_init()

    # 使用装饰器添加文档字符串到 forward 方法，描述其输入和输出
    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("reader_beam_size, sequence_length"))
    @replace_return_docstrings(output_type=RealmReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        relevance_score: Optional[torch.FloatTensor] = None,
        block_mask: Optional[torch.BoolTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        has_answers: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
REALM_FOR_OPEN_QA_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token (should not be used in this model by design).
            
            [What are token type IDs?](../glossary#token-type-ids)
        answer_ids (`list` of shape `(num_answers, answer_length)`, *optional*):
            Answer ids for computing the marginal log-likelihood loss. Indices should be in `[-1, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-1` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
`RealmForOpenQA` 用于端到端的开放域问答。

该类继承自 `RealmPreTrainedModel`。
"""
class RealmForOpenQA(RealmPreTrainedModel):
    def __init__(self, config, retriever=None):
        """
        初始化方法，用于实例化一个 `RealmForOpenQA` 对象。

        Args:
            config (`PretrainedConfig`): 包含该模型配置信息的配置对象。
            retriever (`Optional`): 用于检索的对象，默认为 `None`。
        """
        super().__init__(config)
        self.embedder = RealmEmbedder(config)  # 实例化一个 `RealmEmbedder` 对象
        self.reader = RealmReader(config)  # 实例化一个 `RealmReader` 对象
        self.register_buffer(
            "block_emb",
            torch.zeros(()).new_empty(
                size=(config.num_block_records, config.retriever_proj_size),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        )
        self.retriever = retriever  # 设置检索器对象

        self.post_init()  # 调用初始化后处理方法

    @property
    def searcher_beam_size(self):
        """
        获取搜索器的 beam size。在训练模式下返回 `config.searcher_beam_size`，
        否则返回 `config.reader_beam_size`。

        Returns:
            `int`: beam size 的大小。
        """
        if self.training:
            return self.config.searcher_beam_size
        return self.config.reader_beam_size

    def block_embedding_to(self, device):
        """
        将 `self.block_emb` 发送到指定的设备。

        Args:
            device (`str` or `torch.device`):
                要发送 `self.block_emb` 的目标设备。
        """
        self.block_emb = self.block_emb.to(device)

    @add_start_docstrings_to_model_forward(REALM_FOR_OPEN_QA_DOCSTRING.format("1, sequence_length"))
    @replace_return_docstrings(output_type=RealmForOpenQAOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        answer_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        模型的前向传播方法，用于执行推断或训练。

        Args:
            input_ids (`Optional[torch.LongTensor]`):
                输入的 token IDs。
            attention_mask (`Optional[torch.FloatTensor]`, optional):
                注意力掩码张量，默认为 `None`。
            token_type_ids (`Optional[torch.LongTensor]`, optional):
                分段 token IDs，默认为 `None`。
            answer_ids (`Optional[torch.LongTensor]`, optional):
                答案 token IDs，默认为 `None`。
            return_dict (`Optional[bool]`, optional):
                是否返回字典作为输出，默认为 `None`。

        Returns:
            `RealmForOpenQAOutput` 或者是一个字典，包含模型输出的各种信息。
        """
```