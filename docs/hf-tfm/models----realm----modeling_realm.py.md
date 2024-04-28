# `.\transformers\models\realm\modeling_realm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指明 REALM 作者和 HuggingFace Inc. 团队
# 根据 Apache License, Version 2.0 授权进行使用，若不遵守则不得使用该文件
# 可以通过 http://www.apache.org/licenses/LICENSE-2.0 获取授权副本
# 除非适用法律要求或书面同意，否则不得分发分发基于 "AS IS" 基础的软件，不提供任何明示或暗示的担保或条件
# 请查看具体语言核实权限和限制
""" PyTorch REALM model."""

# 导入需要的模块
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入其他模块
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig

# 获取日志记录器
logger = logging.get_logger(__name__)
# 以下用于文档的示例路径
_EMBEDDER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-embedder"
_ENCODER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-encoder"
_SCORER_CHECKPOINT_FOR_DOC = "google/realm-cc-news-pretrained-scorer"
_CONFIG_FOR_DOC = "RealmConfig"

# REALM 预训练模型存档列表
REALM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/realm-cc-news-pretrained-embedder",
    "google/realm-cc-news-pretrained-encoder",
    "google/realm-cc-news-pretrained-scorer",
    "google/realm-cc-news-pretrained-openqa",
    "google/realm-orqa-nq-openqa",
    "google/realm-orqa-nq-reader",
    "google/realm-orqa-wq-openqa",
    "google/realm-orqa-wq-reader",
    # 查看所有 REALM 模型 https://huggingface.co/models?filter=realm
]

# 加载 TF 模型权重到 PyTorch 模型
def load_tf_weights_in_realm(model, config, tf_checkpoint_path):
    try:
        # 导入所需模块
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TF 检查点路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    # 返回 model 变量
    return model
# 从transformers.models.bert.modeling_bert.BertEmbeddings复制而来，将Bert->Realm
class RealmEmbeddings(nn.Module):
    """构建来自单词、位置和标记类型嵌入的嵌入。"""

    def __init__(self, config):
        super().__init__()
        # 单词嵌入：用于将单词转换为隐藏表示
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入：用于表示单词在句子中的位置
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 标记类型嵌入：用于区分不同类型的标记（例如句子A和句子B）
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm没有采用蛇形命名法，以保持与TensorFlow模型变量名称的一致性，并能够加载任何TensorFlow检查点文件
        # 层正则化：在隐藏表示的每个位置应用正则化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 丢失层：用于在模型中引入随机性，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids（1，长度位置嵌入）在内存中是连续的，并在序列化时导出
        # 位置嵌入类型：确定使用哪种类型的位置嵌入（绝对或相对）
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区：注册位置标识和标记类型标识的缓冲区，不会随着模型参数的更新而更新
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
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
    # 此函数用于根据输入的 input_ids 或 inputs_embeds 创建对应的 token 嵌入
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果输入了 input_ids，获取输入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 否则，获取 inputs_embeds 的形状，去除最后一个维度
        else:
            input_shape = inputs_embeds.size()[:-1]
    
        # 计算序列长度
        seq_length = input_shape[1]
    
        # 如果没有提供 position_ids，则使用预定义的 position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    
        # 如果没有提供 token_type_ids，则使用预定义的 token_type_ids
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    
        # 如果没有提供 inputs_embeds，则使用 word_embeddings 计算 inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 计算 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
    
        # 将 inputs_embeds 和 token_type_embeddings 相加得到最终的 embeddings
        embeddings = inputs_embeds + token_type_embeddings
        # 如果使用绝对位置编码，则添加位置编码
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行层归一化和dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # 返回最终的 embeddings
        return embeddings
# 这是 Realm 模型自注意力层的实现，用于计算每个输入位置的注意力分数
class RealmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，且没有配置 embedding_size，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量和大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义注意力权重的 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 设置位置嵌入的类型
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入tensor reshape 到注意力头的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 计算注意力分数并返回
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

# 这是 Realm 模型自注意力输出层的实现，用于整合注意力计算结果
class RealmSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义线性变换层、LayerNorm 层和 dropout 层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 计算注意力输出并应用 LayerNorm 和 dropout
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 定义 RealmAttention 类，继承自 nn.Module
class RealmAttention(nn.Module):
    # 定义初始化方法
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 创建 self 属性，并初始化为 RealmSelfAttention 类的实例
        self.self = RealmSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 output 属性，并初始化为 RealmSelfOutput 类的实例
        self.output = RealmSelfOutput(config)
        # 创建 pruned_heads 属性，并初始化为一个空的集合
        self.pruned_heads = set()

    # 定义剪枝操作方法
    def prune_heads(self, heads):
        # 如果待剪枝的 head 数为 0，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 方法找到可剪枝的 heads 和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的 heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义前向传播方法
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
        # 调用 self 属性的前向传播方法，获取 self_outputs
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 对 self_outputs 进行处理并传递给 output 属性的前向传播方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构造输出元组，附加注意力信息（如果有的话）
        outputs = (attention_output,) + self_outputs[1:]
        # 返回输���元组
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate with Bert->Realm 复制的 RealmIntermediate 类
class RealmIntermediate(nn.Module):
    # 定义初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建 dense 属性，并初始化为 nn.Linear 类的实例
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 hidden_act 是字符串类型，则使用 ACT2FN 字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # 否则直接使用 config 中定义的激活函数
        else:
            self.intermediate_act_fn = config.hidden_act

    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性层和激活函数进行变换
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回变换后的 hidden_states
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput with Bert->Realm 复制的 RealmOutput 类
class RealmOutput(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，将中间尺寸变换为隐藏尺寸
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNormalization 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态张量和输入张量作为参数，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 对丢弃后的隐藏状态与输入张量进行残差连接，并进行 LayerNormalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer中复制代码，并将Bert->Realm
class RealmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建RealmAttention对象
        self.attention = RealmAttention(config)
        # 是否是解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力且不是解码器，则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建具有绝对位置嵌入类型的RealmAttention对象
            self.crossattention = RealmAttention(config, position_embedding_type="absolute")
        # 创建RealmIntermediate对象
        self.intermediate = RealmIntermediate(config)
        # 创建RealmOutput对象
        self.output = RealmOutput(config)

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
        # 如果过去的键/值对存在，则将decoder自注意力缓存的键/值对放置在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
       
        cross_attn_present_key_value = None
        # 如果是decoder并且有encoder的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有跨注意力层，则引发值错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 跨注意力的缓存的键/值对在过去的键/值对元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行跨注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取跨注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加跨注意力

            # 将跨注意力缓存添加到现在的键/值对元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块技术对前向传播函数进行处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是decoder，则将注意力键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 前向传播函数分块
    def feed_forward_chunk(self, attention_output):
        # 经过中间层
        intermediate_output = self.intermediate(attention_output)
        # 经过输出层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码到Realm
class RealmEncoder(nn.Module):
    # 初始化函数，初始化RealmEncoder类
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 使用ModuleList创建一个包含多个RealmLayer对象的列表，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([RealmLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    # 前向传播函数
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
        # 如果设定要输出隐藏状态，则初始化一个空元组用于存储所有的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果设定要输出注意力分数，则初始化一个空元组用于存储所有的自注意力分数
        all_self_attentions = () if output_attentions else None
        # 如果设定要输出交叉注意力分数并且模型配置允许，则初始化一个空元组用于存储所有的交叉注意力分数
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用梯度检查点并且在训练阶段，并且设置了使用缓存，则发出警告并设定使用缓存为 False
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 根据是否使用缓存创建一个用于存储下一步解码器缓存的元组，若不使用缓存则为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果设定要输出隐藏状态，则将当前隐藏状态添加到存储所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 若存在头部遮罩，则获取当前层的头部遮罩，否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 若存在历史键值对，则获取当前层的历史键值对，否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用梯度检查点并且在训练阶段，则通过梯度检查点函数计算当前层输出
            if self.gradient_checkpointing and self.training:
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
            # 否则正常调用当前层计算当前层输出
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层输出的最后一个元素添加到下一步解码器缓存的元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力分数，则将当前层输出的注意力分数加入到存储所有自注意力分数的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型允许输出交叉注意力分数，则将当前层输出的交叉注意力分数加入到存储所有交叉注意力分数的元组中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果设定要输出隐藏状态，则将最终隐藏状态添加到存储所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 若不返回字典结果，则返回包含隐藏状态、下一步解码器缓存、所有隐藏状态、所有自注意力分数和所有交叉注意力分数的元组
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
        # 返回带过去键值对和交叉注意力信息的基础模型输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从 transformers.models.bert.modeling_bert.BertPooler 复制并修改为 RealmPooler 类
class RealmPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出大小都是隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取与第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记对应的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 经过激活函数处理
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


@dataclass
class RealmEmbedderOutput(ModelOutput):
    """
    [`RealmEmbedder`] 模型的输出。

    Args:
        projected_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):

            投影分数。
        hidden_states (`tuple(torch.FloatTensor)`, *可选*, 当传递了 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor` 元组
            模型在每一层输出的隐藏状态和初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *可选*, 当传递了 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `torch.FloatTensor` 元组
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    projected_score: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class RealmScorerOutput(ModelOutput):
    """
    [`RealmScorer`] 模型的输出。

    Args:
        relevance_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`):
            文档候选项的相关性分数（softmax 之前）。
        query_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):
            从查询嵌入器派生的查询分数。
        candidate_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`):
            从嵌入器派生的候选分数。
    """

    relevance_score: torch.FloatTensor = None
    query_score: torch.FloatTensor = None
    candidate_score: torch.FloatTensor = None


@dataclass
class RealmReaderOutput(ModelOutput):
    """
    [`RealmReader`] 模型的输出。
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Total loss.
        retriever_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Retriever loss.
        reader_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `start_positions`, `end_positions`, `has_answers` are provided):
            Reader loss.
        retriever_correct (`torch.BoolTensor` of shape `(config.searcher_beam_size,)`, *optional*):
            Whether or not an evidence block contains answer.
        reader_correct (`torch.BoolTensor` of shape `(config.reader_beam_size, num_candidates)`, *optional*):
            Whether or not a span candidate contains answer.
        block_idx (`torch.LongTensor` of shape `()`):
            The index of the retrieved evidence block in which the predicted answer is most likely.
        candidate (`torch.LongTensor` of shape `()`):
            The index of the retrieved span candidates in which the predicted answer is most likely.
        start_pos (`torch.IntTensor` of shape `()`):
            Predicted answer starting position in *RealmReader*'s inputs.
        end_pos (`torch.IntTensor` of shape `()`):
            Predicted answer ending position in *RealmReader*'s inputs.
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

    # 定义各个变量并初始化为 None
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
    [`RealmForOpenQA`] 模型的输出。

    Args:
        reader_output (`dict`):
            读取器输出。
        predicted_answer_ids (`torch.LongTensor` of shape `(answer_sequence_length)`):
            预测的答案 id。
    """

    reader_output: dict = None
    predicted_answer_ids: torch.LongTensor = None


class RealmPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果隐藏激活函数是字符串类型，则使用相应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 全连接层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数变换隐藏状态
        hidden_states = self.transform_act_fn(hidden_states)
        # 使用 LayerNorm 归一化隐藏状态
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RealmLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 实例化一个 RealmPredictionHeadTransform 对象，用于隐藏状态的变换
        self.transform = RealmPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个 token 都有一个仅输出的偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 为了正确调整 `resize_token_embeddings` 的偏置，需要这两个变量之间的链接
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 隐藏状态变换
        hidden_states = self.transform(hidden_states)
        # 预测下一个 token
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class RealmOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 实例化一个 RealmLMPredictionHead 对象，用于预测下一个 token
        self.predictions = RealmLMPredictionHead(config)

    def forward(self, sequence_output):
        # 输入序列的输出，用于预测下一个 token
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RealmScorerProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 实例化一个 RealmLMPredictionHead 对象，用于预测下一个 token
        self.predictions = RealmLMPredictionHead(config)
        # 定义一个全连接层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.retriever_proj_size)
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.retriever_proj_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 隐藏状态变换
        hidden_states = self.dense(hidden_states)
        # 使用 LayerNorm 归一化隐藏状态
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RealmReaderProjection(nn.Module):
```  
    # 该类继承自 nn.Module，用于构建一个 PyTorch 模型
    class __init__(self, config):
        # 调用父类的 __init__ 方法
        super().__init__()
        # 保存输入的配置参数
        self.config = config
        # 构建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.span_hidden_size * 2
        self.dense_intermediate = nn.Linear(config.hidden_size, config.span_hidden_size * 2)
        # 构建一个全连接层，输入大小为 config.span_hidden_size，输出大小为 1
        self.dense_output = nn.Linear(config.span_hidden_size, 1)
        # 构建一个层归一化层，归一化维度为 config.span_hidden_size，使用 config.reader_layer_norm_eps 作为参数
        self.layer_normalization = nn.LayerNorm(config.span_hidden_size, eps=config.reader_layer_norm_eps)
        # 构建一个 ReLU 激活函数
        self.relu = nn.ReLU()
    
    # 定义前向传播过程
    def forward(self, hidden_states, block_mask):
        # 定义一个函数，用于生成候选 span
        def span_candidates(masks):
            """
            生成候选 span。
    
            参数:
                masks: <bool> [num_retrievals, max_sequence_len] 是否属于 evidence block 的掩码
    
            返回:
                starts: <int32> [num_spans] 候选 span 的起始位置
                ends: <int32> [num_spans] 候选 span 的结束位置
                span_masks: <int32> [num_retrievals, num_spans] 是否属于 evidence block 的掩码
            """
            # 获取 masks 的最大序列长度
            _, max_sequence_len = masks.shape
    
            # 定义一个函数，给定 width 生成 start 和 end 位置
            def _spans_given_width(width):
                # 计算当前 start 位置
                current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
                # 计算当前 end 位置
                current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
                return current_starts, current_ends
    
            # 遍历不同 width 并获取 start 和 end 位置
            starts, ends = zip(*(_spans_given_width(w + 1) for w in range(self.config.max_span_width)))
    
            # 将所有的 start 和 end 位置拼接起来
            starts = torch.cat(starts, 0)
            ends = torch.cat(ends, 0)
    
            # 根据 start 和 end 位置从 masks 中选出对应的值，得到 span_masks
            start_masks = torch.index_select(masks, dim=-1, index=starts)
            end_masks = torch.index_select(masks, dim=-1, index=ends)
            span_masks = start_masks * end_masks
    
            return starts, ends, span_masks
    
        # 定义一个函数，将掩码转化为分数
        def mask_to_score(mask, dtype=torch.float32):
            # 将掩码转化为浮点型，并取负无穷大的值作为分数
            return (1.0 - mask.type(dtype)) * torch.finfo(dtype).min
    
        # 使用全连接层获得 start 和 end 的预测分数
        hidden_states = self.dense_intermediate(hidden_states)
        start_projection, end_projection = hidden_states.chunk(2, dim=-1)
    
        # 生成候选 span
        candidate_starts, candidate_ends, candidate_mask = span_candidates(block_mask)
    
        # 根据候选 span 的位置从 start_projection 和 end_projection 中选出对应的值
        candidate_start_projections = torch.index_select(start_projection, dim=1, index=candidate_starts)
        candidate_end_projections = torch.index_select(end_projection, dim=1, index=candidate_ends)
        candidate_hidden = candidate_start_projections + candidate_end_projections
    
        # 对候选 span 的隐藏表示进行 ReLU 激活和层归一化
        candidate_hidden = self.relu(candidate_hidden)
        candidate_hidden = self.layer_normalization(candidate_hidden)
    
        # 使用全连接层计算每个候选 span 的分数
        reader_logits = self.dense_output(candidate_hidden).squeeze(-1)
    
        # 根据 candidate_mask 调整分数
        reader_logits += mask_to_score(candidate_mask, dtype=reader_logits.dtype)
    
        # 返回分数、起始位置和结束位置
        return reader_logits, candidate_starts, candidate_ends
# This docstring provides a brief description of the REALM (Retrieval-Augmented Language Model) model,
# which is a PyTorch module. It explains that the model is a subclass of torch.nn.Module, and
# provides instructions on how to use it as a regular PyTorch module, as well as information
# about the configuration parameters required to initialize the model.
REALM_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RealmConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# This docstring is likely intended to provide information about the input requirements of the
# REALM model, but it is currently empty.
REALM_INPUTS_DOCSTRING = r"""
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        token_type_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        head_mask: torch.FloatTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.FloatTensor]]:
        """
        The forward pass of the model.
    
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
                - 1 corresponds to a *sentence B* token.
    
                [What are token type IDs?](../glossary#token-type-ids)
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
    
        Returns:
            Union[BaseModelOutputWithPastAndCrossAttentions, Tuple[torch.FloatTensor]]: A tuple or a BaseModelOutputWithPastAndCrossAttentions object containing various elements depending on the configuration (`config`) and inputs (each can be `torch.FloatTensor` or `None`).
    
            With the ``config`` being the configuration of the model and inputs being the inputs you pass to the model, the outputs are:
    
            - ``last_hidden_state``: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``:
    
                Sequence of hidden-states at the output of the last layer of the model.
    
                If `config.output_hidden_states=True`, also contains all hidden states of the model at the output of each layer plus the initial embedding outputs.
    
                If `config.is_decoder=True` and `config.add_cross_attention=True`, contains the hidden states of the decoder plus the initial embedding outputs and the cross-attention hidden states of each layer.
    
            - ``past_key_values`` (present in `BaseModelOutputWithPastAndCrossAttentions` if Pruned Attention is used):
    
                List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`
    
            - ``hidden_states`` (present when ``config.output_hidden_states=True``):
    
                list of :obj:`torch.FloatTensor` (one for the output of each layer + the output of the embeddings)
    
                The length of the list is ``config.n_layers + 1`` (or ``config.n_layers`` if ``config.output_hidden_states=False``).
    
                Each tensor of shape ``torch.FloatTensor((batch_size, sequence_length, hidden_size)``
    
            - ``attentions`` (present when ``config.output_attentions=True``):
    
                list of :obj:`torch.FloatTensor` (one for each layer)
    
                of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
        """
class RealmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = RealmConfig
    # 指定加载 TensorFlow 权重的方法
    load_tf_weights = load_tf_weights_in_realm
    # 指定基础模型的前缀
    base_model_prefix = "realm"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化线性层权重
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 初始化嵌入层权重
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 初始化 LayerNorm 层权重
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为 1
            module.weight.data.fill_(1.0)

    def _flatten_inputs(self, *inputs):
        """Flatten inputs' shape to (-1, input_shape[-1])"""
        # 将输入张量的形状展平为 (-1, input_shape[-1])
        flattened_inputs = []
        for tensor in inputs:
            if tensor is None:
                flattened_inputs.append(None)
            else:
                input_shape = tensor.shape
                # 如果张量的维度大于 2，则展平
                if len(input_shape) > 2:
                    tensor = tensor.view((-1, input_shape[-1]))
                flattened_inputs.append(tensor)
        return flattened_inputs


class RealmBertModel(RealmPreTrainedModel):
    """
    Same as the original BertModel but remove docstrings.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 RealmEmbeddings
        self.embeddings = RealmEmbeddings(config)
        # 初始化 RealmEncoder
        self.encoder = RealmEncoder(config)

        # 如果指定添加池化层，则初始化 RealmPooler
        self.pooler = RealmPooler(config) if add_pooling_layer else None

        # 权重初始化由其他 Realm 模型管理，但为了保持一致性，这里也进行初始化
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的权重
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        # 输入的 token IDs
        input_ids=None,
        # 注意力遮罩，用于指示哪些位置是要被注意的
        attention_mask=None,
        # token 类型 IDs，用于区分不同句子的 token
        token_type_ids=None,
        # 位置 IDs，指示每个 token 在句子中的位置
        position_ids=None,
        # 头部遮罩，用于指定哪些注意力头是要被屏蔽的
        head_mask=None,
        # 输入的嵌入式表示，如果不是 None，则忽略 input_ids 参数
        inputs_embeds=None,
        # 编码器隐藏状态，用于注意力机制的输入
        encoder_hidden_states=None,
        # 编码器注意力遮罩，用于指示哪些位置是要被注意的（编码器的注意力遮罩）
        encoder_attention_mask=None,
        # 过去的键值对，用于存储 Transformer 模型中的中间状态
        past_key_values=None,
        # 是否使用缓存，用于指定是否缓存中间状态
        use_cache=None,
        # 是否输出注意力权重
        output_attentions=None,
        # 是否输出隐藏状态
        output_hidden_states=None,
        # 是否返回一个字典作为输出
        return_dict=None,
# 使用自定义的装饰器添加模型文档注释和特定模型文档注释
@add_start_docstrings(
    "The embedder of REALM outputting projected score that will be used to calculate relevance score.",
    REALM_START_DOCSTRING,
)
# 定义 RealmEmbedder 类，继承自 RealmPreTrainedModel 类
class RealmEmbedder(RealmPreTrainedModel):
    # 定义与权重相关的键名列表
    _tied_weights_keys = ["cls.predictions.decoder.bias"]

    # 初始化函数，根据给定配置初始化模型
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 实例化 RealmBertModel 类并赋给 realm 属性
        self.realm = RealmBertModel(self.config)
        # 实例化 RealmScorerProjection 类并赋给 cls 属性
        self.cls = RealmScorerProjection(self.config)
        # 进行其他初始化操作
        self.post_init()

    # 获取输入嵌入的函数
    def get_input_embeddings(self):
        # 返回 realm 对象的 word_embeddings 属性
        return self.realm.embeddings.word_embeddings

    # 设置输入嵌入的函数
    def set_input_embeddings(self, value):
        # 设置 realm 对象的 word_embeddings 属性值为给定 value
        self.realm.embeddings.word_embeddings = value

    # forward 函数，实现模型的前向传播
    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        r"""
        前向传播函数，返回模型输出

        返回结果:
        包含投影分数的输出结果

        示例:
        ```py
        >>> from transformers import AutoTokenizer, RealmEmbedder
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/realm-cc-news-pretrained-embedder")
        >>> model = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> projected_score = outputs.projected_score
        ```
        """

        # 初始化 return_dict 参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 realm 对象的前向传播函数，获取 realm_outputs
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

        # 获取 pooler_output
        pooler_output = realm_outputs[1]  # [batch_size, hidden_size]
        # 获取 projected_score
        projected_score = self.cls(pooler_output)  # [batch_size, retriever_proj_size]

        # 根据 return_dict 的值返回指定结果
        if not return_dict:
            return (projected_score,) + realm_outputs[2:4]
        else:
            return RealmEmbedderOutput(
                projected_score=projected_score,
                hidden_states=realm_outputs.hidden_states,
                attentions=realm_outputs.attentions,
            )


# 使用自定义装饰器添加模型文档注释
@add_start_docstrings(
    # 定义一个字符串，描述了 REALM 输出的评分是对文档候选项的相关性评分（softmax 之前的分数）。
    "The scorer of REALM outputting relevance scores representing the score of document candidates (before softmax).",
    
    # REALM_START_DOCSTRING 是一个预定义的变量或常量，可能用于生成文档字符串的起始部分。
    REALM_START_DOCSTRING,
# 定义一个类 RealmScorer，继承自 RealmPreTrainedModel
class RealmScorer(RealmPreTrainedModel):
    r"""
    Args:
        query_embedder ([`RealmEmbedder`]):
            Embedder for input sequences. If not specified, it will use the same embedder as candidate sequences.
    """

    def __init__(self, config, query_embedder=None):
        # 调用父类的构造方法
        super().__init__(config)

        # 创建一个 RealmEmbedder 对象
        self.embedder = RealmEmbedder(self.config)

        # 设置查询嵌入器，如果未指定，则使用与候选序列相同的嵌入器
        self.query_embedder = query_embedder if query_embedder is not None else self.embedder

        # 调用后续初始化方法
        self.post_init()

    # 前向传播方法
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
        
# 添加文档字符串说明
@add_start_docstrings(
    "The knowledge-augmented encoder of REALM outputting masked language model logits and marginal log-likelihood"
    " loss.",
    REALM_START_DOCSTRING,
)
# 定义类 RealmKnowledgeAugEncoder，继承自 RealmPreTrainedModel
class RealmKnowledgeAugEncoder(RealmPreTrainedModel):
    # 静态成员变量
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        # 调用父类的构造方法
        super().__init__(config)
        # 创建 RealmBertModel 对象
        self.realm = RealmBertModel(self.config)
        # 创建 RealmOnlyMLMHead 对象
        self.cls = RealmOnlyMLMHead(self.config)
        # 调用后续初始化方法
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.realm.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.realm.embeddings.word_embeddings = value

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(
        REALM_INPUTS_DOCSTRING.format("batch_size, num_candidates, sequence_length")
    )
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,  # 输入的token id序列，可选参数
            attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，可选参数
            token_type_ids: Optional[torch.LongTensor] = None,  # token类型 id，可选参数
            position_ids: Optional[torch.LongTensor] = None,  # 位置 id，可选参数
            head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，可选参数
            inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入向量，可选参数
            relevance_score: Optional[torch.FloatTensor] = None,  # 相关性得分，可选参数
            labels: Optional[torch.LongTensor] = None,  # 标签，可选参数
            mlm_mask: Optional[torch.LongTensor] = None,  # MLM掩码，可选参数
            output_attentions: Optional[bool] = None,  # 是否输出注意力矩阵，可选参数
            output_hidden_states: Optional[bool] = None,  # 是否输出隐藏层状态，可选参数
            return_dict: Optional[bool] = None,  # 是否返回字典，可选参数
# 这是一个用于 REALM 模型的 RealmReader 类
@add_start_docstrings("The reader of REALM.", REALM_START_DOCSTRING)
class RealmReader(RealmPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置分类标签数量
        self.num_labels = config.num_labels
        # 实例化 RealmBertModel 和其他相关模块
        self.realm = RealmBertModel(config)
        self.cls = RealmOnlyMLMHead(config)
        self.qa_outputs = RealmReaderProjection(config)
        # 执行模型初始化后的操作
        self.post_init()

    # 定义模型前向计算逻辑
    @add_start_docstrings_to_model_forward(REALM_INPUTS_DOCSTRING.format("reader_beam_size, sequence_length"))
    @replace_return_docstrings(output_type=RealmReaderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # 输入张量
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 额外的输入参数
        relevance_score: Optional[torch.FloatTensor] = None,
        block_mask: Optional[torch.BoolTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        has_answers: Optional[torch.BoolTensor] = None,
        # 输出控制参数
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            输入序列标记在词汇表中的索引。

            索引可以使用 [`AutoTokenizer`] 获得。查看 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`] 获取更多细节。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            避免对填充令牌索引执行注意力的掩码。掩码值选定在 `[0, 1]` 之间：

            - 对于**未被屏蔽**的标记，值为1，
            - 对于**被屏蔽**的标记，值为0。

            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            段标记索引，用于指示输入的第一部分和第二部分。索引在 `[0, 1]` 中选定：

            - 0 对应于*句子A* 标记,
            - 1 对应于*句子B* 标记（根据设计，此模型不使用此标记）。

            [什么是标记类型 ID？](../glossary#token-type-ids)
        answer_ids (`list` of shape `(num_answers, answer_length)`, *optional*):
            用于计算边际对数似然损失的答案 ID。索引应在 `[-1, 0, ..., config.vocab_size]` 中（参见 `input_ids` 文档字符串）。索引设置为 `-1` 的标记将被忽略（屏蔽），仅计算标签为 `[0, ..., config.vocab_size]` 的标记的损失。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

@add_start_docstrings(
    "`RealmForOpenQA` for end-to-end open domain question answering.",
    REALM_START_DOCSTRING,
)
class RealmForOpenQA(RealmPreTrainedModel):

    # 初始化函数，接收一个config配置对象和一个retriever对象作为参数
    def __init__(self, config, retriever=None):
        super().__init__(config)
        self.embedder = RealmEmbedder(config) # 实例化一个RealmEmbedder对象
        self.reader = RealmReader(config) # 实例化一个RealmReader对象

        # 创建一个torch的tensor，用于存放block_emb，初始化为0
        self.register_buffer(
            "block_emb",
            torch.zeros(()).new_empty(
                size=(config.num_block_records, config.retriever_proj_size),
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
        )

        self.retriever = retriever # 将retriever赋值给self.retriever

        self.post_init()

    # 定义searcher_beam_size属性
    @property
    def searcher_beam_size(self):
        # 若处于训练阶段，则返回config中的searcher_beam_size
        if self.training:
            return self.config.searcher_beam_size
        # 否则返回config中的reader_beam_size
        return self.config.reader_beam_size

    # 将self.block_emb发送到指定的device
    def block_embedding_to(self, device):
        """Send `self.block_emb` to a specific device.

        Args:
            device (`str` or `torch.device`):
                The device to which `self.block_emb` will be sent.
        """

        self.block_emb = self.block_emb.to(device)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(REALM_FOR_OPEN_QA_DOCSTRING.format("1, sequence_length"))
    @replace_return_docstrings(output_type=RealmForOpenQAOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        answer_ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
"""
```