# `.\transformers\models\bros\modeling_bros.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则不得使用此文件
# 根据许可证分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
""" PyTorch Bros model."""

# 导入所需的库
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入相关模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bros import BrosConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "jinho8345/bros-base-uncased"
_CONFIG_FOR_DOC = "BrosConfig"

# Bros 预训练模型存档列表
BROS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "jinho8345/bros-base-uncased",
    "jinho8345/bros-large-uncased",
    # 查看所有 Bros 模型 https://huggingface.co/models?filter=bros
]

# Bros 模型的起始文档字符串
BROS_START_DOCSTRING = r"""
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BrosConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Bros 模型的输入文档字符串
BROS_INPUTS_DOCSTRING = r"""
"""


@dataclass
class BrosSpadeOutput(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            分类损失。
            如果提供了`labels`，则返回分类损失。
        initial_token_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            实体初始标记的分类分数（SoftMax之前）。
            形状为`(batch_size, sequence_length, config.num_labels)`的张量。
        subsequent_token_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, sequence_length+1)`):
            实体序列标记的分类分数（SoftMax之前）。
            形状为`(batch_size, sequence_length, sequence_length+1)`的张量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型隐藏状态的元组。
            如果传递了`output_hidden_states=True`或者`config.output_hidden_states=True`，则返回一个元组，其中包含模型每一层的隐藏状态。
            元组的长度为2，包含了形状为`(batch_size, sequence_length, hidden_size)`的张量，一个用于嵌入层的输出，如果模型有嵌入层，还有每一层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            模型注意力权重的元组。
            如果传递了`output_attentions=True`或者`config.output_attentions=True`，则返回一个元组，其中包含每一层的注意力权重。
            元组的长度为2，每个元素是形状为`(batch_size, num_heads, sequence_length, sequence_length)`的张量。
            这些是经过注意力SoftMax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None  # 分类损失，默认为None
    initial_token_logits: torch.FloatTensor = None  # 实体初始标记的分类分数，默认为None
    subsequent_token_logits: torch.FloatTensor = None  # 实体序列标记的分类分数，默认为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型隐藏状态，默认为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 模型注意力权重，默认为None
class BrosPositionalEmbedding1D(nn.Module):
    # 定义一个继承自 nn.Module 的类 BrosPositionalEmbedding1D

    def __init__(self, config):
        # 初始化方法，接受一个配置参数 config
        super(BrosPositionalEmbedding1D, self).__init__()
        # 调用父类的初始化方法

        self.dim_bbox_sinusoid_emb_1d = config.dim_bbox_sinusoid_emb_1d
        # 将配置参数中的 dim_bbox_sinusoid_emb_1d 赋值给类属性

        inv_freq = 1 / (
            10000 ** (torch.arange(0.0, self.dim_bbox_sinusoid_emb_1d, 2.0) / self.dim_bbox_sinusoid_emb_1d)
        )
        # 计算频率的倒数
        self.register_buffer("inv_freq", inv_freq)
        # 将频率的倒数注册为缓冲区

    def forward(self, pos_seq: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，接受一个位置序列 pos_seq，返回一个张量
        seq_size = pos_seq.size()
        # 获取位置序列的大小
        b1, b2, b3 = seq_size
        # 将大小拆分为 b1, b2, b3
        sinusoid_inp = pos_seq.view(b1, b2, b3, 1) * self.inv_freq.view(1, 1, 1, self.dim_bbox_sinusoid_emb_1d // 2)
        # 计算正弦输入
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        # 将正弦和余弦值拼接在一起
        return pos_emb
        # 返回位置嵌入张量


class BrosPositionalEmbedding2D(nn.Module):
    # 定义一个继承自 nn.Module 的类 BrosPositionalEmbedding2D

    def __init__(self, config):
        # 初始化方法，接受一个配置参数 config
        super(BrosPositionalEmbedding2D, self).__init__()
        # 调用父类的初始化方法

        self.dim_bbox = config.dim_bbox
        # 将配置参数中的 dim_bbox 赋值给类属性
        self.x_pos_emb = BrosPositionalEmbedding1D(config)
        # 创建一个一维位置嵌入对象
        self.y_pos_emb = BrosPositionalEmbedding1D(config)
        # 创建另一个一维位置嵌入对象

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，接受一个边界框张量 bbox，返回一个张量
        stack = []
        # 创建一个空列表 stack
        for i in range(self.dim_bbox):
            # 遍历边界框的维度
            if i % 2 == 0:
                stack.append(self.x_pos_emb(bbox[..., i]))
            # 如果是偶数维度，使用 x_pos_emb 进行处理
            else:
                stack.append(self.y_pos_emb(bbox[..., i]))
            # 如果是奇数维度，使用 y_pos_emb 进行处理
        bbox_pos_emb = torch.cat(stack, dim=-1)
        # 沿指��维度拼接处理后的结果
        return bbox_pos_emb
        # 返回边界框位置嵌入张量


class BrosBboxEmbeddings(nn.Module):
    # 定义一个继承自 nn.Module 的类 BrosBboxEmbeddings

    def __init__(self, config):
        # 初始化方法，接受一个配置参数 config
        super(BrosBboxEmbeddings, self).__init__()
        # 调用父类的初始化方法
        self.bbox_sinusoid_emb = BrosPositionalEmbedding2D(config)
        # 创建一个二维位置嵌入对象
        self.bbox_projection = nn.Linear(config.dim_bbox_sinusoid_emb_2d, config.dim_bbox_projection, bias=False)
        # 创建一个线性层用于投影

    def forward(self, bbox: torch.Tensor):
        # 前向传播方法，接受一个边界框张量 bbox
        bbox_t = bbox.transpose(0, 1)
        # 转置边界框张量
        bbox_pos = bbox_t[None, :, :, :] - bbox_t[:, None, :, :]
        # 计算边界框之间的位置关系
        bbox_pos_emb = self.bbox_sinusoid_emb(bbox_pos)
        # 使用二维位置嵌入对象处理位置关系
        bbox_pos_emb = self.bbox_projection(bbox_pos_emb)
        # 使用线性层��行投影

        return bbox_pos_emb
        # 返回处理后的边界框位置嵌入张量


class BrosTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 定义一个继承自 nn.Module 的类 BrosTextEmbeddings，用于构建文本嵌入
    # 初始化方法，用于初始化模型参数和各种嵌入层
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
    
        # 词嵌入层，将词汇索引映射到向量空间
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，将位置索引映射到向量空间
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 类型嵌入层，将标记类型映射到向量空间
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
        # LayerNorm 层，用于归一化模型的输出
        # 属性名没有使用蛇形命名法以保持与 TensorFlow 模型变量名一致，以便加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃输入数据以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，绝对位置还是相对位置
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置索引缓冲区，用于存储位置索引
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # 注册标记类型索引缓冲区，用于存储标记类型索引
        self.register_buffer(
            "token_type_ids",
            torch.zeros(
                self.position_ids.size(),
                dtype=torch.long,
                device=self.position_ids.device,
            ),
            persistent=False,
        )
    
    # 前向传播方法，用于模型的正向计算
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果输入数据的词汇索引不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
    
        # 获取序列长度
        seq_length = input_shape[1]
    
        # 如果位置索引为空，则根据序列长度和历史键值的长度计算位置索引
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    
        # 如果标记类型索引为空，则根据情况进行处理
        if token_type_ids is None:
            # 如果模型中有标记类型索引缓冲区，则使用缓冲区中的索引
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则，创建全零的标记类型索引
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    
        # 如果输入的嵌入数据为空，则使用词嵌入层获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取标记类型的嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
    
        # 将词嵌入和标记类型嵌入相加得到总的嵌入向量
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为绝对位置，则获取位置嵌入向量并加到总的嵌入向量中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对总的嵌入向量进行 LayerNorm 归一化
        embeddings = self.LayerNorm(embeddings)
        # 对归一化后的向量进行 Dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的向量
        return embeddings
# 定义一个自注意力机制的类，继承自 nn.Module
class BrosSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否可以被注意力头数整除，并且没有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 存储注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入层
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标记是否是解码器层
        self.is_decoder = config.is_decoder

    # 将输入张量转换为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[torch.Tensor] = False,



# 定义一个自注意力层的输出类，继承自 nn.Module
class BrosSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义全连接层、LayerNorm 层和 dropout 层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



# 定义一个注意力层的类，继承自 nn.Module
class BrosAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义自注意力和输出层
        self.self = BrosSelfAttention(config)
        self.output = BrosSelfOutput(config)
        # 存储被剪枝的注意力头的集合
        self.pruned_heads = set()
    # 对给定的头部进行修剪操作
    def prune_heads(self, heads):
        # 如果头部列表为空，则直接返回
        if len(heads) == 0:
            return
        # 找到可修剪的头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用 self 模块进行前向传播
        self_outputs = self.self(
            hidden_states=hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        # 使用 output 模块处理 self 模块的输出和原始隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制代码，将类名由 BertIntermediate 改为 BrosIntermediate
class BrosIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入维度变为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则将其转换为相应的激活函数；否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 经过激活函数变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回变换后的 hidden_states
        return hidden_states


# 定义 BrosOutput 类
class BrosOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入维度变为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对输入进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，用于随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # hidden_states 经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 经过 dropout 层
        hidden_states = self.dropout(hidden_states)
        # 将 hidden_states 和 input_tensor 相加，然后经过 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回变换后的 hidden_states
        return hidden_states


# 定义 BrosLayer 类
class BrosLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 初始化注意力机制
        self.attention = BrosAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加跨层注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨层注意力机制且不是解码器，抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise Exception(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨层注意力机制
            self.crossattention = BrosAttention(config)
        # 初始化中间层
        self.intermediate = BrosIntermediate(config)
        # 初始化输出层
        self.output = BrosOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 如果过去的键/值缓存不为空，则解码器的单向自注意力的缓存键/值元组在位置1、2处
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力层
        self_attention_outputs = self.attention(
            hidden_states,
            bbox_pos_emb=bbox_pos_emb,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 将输出设置为除自注意力缓存之外的所有输出
            outputs = self_attention_outputs[1:-1]
            # 获取当前的键/值缓存
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果需要输出注意力权重，则添加自注意力
            outputs = self_attention_outputs[1:]

        # 初始化交叉注意力的键/值缓存
        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 检查是否存在交叉注意力层
            if hasattr(self, "crossattention"):
                raise Exception(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 获取过去的交叉注意力的键/值缓存
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 调用交叉注意力层
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力层的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力层的输出添加到总输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的键/值缓存添加到当前键/值缓存中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对注意力输出应用分块操作
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        # 将分块后的输出添加到总输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 对自注意力输出进行前馈传播
    def feed_forward_chunk(self, attention_output):
        # 通过中间层
        intermediate_output = self.intermediate(attention_output)
        # 经过输出层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义 BrosEncoder 类，继承自 nn.Module
class BrosEncoder(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建 nn.ModuleList，包含多个 BrosLayer 实例，数量由 config.num_hidden_layers 决定
        self.layer = nn.ModuleList([BrosLayer(config) for _ in range(config.num_hidden_layers)])

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        bbox_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 从 transformers.models.bert.modeling_bert.BertPooler 复制并修改为 BrosPooler 类
class BrosPooler(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 创建 nn.Linear 层，用于降维
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建激活函数层，使用 Tanh 函数
        self.activation = nn.Tanh()

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地取第一个 token 对应的隐藏状态来“汇聚”模型
        first_token_tensor = hidden_states[:, 0]
        # 对第一个 token 的隐藏状态进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 对线性变换后的结果进行 Tanh 激活
        pooled_output = self.activation(pooled_output)
        # 返回汇聚后的输出
        return pooled_output


# 定义 BrosRelationExtractor 类，用于关系提取
class BrosRelationExtractor(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 初始化属性
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.head_hidden_size = config.hidden_size
        self.classifier_dropout_prob = config.classifier_dropout_prob

        # 创建 Dropout 层，用于模型训练时的随机丢弃
        self.drop = nn.Dropout(self.classifier_dropout_prob)
        # 创建线性变换层，用于生成查询向量
        self.query = nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        # 创建线性变换层，用于生成键向量
        self.key = nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)

        # 创建虚拟节点参数，用于处理输入 key_layer 与查询向量的维度不匹配问题
        self.dummy_node = nn.Parameter(torch.zeros(1, self.backbone_hidden_size))

    # 前向传播方法
    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        # 对查询向量进行线性变换和 Dropout
        query_layer = self.query(self.drop(query_layer))

        # 创建虚拟节点向量，并将其拼接到输入的 key_layer 上
        dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, key_layer.size(1), 1)
        key_layer = torch.cat([key_layer, dummy_vec], axis=0)
        # 对键向量进行线性变换和 Dropout
        key_layer = self.key(self.drop(key_layer))

        # 将查询向量和键向量进行维度重塑，以便进行矩阵乘法
        query_layer = query_layer.view(
            query_layer.size(0), query_layer.size(1), self.n_relations, self.head_hidden_size
        )
        key_layer = key_layer.view(key_layer.size(0), key_layer.size(1), self.n_relations, self.head_hidden_size)

        # 计算查询向量与键向量之间的关系分数
        relation_score = torch.matmul(
            query_layer.permute(2, 1, 0, 3), key_layer.permute(2, 1, 3, 0)
        )  # equivalent to torch.einsum("ibnd,jbnd->nbij", (query_layer, key_layer))

        # 返回关系分数
        return relation_score
class BrosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义 BrosPreTrainedModel 类，用于处理权重初始化和预训练模型的下载和加载
    config_class = BrosConfig
    base_model_prefix = "bros"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化模型的权重
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，初始化偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare Bros Model transformer outputting raw hidden-states without any specific head on top.",
    BROS_START_DOCSTRING,
)
class BrosModel(BrosPreTrainedModel):
    # 定义 BrosModel 类，继承自 BrosPreTrainedModel
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化模型的各个组件
        self.embeddings = BrosTextEmbeddings(config)
        self.bbox_embeddings = BrosBboxEmbeddings(config)
        self.encoder = BrosEncoder(config)

        # 如果需要添加池化层，则初始化池化层
        self.pooler = BrosPooler(config) if add_pooling_layer else None

        # 初始化模型的权重
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 剪枝模型的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，用于模型的前向计算，接收多个参数
    def forward(
        self,
        # 输入的标记序列的张量，可选
        input_ids: Optional[torch.Tensor] = None,
        # 目标框的张量表示，可选
        bbox: Optional[torch.Tensor] = None,
        # 注意力遮罩张量，可选
        attention_mask: Optional[torch.Tensor] = None,
        # 标记类型标识的张量，可选
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置编码标识的张量，可选
        position_ids: Optional[torch.Tensor] = None,
        # 头部遮罩的张量，可选
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入的张量表示，可选
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器隐藏状态的张量表示，可选
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器注意力遮罩的张量，可选
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 过去键值对的列表，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否使用缓存，可选
        use_cache: Optional[bool] = None,
        # 是否输出注意力分布，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，可选
        return_dict: Optional[bool] = None,
# 为标记分类任务设计的 Bros 模型，包含一个标记分类头部（在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    Bros Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BROS_START_DOCSTRING,
)
class BrosForTokenClassification(BrosPreTrainedModel):
    # 在加载模型时忽略的键列表，即不期望出现的键
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels

        # 初始化 Bros 模型
        self.bros = BrosModel(config)
        # 分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型权重
        self.init_weights()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_first_token_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        返回值类型注解，可以是 torch.Tensor 元组或 TokenClassifierOutput 类型

        Returns:
        返回值说明

        Examples:
        示例代码

        ```python
        >>> import torch
        >>> from transformers import BrosProcessor, BrosForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```py"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法
        outputs = self.bros(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 对输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            if bbox_first_token_mask is not None:
                bbox_first_token_mask = bbox_first_token_mask.view(-1)
                # 计算损失
                loss = loss_fct(
                    logits.view(-1, self.num_labels)[bbox_first_token_mask], labels.view(-1)[bbox_first_token_mask]
                )
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则返回元组
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有标记分类头的 Bros 模型，用于在隐藏状态输出之上进行标记分类，例如用于命名实体识别（NER）任务。
# initial_token_classifier 用于预测每个实体的第一个标记，subsequent_token_classifier 用于预测实体内后续的标记。
# 与 BrosForTokenClassification 相比，这个模型对序列化错误更加健壮，因为它从一个标记预测下一个标记。
class BrosSpadeEEForTokenClassification(BrosPreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size

        # 初始化 BrosModel
        self.bros = BrosModel(config)
        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob
        )

        # 用于实体提取（NER）的初始标记分类
        self.initial_token_classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(classifier_dropout),
            nn.Linear(config.hidden_size, config.num_labels),
        )

        # 用于实体提取（NER）的后续标记分类
        self.subsequent_token_classifier = BrosRelationExtractor(config)

        self.init_weights()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_first_token_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        initial_token_labels: Optional[torch.Tensor] = None,
        subsequent_token_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

``` 
# 定义一个带有标记分类头的 Bros 模型，用于在隐藏状态输出之上进行标记分类，例如用于实体链接任务。
# entity_linker 用于预测实体内部链接（一个实体到另一个实体）。
class BrosSpadeELForTokenClassification(BrosPreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置对象保存在实例属性中
        self.config = config
        # 将配置中的标签数量保存在实例属性中
        self.num_labels = config.num_labels
        # 将配置中的关系数量保存在实例属性中
        self.n_relations = config.n_relations
        # 将配置中的隐藏层大小保存在实例属性中
        self.backbone_hidden_size = config.hidden_size

        # 创建 BrosModel 的实例，传入配置对象
        self.bros = BrosModel(config)
        # 如果配置中包含 classifier_dropout 属性，则使用其值，否则使用隐藏层 dropout 的默认值
        (config.classifier_dropout if hasattr(config, "classifier_dropout") else config.hidden_dropout_prob)

        # 创建 BrosRelationExtractor 的实例，传入配置对象
        self.entity_linker = BrosRelationExtractor(config)

        # 初始化模型的权重
        self.init_weights()

    # 重写 forward 方法，添加了一些输入和输出的文档字符串
    @add_start_docstrings_to_model_forward(BROS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bbox_first_token_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        返回值类型为 torch.Tensor 元组或 TokenClassifierOutput 对象的联合类型

        示例:

        ```py
        >>> import torch
        >>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification

        >>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

        >>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

        >>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
        >>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
        >>> encoding["bbox"] = bbox

        >>> outputs = model(**encoding)
        ```"""
        # 确定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型进行推断
        outputs = self.bros(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 调整输出张量的形状
        last_hidden_states = outputs[0]
        last_hidden_states = last_hidden_states.transpose(0, 1).contiguous()

        # 通过实体链接器获取实体链接的对数概率
        logits = self.entity_linker(last_hidden_states, last_hidden_states).squeeze(0)

        # 计算损失函数
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            batch_size, max_seq_length = attention_mask.shape
            device = attention_mask.device

            # 创建自我标记的掩码，用于排除对角线元素
            self_token_mask = torch.eye(max_seq_length, max_seq_length + 1).to(device).bool()

            mask = bbox_first_token_mask.view(-1)
            bbox_first_token_mask = torch.cat(
                [
                    ~bbox_first_token_mask,
                    torch.zeros([batch_size, 1], dtype=torch.bool).to(device),
                ],
                axis=1,
            )
            logits = logits.masked_fill(bbox_first_token_mask[:, None, :], torch.finfo(logits.dtype).min)
            logits = logits.masked_fill(self_token_mask[None, :, :], torch.finfo(logits.dtype).min)

            # 计算损失
            loss = loss_fct(logits.view(-1, max_seq_length + 1)[mask], labels.view(-1)[mask])

        # 如果不返回字典形式的结果，则组装输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```