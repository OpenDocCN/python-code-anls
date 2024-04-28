# `.\models\layoutlm\modeling_layoutlm.py`

```
# 设置编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 在遵守许可证的情况下，禁止使用此文件
# 您可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则在"AS IS"的基础上分发软件
# 没有任何形式的保证或条件，明示或暗示
# 请查看许可证以获取具体语言、权利和限制
""" PyTorch LayoutLM 模型."""
# 导入所需的库及类型提示
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 HuggingFace 库中导入激活函数映射表
from ...activations import ACT2FN
# 从 HuggingFace 库中导入模型输出类
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 从 HuggingFace 库中导入模型工具类
from ...modeling_utils import PreTrainedModel
# 从 HuggingFace 库中导入 PyTorch 工具类
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 从 HuggingFace 库中导入常用方法
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从 HuggingFace 配置中导入 LayoutLM 配置
from .configuration_layoutlm import LayoutLMConfig

# 创建日志记录器
logger = logging.get_logger(__name__)

# 以下为文档中对应的定义
_CONFIG_FOR_DOC = "LayoutLMConfig"
_CHECKPOINT_FOR_DOC = "microsoft/layoutlm-base-uncased"

# 定义已经发布的 LayoutLM 预训练模型
LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlm-base-uncased",
    "layoutlm-large-uncased",
]

# 定义 LayoutLM 层规范化
LayoutLMLayerNorm = nn.LayerNorm

# 定义 LayoutLM 嵌入层
class LayoutLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutLMEmbeddings, self).__init__()
        # 定义词嵌入、位置嵌入和 token 类型嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 定义规范化层和 dropout 层
        self.LayerNorm = LayoutLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册缓冲区用于存储位置 id
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义一个前向传播方法，用于模型的前向推断
    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        # 如果输入的词索引不为空，则获取输入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取输入嵌入的形状，不包括最后一维（词嵌入维度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 获取输入数据的设备（CPU/GPU）
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供位置信息，则使用模型内部的位置嵌入
        if position_ids is None:
            # 从模型内部的位置嵌入中获取与序列长度相匹配的位置信息
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供token类型信息，则初始化为全0的tensor
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 如果未提供嵌入输入，则根据输入的词索引获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 获取词嵌入
        words_embeddings = inputs_embeds
        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)
        
        # 获取bbox中的左上右下位置坐标，并分别获取对应的位置嵌入
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            # 如果bbox坐标值超出范围，则抛出错误
            raise IndexError("The `bbox`coordinate values should be within 0-1000 range.") from e

        # 获取bbox的高度和宽度，并分别获取对应的位置嵌入
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        # 获取token类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将所有嵌入相加得到最终嵌入表示
        embeddings = (
            words_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        
        # 对嵌入进行Layer Normalization
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入进行Dropout处理
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入表示
        return embeddings
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制并修改为LayoutLMSelfAttention类
class LayoutLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否可以被注意力头的数量整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化各种参数
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化query、key和value的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置编码，需要额外的距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 判断该层是否为decoder
        self.is_decoder = config.is_decoder

    # 调整输入形状以便分组，并置换位置
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并修改为LayoutLMSelfOutput类
class LayoutLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层dense、LayerNorm层和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->LayoutLM
# 创建 LayoutLMAttention 类，继承自 nn.Module
class LayoutLMAttention(nn.Module):
    # 初始化函数
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建 LayoutLMSelfAttention 对象
        self.self = LayoutLMSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 LayoutLMSelfOutput 对象
        self.output = LayoutLMSelfOutput(config)
        # 初始化可剪枝的头部集合
        self.pruned_heads = set()

    # 头部剪枝函数
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
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
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果有输出注意力，将注意力也加入到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
# 创建 LayoutLMIntermediate 类，继承自 nn.Module
class LayoutLMIntermediate(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 创建线性层对象
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断隐藏层激活函数类型，选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 使用选择的激活函数进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM
# 创建 LayoutLMOutput 类，继承自 nn.Module
class LayoutLMOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受hidden_states和input_tensor两个torch张量，返回一个torch张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对hidden_states进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的hidden_states进行丢弃操作
        hidden_states = self.dropout(hidden_states)
        # 对丢弃后的hidden_states进行LayerNorm操作，并加上input_tensor
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的hidden_states
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLayer 复制代码并将 Bert 改为 LayoutLM
class LayoutLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前馈网络的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度，用于注意力计算
        self.seq_len_dim = 1
        # 创建 LayoutLMAttention 层
        self.attention = LayoutLMAttention(config)
        # 是否为解码器模型
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力层
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # 如果添加了交叉注意力层，则验证是否为解码器模型，因为交叉注意力通常在解码器中使用
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建交叉注意力层，使用绝对位置编码
            self.crossattention = LayoutLMAttention(config, position_embedding_type="absolute")
        # 创建 LayoutLMIntermediate 层，用于中间表示
        self.intermediate = LayoutLMIntermediate(config)
        # 创建 LayoutLMOutput 层，用于最终输出
        self.output = LayoutLMOutput(config)

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
        # 函数签名，指定了函数的输入和输出类型

        # 如果过去的键/值缓存不为空，则只使用缓存的前两个元素
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用 self-attention 模块进行注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取 self-attention 的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是 self-attention 缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 截取除了最后一个元素之外的所有元素
            present_key_value = self_attention_outputs[-1]  # 获取最后一个元素
        else:
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则添加 self-attention
                                                  # 的输出到输出中

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模块没有交叉注意力层，则抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值缓存不为空，则只使用缓存的最后两个元素
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模块进行注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力的输出添加到输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的缓存添加到当前的键/值缓存中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向传播的分块
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，则将注意力键/值作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 过前向传播的一块
        intermediate_output = self.intermediate(attention_output)
        # 输出层的前向传播
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制并修改成LayoutLM
class LayoutLMEncoder(nn.Module):
    # 初始化LayoutLMEncoder类
    def __init__(self, config):
        # 继承父类的初始化方法
        super().__init__()
        # 将config参数保存到对象中
        self.config = config
        # 创建包含多个LayoutLMLayer对象的列表，数量为config中指定的隐藏层数量
        self.layer = nn.ModuleList([LayoutLMLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点为False
        self.gradient_checkpointing = False

    # 正向传播函数
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
    # 定义函数的返回值类型，可以是元组或者BaseModelOutputWithPastAndCrossAttentions类型
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 定义是否输出所有隐藏状态的变量，如果不输出则为None
        all_hidden_states = () if output_hidden_states else None
        # 定义是否输出所有自注意力矩阵的变量，如果不输出则为None
        all_self_attentions = () if output_attentions else None
        # 定义是否输出所有交叉注意力矩阵的变量，如果不输出或者当前模型不支持，则为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果配置允许梯度检查点且当前处于训练状态
        if self.gradient_checkpointing and self.training:
            # 如果使用了缓存，则不支持梯度检查点，发出警告并将use_cache设置为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化下一个解码器缓存，如果不使用缓存则为None
        next_decoder_cache = () if use_cache else None
        # 遍历所有的解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果没有则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取先前的键值对，如果没有则为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果允许梯度检查点且当前处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点方法调用当前层的模块
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
                # 调用当前层的模块
                # hidden_states: 隐藏状态
                # attention_mask:  注意力掩码
                # layer_head_mask: 层头部掩码
                # encoder_hidden_states: 编码器的隐藏状态
                # encoder_attention_mask: 编码器的注意力掩码
                # past_key_value: 先前的键值对
                # output_attentions: 是否输出注意力
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层模块的输出第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层输出的最后一个元素添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力信息，则添加自注意力和交叉注意力矩阵
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果
        if not return_dict:
            # 返回不为None的元素
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
        # 返回已声明的BaseModelOutputWithPastAndCrossAttentions类型
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler中复制而来，定义了LayoutLMPooler类
class LayoutLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入和输出大小都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出隐藏状态中对应于第一个标记的张量
        first_token_tensor = hidden_states[:, 0]
        # 通过线性变换层处理第一个标记张量
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数处理输出结果
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform中复制而来，将Bert替换为LayoutLM
class LayoutLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性变换层，输入和输出大小都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 确定激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性变换层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理输出结果
        hidden_states = self.transform_act_fn(hidden_states)
        # 使用LayerNorm层处理输出结果
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead中复制而来，将Bert替换为LayoutLM
class LayoutLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化LayoutLMPredictionHeadTransform对象
        self.transform = LayoutLMPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但对于每个标记都有一个仅存在于输出的偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化偏置项为0
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要将这两个变量链接起来，以便在resize_token_embeddings时正确调整偏置项的大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead中复制而来，将Bert替换为LayoutLM
class LayoutLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化LayoutLMLMPredictionHead对象
        self.predictions = LayoutLMLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 调用predictions，得到预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和预训练模型下载加载的抽象类。
    """

    # 设置config_class为LayoutLMConfig
    config_class = LayoutLMConfig
    # 预训练模型存档映射，将预训练模型存档列表赋值给预训练模型存档映射变量
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST
    # 基础模型前缀，指定为"layoutlm"
    base_model_prefix = "layoutlm"
    # 支持梯度检查点，设置为True
    supports_gradient_checkpointing = True

    # 初始化模型参数的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果设置了填充索引，将对应索引位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayoutLMLayerNorm 类型的层
        elif isinstance(module, LayoutLMLayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
LAYOUTLM_START_DOCSTRING = r"""
    LayoutLM 模型由 Yiheng Xu、Minghao Li、Lei Cui、Shaohan Huang、Furu Wei 和 Ming Zhou 在论文
    [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) 中提出。

    该模型是 PyTorch 的 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其视为普通的 PyTorch 模块，并参考 PyTorch 文档了解与一般用法和行为相关的所有内容。

    参数:
        config ([`LayoutLMConfig`]): 模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

LAYOUTLM_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "不带任何特定头部的原始隐藏状态输出的 LayoutLM 模型。",
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMModel(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super(LayoutLMModel, self).__init__(config)
        self.config = config

        self.embeddings = LayoutLMEmbeddings(config)  # 初始化嵌入层
        self.encoder = LayoutLMEncoder(config)  # 初始化编码器
        self.pooler = LayoutLMPooler(config)  # 初始化池化器

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        对模型的头部进行修剪。heads_to_prune: {layer_num: 该层要修剪的头部列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings("""带有顶部 `语言建模` 头部的 LayoutLM 模型。""", LAYOUTLM_START_DOCSTRING)
```  
# LayoutLMForMaskedLM 类定义，继承自 LayoutLMPreTrainedModel
class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    # 定义 tied_weights_keys，用于指定共享权重的键值列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 实例化 LayoutLMModel
        self.layoutlm = LayoutLMModel(config)
        # 实例化 LayoutLMOnlyMLMHead
        self.cls = LayoutLMOnlyMLMHead(config)

        # 调用 post_init 函数，初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入的函数
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # 获取输出嵌入的函数
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的函数
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# LayoutLMForSequenceClassification 类定义，继承自 LayoutLMPreTrainedModel
@add_start_docstrings(
    """
    LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
    document image classification tasks such as the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置类别数
        self.num_labels = config.num_labels
        # 实例化 LayoutLMModel
        self.layoutlm = LayoutLMModel(config)
        # 实例化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 线性分类器，用于分类任务
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用 post_init 函数，初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入的函数
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，用于模型的前向推断过程
    def forward(
        # 输入的 token IDs，类型为长整型张量，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 边界框信息，类型为长整型张量，默认为 None
        bbox: Optional[torch.LongTensor] = None,
        # 注意力掩码，类型为浮点型张量，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,
        # token 类型 IDs，类型为长整型张量，默认为 None
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 IDs，类型为长整型张量，默认为 None
        position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，类型为浮点型张量，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入的嵌入张量，类型为浮点型张量，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，类型为长整型张量，默认为 None
        labels: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重，类型为布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的结果，类型为布尔值，默认为 None
        return_dict: Optional[bool] = None,
# 使用 add_start_docstrings 装饰器添加模型的文档字符串，描述模型的作用和适用场景
# 这是一个在 LayoutLM 模型基础上增加了一个标记分类头部的模型，
# 用于序列标记（信息提取）任务，比如 FUNSD 数据集和 SROIE 数据集
# 添加了链接到 FUNSD 和 SROIE 数据集的说明
class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设定类别数
        self.num_labels = config.num_labels
        # 创建 LayoutLM 模型
        self.layoutlm = LayoutLMModel(config)
        # 创建一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # 使用装饰器添加模型前向传播方法的文档字符串，描述输入参数和返回值
    # 并使用装饰器替换返回值的文档字符串的输出类型和配置类
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LayoutLMForTokenClassification
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        >>> model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

        >>> words = ["Hello", "world"]
        >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

        >>> token_boxes = []
        >>> for word, box in zip(words, normalized_word_boxes):
        ...     word_tokens = tokenizer.tokenize(word)
        ...     token_boxes.extend([box] * len(word_tokens))
        >>> # add bounding boxes of cls + sep tokens
        >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
        >>> input_ids = encoding["input_ids"]
        >>> attention_mask = encoding["attention_mask"]
        >>> token_type_ids = encoding["token_type_ids"]
        >>> bbox = torch.tensor([token_boxes])
        >>> token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  # batch size of 1

        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     bbox=bbox,
        ...     attention_mask=attention_mask,
        ...     token_type_ids=token_type_ids,
        ...     labels=token_labels,
        ... )

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        # 如果 return_dict 参数不是 None，则使用其值；否则使用模型配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 layoutlm 模型进行前向传播
        outputs = self.layoutlm(
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

        # 获取模型输出中的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对处理后的序列输出进行分类，得到分类结果
        logits = self.classifier(sequence_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回不同的输出形式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 LayoutLM 模型，为提取式问答任务添加一个跨度分类头部
# 例如 [DocVQA](https://rrc.cvc.uab.es/?ch=17) （在最终隐藏状态输出之上的线性层，用于计算“跨度开始对数”和“跨度结束对数”）
class LayoutLMForQuestionAnswering(LayoutLMPreTrainedModel):
    def __init__(self, config, has_visual_segment_embedding=True):
        # 初始化 LayoutLMForQuestionAnswering 类
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 实例化 LayoutLM 模型
        self.layoutlm = LayoutLMModel(config)
        # 初始化问题-回答输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # 前向传播函数
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
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
```