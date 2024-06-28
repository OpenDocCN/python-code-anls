# `.\models\data2vec\modeling_data2vec_text.py`

```py
# 设置文件编码为UTF-8，确保支持中文等多种字符集
# 版权声明，告知代码的版权归属于The HuggingFace Inc.团队
#
# 根据Apache许可证2.0版授权使用本文件
# 除非符合许可证的要求，否则不得使用本文件
# 可以从以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证了解具体语言的权限和限制
"""PyTorch Data2VecText model."""

# 导入数学库，用于数学运算
import math
# 导入类型提示工具，用于函数参数和返回值的类型注释
from typing import List, Optional, Tuple, Union

# 导入PyTorch相关库
import torch
# 导入PyTorch中的checkpoint工具
import torch.utils.checkpoint
# 导入PyTorch中的神经网络模块
from torch import nn
# 导入PyTorch中的损失函数：二分类交叉熵损失、多分类交叉熵损失、均方误差损失
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数映射表和GELU激活函数
from ...activations import ACT2FN, gelu
# 导入模型输出类，包括基础输出、带过去和交叉注意力的基础输出、带池化和交叉注意力的基础输出、因果语言模型输出和交叉注意力、掩码语言模型输出、多选模型输出、问答模型输出、序列分类器输出、标记分类器输出
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
# 导入模型工具类，包括预训练模型和一些工具函数
from ...modeling_utils import PreTrainedModel
# 导入PyTorch工具类，应用前向传播分块、找到可修剪头和索引、修剪线性层
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入通用工具，包括添加代码示例文档字符串、添加起始文档字符串、将起始文档字符串添加到模型前向方法、日志记录、替换返回文档字符串
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入Data2VecText的配置类
from .configuration_data2vec_text import Data2VecTextConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# Data2VecText模型中隐藏状态的起始位置常量
_HIDDEN_STATES_START_POSITION = 2

# 文档中常用的检查点示例
_CHECKPOINT_FOR_DOC = "facebook/data2vec-text-base"
# 文档中常用的配置示例
_CONFIG_FOR_DOC = "Data2VecTextConfig"

# Data2VecText预训练模型的存档列表
DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-text-base",
    # 更多Data2VecText模型示例请查看 https://huggingface.co/models?filter=data2vec-text
]

# 从transformers.models.roberta.modeling_roberta.RobertaEmbeddings复制并修改为Data2VecText
class Data2VecTextForTextEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    
    # 从transformers.models.bert.modeling_bert.BertEmbeddings.__init__复制
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化词嵌入层，根据配置文件指定词汇表大小、隐藏层大小，并设置填充标记索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，根据配置文件指定最大位置嵌入数和隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化标记类型嵌入层，根据配置文件指定类型词汇表大小和隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 的命名不使用蛇形命名法，以便与 TensorFlow 模型变量名保持一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，使用配置文件中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册一个缓冲区变量 position_ids，包含从 0 到最大位置嵌入数的序列，不持久化
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册一个缓冲区变量 token_type_ids，初始化为与 position_ids 相同形状的全零张量，不持久化
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # 设置填充标记索引为配置文件中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 初始化位置嵌入层，根据配置文件指定最大位置嵌入数和隐藏层大小，使用与 padding_idx 相同的填充索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        ):
            如果没有提供位置 id：
                如果提供了输入 token id：
                    # 根据输入 token id 创建位置 id。任何填充的 token 保持填充状态。
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                否则：
                    # 根据输入的嵌入张量创建位置 id
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        如果提供了输入 token id：
            # 获取输入 token id 的形状
            input_shape = input_ids.size()
        否则：
            # 获取输入嵌入张量的形状（去掉最后一个维度，即序列长度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，通常情况下全为零。这有助于用户在跟踪模型时不传递 token_type_ids，解决问题 #5664
        如果 token_type_ids 为空：
            如果 self 中有 "token_type_ids" 属性：
                # 使用已注册的缓冲区的 token_type_ids，截取到序列长度的部分并扩展为与输入形状相同大小
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            否则：
                # 创建全零的 token_type_ids，其形状与输入相同，数据类型为 long，设备为 self.position_ids 的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        如果 inputs_embeds 为空：
            # 使用 word_embeddings 方法根据输入 token id 获取嵌入张量
            inputs_embeds = self.word_embeddings(input_ids)
        # 使用 token_type_embeddings 方法根据 token_type_ids 获取 token type 嵌入张量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入张量和 token type 嵌入张量相加得到最终嵌入张量
        embeddings = inputs_embeds + token_type_embeddings

        如果 self.position_embedding_type == "absolute"：
            # 如果使用绝对位置嵌入类型，则根据位置 ids 获取位置嵌入张量并加到最终嵌入张量上
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对最终嵌入张量进行 LayerNorm 规范化
        embeddings = self.LayerNorm(embeddings)
        # 对最终嵌入张量进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回最终嵌入张量
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        直接提供嵌入张量，无法推断哪些是填充的，因此生成顺序的位置 id。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入张量的形状（去掉最后一个维度，即序列长度）
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 创建顺序的位置 id，从 self.padding_idx + 1 开始，到 sequence_length + self.padding_idx + 1 结束
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 扩展位置 id 的维度，使其与输入张量形状相同
        return position_ids.unsqueeze(0).expand(input_shape)
# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->Data2VecText
class Data2VecTextSelfAttention(nn.Module):
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

        # Linear transformation for query, key, and value tensors
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # Conditionally initialize distance embeddings based on position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # Reshape the tensor for multi-head attention computation
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class Data2VecTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Fully connected layer for self-output transformation
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization to stabilize learning
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout regularization to prevent overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # Linear transformation followed by dropout
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Residual connection followed by layer normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Data2VecText
# 定义 Data2VecTextAttention 类，继承自 nn.Module，用于处理 Data2VecText 模型的自注意力机制

class Data2VecTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 self 层，使用 Data2VecTextSelfAttention 类处理自注意力机制
        self.self = Data2VecTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 output 层，使用 Data2VecTextSelfOutput 类处理自注意力机制的输出
        self.output = Data2VecTextSelfOutput(config)
        # 初始化一个空集合，用于存储被剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可以剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        # 调用 self 层处理输入的隐藏状态和相关的参数
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 output 层处理 self 层的输出和原始的隐藏状态，得到注意力机制的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将它们添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，添加注意力权重
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
# 定义 Data2VecTextIntermediate 类，继承自 nn.Module，用于处理 Data2VecText 模型的中间层

class Data2VecTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个线性层，将输入的隐藏状态映射到中间层的大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则使用对应的激活函数，否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过线性层映射到中间层的大小
        hidden_states = self.dense(hidden_states)
        # 使用中间层的激活函数处理映射后的结果
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
# 定义 Data2VecTextOutput 类，继承自 nn.Module，用于处理 Data2VecText 模型的输出层
class Data2VecTextOutput(nn.Module):
    # 初始化方法，接受一个名为 config 的参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，将输入特征的大小设为 config.intermediate_size，输出特征的大小设为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入特征的大小为 config.hidden_size，使用 config.layer_norm_eps 作为 epsilon 参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，使用 config.hidden_dropout_prob 作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受两个参数 hidden_states 和 input_tensor，返回一个 torch.Tensor 类型的值
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到 self.dense 线性层中，得到输出 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对 hidden_states 应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的 hidden_states 与 input_tensor 相加，并输入到 self.LayerNorm 层中进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的 hidden_states
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLayer 复制并修改为 Data2VecTextLayer
class Data2VecTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化模型的配置参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设定为 1
        self.seq_len_dim = 1
        # 初始化自注意力层
        self.attention = Data2VecTextAttention(config)
        # 是否为解码器模型
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力但不是解码器模型，则引发错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力层，并使用绝对位置编码
            self.crossattention = Data2VecTextAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = Data2VecTextIntermediate(config)
        # 初始化输出层
        self.output = Data2VecTextOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention mechanism using the stored key/value pairs from previous steps if available
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # Extract the attention output from self-attention mechanism
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Exclude the first and the last element of self_attention_outputs which are the attention output
            # and the present_key_value respectively
            outputs = self_attention_outputs[1:-1]
            # Retrieve the present key/value tuple from self-attention outputs
            present_key_value = self_attention_outputs[-1]
        else:
            # Include all elements except the first element (attention_output) if output_attentions is enabled
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # Raise an error if cross-attention layers are expected but not instantiated
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention mechanism using stored key/value pairs from previous steps if available
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # Extract the attention output from cross-attention mechanism
            attention_output = cross_attention_outputs[0]
            # Combine outputs with cross-attention outputs excluding the first and the last element
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            # Concatenate present_key_value with cross-attn present_key_value
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking to the forward pass of the feed forward layer
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # Combine layer_output with outputs
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            # Append present_key_value to outputs if the model is a decoder
            outputs = outputs + (present_key_value,)

        # Return all outputs of the transformer layer
        return outputs

    def feed_forward_chunk(self, attention_output):
        # Apply feed forward chunk processing using intermediate and output layers
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制而来，将 Bert 替换为 Data2VecText
class Data2VecTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个 nn.ModuleList，包含 config.num_hidden_layers 个 Data2VecTextLayer 的实例
        self.layer = nn.ModuleList([Data2VecTextLayer(config) for _ in range(config.num_hidden_layers)])
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
        # 如果不需要输出隐藏状态，则初始化为空元组；否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组；否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出交叉注意力权重或配置不支持，则初始化为空元组；否则为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 设置为 True，则发出警告并将其设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果 use_cache 为 True，则初始化下一个解码器缓存为空元组；否则为 None
        next_decoder_cache = () if use_cache else None

        # 遍历所有的解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则添加当前层的隐藏状态到 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有头部掩码，则使用当前层对应的头部掩码；否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果有过去的键值对，则使用当前层对应的过去键值对；否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播计算
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
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层输出的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果 use_cache 为 True，则更新下一个解码器缓存
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则添加当前层输出的注意力权重到 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置支持交叉注意力，则添加当前层输出的交叉注意力到 all_cross_attentions
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则添加最终隐藏状态到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回一个元组，包含非空值
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
        # 否则返回一个 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler
class Data2VecTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从隐藏状态中取出第一个 token 对应的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数处理全连接层的输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


class Data2VecTextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Data2VecTextConfig
    base_model_prefix = "data2vec_text"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Data2VecTextForTextEmbeddings", "Data2VecTextLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果指定了 padding_idx，则将其对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                # 如果有偏置项，则将其初始化为零
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                # 如果有权重项，则将其初始化为全 1
                module.weight.data.fill_(1.0)


DATA2VECTEXT_START_DOCSTRING = r"""
    Data2VecText was proposed in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and
    Language](https://arxiv.org/pdf/2202.03555) by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and
    Michael Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
"""
    Parameters:
        config ([`Data2VecTextConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
定义了一个多行字符串常量，用于文档字符串的输入参数说明。
"""

@add_start_docstrings(
    "The bare Data2VecText Model for text transformer outputting raw hidden-states without any specific head on top.",
    DATA2VECTEXT_START_DOCSTRING,
)
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as a decoder, the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    # 初始化函数，用于初始化模型
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置信息保存在实例中
        self.config = config

        # 初始化词嵌入层
        self.embeddings = Data2VecTextForTextEmbeddings(config)
        # 初始化文本编码器
        self.encoder = Data2VecTextEncoder(config)

        # 根据需要添加池化层
        self.pooler = Data2VecTextPooler(config) if add_pooling_layer else None

        # 执行后续的初始化操作
        self.post_init()

    # 获取输入词嵌入的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入词嵌入的方法
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中注意力头部的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头部进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 覆盖的前向传播方法，实现模型的前向计算
    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制过来的
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """Data2VecText Model with a `language modeling` head on top for CLM fine-tuning.""", DATA2VECTEXT_START_DOCSTRING
)
class Data2VecTextForCausalLM(Data2VecTextPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `Data2VecTextLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 Data2VecTextModel，不包含池化层
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        # 初始化语言模型头部 Data2VecTextLMHead
        self.lm_head = Data2VecTextLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回语言模型头部的解码器权重
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的解码器权重
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 模型前向传播函数，详细参数说明参见 add_start_docstrings_to_model_forward 的注释
        ...

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力遮罩，创建全为1的遮罩
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果传入了过去的键值对，裁剪输入的 input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认的旧行为：只保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含准备好的输入信息的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 定义一个方法 `_reorder_cache`，用于重排序缓存中的过去键值
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空的元组用于存储重排序后的过去键值
        reordered_past = ()
        # 遍历传入的 past_key_values 中的每一层的过去状态
        for layer_past in past_key_values:
            # 对于每一层的过去状态，按照 beam_idx 给定的顺序进行索引选择，并转移到对应的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                # 将重排序后的每一层的过去状态添加到 reordered_past 中
            )
        # 返回重排序后的 past_key_values
        return reordered_past
# 给 Data2VecTextForMaskedLM 类添加文档字符串，描述其作为一个在顶部带有语言建模头部的 data2vec 模型
@add_start_docstrings("""data2vec Model with a `language modeling` head on top.""", DATA2VECTEXT_START_DOCSTRING)
class Data2VecTextForMaskedLM(Data2VecTextPreTrainedModel):
    # 定义与权重绑定的关键字列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置指定为解码器，发出警告提示
        if config.is_decoder:
            logger.warning(
                "If you want to use `Data2VecTextForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 data2vec_text 模型和 lm_head
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        self.lm_head = Data2VecTextLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回 lm_head 的解码器
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置 lm_head 的解码器的新嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 为 forward 方法添加模型输入的文档字符串和代码示例的文档字符串
    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 根据 return_dict 是否为 None，决定是否使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 data2vec_text 方法，生成预测输出
        outputs = self.data2vec_text(
            input_ids,
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
        # 获取预测输出的序列部分
        sequence_output = outputs[0]
        # 将序列输出送入语言模型头部，生成预测分数
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果 labels 不为 None，则计算 masked language modeling 损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()

            # 将 labels 移动到与 prediction_scores 相同的设备上
            labels = labels.to(prediction_scores.device)
            # 计算 masked language modeling 损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回一个包含预测分数和其他输出的元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则返回一个 MaskedLMOutput 对象，包含损失、预测分数、隐藏状态和注意力
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaLMHead复制并将Roberta改为Data2VecText
class Data2VecTextLMHead(nn.Module):
    """Data2VecText Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 线性层，将隐藏状态映射回词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 使用dense层进行线性变换
        x = gelu(x)  # 应用GELU激活函数
        x = self.layer_norm(x)  # 应用LayerNorm

        # 使用decoder层将特征映射回词汇表大小
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 当这两个权重断开连接时（在TPU上或者当偏置被重新调整大小时），用于绑定这两个权重
        # 为了加速兼容性和不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    Data2VecText模型变换器，顶部带有序列分类/回归头（汇总输出的线性层），例如用于GLUE任务。
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextForSequenceClassification(Data2VecTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)  # 初始化Data2VecText模型
        self.classifier = Data2VecTextClassificationHead(config)  # 初始化Data2VecText分类头部

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
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
        # 根据需要决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 data2vec_text 方法，获取模型的输出
        outputs = self.data2vec_text(
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传入分类器获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值
        loss = None
        # 如果提供了标签，进行损失计算
        if labels is not None:
            # 将标签移动到 logits 的设备上
            labels = labels.to(logits.device)

            # 根据问题类型设置配置的问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据不同的问题类型选择损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归任务，计算 MSE 损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算 MSE 损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，计算交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，计算带 logits 的 BCE 损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典形式的输出，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、logits、隐藏状态和注意力的 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串描述模型基础信息和任务应用场景
@add_start_docstrings(
    """
    Data2VecText Model with a multiple choice classification head on top (a linear layer on top of the pooled output
    and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
# 定义一个新的类 Data2VecTextForMultipleChoice，继承自 Data2VecTextPreTrainedModel
class Data2VecTextForMultipleChoice(Data2VecTextPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 Data2VecTextModel 对象
        self.data2vec_text = Data2VecTextModel(config)
        # 添加一个 dropout 层，使用配置中的隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个线性层用于分类，输入大小为配置中的隐藏层大小，输出为1（用于二分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串描述模型前向传播的输入参数
    @add_start_docstrings_to_model_forward(
        DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加代码示例文档字符串，包含模型输出类型、检查点和配置信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法定义
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 后续还有更多的参数，但这里不对其进行注释
    ):
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 如果 return_dict 参数为 None，则使用 self.config.use_return_dict 决定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择题数量，根据 input_ids 的第二维度确定，如果 input_ids 为 None，则根据 inputs_embeds 的第二维度确定
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 input_ids 展平为二维张量，用于模型输入，如果 input_ids 为 None，则 flat_input_ids 也为 None
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将 position_ids 展平为二维张量，如果 position_ids 为 None，则 flat_position_ids 也为 None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将 token_type_ids 展平为二维张量，如果 token_type_ids 为 None，则 flat_token_type_ids 也为 None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 attention_mask 展平为二维张量，如果 attention_mask 为 None，则 flat_attention_mask 也为 None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 inputs_embeds 展平为三维张量，如果 inputs_embeds 为 None，则 flat_inputs_embeds 也为 None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用模型的 data2vec_text 方法，传入展平后的张量作为参数，并获取模型输出
        outputs = self.data2vec_text(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取模型输出中的汇聚输出，即模型的汇总表示
        pooled_output = outputs[1]

        # 对汇聚输出应用 dropout 操作，以防止过拟合
        pooled_output = self.dropout(pooled_output)
        # 将汇聚输出传入分类器，计算分类 logits
        logits = self.classifier(pooled_output)
        # 重新调整 logits 的形状为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为 None，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 将 labels 转移到 reshaped_logits 的设备上，计算交叉熵损失
            labels = labels.to(reshaped_logits.device)
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回一个 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Data2VecText Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextForTokenClassification(Data2VecTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数目

        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)  # 初始化Data2VecText模型，不添加池化层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 使用分类器的dropout或者隐藏层的dropout概率
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 线性层，将隐藏状态映射到标签数目

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        """
        Perform a forward pass through the model with optional inputs and outputs.

        Args:
            input_ids (Optional[torch.LongTensor]): The input tensor of token indices.
            attention_mask (Optional[torch.FloatTensor]): The attention mask tensor.
            token_type_ids (Optional[torch.LongTensor]): The token type IDs tensor.
            position_ids (Optional[torch.LongTensor]): The position IDs tensor.
            head_mask (Optional[torch.FloatTensor]): The head mask tensor.
            inputs_embeds (Optional[torch.FloatTensor]): The embedded input tensors.
            labels (Optional[torch.LongTensor]): The tensor of labels for classification.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return outputs as a dictionary.

        Returns:
            TokenClassifierOutput: Output object with logits and optional additional outputs.
        """
        # 实现模型的前向传播逻辑，生成对应的输出对象
        pass  # placeholder, 实际逻辑应填充在这里
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 决定返回值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 data2vec_text 方法，获取输出结果
        outputs = self.data2vec_text(
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

        # 从输出中获取序列输出
        sequence_output = outputs[0]

        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 将 dropout 后的输出传递给分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # 将标签转移到 logits 的设备上，并计算损失
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则构造输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构造 TokenClassifierOutput 对象并返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制并修改为Data2VecTextClassificationHead
class Data2VecTextClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    def __init__(self, config):
        super().__init__()
        # 全连接层，输入和输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的dropout率，如果未指定则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # Dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出投影层，将hidden_size映射到num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取features的第一个token（等同于[CLS]）
        x = features[:, 0, :]
        # 应用dropout
        x = self.dropout(x)
        # 全连接层
        x = self.dense(x)
        # 使用tanh激活函数
        x = torch.tanh(x)
        # 再次应用dropout
        x = self.dropout(x)
        # 输出投影层
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    Data2VecText模型的问题回答任务头部，用于像SQuAD这样的抽取式问答任务（在隐藏状态输出之上使用线性层来计算“起始位置logits”和“结束位置logits”）。
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextForQuestionAnswering(Data2VecTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels

        # Data2VecText模型的实例，不包含池化层
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        # 问题回答输出层，全连接层将hidden_size映射到num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
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
        # 初始化 return_dict，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 data2vec_text 方法，将输入数据转换为向量表示
        outputs = self.data2vec_text(
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

        # 从输出中获取序列输出（通常是模型的最后一层隐藏状态）
        sequence_output = outputs[0]

        # 将序列输出传递给 qa_outputs 模型，获得问题回答的 logits
        logits = self.qa_outputs(sequence_output)

        # 将 logits 拆分为开始和结束位置的预测 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # 去除多余的维度并保持连续性
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 的维度大于 1，则去除多余的维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 忽略超出模型输入范围的 start/end positions
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定的 ignore_index
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 计算开始位置和结束位置损失的平均值作为总损失
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不要求返回字典，则返回 start_logits, end_logits 和其它可能的输出
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回一个 QuestionAnsweringModelOutput 对象，包含损失、开始和结束位置的 logits，以及其它可能的输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: torch.Tensor, input tensor containing token IDs
        padding_idx: int, index of padding token
        past_key_values_length: int, optional, length of past key values

    Returns:
        torch.Tensor, tensor of position IDs corresponding to input_ids
    """
    # 创建一个掩码，标记非填充符号的位置为1，填充符号位置为0
    mask = input_ids.ne(padding_idx).int()
    # 根据掩码累积计数，并添加过去键值长度，然后乘以掩码，以得到增量索引
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将增量索引转换为长整型，并加上填充索引，得到最终的位置 ID
    return incremental_indices.long() + padding_idx
```