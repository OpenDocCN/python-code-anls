# `.\transformers\models\xlm_roberta_xl\modeling_xlm_roberta_xl.py`

```py
# 导入所需的模块和函数
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
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_xlm_roberta_xl import XLMRobertaXLConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型检查点和配置文件
_CHECKPOINT_FOR_DOC = "facebook/xlm-roberta-xl"
_CONFIG_FOR_DOC = "XLMRobertaXLConfig"

# 预训练模型列表
XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xlm-roberta-xl",
    "facebook/xlm-roberta-xxl",
    # See all RoBERTa models at https://huggingface.co/models?filter=xlm-roberta-xl
]

# 定义 XLMRobertaXLEmbeddings 类
class XLMRobertaXLEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """


该代码定义了一个名为 `XLMRobertaXLEmbeddings` 的类,并导入了许多必要的模块和函数。主要功能如下:

1. 导入所需的 PyTorch 模块和函数,如 `nn`、`BCEWithLogitsLoss`、`CrossEntropyLoss` 等。
2. 导入 Hugging Face 库中的一些模型输出类,如 `BaseModelOutputWithPastAndCrossAttentions`、`MaskedLMOutput` 等。
3. 导入一些工具函数,如 `add_start_docstrings`、`logging` 等。
4. 定义了 `XLMRobertaXLEmbeddings` 类,这个类是 `BertEmbeddings` 的一个变体,主要用于处理位置嵌入。
5. 定义了一些常量,如预训练模型的检查点和配置文件路径,以及预训练模型的列表。

总的来说,这段代码是一个 PyTorch 模型的定义和导入部分,为后续的模型构建和使用做好了准备。
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建词嵌入的嵌入层，根据词汇表大小和隐藏层大小创建嵌入层，并设置填充标记ID
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入的嵌入层，根据最大位置嵌入数量和隐藏层大小创建嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建token类型的嵌入层，根据token类型词汇表大小和隐藏层大小创建嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
        # self.LayerNorm不改为蛇形命名是为了与TensorFlow模型变量名保持一致，以便加载任何TensorFlow检查点文件
        # 创建丢弃层，使用隐藏层丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids（1，len position emb）在内存中是连续的，并在序列化时导出
        # 根据配置文件中的位置嵌入类型，设置为绝对或相对
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置ID缓冲区，包含从0到最大位置嵌入数量的张量
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册token类型ID缓冲区，包含与位置ID相同大小的零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    
        # End copy
        # 设置填充标记ID
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入的嵌入层，根据最大位置嵌入数量和隐藏层大小创建嵌入层，并设置填充标记ID
    
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果位置 id 为空
        if position_ids is None:
            # 如果输入 id 不为空
            if input_ids is not None:
                # 从输入的 token id 创建位置 id。任何填充的 token 保持填充状态
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 从输入嵌入创建位置 id
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中所注册的缓冲区，其中全部为零，默认情况下
        # 当它是自动生成的，注册的缓冲区可以在未传递 token_type_ids 的情况下帮助用户跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算嵌入
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 应用丢弃
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings

    # 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings.create_position_ids_from_inputs_embeds 复制而来
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        我们直接提供嵌入。我们无法推断哪些是填充的，因此只需生成顺序位置 id。

        Args:
            inputs_embeds: torch.Tensor

        返回: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序的位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制并修改为 XLMRobertaXLSelfAttention 类
class XLMRobertaXLSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不是注意力头数的倍数且没有嵌入大小，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和注意力头大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为"absolute"
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为"relative_key"或"relative_key_query"，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器的标志
        self.is_decoder = config.is_decoder

    # 将输入张量 x 转置为得分张量
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

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



# 定义 XLMRobertaXLSelfOutput 类
class XLMRobertaXLSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层和 Dropout 层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states, input_tensor):
        # 经过全连接层和 Dropout 层处理
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 添加残差连接
        hidden_states = hidden_states + input_tensor
        return hidden_states


# 定义 XLMRobertaXLAttention 类
class XLMRobertaXLAttention(nn.Module):
    # 初始化函数，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层归一化
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化自注意力层
        self.self = XLMRobertaXLSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化自注意力层输出
        self.output = XLMRobertaXLSelfOutput(config)
        # 初始化被修剪的注意力头集合
        self.pruned_heads = set()

    # 修剪头部的函数
    def prune_heads(self, heads):
        # 如果没有需要修剪的头部则返回
        if len(heads) == 0:
            return
        # 找到需要修剪的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
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
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 应用自注意力层的归一化
        intermediate = self.self_attn_layer_norm(hidden_states)
        # 使用自注意力层进行前向传播
        self_outputs = self.self(
            intermediate,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 获取注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，则将注意力加入到输出中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回输出
        return outputs
# 定义 XLMRobertaXLIntermediate 类，继承自 nn.Module
class XLMRobertaXLIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将隐藏状态的大小转换为中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串类型，则将其映射为对应的函数；否则直接使用给定的隐藏激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 XLMRobertaXLOutput 类，继承自 nn.Module
class XLMRobertaXLOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将中间大小转换为隐藏状态大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    # 前向传播函数
    def forward(self, hidden_states, input_tensor):
        # 通过线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 将线性层处理后的结果与输入张量相加
        hidden_states = hidden_states + input_tensor
        # 返回处理后的隐藏状态
        return hidden_states


# 定义 XLMRobertaXLLayer 类，继承自 nn.Module
class XLMRobertaXLLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一些属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建注意力层
        self.attention = XLMRobertaXLAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果需要使用交叉注意力，创建相应的交叉注意力层
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = XLMRobertaXLAttention(config, position_embedding_type="absolute")
        # 创建中间层，输出层和 LayerNorm 层
        self.intermediate = XLMRobertaXLIntermediate(config)
        self.output = XLMRobertaXLOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 如果过去的键/值元组不为空，解码器的自注意力缓存的键/值元组位于位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力机制处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意权重，则添加自注意力
          
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组位于过去键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力机制处理自注意力输出、注意掩码、头部掩码、编码器隐藏状态、编码器注意掩码等
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意权重，则添加交叉注意力

            # 将交叉注意力缓存添加到现在的键/值元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对前向传播应用分块处理，并返回结果
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 处理前向传播的分块，返回层输出
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.LayerNorm(attention_output)
        intermediate_output = self.intermediate(intermediate_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义一个 XLMRobertaXLEncoder 类，继承自 nn.Module 类
class XLMRobertaXLEncoder(nn.Module):
    # 初始化方法，接收配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 创建包含多个 XLMRobertaXLLayer 的列表，数量为配置参数中指定的隐藏层数量
        self.layer = nn.ModuleList([XLMRobertaXLLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建一个 LayerNorm 层，输入大小为隐藏层大小，使用配置参数中的 eps 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收多个输入参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):  
        # 如果设置了渐变检查点并且处于训练状态，则执行以下操作
        if self.gradient_checkpointing and self.training:
            # 如果使用缓存，则发出警告信息并将use_cache设置为False，因为与渐变检查点不兼容
            if use_cache:  
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        # 初始化存储隐藏状态、注意力权重和交叉注意力权重的变量 
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果使用缓存则初始化下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历每个层，并执行以下操作
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态存储到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头掩码和过去的键值
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果设置了渐变检查点并且处于训练状态，则使用渐变检查点函数来执行当前层的操作
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
            else:
                # 否则直接调用当前层的操作
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出加入到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的注意力权重加入到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中添加了交叉注意力，则将当前层的交叉注意力加入到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 对当前隐藏状态进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)

        # 如果输出隐藏状态，则将最终隐藏状态加入到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重和交叉注意力
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
        # 返回包含过去和交叉注意力的 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler中复制过来的代码，定义了XLMRobertaXLPooler类
class XLMRobertaXLPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度均为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个Tanh激活函数层
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取第一个标记对应的隐藏状态来"池化"模型。
        first_token_tensor = hidden_states[:, 0]
        # 使用全连接层处理第一个标记对应的隐藏状态
        pooled_output = self.dense(first_token_tensor)
        # 使用Tanh激活函数激活处理后的输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 定义了XLMRobertaXLPreTrainedModel类，这是一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口
class XLMRobertaXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置了配置类和基础模型前缀
    config_class = XLMRobertaXLConfig
    base_model_prefix = "roberta"

    # 从transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights中复制过来的代码
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 权重初始化，与TF版本略有不同，TF版本使用truncated_normal进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
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


# 设置XLM_ROBERTA_XL_START_DOCSTRING文档字符串
XLM_ROBERTA_XL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`XLMRobertaXLConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 设置XLM_ROBERTA_XL_INPUTS_DOCSTRING文档字符串
XLM_ROBERTA_XL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记的索引，可以使用 AutoTokenizer 获取。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 获取更多详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免对填充标记索引执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 之间：
            # - 0 对应*句子 A* 标记，
            # - 1 对应*句子 B* 标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]`。 
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中选定头部无效的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示头部**未被掩码**，
            # - 0 表示头部**被掩码**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以选择直接传递一个嵌入表示形式，而不是传递 input_ids。如果您想要更多控制如何将 input_ids 索引转换为关联向量，
            # 而不是使用模型的内部嵌入查找矩阵，则这很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详情请参见返回张量下的 'attentions'。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详情请参见返回张量下的 'hidden_states'。
        return_dict (`bool`, *optional*):
            # 是否返回一个 util.ModelOutput 而不是一个普通元组。
# 导入必要的库
from transformers import XLMRobertaXLPreTrainedModel, XLMRobertaXLEmbeddings, XLMRobertaXLEncoder, XLMRobertaXLPooler, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging
from .configuration_xlm_roberta import XLMRobertaConfig
from .modeling_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward

logger = logging.get_logger(__name__)

# 定义 XLM-RoBERTa-XL 模型类，继承自 XLMRobertaXLPreTrainedModel
@add_start_docstrings(
    "The bare XLM-RoBERTa-XL Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLModel(XLMRobertaXLPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin. To behave as an decoder the model needs to be initialized with the `is_decoder`
    argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with
    both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as
    an input to the forward pass. .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    # 初始化方法
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置配置信息
        self.config = config

        # 初始化词嵌入层和编码器
        self.embeddings = XLMRobertaXLEmbeddings(config)
        self.encoder = XLMRobertaXLEncoder(config)

        # 添加池化层（可选）
        self.pooler = XLMRobertaXLPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入词嵌入层
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入词嵌入层
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 精简模型的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    # 前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 拷贝自 transformers.models.bert.modeling_bert.BertModel.forward
    # 定义一个方法forward，用于模型的前向传播
    def forward(
        # 输入的token编号，类型为torch.Tensor，可选参数
        input_ids: Optional[torch.Tensor] = None,
        # 输入的attention mask，类型为torch.Tensor，可选参数
        attention_mask: Optional[torch.Tensor] = None,
        # 输入的token类型编号，类型为torch.Tensor，可选参数
        token_type_ids: Optional[torch.Tensor] = None,
        # 输入的位置编号，类型为torch.Tensor，可选参数
        position_ids: Optional[torch.Tensor] = None,
        # 头部遮罩，类型为torch.Tensor，可选参数
        head_mask: Optional[torch.Tensor] = None,
        # 输入的嵌入向量，类型为torch.Tensor，可选参数
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器的隐藏状态，类型为torch.Tensor，可选参数
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器的attention mask，类型为torch.Tensor，可选参数
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 之前的键值对，类型为List[torch.FloatTensor]，可选参数
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否使用缓存，类型为bool，可选参数
        use_cache: Optional[bool] = None,
        # 是否输出attention的结果，类型为bool，可选参数
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为bool，可选参数
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典式输出，类型为bool，可选参数
        return_dict: Optional[bool] = None,
# 使用自定义的文档字符串装饰器，给模型添加描述
# 使用XLM-RoBERTa-XL模型，添加顶部的语言模型头进行CLM微调
# 引用父类的文档字符串和配置信息
class XLMRobertaXLForCausalLM(XLMRobertaXLPreTrainedModel):
    # 定义被绑定权重的键
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 如果不是解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")
        # 初始化XLM-RoBERTa-XL模型和语言模型头
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaXLLMHead(config)
        # 初始化模型权重
        self.init_weights()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播方法，接受多种输入参数，并设置输出文档字符串和返回类型
    # 使用替换返回字符串的装饰器，指定输出类型和配置类
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
        pass

    # 为生成准备输入，根据输入ID和关键值准备输入
    # 如果没有关注力掩码，则创建一个全是1的掩码
    # 如果存在关键值，裁剪输入ID
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新组织缓存中的数据，以适应新的beam搜索结果
    def _reorder_cache(self, past_key_values, beam_idx):
        # 创建一个空元组用于存储重新组织后的缓存数据
        reordered_past = ()
        # 遍历过去的键-值对
        for layer_past in past_key_values:
            # 将每一层的过去状态按照beam搜索结果重新排序，并添加到reordered_past中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新组织后的缓存数据
        return reordered_past
# 使用装饰器添加文档字符串，定义一个带有语言模型头部的 XLM-RoBERTa-XL 模型类
@add_start_docstrings(
    """XLM-RoBERTa-XL Model with a `language modeling` head on top.""", XLM_ROBERTA_XL_START_DOCSTRING
)
# 继承自 XLMRobertaXLPreTrainedModel 的类，具备预训练模型的功能
class XLMRobertaXLForMaskedLM(XLMRobertaXLPreTrainedModel):
    # 定义模型中可能存在的绑定权重的键列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化模型
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中指定了该模型是一个解码器，给出警告提示
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 实例化 XLM-RoBERTa-XL 模型，不带池化层
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # 创建语言模型头部
        self.lm_head = XLMRobertaXLLMHead(config)

        # 初始化模型权重
        self.init_weights()

    # 返回输出嵌入的解码器
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入的解码器
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 使用装饰器添加文档和代码示例，定义模型的前向传播方法
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    # 定义前向传播方法，支持多种输入参数
    def forward(
        self,
        # 输入序列的张量
        input_ids: Optional[torch.LongTensor] = None,
        # 用于指定注意力掩码
        attention_mask: Optional[torch.FloatTensor] = None,
        # 用于指定标记类型的张量
        token_type_ids: Optional[torch.LongTensor] = None,
        # 用于指定位置 ID 的张量
        position_ids: Optional[torch.LongTensor] = None,
        # 用于指定头部掩码
        head_mask: Optional[torch.FloatTensor] = None,
        # 用于指定输入嵌入的张量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 用于指定编码器的隐藏状态
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 用于指定编码器注意力掩码
        encoder_attention_mask: Optional[torch.FloatFloat] = None,
        # 用于指定标签的张量
        labels: Optional[torch.LongTensor] = None,
        # 用于指定是否输出注意力
        output_attentions: Optional[bool] = None,
        # 用于指定是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 用于指定是否返回字典格式结果
        return_dict: Optional[bool] = None,
    # 定义一个函数，输入参数为input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, output_attentions, output_hidden_states, return_dict，输出结果为含有MaskedLMOutput类型元素的元组
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        # 确定返回结果类型
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 用于计算掩码语言模型损失的标签。索引应为`[-100, 0, ..., config.vocab_size]`(见`input_ids`文档)。索引设置为`-100`的标记将被忽略(掩码)，损失仅计算标签在`[0, ..., config.vocab_size]`中的标记
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            # 用于隐藏已被废弃的旧参数
            Used to hide legacy arguments that have been deprecated.
        """
        # 如果return_dict不是None，则将其赋值给return_dict，否则使用self.config.use_return_dict的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用self.roberta处理输入的参数，得到输出结果
        outputs = self.roberta(
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
        # 获取输出结果中的第一个元素
        sequence_output = outputs[0]
        # 用lm_head对sequence_output进行预测
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果labels不为None，则计算masked_lm_loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算损失函数
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果return_dict为False，则输出为(prediction_scores,)加上outputs的第三个元素开始的内容组成的元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回含有MaskedLMOutput类型元素的MaskedLMOutput对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class XLMRobertaXLLMHead(nn.Module):
    """XLM-RoBERTa-XL Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个线性层，输入和输出维度为config.hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建一个LayerNorm层，对输入进行归一化处理

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)  # 创建一个线性层，用于解码，输入维度为config.hidden_size，输出维度为config.vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个参数Tensor，用于加入到decoder的输出中作为偏置
        self.decoder.bias = self.bias  # 将偏置参数添加到decoder层中

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 将features输入到dense层中
        x = gelu(x)  # 使用Gaussian Error Linear Unit (GELU)激活函数对x进行激活
        x = self.layer_norm(x)  # 将x输入到LayerNorm层中进行归一化处理

        # project back to size of vocabulary with bias
        x = self.decoder(x)  # 将x输入到decoder层中进行解码得到输出

        return x  # 返回输出x作为forward的结果

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias  # 将decoder层的偏置参数赋值给bias


@add_start_docstrings(
    """
    XLM-RoBERTa-XL Model transformer with a sequence classification/regression head on top (a linear layer on top
    of the pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLForSequenceClassification(XLMRobertaXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 设置num_labels为传入的config的num_labels
        self.config = config  # 保存传入的config

        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)  # 创建XLMRobertaXLModel实例，并传入config和add_pooling_layer参数
        self.classifier = XLMRobertaXLClassificationHead(config)  # 创建XLMRobertaXLClassificationHead实例，并传入config

        self.init_weights()  # 初始化模型的权重参数

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保返回结果字典不为空，根据配置决定是否使用返回结果字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给预训练 RoBERTa 模型
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
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加起始文档字符串，描述了该模型是基于 XLM-RoBERTa-XL 模型的多选分类模型
# 在其顶部有一个用于多选分类任务的线性层（在池化输出之上）和一个 softmax 激活函数
# 例如，用于 RocStories/SWAG 任务
class XLMRobertaXLForMultipleChoice(XLMRobertaXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 XLM-RoBERTa-XL 模型
        self.roberta = XLMRobertaXLModel(config)
        # 添加一个 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个线性层，用于多选分类任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重
        self.init_weights()

    # 添加模型前向传播的起始文档字符串，描述了模型接受的输入参数
    # 以及输入参数的形状
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
    # 这是一个多选模型的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        # 如果 return_dict 没有设置，则使用配置中的默认值
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选项数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    
        # 将输入展平为单一的序列
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
    
        # 将展平的输入传递给 RoBERTa 模型
        outputs = self.roberta(
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
        # 获取池化输出
        pooled_output = outputs[1]
    
        # 对池化输出进行 dropout 操作
        pooled_output = self.dropout(pooled_output)
        # 将池化输出传递给分类器得到logits
        logits = self.classifier(pooled_output)
        # 将logits重塑为多选格式
        reshaped_logits = logits.view(-1, num_choices)
    
        # 如果提供了标签，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
    
        # 根据 return_dict 返回不同的输出格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 在XLM-RoBERTa-XL模型上添加一个用于标记分类的头部, 例如用于命名实体识别（NER）任务的线性层，该层在隐藏状态输出之上
@add_start_docstrings(
    """
    XLM-RoBERTa-XL Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_XL_START_DOCSTRING,
)
# 定义XLMRobertaXLForTokenClassification类，继承自XLMRobertaXLPreTrainedModel
class XLMRobertaXLForTokenClassification(XLMRobertaXLPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取配置中的标签数量
        self.num_labels = config.num_labels

        # 定义一个XLMRobertaXLModel对象
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # 获取配置中的分类器dropout，如果没有配置，则使用隐藏层dropout值
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个dropout层，用于在下面的分类器之前进行特征层的随机失活
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个全连接层，用于将隐藏层特征映射到标签数量的维度上
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重
        self.init_weights()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoBERTa 模型进行 forward pass
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

        # 获取模型输出中的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的序列输出传入分类器得到 logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果有提供标签，则计算分类损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只保留损失的激活部分
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典，构造输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class XLMRobertaXLClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 一个全连接层，输入维度为config.hidden_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的dropout，如果config.classifier_dropout不为None则使用它，否则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # Dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取features中的第一个token（等同于[CLS]）
        x = features[:, 0, :]
        # 应用dropout
        x = self.dropout(x)
        # 全连接层
        x = self.dense(x)
        # tanh激活函数
        x = torch.tanh(x)
        # 再次应用dropout
        x = self.dropout(x)
        # 输出层
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    XLM-RoBERTa-XL Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLForQuestionAnswering(XLMRobertaXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 标签的数量
        self.num_labels = config.num_labels

        # XLM-RoBERTa模型
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # 线性输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重
        self.init_weights()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    # 定义问答模型的前向传播过程
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
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
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        # 如果 return_dict 为 None，则使用配置文件中的默认值
        r"""
        # 定义输入参数的文档说明:
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            标注答案起始位置的标签，用于计算token分类损失。
            位置值会被限制在序列长度(`sequence_length`)之内，超出的位置不会被考虑在损失计算中。
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            标注答案结束位置的标签，用于计算token分类损失。
            位置值会被限制在序列长度(`sequence_length`)之内，超出的位置不会被考虑在损失计算中。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过 Roberta 编码器获取序列输出
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
    
        # 获取序列输出
        sequence_output = outputs[0]
    
        # 通过qa_outputs层获取start和end logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    
        # 计算total loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果是多GPU训练，去掉最后一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时start/end位置会超出模型输入长度，我们忽略这些
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
    
            # 使用交叉熵loss计算start和end loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
    
        # 根据 return_dict 返回相应的输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从给定的输入ids中创建位置ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    用非填充符号替换它们的位置数字。位置数字从padding_idx+1开始。填充符号被忽略。这是改编自fairseq的`utils.make_positions`。

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # 将非填充符号替换为它们的位置数。位置数字从padding_idx+1开始。填充符号被忽略。
    mask = input_ids.ne(padding_idx).int()
    # 增量索引用于确定输入ids中每个元素的位置。past_key_values_length用于确定之前的键值对的长度。
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回增量索引加上填充索引后的结果
    return incremental_indices.long() + padding_idx
```