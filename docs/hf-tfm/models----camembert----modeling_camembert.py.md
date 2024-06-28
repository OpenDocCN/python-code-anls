# `.\models\camembert\modeling_camembert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：2019 年由 Inria、Facebook AI Research 和 HuggingFace Inc. 团队创建
# 版权声明：2018 年，NVIDIA CORPORATION 版权所有
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“按现状”提供的，
# 没有任何形式的明示或暗示保证，包括但不限于对适销性或特定用途适用性的暗示保证。
# 有关详细信息，请参阅许可证。
"""PyTorch CamemBERT 模型。"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从外部导入一些自定义模块和类
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
# 从模型配置文件中导入 CamembertConfig 类
from .configuration_camembert import CamembertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "almanach/camembert-base"
_CONFIG_FOR_DOC = "CamembertConfig"

# 预训练模型存档列表
CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "almanach/camembert-base",
    "Musixmatch/umberto-commoncrawl-cased-v1",
    "Musixmatch/umberto-wikipedia-uncased-v1",
    # 查看所有 CamemBERT 模型：https://huggingface.co/models?filter=camembert
]

# CamemBERT 模型起始文档字符串
CAMEMBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CamembertConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->Camembert
class CamembertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        # 创建词嵌入层，用于将输入的词索引转换为词向量表示
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，用于存储位置信息的嵌入表示
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建类型嵌入层，用于存储token的类型信息的嵌入表示
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 创建 LayerNorm 层，用于归一化隐藏状态向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，用于在训练过程中随机置零一部分输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 初始化位置嵌入类型，指定为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册 position_ids 缓冲区，用于存储位置嵌入的位置索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册 token_type_ids 缓冲区，用于存储类型嵌入的 token 类型索引
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置 padding_idx 为 config.pad_token_id，用于指定 padding 位置的索引
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层，用于存储位置信息的嵌入表示，指定 padding 索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果位置标识符为空，则根据输入的标记标识符创建位置标识符。任何填充的标记仍然保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        else:
            # 否则，根据输入的嵌入张量创建位置标识符
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            # 如果输入标记标识符不为空，则获取其形状
            input_shape = input_ids.size()
        else:
            # 否则，获取输入嵌入张量的形状，但不包括最后一维
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，通常情况下为全零，这有助于用户在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 如果模型有 token_type_ids 属性，则使用其注册的缓冲区，并扩展以匹配输入的形状
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，创建全零的 token_type_ids 张量，其形状与输入形状相同，并使用与 position_ids 相同的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果输入嵌入张量为空，则通过输入标记标识符获取单词嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的嵌入向量：输入嵌入加上 token_type_embeddings
        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型是 "absolute"，则添加位置嵌入到最终的嵌入向量中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 应用 LayerNorm 层对嵌入向量进行归一化
        embeddings = self.LayerNorm(embeddings)

        # 对归一化后的嵌入向量进行 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量作为输出
        return embeddings
# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->Camembert
class CamembertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否能够被注意力头数整除，如果不行且配置中没有嵌入大小，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置嵌入，初始化距离嵌入的 Embedding
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标识是否为解码器
        self.is_decoder = config.is_decoder

    # 调整张量形状以便进行注意力计算
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
# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput with Roberta->Camembert
class CamembertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化密集连接层、LayerNorm 层和 dropout 层
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
# Copied from transformers.models.roberta.modeling_roberta.RobertaAttention with Roberta->Camembert
class CamembertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层，使用CamembertSelfAttention类
        self.self = CamembertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化输出层，使用CamembertSelfOutput类
        self.output = CamembertSelfOutput(config)
        # 存储被修剪的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的注意力头和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的注意力头
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
        # 调用自注意力层的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用输出层的前向传播，得到注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力信息，则将其加入到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Roberta->Camembert
class CamembertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层，将隐藏状态维度转换为中间状态维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置初始化中间激活函数
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Roberta->Camembert
class CamembertOutput(nn.Module):
    # 初始化函数，用于初始化一个神经网络层
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，将输入特征的大小设为 config.intermediate_size，输出特征的大小设为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对输入进行归一化处理，归一化的特征维度为 config.hidden_size，设置归一化的 epsilon 值为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，以 config.hidden_dropout_prob 的概率随机将输入置为 0，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受两个输入张量，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入 hidden_states 经过全连接层 self.dense，得到新的 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对新的 hidden_states 应用 Dropout，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将经过 Dropout 的 hidden_states 与输入张量 input_tensor 相加，然后经过 LayerNorm 层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回最终处理后的 hidden_states 作为输出结果
        return hidden_states
# 从transformers.models.roberta.modeling_roberta.RobertaLayer复制的代码，将Roberta替换为Camembert
class CamembertLayer(nn.Module):
    # 初始化函数，接受一个配置对象config作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置前向传播中用于分块的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度的索引，通常为1
        self.seq_len_dim = 1
        # 使用配置对象创建CamembertAttention层
        self.attention = CamembertAttention(config)
        # 是否作为解码器使用的标志
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力的标志
        self.add_cross_attention = config.add_cross_attention
        # 如果设置了添加交叉注意力，且不是解码器模型，则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置编码类型创建交叉注意力层
            self.crossattention = CamembertAttention(config, position_embedding_type="absolute")
        # CamembertIntermediate中间层
        self.intermediate = CamembertIntermediate(config)
        # CamembertOutput输出层
        self.output = CamembertOutput(config)

    # 前向传播函数，接受多个参数作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码张量，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码张量，可选
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态张量，可选
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器注意力掩码张量，可选
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值元组，可选
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，缺省为False
    ) -> Tuple[torch.Tensor]:
        # 声明函数的返回类型为一个包含单个 torch.Tensor 的元组
        # 如果有过去的注意力缓存，获取解码器单向自注意力的缓存键/值元组，位置在1,2处
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用当前模块中的注意力层进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果当前模块是解码器模块，最后一个输出为自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 否则将自注意力计算的输出作为结果之一，并添加自注意力权重输出
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        # 如果当前模块是解码器且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果当前模块没有交叉注意力层，则引发值错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 获取解码器交叉注意力缓存的键/值元组，位置在3,4处
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力层计算交叉注意力输出
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
            # 将交叉注意力计算的输出添加到结果之一，并添加交叉注意力权重输出
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的当前键/值元组添加到当前键/值元组中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用前向传播的分块策略到注意力输出上
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将分块后的结果作为输出之一
        outputs = (layer_output,) + outputs

        # 如果当前模块是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回所有的输出
        return outputs

    # 定义一个处理注意力输出的分块函数
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层的输出，得到最终的层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的层输出
        return layer_output
# 从transformers.models.roberta.modeling_roberta.RobertaEncoder复制并修改为CamembertEncoder
class CamembertEncoder(nn.Module):
    # 初始化函数，接收一个配置对象config作为参数
    def __init__(self, config):
        super().__init__()
        # 将传入的配置对象保存到成员变量self.config中
        self.config = config
        # 使用列表推导式创建一个由CamembertLayer对象组成的ModuleList，长度为config.num_hidden_layers
        self.layer = nn.ModuleList([CamembertLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点功能
        self.gradient_checkpointing = False

    # 前向传播函数，接收多个输入参数，具体功能在后续方法体中实现
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
        # 初始化存储所有隐藏状态的元组，如果不需要输出隐藏状态则为None
        all_hidden_states = () if output_hidden_states else None
        # 初始化存储所有自注意力机制结果的元组，如果不需要输出注意力则为None
        all_self_attentions = () if output_attentions else None
        # 初始化存储所有交叉注意力机制结果的元组，如果不需要输出交叉注意力则为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启了梯度检查点且处于训练阶段
        if self.gradient_checkpointing and self.training:
            # 如果使用了缓存，则给出警告并设置use_cache为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果需要使用缓存，则初始化下一个解码器缓存的元组，否则设为None
        next_decoder_cache = () if use_cache else None
        # 遍历所有解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入all_hidden_states元组
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部遮罩，如果没有则设为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对，如果没有则设为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点且处于训练阶段
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数计算当前层的输出
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
                # 否则直接调用当前层模块计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层输出的最后一个元素加入下一个解码器缓存元组
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力，则将当前层输出的第二个元素加入all_self_attentions元组
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置要求添加交叉注意力，则将当前层输出的第三个元素加入all_cross_attentions元组
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入all_hidden_states元组
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不使用返回字典结构，则按照顺序返回相关的输出元组
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
        # 否则，返回带有过去键值和交叉注意力的基本模型输出对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler
class CamembertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 获取第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态通过线性层
        pooled_output = self.dense(first_token_tensor)
        # 应用 Tanh 激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CamembertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CamembertConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # 使用正态分布初始化线性层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果定义了 padding_idx，则将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 层的偏置项初始化为零
            module.bias.data.zero_()
            # 将 LayerNorm 层的权重初始化为全1
            module.weight.data.fill_(1.0)


CAMEMBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取这些索引。
            # 参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取详细信息。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免在填充的标记索引上执行注意力操作。
            # 遮罩值选取在 `[0, 1]` 之间：
            # - 1 表示**未遮罩**的标记，
            # - 0 表示**遮罩**的标记。
            # [什么是注意力遮罩？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一和第二部分。
            # 索引选取在 `[0, 1]` 之间：
            # - 0 对应*句子 A* 的标记，
            # - 1 对应*句子 B* 的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列中每个标记在位置嵌入中的位置索引。
            # 索引选取在 `[0, config.max_position_embeddings - 1]` 范围内。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块中选择的头部置空的遮罩。
            # 遮罩值选取在 `[0, 1]` 之间：
            # - 1 表示**未遮罩**的头部，
            # - 0 表示**遮罩**的头部。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，代替传递 `input_ids`，您可以直接传递嵌入表示。
            # 如果您希望更加控制将 `input_ids` 索引转换为关联向量的方式，这将会很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 有关更多详细信息，请参见返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 有关更多详细信息，请参见返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
    """

    # 从 transformers.models.roberta.modeling_roberta.RobertaClassificationHead 复制并修改为支持 Camembert
    class CamembertClassificationHead(nn.Module):
        """用于句子级分类任务的头部模块。"""

        def __init__(self, config):
            super().__init__()
            # 密集连接层，将输入特征映射到隐藏层大小
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            # 分类器的 dropout 操作
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
            self.dropout = nn.Dropout(classifier_dropout)
            # 输出投影层，将隐藏层映射到标签数量大小
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            # 取特征的第一个位置处的向量，对应于 <s> 标记（等同于 [CLS]）
            x = features[:, 0, :]
            x = self.dropout(x)  # 应用 dropout
            x = self.dense(x)  # 密集连接层
            x = torch.tanh(x)  # 使用双曲正切激活函数
            x = self.dropout(x)  # 再次应用 dropout
            x = self.out_proj(x)  # 输出投影层映射到标签数量大小
            return x

    # 从 transformers.models.roberta.modeling_roberta.RobertaLMHead 复制并修改为支持 Camembert
    class CamembertLMHead(nn.Module):
        """用于掩码语言建模的 Camembert 头部模块。"""

        def __init__(self, config):
            super().__init__()
            # 密集连接层，将输入特征映射到隐藏层大小
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            # LayerNorm 层，用于归一化隐藏层特征
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

            # 解码层，将隐藏层映射回词汇表大小
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))
            self.decoder.bias = self.bias

        def forward(self, features, **kwargs):
            x = self.dense(features)  # 密集连接层
            x = gelu(x)  # 使用 GELU 激活函数
            x = self.layer_norm(x)  # LayerNorm 归一化

            # 使用偏置将特征映射回词汇表大小
            x = self.decoder(x)

            return x

        def _tie_weights(self):
            # 如果权重断开连接（在 TPU 上或当偏置被调整大小时），则重新绑定这两个权重
            # 为了加速兼容性和不破坏向后兼容性
            if self.decoder.bias.device.type == "meta":
                self.decoder.bias = self.bias
            else:
                self.bias = self.decoder.bias

    @add_start_docstrings(
        "The bare CamemBERT Model transformer outputting raw hidden-states without any specific head on top.",
        CAMEMBERT_START_DOCSTRING,
    )
    # 从 CamembertPreTrainedModel 继承的 CamembertModel 类
    class CamembertModel(CamembertPreTrainedModel):
        """
        模型可以作为编码器（仅自注意力）或解码器使用，此时在自注意力层之间添加了一层交叉注意力层，遵循 *Attention is
        all you need*_ 中描述的架构，作者是 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion
        Jones、Aidan N. Gomez、Lukasz Kaiser 和 Illia Polosukhin。

        要作为解码器使用，模型需要使用配置设置中的 `is_decoder` 参数初始化为 `True`。要用于 Seq2Seq 模型，
        模型需要同时使用 `is_decoder` 参数和
    ```
    """
    add_cross_attention 设置为 True；预期在前向传播中作为输入传入 encoder_hidden_states。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _no_split_modules = []

    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制并修改为 Camembert
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层和编码器
        self.embeddings = CamembertEmbeddings(config)
        self.encoder = CamembertEncoder(config)

        # 如果需要添加池化层，则初始化池化器
        self.pooler = CamembertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        对模型的注意力头进行修剪。heads_to_prune: {layer_num: 要在该层中修剪的头列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制
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
# 使用装饰器将文档字符串添加到模型类的定义中，描述了此类是一个带有语言建模头部的CamemBERT模型。
# 这些文档字符串是从CAMEMBERT_START_DOCSTRING导入的基础信息后面增加的。
@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top.""",
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForMaskedLM复制过来，将Roberta改为Camembert，ROBERTA改为CAMEMBERT。
class CamembertForMaskedLM(CamembertPreTrainedModel):
    # 定义了一个列表，包含了lm_head.decoder.weight和lm_head.decoder.bias，这些权重是被绑定的。
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化方法，接受一个config参数，并调用其父类的初始化方法。
    def __init__(self, config):
        super().__init__(config)

        # 如果config.is_decoder为True，给出警告，建议在使用CamembertForMaskedLM时将其设为False，以使用双向自注意力。
        if config.is_decoder:
            logger.warning(
                "If you want to use `CamembertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化一个CamembertModel对象，并禁用添加池化层。
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 初始化一个CamembertLMHead对象。
        self.lm_head = CamembertLMHead(config)

        # 初始化权重并应用最终处理。
        self.post_init()

    # 返回语言建模头部的输出嵌入。
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置语言建模头部的输出嵌入为新的嵌入。
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播方法，接受多个输入参数，并且被装饰器修饰，添加了一些模型前向传播的文档字符串。
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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
        # 以下是方法参数的描述，注释解释了每个参数的作用和类型。
    ):
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # Determine whether to use a return dictionary based on the provided argument or the default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the input data through the Roberta model to obtain outputs
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
        # Retrieve the sequence output from the Roberta model's outputs
        sequence_output = outputs[0]
        # Generate prediction scores using the language modeling head
        prediction_scores = self.lm_head(sequence_output)

        # Initialize the masked language modeling loss variable
        masked_lm_loss = None
        # Calculate the masked language modeling loss if labels are provided
        if labels is not None:
            # Move labels to the device where prediction_scores tensor resides for model parallelism
            labels = labels.to(prediction_scores.device)
            # Define the loss function as Cross Entropy Loss
            loss_fct = CrossEntropyLoss()
            # Compute the masked LM loss based on prediction scores and labels
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # If return_dict is False, prepare the output tuple with prediction scores and additional outputs
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # If return_dict is True, construct a MaskedLMOutput object with specific attributes
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    CamemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 基于 transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification 复制修改，将所有 Roberta 替换为 Camembert，所有 ROBERTA 替换为 CAMEMBERT
class CamembertForSequenceClassification(CamembertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数目
        self.config = config  # 存储配置信息

        self.roberta = CamembertModel(config, add_pooling_layer=False)  # 初始化 Camembert 模型，不添加汇聚层
        self.classifier = CamembertClassificationHead(config)  # 初始化 Camembert 分类头部

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用指定的值；否则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoBERTa 模型进行前向传播
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
        # 获取 RoBERTa 输出的序列输出
        sequence_output = outputs[0]
        # 经过分类器得到 logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            # 确定问题类型，根据 num_labels 和 labels 的数据类型进行分类
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数
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

        # 如果不需要返回字典，则返回模型的输出和损失
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则构造 SequenceClassifierOutput 对象并返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述了该类是基于CamemBERT模型的多选分类器，适用于例如RocStories/SWAG任务。
@add_start_docstrings(
    """
    CamemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice中复制的代码，将Roberta替换为Camembert，ROBERTA替换为CAMEMBERT
class CamembertForMultipleChoice(CamembertPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化对象
        super().__init__(config)

        # 初始化Camembert模型
        self.roberta = CamembertModel(config)
        # 使用config中定义的hidden_dropout_prob初始化一个Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层用于多选分类，输入维度为config中定义的hidden_size，输出维度为1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法，接收多个输入参数并返回一个包含输出的字典或者一个元组
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
        # 描述输入参数的文档字符串，指定了输入的形状和含义
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 `return_dict` 参数确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入 `input_ids` 的第二维大小作为选项数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果 `input_ids` 不为空，则展平为二维张量
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果 `position_ids` 不为空，则展平为二维张量
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果 `token_type_ids` 不为空，则展平为二维张量
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果 `attention_mask` 不为空，则展平为二维张量
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果 `inputs_embeds` 不为空，则展平为三维张量
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 RoBERTa 模型进行前向传播
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
        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 对池化后的输出应用分类器得到 logits
        logits = self.classifier(pooled_output)
        # 重塑 logits 的形状为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签 `labels`
        if labels is not None:
            # 将标签移动到正确的设备以支持模型并行计算
            labels = labels.to(reshaped_logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(reshaped_logits, labels)

        # 如果不使用返回字典，则返回扁平化后的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用返回字典形式输出结果
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
CamemBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
for Named-Entity-Recognition (NER) tasks.
"""

# 从transformers.models.roberta.modeling_roberta.RobertaForTokenClassification复制，将Roberta替换为Camembert，ROBERTA替换为CAMEMBERT
@add_start_docstrings(
    """
    CamemBERT模型，顶部带有一个标记分类头（在隐藏状态输出的顶部增加了一个线性层），例如用于命名实体识别（NER）任务。
    """,
    CAMEMBERT_START_DOCSTRING,
)
class CamembertForTokenClassification(CamembertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化Camembert模型，不包括池化层
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        
        # 分类器的dropout率，如果未指定，则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 线性分类器，将隐藏状态的输出映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
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
        CamemBERT模型的前向传播方法。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入的token索引序列. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): 注意力遮罩，指示哪些元素是填充值而不是实际数据. Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): token类型ids，用于区分不同的句子. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): 位置ids，指示每个token在输入中的位置. Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): 头部遮罩，用于指定哪些注意力头部被屏蔽. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): 嵌入的输入，而不是使用input_ids. Defaults to None.
            labels (Optional[torch.LongTensor], optional): 标签，用于训练时的监督. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出所有注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出所有隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典格式的输出. Defaults to None.

        Returns:
            TokenClassifierOutput or Tuple[torch.FloatTensor]: 模型的输出结果或元组，根据return_dict参数决定输出形式.
        """
        # 实现CamemBERT模型的前向传播逻辑，详细解释见上文
        pass  # forward方法的具体实现在实际代码中，这里暂时不作展示
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 不为 None，则使用传入的 return_dict，否则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 模型处理输入数据
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

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        
        # 使用分类器对处理后的序列输出进行分类得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            # 将标签移到与 logits 相同的设备上，以支持模型并行计算
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用 return_dict，按顺序返回 logits 和额外的模型输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 类构建返回结果，包括损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    CamemBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`
    """,
    CAMEMBERT_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering 复制而来，将所有 Roberta 替换为 Camembert，所有 ROBERTA 替换为 CAMEMBERT
class CamembertForQuestionAnswering(CamembertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 Camembert 模型，禁用 pooling 层
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 线性层，用于输出分类 logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
    )
    # 定义前向传播方法，接受多种输入参数并返回结果
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 模型进行前向传播
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

        # 对序列输出进行问答任务的输出
        logits = self.qa_outputs(sequence_output)
        # 将输出分割为开始位置和结束位置的 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # 去除维度为 1 的维度，并保证连续性
        end_logits = end_logits.squeeze(-1).contiguous()  # 去除维度为 1 的维度，并保证连续性

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果输入的 start_positions 或 end_positions 是多维的，在 GPU 上处理时需要进行调整
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将超出模型输入长度的位置索引设置为忽略索引
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略忽略索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # 计算开始位置和结束位置的损失
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总体损失
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典，则返回一个元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 对象，包含损失、开始位置 logits、结束位置 logits、隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用自定义的装饰器添加模型文档字符串，说明这是一个带有语言建模头部的CamemBERT模型，用于条件语言建模（CLM）微调
@add_start_docstrings(
    """CamemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", CAMEMBERT_START_DOCSTRING
)
# 从transformers.models.roberta.modeling_roberta.RobertaForCausalLM复制并修改为CamembertForCausalLM，替换了相关引用和模型名称
# 将FacebookAI/roberta-base替换为almanach/camembert-base
class CamembertForCausalLM(CamembertPreTrainedModel):
    # 指定权重共享的键列表，这些键将与lm_head.decoder的权重和偏置相关联
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 如果配置不是解码器，则发出警告，建议添加"is_decoder=True"以独立使用CamembertLMHeadModel
        if not config.is_decoder:
            logger.warning("If you want to use `CamembertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化Camembert模型部分，不包括池化层
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        # 初始化Camembert语言建模头部
        self.lm_head = CamembertLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输出嵌入层的方法，返回lm_head.decoder，即语言建模头部的解码器
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入层的方法，更新lm_head.decoder的值为新的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 重写forward方法，根据参数文档说明进行详细的输入和输出注释
    @add_start_docstrings_to_model_forward(CAMEMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成准备输入数据，根据给定参数设置输入形状
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入数据的形状
        input_shape = input_ids.shape

        # 如果未提供注意力掩码，则创建一个全为1的掩码，长度与输入相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果已提供过去的键值（用于缓存），则根据过去键值裁剪输入的ID序列
        if past_key_values is not None:
            # 获取过去键值的长度（通常是序列长度）
            past_length = past_key_values[0][0].shape[2]

            # 如果输入ID序列长度大于过去键值长度，裁剪序列，保留后面部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 裁剪输入ID序列
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回准备好的输入参数字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的过去键值，根据给定的beam索引
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排序后的过去键值元组
        reordered_past = ()

        # 遍历每一层的过去键值
        for layer_past in past_key_values:
            # 对每个过去状态，根据beam索引重新排序，并加入元组中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )

        # 返回重新排序后的过去键值
        return reordered_past
# 从输入的input_ids中创建位置标识符，用于Transformer模型的位置编码
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: 输入的整数张量，包含了模型的输入内容
        padding_idx: 表示填充的索引，用于识别填充符号
        past_key_values_length: 过去键值的长度，用于增量索引计算

    Returns:
        torch.Tensor: 包含了每个位置的标识符的长整型张量
    """
    # 创建一个掩码张量，将非填充符号的位置标记为1，填充符号标记为0
    mask = input_ids.ne(padding_idx).int()
    # 计算每个位置的增量索引，忽略填充符号
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将增量索引转换为长整型，并加上填充索引，以获得最终的位置标识符
    return incremental_indices.long() + padding_idx
```