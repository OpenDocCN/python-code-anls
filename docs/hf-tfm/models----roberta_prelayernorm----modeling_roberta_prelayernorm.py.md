# `.\transformers\models\roberta_prelayernorm\modeling_roberta_prelayernorm.py`

```
# 这是一个 PyTorch 实现的 RoBERTa-PreLayerNorm 模型的代码
# 这个模型是基于 Google AI Language Team 和 HuggingFace Inc. 团队的工作开发的
# 它遵循 Apache 2.0 许可协议

# 引入必要的库和模块
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
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig


# 获取日志器
logger = logging.get_logger(__name__)

# 模型的一些常量
_CHECKPOINT_FOR_DOC = "andreasmadsen/efficient_mlm_m0.40"
_CONFIG_FOR_DOC = "RobertaPreLayerNormConfig"

# RoBERTa-PreLayerNorm 预训练模型的列表
ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "andreasmadsen/efficient_mlm_m0.15",
    "andreasmadsen/efficient_mlm_m0.20",
    "andreasmadsen/efficient_mlm_m0.30",
    "andreasmadsen/efficient_mlm_m0.40",
    "andreasmadsen/efficient_mlm_m0.50",
    "andreasmadsen/efficient_mlm_m0.60",
    "andreasmadsen/efficient_mlm_m0.70",
    "andreasmadsen/efficient_mlm_m0.80",
    # See all RoBERTaWithPreLayerNorm models at https://huggingface.co/models?filter=roberta_with_prelayernorm
]

# 定义 RobertaPreLayerNormEmbeddings 类，它继承自 nn.Module
# 这个类实现了 RoBERTa 模型的词嵌入层
class RobertaPreLayerNormEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 复制自 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 的 __init__ 方法
    # 这个方法初始化了词嵌入层的各个组成部分
    # 初始化函数，用于初始化模型参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建词嵌入层，将词汇索引映射为隐藏状态向量，其中包括词汇表大小、隐藏状态大小、填充索引等配置
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，用于表示输入序列中每个位置的位置编码
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，用于区分不同类型的输入标记，例如句子A和句子B
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 为了与 TensorFlow 模型变量名保持一致，并能够加载任何 TensorFlow 检查点文件，self.LayerNorm 没有使用蛇形命名
        # 创建层归一化层，用于归一化输入的隐藏状态向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，用于随机屏蔽部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 在内存中是连续的，并在序列化时导出
        # 设置位置编码类型，绝对位置编码或相对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置 ID 缓冲区，用于存储位置编码
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册标记类型 ID 缓冲区，用于存储标记类型编码
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # self.padding_idx 是用于填充的索引
        # 重新创建位置嵌入层，用于表示输入序列中每个位置的位置编码，指定填充索引
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数，用于计算模型的输出
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果没有提供位置 ID，则根据输入的 token ID 创建位置 ID。任何填充的 token 仍然保持填充状态
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果提供了输入 token ID，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中全是零，这通常在自动生成时发生，
        # 注册的缓冲区有助于用户在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入的嵌入，则使用词嵌入层获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 融合输入嵌入和 token 类型嵌入
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为“absolute”，则获取位置嵌入并添加到 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 运行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 应用 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings

    # 从输入的嵌入中创建位置 ID
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序的位置 ID
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制代码，将Bert->RobertaPreLayerNorm
class RobertaPreLayerNormSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果hidden_size不能被num_attention_heads整除，或者没有embedding_size属性，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 设置dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果position_embedding_type为"relative_key"或"relative_key_query"，则设置相对位置编码的最大位置嵌入数量和距离编码
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder

    # 重新排列张量形状以便计算得分
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



# RobertaPreLayerNormSelfOutput类定义
class RobertaPreLayerNormSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过密集连接层传递隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用dropout
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的结果与输入张量相加
        hidden_states = hidden_states + input_tensor
        return hidden_states
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建一个RobertaPreLayerNormSelfAttention对象，并传递相关配置参数和位置嵌入类型
        self.self = RobertaPreLayerNormSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建一个RobertaPreLayerNormSelfOutput对象，并传递相关配置参数
        self.output = RobertaPreLayerNormSelfOutput(config)
        # 创建一个LayerNorm对象，参数为隐藏层的大小和层归一化的epsilon值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个空集合，用于存储需要剪枝的头
        self.pruned_heads = set()

    # 根据给定的头索引列表，剪枝注意力机制
    def prune_heads(self, heads):
        # 如果头索引列表为空，则直接返回
        if len(heads) == 0:
            return
        # 寻找可剪枝的头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝的头
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
        # 对隐藏状态进行层归一化
        hidden_states_pre_layer_norm = self.LayerNorm(hidden_states)
        # 调用self.self的前向传播函数，得到self_outputs
        self_outputs = self.self(
            hidden_states_pre_layer_norm,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将self_outputs与hidden_states作为输入，调用self.output的前向传播函数，得到attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构造输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
# 定义一个类，该类实现了Roberta模型中的PreLayerNorm中间层
class RobertaPreLayerNormIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # LayerNorm层，用于对输入进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 全连接层，将输入的隐藏状态映射到中间层的维度上
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断配置中的激活函数类型，根据类型选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入进行LayerNorm归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        # 将归一化后的输入通过全连接层得到中间层的输出
        hidden_states = self.dense(hidden_states)
        # 使用中间层的激活函数对输出进行激活处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义一个类，该类实现了Roberta模型中的PreLayerNorm输出层
class RobertaPreLayerNormOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将中间层的输出映射回隐藏层的维度上
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将中间层的输出通过全连接层得到最终层的输出
        hidden_states = self.dense(hidden_states)
        # 对最终层的输出进行dropout处理
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的输出与输入进行相加
        hidden_states = hidden_states + input_tensor
        return hidden_states


# 从transformers.models.bert.modeling_bert中复制过来的代码，将BertLayer改为RobertaPreLayerNorm
# 定义一个类，该类实现了Roberta模型中的PreLayerNorm层
class RobertaPreLayerNormLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 前馈网络的批处理尺寸
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # SelfAttention层
        self.attention = RobertaPreLayerNormAttention(config)
        # 判断是否是解码器
        self.is_decoder = config.is_decoder
        # 判断是否添加跨层Attention
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果添加跨层Attention，则要求必须是解码器模型
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 添加跨层Attention的SelfAttention层
            self.crossattention = RobertaPreLayerNormAttention(config, position_embedding_type="absolute")
        # PreLayerNorm中间层
        self.intermediate = RobertaPreLayerNormIntermediate(config)
        # PreLayerNorm输出层
        self.output = RobertaPreLayerNormOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    # 这个函数是 Transformer 模型的一部分，它负责进行多头注意力机制和前馈神经网络的计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        past_key_value: Tuple[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        # 如果存在过去的注意力机制 key 和 value，则从中提取出 self-attention 的 key 和 value
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 计算 self-attention 的输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 将 self-attention 的输出赋给 attention_output
        attention_output = self_attention_outputs[0]
    
        # 如果是解码器模型，则将 self-attention 的结果和 past_key_value 分别存放在不同的变量中
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器模型，则只需要返回 self-attention 的结果
            outputs = self_attention_outputs[1:]
    
        # 如果是解码器模型且提供了编码器隐藏状态，则计算跨注意力的输出
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            # 从 past_key_value 中提取出跨注意力的 key 和 value
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
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
            outputs = outputs + cross_attention_outputs[1:-1]
            # 将跨注意力的 key 和 value 添加到 present_key_value 中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value
    
        # 应用前馈网络
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
    
        # 如果是解码器模型，则将 present_key_value 作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
    
        return outputs
    
    # 这个函数是 Transformer 模型的前馈神经网络部分
    def feed_forward_chunk(self, attention_output):
        # 计算中间输出
        intermediate_output = self.intermediate(attention_output)
        # 计算最终输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义了一个名为 RobertaPreLayerNormEncoder 的类，继承自 nn.Module，用于实现 RoBERTa 的预层归一化编码器
class RobertaPreLayerNormEncoder(nn.Module):
    # 类构造器方法，初始化一个新的实例
    def __init__(self, config):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 将传入的配置对象保存在实例变量中
        self.config = config
        # 创建一个模块列表，包含多个 RobertaPreLayerNormLayer，数量由配置中的 num_hidden_layers 决定
        self.layer = nn.ModuleList([RobertaPreLayerNormLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点变量为 False
        self.gradient_checkpointing = False

    # 定义前向传播函数，用于在网络中传递数据和计算
    def forward(
        # 第一个参数是隐藏状态，是一个张量
        hidden_states: torch.Tensor,
        # 可选参数，注意力掩码，用浮点张量表示
        attention_mask: Optional[torch.FloatTensor] = None,
        # 可选参数，头部掩码，也是一个浮点张量
        head_mask: Optional[torch.FloatTensor] = None,
        # 可选参数，编码器的隐藏状态，用于可能的跨层连接
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 可选参数，编码器的注意力掩码
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 可选参数，包含先前的键值对的元组，用于注意力机制的缓存
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 可选参数，指示是否使用缓存来保存某些中间结果以加速后续的计算
        use_cache: Optional[bool] = None,
        # 可选参数，指示是否输出注意力权重
        output_attentions: Optional[bool] = False,
        # 可选参数，指示是否输出隐藏状态
        output_hidden_states: Optional[bool] = False,
        # 可选参数，指示是否以字典形式返回输出
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
            # 如果需要输出隐藏状态，则初始化一个空元组
            all_hidden_states = () if output_hidden_states else None
            # 如果需要输出自我注意力分数，则初始化一个空元组
            all_self_attentions = () if output_attentions else None
            # 如果需要输出交叉注意力分数，并且模型配置允许交叉注意力，则初始化一个空元组
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    
            # 如果启用了梯度检查点且处于训练模式，并且设置了`use_cache=True`，则警告并设置`use_cache=False`
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
    
            # 如果`use_cache=True`，则初始化一个空元组作为下一个解码器缓存（用于存储前向传播的中间结果）
            next_decoder_cache = () if use_cache else None
            # 遍历每个解码器层
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
    
                # 获取当前层的注意力掩码（如果有的话）
                layer_head_mask = head_mask[i] if head_mask is not None else None
                # 获取当前层的过去键值（如果有的话）
                past_key_value = past_key_values[i] if past_key_values is not None else None
    
                # 如果启用了梯度检查点且处于训练模式，则使用梯度检查点函数进行前向传播
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
                    # 否则，调用当前层的前向传播函数进行前向传播
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )
    
                # 更新隐藏状态为当前层的前向传播���出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果`use_cache=True`，则将当前层前向传播输出的最后一个元素添加到下一个解码器缓存中
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                # 如果需要输出自我注意力分数，则将当前层前向传播输出的第二个元素添加到所有自我注意力分数中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    # 如果模型配置允许交叉注意力，则将当前层前向传播输出的第三个元素添加到所有交叉注意力分数中
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 如果`return_dict=False`，则以元组的形式返回结果
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
            # 否则，以`BaseModelOutputWithPastAndCrossAttentions`的形式返回结果
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
# 从 transformers.models.bert.modeling_bert.BertPooler 复制过来的类，用于 RoBERTa 模型的预处理层归一化池化
class RobertaPreLayerNormPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为 Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地选择第一个 token 对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 对应的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数进行激活
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 从 transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel 复制过来的类，用于 RoBERTa 预处理层归一化预训练模型
class RobertaPreLayerNormPreTrainedModel(PreTrainedModel):
    """
    一个用于处理权重初始化以及下载和加载预训练模型的简单接口的抽象类。
    """

    # 指定配置类
    config_class = RobertaPreLayerNormConfig
    # 模型参数前缀
    base_model_prefix = "roberta_prelayernorm"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["RobertaPreLayerNormEmbeddings", "RobertaPreLayerNormSelfAttention"]

    # 从 transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights 复制过来的方法，用于初始化权重
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 对于线性层，使用均值为 0，标准差为 self.config.initializer_range 的正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置，将偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用均值为 0，标准差为 self.config.initializer_range 的正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，将填充索引位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，将偏置初始化为零，权重初始化为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ROBERTA_PRELAYERNORM_START_DOCSTRING = r"""

    此模型继承自 [`PreTrainedModel`]。查看超类文档以获取库实现的所有模型的通用方法（例如下载或保存、调整输入嵌入、修剪头等）。

    此模型还是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。
    将其用作常规 PyTorch 模块，并参考 PyTorch 文档以获取与一般用法和行为相关的所有事项。
    Parameters:
        # 接受一个名为 config 的参数，该参数是一个包含模型所有参数的配置类
        # 通过使用配置文件初始化模型，不会加载模型关联的权重，只加载配置。
        # 可以使用 `~PreTrainedModel.from_pretrained` 方法加载模型权重。
        config ([`RobertaPreLayerNormConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING 定义了 RoBERTa-PreLayerNorm 模型的输入参数文档字符串
ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = r"""
    Args:
        # input_ids 是输入序列标记在词汇表中的索引，类型为 torch.LongTensor，形状为 ({0})
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            
            [What are input IDs?](../glossary#input-ids)
        # attention_mask 是一个避免在填充标记索引上执行注意力的掩码，类型为 torch.FloatTensor，形状为 ({0})，可选参数
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            
            [What are attention masks?](../glossary#attention-mask)
        # token_type_ids 是段标记索引，指示输入的第一部分和第二部分，类型为 torch.LongTensor，形状为 ({0})，可选参数
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
            
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.
            
            [What are token type IDs?](../glossary#token-type-ids)
        # position_ids 是输入序列标记在位置嵌入中的位置索引，类型为 torch.LongTensor，形状为 ({0})，可选参数
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            
            [What are position IDs?](../glossary#position-ids)
        # head_mask 是一个用于将自注意力模块的选定头部置零的掩码，类型为 torch.FloatTensor，形状为 `(num_heads,)` 或 `(num_layers, num_heads)`，可选参数
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            
        # inputs_embeds 是一个直接传递嵌入表示的选项，而不是传递 input_ids，类型为 torch.FloatTensor，形状为 `({0}, hidden_size)`，可选参数
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        # output_attentions 是一个布尔值，表示是否返回所有注意力层的注意力张量，默认为 False，可选参数
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # output_hidden_states 是一个布尔值，表示是否返回所有层的隐藏状态，默认为 False，可选参数
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # return_dict 是一个布尔值，表示是否返回 [`~utils.ModelOutput`] 而不是普通元组，默认为 False，可选参数
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    ROBERTA_PRELAYERNORM_START_DOCSTRING,



    # ROBERTA_PRELAYERNORM_START_DOCSTRING 是一个预定义的常量，用于指示 RoBERTa 模型的文档字符串的起始位置
# 定义一个自定义的 RoBERTa 预层归一化模型，继承自 RoBERTa 预层归一化预训练模型
class RobertaPreLayerNormModel(RobertaPreLayerNormPreTrainedModel):

    """
    该模型既可以作为一个编码器（只有自注意力），也可以作为一个解码器，此时在自注意力层之间添加一个交叉注意力层，
    遵循 *Attention is all you need* 中描述的架构，作者为 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin。

    要作为解码器，模型需要使用设置为 `True` 的 `is_decoder` 参数来初始化配置。要在 Seq2Seq 模型中使用，模型需要同时
    使用 `is_decoder` 参数和 `add_cross_attention` 设置为 `True` 来初始化；然后期望将 `encoder_hidden_states`
    作为前向传递的输入。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的构造函数
        super().__init__(config)
        # 保存配置
        self.config = config

        # 初始化 RoBERTa 预层归一化 embeddings
        self.embeddings = RobertaPreLayerNormEmbeddings(config)
        # 初始化 RoBERTa 预层归一化 encoder
        self.encoder = RobertaPreLayerNormEncoder(config)
        # 初始化层归一化层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果 add_pooling_layer 为 True，则初始化 RoBERTa 预层归一化 pooler
        self.pooler = RobertaPreLayerNormPooler(config) if add_pooling_layer else None

        # 初始化模型权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数
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
    """RoBERTa-PreLayerNorm Model with a `language modeling` head on top for CLM fine-tuning.""",
    # RoBERTa-PreLayerNorm 模型，顶部带有语言建模头部，用于 CLM 微调。
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
# 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM 复制而来，进行了一系列修改
# 修改包括将“roberta-base”更改为“andreasmadsen/efficient_mlm_m0.40”，将“ROBERTA”更改为“ROBERTA_PRELAYERNORM”，将“Roberta”更改为“RobertaPreLayerNorm”，将“roberta”更改为“roberta_prelayernorm”，将“RobertaPreLayerNormTokenizer”更改为“RobertaTokenizer”
# 这个类是用于预测生成下一个单词的 RoBERTa 模型，其中包含了预先的 LayerNorm 处理
class RobertaPreLayerNormForCausalLM(RobertaPreLayerNormPreTrainedModel):
    # 这是一个 tied weights 的关键字列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 如果不是 decoder 模型，会发出警告
        if not config.is_decoder:
            logger.warning(
                "If you want to use `RobertaPreLayerNormLMHeadModel` as a standalone, add `is_decoder=True.`"
            )

        # 初始化 RoBERTaPreLayerNormModel 和 RoBERTaPreLayerNormLMHead
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        self.lm_head = RobertaPreLayerNormLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，接收多个输入参数
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    # 准备生成的输入数据，处理输入数据的形状和注意力掩码
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入数据的形状
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果存在过去的键值对，则对输入数据进行裁剪
        if past_key_values is not None:
            # 获取过去的键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入数据的长度大于过去键值对的长度，则需要移除前缀内容
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 对输入数据进行裁剪
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回处理后的输入数据
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的键值对
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排序后的过去键值对
        reordered_past = ()
        # 遍历每个层的过去键值对
        for layer_past in past_key_values:
            # 对每个过去键值对的状态进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 使用 add_start_docstrings 添加模型描述文档
@class装饰器，用于添加模型描述文档
class RobertaPreLayerNormForMaskedLM(RobertaPreLayerNormPreTrainedModel):
    # 定义权重绑定的关键字
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 从原始代码中复制的初始化函数，用于初始化 RoBERTa-PreLayerNorm Model
    def __init__(self, config):
        调用父类的初始化函数
        super().__init__(config)
        如果配置要求是decoder，则产生警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaPreLayerNormForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        
        # 初始化 RoBERTa-PreLayerNorm 模型和 lm_head
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        self.lm_head = RobertaPreLayerNormLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 使用 add_start_docstrings_to_model_forward 添加模型输入描述和代码示例描述
    # 从原始代码中复制的前向传播函数，用于 RoBERTa-PreLayerNorm 模型的前向传播
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
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 确定是否返回字典格式的输出，若未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoBERTa 的预处理层和正则化层，生成模型的输出
        outputs = self.roberta_prelayernorm(
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
        # 获取 RoBERTa 模型输出的序列特征
        sequence_output = outputs[0]
        # 通过语言模型头部预测下一个词的概率分布
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果提供了标签，则计算掩码语言建模损失
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            # 计算掩码语言建模的交叉熵损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典格式的输出
        if not return_dict:
            # 将输出装配成元组形式
            output = (prediction_scores,) + outputs[2:]
            # 如果存在掩码语言建模损失，则将其加入输出中
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回掩码语言建模的输出，包括损失、预测的logits、隐藏状态以及注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaLMHead复制并将Roberta->RobertaPreLayerNorm
class RobertaPreLayerNormLMHead(nn.Module):
    """用于掩盖语言建模的RobertaPreLayerNorm头部。"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 初始化一个线性层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化LayerNorm层

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)  # 初始化一个线性层
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个代表偏置的参数
        self.decoder.bias = self.bias  # 将偏置参数赋值给解码器的偏置

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 使用线性层对特征进行转换
        x = gelu(x)  # 使用gelu激活函数
        x = self.layer_norm(x)  # 对特征进行LayerNorm处理

        # 将特征映射回词汇表大小并加上偏置
        x = self.decoder(x)

        return x  # 返回结果

    def _tie_weights(self):
        # 如果这两个权重被断开连接（在TPU上或当偏置被重新调整时），则将它们连接起来
        # 用于加速兼容性和不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":  # 检查偏置的设备类型
            self.decoder.bias = self.bias  # 如果设备类型为"meta"，则将解码器的偏置赋值为当前层的偏置
        else:
            self.bias = self.decoder.bias  # 否则将当前层的偏置赋值为解码器的偏置


@add_start_docstrings(
    """
    在顶部添加了一个序列分类/回归头部的RoBERTa-PreLayerNorm模型变压器（在池化输出之上添加了一个线性层）例如，用于GLUE任务。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class RobertaPreLayerNormForSequenceClassification(RobertaPreLayerNormPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 类别数量
        self.config = config  # 配置文件

        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)  # 初始化RobertaPreLayerNormModel模型，不添加池化层
        self.classifier = RobertaPreLayerNormClassificationHead(config)  # 分类器头部

        # 初始化权重并进行最终处理
        self.post_init()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        Forward pass of the model.
    
        Args:
            input_ids (torch.LongTensor of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.Tensor of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding tokens.
            token_type_ids (torch.LongTensor of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs.
            position_ids (torch.LongTensor of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
            head_mask (torch.Tensor of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.Tensor of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            output_attentions (bool, *optional*):
                Whether to also return the attentions tensors of all attention layers.
            output_hidden_states (bool, *optional*):
                Whether to also return the hidden states of all layers.
            labels (torch.LongTensor of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`.
                If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` 
                a classification loss is computed (Cross-Entropy).
            return_dict (bool, *optional*):
                Whether to return outputs as a :class:`~transformers.file_utils.ModelOutput` class with all outputs.
    
        Returns:
            :class:`~transformers.file_utils.SequenceClassifierOutput`: if ``return_dict=True``
                A sequence classifier output consisting of:
                - loss (:obj:`torch.FloatTensor` of shape `(1,)`, *optional*):
                    Classification or regression loss.
                - logits (:obj:`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
                    Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
                - hidden_states (:obj:`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                    Hidden states of the model at the output of each layer plus 
                    the initial embedding outputs.
                - attentions (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`)
                    Attentions weights after the attention softmax, used to calculate the weighted average 
                    in the self-attention heads.
    
                When `return_dict=True`, fields include additional fields.
    
        """
        # If the `return_dict` argument is not specified, set it to the value of `use_return_dict` in the model's configuration.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Forward pass through the Roberta model.
        outputs = self.roberta_prelayernorm(
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
    
        # Get the sequence output from the Roberta model.
        sequence_output = outputs[0]
    
        # Pass the sequence output through the classification layer to get logits.
        logits = self.classifier(sequence_output)
    
        # Initialize the loss.
        loss = None
    
        # If labels are provided, compute the loss.
        if labels is not None:
            # Move labels to the correct device to enable model parallelism.
            labels = labels.to(logits.device)
    
            # Determine the problem type if it was not specified in the model's configuration.
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
    
            # Calculate the loss based on the problem type.
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
    
        # If `return_dict` is False, return the output as a tuple.
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # If `return_dict` is True, return the output as a SequenceClassifierOutput object.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 引入所需的模块或类
@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a multiple choice classification head on top (a linear layer on top of the pooled
    output and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice复制过来的类，修改部分名称以匹配当前类名和模块名
class RobertaPreLayerNormForMultipleChoice(RobertaPreLayerNormPreTrainedModel):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建RobertaPreLayerNormModel模型对象
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config)
        # 创建一个Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，用于分类任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数，返回结果
    @add_start_docstrings_to_model_forward(
        ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否要返回字典格式的输出，若未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选项数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 对输入进行展平，方便处理
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 通过预处理层获取输出
        outputs = self.roberta_prelayernorm(
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

        # 对池化输出进行dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器获取logits
        logits = self.classifier(pooled_output)
        # 重塑logits的形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 将标签移到正确的设备上以启用模型并行化
            labels = labels.to(reshaped_logits.device)
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            # 如果不需要返回字典格式的结果，则返回reshaped_logits及其它输出
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
``` 
# 为 RoBERTa 预层标准化模型添加一个标记分类头（在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    RobertaPreLayerNorm 模型，其顶部带有一个标记分类头（在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class RobertaPreLayerNormForTokenClassification(RobertaPreLayerNormPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签的数量
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 预层标准化模型，不添加池化层
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        # 如果分类器的丢弃率不为 None，则使用该值，否则使用隐藏层的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 添加线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加输入文档字符串
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.forward 复制而来，将 roberta->roberta_prelayernorm
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        定义函数的输入和输出类型注释。函数接受input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict作为输入，返回一个Tuple[torch.Tensor]或TokenClassifierOutput对象。
        inputs中的labels是一个torch.LongTensor类型的二维张量，形状为(batch_size, sequence_length)，可选参数。用于计算标记分类损失，索引应在[0, ..., config.num_labels - 1]之间。
        """
        设置return_dict的默认值，如果调用函数时提供了return_dict参数，则使用提供的值，否则使用self.config.use_return_dict的值
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        调用self.roberta_prelayernorm方法，传入input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict作为参数，并将返回值保存到outputs变量中
        """
        outputs = self.roberta_prelayernorm(
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
        获取outputs变量中的第一个元素，并将其保存到sequence_output变量中
        """
        sequence_output = outputs[0]
        对sequence_output进行dropout操作
        """
        sequence_output = self.dropout(sequence_output)
        将sequence_output传入self.classifier，得到logits
        """
        logits = self.classifier(sequence_output)
        初始化loss变量为空
        """
        loss = None
        labels不为空时
        """
        if labels is not None:
            # 将labels移动到正确的设备上以启用模型并行计算
            labels = labels.to(logits.device)
            初始化损失函数为交叉熵损失函数
            """
            loss_fct = CrossEntropyLoss()
            计算损失，并将其保存到loss变量中
            """
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return_dict为False时
        """
        if not return_dict:
            output变量为logits和outputs中第2个元素开始的所有元素组成的元组
            """
            output = (logits,) + outputs[2:]
            如果loss不为空，则将loss添加到output的开头，并返回output；否则，直接返回output
            """
            return ((loss,) + output) if loss is not None else output
        return_dict为True时
        """
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制而来，将Roberta->RobertaPreLayerNorm
class RobertaPreLayerNormClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # 取<s> token (相当于[CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    在RobertaPreLayerNorm模型顶部添加了一个跨度分类头，用于提取式问答任务，如SQuAD
    （在隐藏状态输出上方添加线性层，以计算`span start logits`和`span end logits`）。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class RobertaPreLayerNormForQuestionAnswering(RobertaPreLayerNormPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.forward复制而来，将roberta->roberta_prelayernorm
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
        # 确保返回字典的设置正确
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 将输入传递给 RoBERTa 模型的前处理部分
        outputs = self.roberta_prelayernorm(
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

        # 获取 RoBERTa 模型的输出序列
        sequence_output = outputs[0]

        # 将序列输出传递给问答模型的输出层以获得 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 按照最后一个维度拆分成起始位置和结束位置的 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 移除不必要的维度
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 计算总损失
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果是在多 GPU 上运行，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不返回字典，则返回各种输出项
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果返回字典，则返回 QuestionAnsweringModelOutput 类型的对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 根据输入的输入 ID 创建位置 ID
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x: 输入的张量

    Returns: torch.Tensor 返回张量
    """
    # 创建一个掩码，将非填充符号替换为它们的位置编号。位置编号从 padding_idx+1 开始。填充符号被忽略。
    mask = input_ids.ne(padding_idx).int()
    # 累加掩码，得到增量索引，乘以掩码，将填充符号的索引设为0，并加上过去键值长度
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将增量索引转换为长整型，并加上填充索引
    return incremental_indices.long() + padding_idx
```