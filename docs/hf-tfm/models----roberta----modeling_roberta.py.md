# `.\transformers\models\roberta\modeling_roberta.py`

```py
# 导入必要的库
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从内部库中导入一些函数和类
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

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预定义一些文档中要用到的变量
_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

# 预定义的 RoBERTa 模型列表
ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # 查看所有 RoBERTa 模型: https://huggingface.co/models?filter=roberta
]

# 定义 RoBERTaEmbeddings 类，用于处理 RoBERTa 模型的输入嵌入
class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 这是一个与 BertEmbeddings 类似的类，稍微调整了位置嵌入的索引方式
    # 但这里没有实现具体内容，只是作为一个占位符
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，词汇量大小为config.vocab_size，隐藏层大小为config.hidden_size
        # 使用padding_idx参数指定填充的token id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，最大位置嵌入长度为config.max_position_embeddings，隐藏层大小为config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建token类型嵌入层，token类型数目为config.type_vocab_size，隐藏层大小为config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm不采用蛇形命名以与TensorFlow模型变量名保持一致，并能够加载任何TensorFlow检查点文件
        # 创建LayerNorm层，输入大小为config.hidden_size，epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册position_ids缓冲，包含从0到config.max_position_embeddings-1的位置id
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册token_type_ids缓冲，大小与position_ids相同，全为0，用于标识token的类型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置padding_idx
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层，与之前的位置嵌入层重复了，似乎是多余的代码
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播方法
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ): 
        # 如果没有指定位置 ID，则根据输入的标记 ID 创建位置 ID。任何填充的标记仍然保持填充状态。
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果指定了输入标记 ID，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中所有值都为零，通常在自动生成时发生，
        # 注册的缓冲区可帮助用户在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果没有指定输入嵌入，则使用 word_embeddings 对输入标记 ID 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 使用 token_type_embeddings 获取 token 类型的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token 类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置嵌入类型为 "absolute"，则计算位置嵌入并加到 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对 embeddings 进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # 返回 embeddings
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序位置 ID
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制并修改为 RobertaSelfAttention 类
class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏大小不是注意力头数的倍数且配置没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对键或相对键查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量转置以匹配注意力分数的形状
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
    ):
        # 省略了前向传播函数的具体实现，由具体的使用场景决定
        pass

# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并修改为 RobertaSelfOutput 类
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm 层和 dropout 层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # dropout 层
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 层并将输入张量加到 hidden_states 中
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# 从 transformers.models.bert.modeling_bert.BertAttention 复制并修改为 RobertaAttention 类
# 定义一个名为RobertaAttention的类，继承自nn.Module类
class RobertaAttention(nn.Module):
    # 初始化函数，接受config和position_embedding_type两个参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个RobertaSelfAttention对象并赋值给self.self
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建一个RobertaSelfOutput对象并赋值给self.output
        self.output = RobertaSelfOutput(config)
        # 创建一个空集合并赋值给self.pruned_heads
        self.pruned_heads = set()

    # 定义一个用于修剪头部的方法
    def prune_heads(self, heads):
        # 如果需要修剪的头部数量为0，则直接返回
        if len(heads) == 0:
            return
        # 调用find_pruneable_heads_and_indices函数获取需要修剪的头部和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
    ) -> Tuple[torch.Tensor]:
        # 调用self.self的前向传播函数，并将结果存储在self_outputs中
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将self_outputs的第一个元素和hidden_states作为输入，调用self.output的前向传播函数，并将结果存储在attention_output中
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将attention_output和self_outputs[1:]存储在outputs中，如果需要输出attentions，则添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回outputs
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate中复制了该类
class RobertaIntermediate(nn.Module):
    # 初始化函数，接受config作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层并赋值给self.dense
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则选择对应的激活函数赋值给self.intermediate_act_fn，否则直接赋值为config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 定义前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用self.dense的前向传播函数，并将结果存储在hidden_states中
        hidden_states = self.dense(hidden_states)
        # 调用self.intermediate_act_fn将hidden_states作为输入，并将结果存储在hidden_states中
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回hidden_states
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput中复制了该类
class RobertaOutput(nn.Module):
    # 略
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为 config 中的 intermediate_size，输出大小为 config 中的 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对隐藏状态的维度进行归一化，epsilon 值为 config 中的 layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，对隐藏状态进行随机丢弃以防止过拟合，丢弃概率为 config 中的 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了该模型的前向计算过程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的隐藏状态与输入张量相加，并进行 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLayer 复制代码，将其中的 Bert 替换为 Roberta
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义前向传播过程中用到的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # 初始化自注意力层
        self.attention = RobertaAttention(config)
        # 是否是解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力层
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = RobertaIntermediate(config)
        # 初始化输出层
        self.output = RobertaOutput(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        head_mask: Optional[torch.FloatTensor] = None,  # 注意力头的掩码
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器的注意力掩码
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重
```  
    ) -> Tuple[torch.Tensor]:
        # 定义函数签名，指定输入和输出类型

        # 如果有历史的self-attention缓存键/值元组，则在位置1,2处
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行self-attention计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取self-attention的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是self-attn缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加self注意力

        cross_attn_present_key_value = None
        # 如果是解码器并且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 如果传入了`encoder_hidden_states`，则通过设置`config.add_cross_attention=True`来实例化带有交叉注意力层的`self`
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存键/值元组在past_key_value元组的位置3,4处
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力计算
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
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到present_key_value元组的位置3,4处
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对前向传播应用分块处理并返回输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，则将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 使用self-attention输出进行前向传播
        intermediate_output = self.intermediate(attention_output)
        # 使用中间输出和self-attention输出进行输出层处理
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert中复制的代码，并将Bert更改为Roberta
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 使用ModuleList创建包含config.num_hidden_layers个RobertaLayer对象的列表
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 定义前向传播方法
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
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化空元组
        all_self_attentions = () if output_attentions else None
        # 如果不输出跨层注意力权重，或者没有启用跨层注意力，则初始化空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    
        # 梯度检查点和训练模式下，若使用缓存则警告并设置 use_cache 为 False
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
    
        # 若不使用缓存，则初始化空元组用于下一个解码器缓存
        next_decoder_cache = () if use_cache else None
    
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则记录当前隐藏状态到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 获取当前解码器层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前解码器层的过去键值对（用于缓存）
            past_key_value = past_key_values[i] if past_key_values is not None else None
    
            # 如果开启梯度检查点且处于训练模式，则调用梯度检查点函数
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
                # 否则正常调用解码器层模块
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
    
            # 更新当前隐藏状态为解码器层模块的输出的第一个元素
            hidden_states = layer_outputs[0]
    
            # 如果使用缓存，则将解码器层模块的输出的最后一个元素添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
    
            # 如果输出注意力权重，则记录当前注意力权重到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中包含跨层注意力，则记录跨层注意力到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
        # 如果输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果不返回字典格式，则返回非空元组的元素
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
        # 否则返回 BaseModelOutputWithPastAndCrossAttentions 类型的对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从 BERT 模型中复制过来的池化层，用于 RoBERTa 模型
class RobertaPooler(nn.Module):
    def __init__(self, config):
        # 继承父类的初始化方法
        super().__init__()
        # 线性层，用于池化
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 简单地通过取第一个标记对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# RoBERTa 预训练模型类，用于处理权重初始化、下载和加载预训练模型
class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # RoBERTa 模型的配置类
    config_class = RobertaConfig
    # 模型的前缀名
    base_model_prefix = "roberta"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不进行参数拆分的模块列表
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention"]

    # 从 BERT 模型中复制过来的方法，用于初始化权重
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 初始化线性层权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    # 输入序列的索引值，表示该序列中每个单词在词汇表中的索引
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
    
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            [What are input IDs?](../glossary#input-ids)
    # 注意力掩码，用于避免对填充令牌进行注意力计算
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            [What are attention masks?](../glossary#attention-mask)
    # 令牌类型ID，用于指示输入的第一部分和第二部分
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:
    
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.
    
            [What are token type IDs?](../glossary#token-type-ids)
    # 位置ID，表示每个输入序列中每个令牌的位置
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
    
            [What are position IDs?](../glossary#position-ids)
    # 注意力头掩码，用于屏蔽部分注意力头
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
    
    # 输入嵌入，可以直接传递嵌入向量而不是输入ID
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
    # 是否输出所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
    # 是否输出所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
    # 是否返回 ModelOutput 对象
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 继承自 RoBERTa 预训练模型基类的 RoBERTa 模型类，用于输出没有特定头部的原始隐藏状态
class RobertaModel(RobertaPreTrainedModel):
    """

    该模型可以作为编码器（仅具有自注意力）或解码器，此时在自注意力层之间添加了交叉注意力层，遵循 *Attention is all you need*_ 中描述的架构
    由 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin。

    要使模型行为类似解码器，需要使用配置中设置 `is_decoder` 参数为 `True` 进行初始化。要在 Seq2Seq 模型中使用该模型，需要同时进行初始化，
    并设置 `is_decoder` 参数和 `add_cross_attention` 参数为 `True`；然后期望输入前向传播的 `encoder_hidden_states`。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制而来，Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 RoBERTa 的嵌入层和编码器
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        # 如果需要添加池化层，则初始化 RoBERTa 的池化层
        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        heads_to_prune: 要剪枝的模型头部的字典 {层编号: 要在此层中剪枝的头部列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数的模型文档说明和代码示例文档说明
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制而来
    # 定义 forward 方法，用于模型的前向传播
    def forward(
        self,
        # 输入 ID 序列，表示输入文本
        input_ids: Optional[torch.Tensor] = None,
        # 输入文本的注意力掩码，用于区分输入和填充部分
        attention_mask: Optional[torch.Tensor] = None,
        # 输入文本的类型 ID，用于区分不同的输入段落
        token_type_ids: Optional[torch.Tensor] = None,
        # 输入文本的位置 ID，用于区分不同位置的输入
        position_ids: Optional[torch.Tensor] = None,
        # 注意力层的掩码，用于控制注意力计算
        head_mask: Optional[torch.Tensor] = None,
        # 输入文本的嵌入表示，可以替代 input_ids
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器的隐藏状态，用于跨层信息传播
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器的注意力掩码，用于控制跨层注意力计算
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 前一个时间步的关键值对，用于快速计算
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出所有隐藏层
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = None,
# 创建一个基于 RoBERTa 的语言模型，用于 CLM fine-tuning
@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)
class RobertaForCausalLM(RobertaPreTrainedModel):
    # 定义需要共享权重的键
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建一个 RoBERTaModel 对象，并指定不添加汇聚层
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 创建一个 RoBERTaLMHead 对象
        self.lm_head = RobertaLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 输出如何替换
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
    ):
        # 准备生成模型输入
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            # 获取输入形状
            input_shape = input_ids.shape
            # 如果没有指定解码器的注意力掩码，则默认为全1
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果使用了过去的键值，则剪裁输入的decoder_input_ids
            if past_key_values is not None:
                # 获取过去的长度
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法只传递了最后一个输入ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认情况下只保留最后一个ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

            # 返回准备好的模型输入
            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
        ...
    # 重新排序缓存的过去的关键值
    def _reorder_cache(self, past_key_values, beam_idx):
        # 创建一个空的元组来存储重新排序的过去的关键值
        reordered_past = ()
        # 遍历每个层的过去的关键值
        for layer_past in past_key_values:
            # 对于每个层的过去的关键值,
            # 使用beam_idx来选择对应的过去状态,并将其添加到重新排序的元组中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序的过去的关键值
        return reordered_past
# 定义 RoBERTa 语言模型的 MaskedLM 版本
@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(RobertaPreTrainedModel):
    # 定义共享权重的键
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)

        # 如果是 decoder，发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 RoBERTa 模型和 LM 头
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，参数详见注释
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    # 返回类型是 Union[Tuple[torch.Tensor], MaskedLMOutput]
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.roberta 方法获得输出
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 使用 self.lm_head 方法计算预测得分
        prediction_scores = self.lm_head(sequence_output)

        # 初始化 masked_lm_loss 为 None
        masked_lm_loss = None
        # 如果提供了 labels，计算 masked language modeling 损失
        if labels is not None:
            # 将 labels 移动到正确的设备上，以支持模型并行
            labels = labels.to(prediction_scores.device)
            # 使用交叉熵损失计算损失
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不使用返回字典
        if not return_dict:
            # 返回预测得分和其他输出
            output = (prediction_scores,) + outputs[2:]
            # 如果有损失，返回损失和输出
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 使用返回字典的方式返回输出
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# RobertaLMHead 类定义了 Roberta 模型中用于 masked language modeling 的头部 (head)
class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        # 调用父类 nn.Module 的构造方法
        super().__init__()
        # 创建一个全连接层，将输入映射到与隐藏层大小相同的大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个层归一化层，用于对全连接层的输出进行归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建一个全连接层，将归一化后的输出映射到词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 创建一个可学习的偏置项
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将偏置项设置为全连接层的偏置项
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 对输入 features 进行全连接变换
        x = self.dense(features)
        # 对变换后的 x 应用 GELU 激活函数
        x = gelu(x)
        # 对激活后的 x 进行层归一化
        x = self.layer_norm(x)

        # 将归一化后的 x 通过最终的全连接层和偏置项映射到词汇表大小
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果解码器的偏置项在 "meta" 设备上，则将其绑定到self.bias
        # 这是为了确保兼容性并不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

# RobertaForSequenceClassification 类定义了 Roberta 模型用于序列分类任务的头部
@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        # 调用父类 RobertaPreTrainedModel 的构造方法
        super().__init__(config)
        # 设置分类任务的标签数量
        self.num_labels = config.num_labels
        # 保存配置信息
        self.config = config

        # 创建 RobertaModel 实例，不添加池化层
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 创建 RobertaClassificationHead 实例
        self.classifier = RobertaClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ):
        # 在这里实现前向传播过程
        pass
    # 定义函数的输入和输出类型信息
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 检查是否返回字典，如果不是，使用配置中的返回字典选项
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型进行推理
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
        # 将输出传递给分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            # 确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用二进制交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            # 如果不返回字典，将结果以元组形式返回
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，以 SequenceClassifierOutput 类的形式返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 Roberta 多选分类模型添加文档字符串，描述其在顶部的多选分类头部的结构（在池化输出之上的线性层和一个 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
# 定义 Roberta 多选分类模型类，继承自 RobertaPreTrainedModel
class RobertaForMultipleChoice(RobertaPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 RobertaModel 对象
        self.roberta = RobertaModel(config)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化分类器线性层，输入维度为隐藏状态的维度，输出维度为 1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
    # 定义函数，用于多选题的模型输出
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择的数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入扁平化
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用RoBERTa模型进行前向传播
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
        # 获取汇聚的输出
        pooled_output = outputs[1]

        # 对池化输出进行dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器预测标签
        logits = self.classifier(pooled_output)
        # 重塑预测的标签
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 将标签移到正确的设备以启用模型并行计算
            labels = labels.to(reshaped_logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多个选择题模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
# 定义了一个 RobertaForTokenClassification 类，用于在 RoBERTa 模型之上添加一个用于标记分类的头部（即在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
class RobertaForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 将标签的数量设置为配置中的标签数量
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 模型
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 如果配置中的分类器丢弃率不为 None，则使用配置中的值，否则使用配置中的隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    # 重写 forward 方法，用于模型前向传播
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
    # 定义函数，用于进行 token 分类任务
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用 config 里的 use_return_dict 决定是否返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入数据传递给 Roberta 模型进行 forward 运算
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
    
        # 获取 Roberta 输出的序列表示
        sequence_output = outputs[0]
    
        # 对序列表示应用 dropout 操作
        sequence_output = self.dropout(sequence_output)
        # 将序列表示输入分类器，得到分类得分
        logits = self.classifier(sequence_output)
    
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备上，以支持模型并行运算
            labels = labels.to(logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失，同时展开 logits 和 labels
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        if not return_dict:
            # 如果不要求返回字典格式的结果，则返回 tuple 格式的结果
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回字典格式的结果，包括损失、分类得分、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class RobertaClassificationHead(nn.Module):
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
        x = features[:, 0, :]  # 取得<s>标记（相当于[CLS]）
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    Roberta 模型的文本抽取问题回答任务的分类头部（在隐藏状态输出之上的线性层，用于计算`span start logits`和`span end logits`）。
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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

        # 使用 RoBERTa 模型处理输入数据
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

        sequence_output = outputs[0]

        # 在 RoBERTa 模型输出的基础上计算起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上，则添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出模型输入范围，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失，取两个损失的平均值
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问题回答模型的输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 根据输入 ID 创建位置 ID
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    根据输入 ID 创建位置 ID，非填充符用其位置号代替，位置号从 padding_idx+1 开始计算。
    忽略填充符。这是从 fairseq 的 `utils.make_positions` 修改而来。

    参数:
        input_ids: 输入 ID 序列
        padding_idx: 填充符在 ID 序列中的索引
        past_key_values_length: 历史位置信息长度，用于增量计算位置

    返回:
        新的位置 ID 序列
    """
    # 下面一系列类型转换和操作是为了同时兼容 ONNX 导出和 XLA 的需求
    # 创建一个掩码，将填充符位置标记为 0，其他位置标记为 1
    mask = input_ids.ne(padding_idx).int()
    # 根据掩码计算每个位置的位置 ID，同时考虑历史位置信息长度
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将位置 ID 加上 padding_idx，使其从 padding_idx+1 开始计数
    return incremental_indices.long() + padding_idx
```