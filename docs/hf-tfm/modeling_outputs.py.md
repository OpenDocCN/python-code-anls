# `.\transformers\modeling_outputs.py`

```
# 导入警告模块
import warnings
# 导入 dataclass 模块
from dataclasses import dataclass
# 导入 Optional 和 Tuple 类型
from typing import Optional, Tuple
# 导入 torch 模块
import torch
# 从当前目录下的 utils 模块中导入 ModelOutput 类
from .utils import ModelOutput

# 定义 BaseModelOutput 类，继承自 ModelOutput 类
@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义 last_hidden_state 属性，默认为 None
    last_hidden_state: torch.FloatTensor = None
    # 定义 hidden_states 属性，类型为 Optional[Tuple[torch.FloatTensor]]，默认为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 attentions 属性，类型为 Optional[Tuple[torch.FloatTensor]]，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义 BaseModelOutputWithNoAttention 类，继承自 ModelOutput 类
@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    # 定义变量 last_hidden_state，表示模型最后一层的隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 定义变量 hidden_states，表示模型每一层的隐藏状态的元组，包括初始嵌入层的输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义一个带有池化的模型输出基类，继承自 ModelOutput 类
class BaseModelOutputWithPooling(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义类属性，包含最后一层隐藏状态、池化输出、隐藏状态和注意力权重
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义一个带有池化但不包含注意力权重的模型输出基类，继承自 ModelOutput 类
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a pooling operation on the spatial dimensions.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    # 定义变量并初始化为 None，用于存储模型的隐藏状态信息
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储模型输出和过去的键/值（以加速序列解码）
@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 最后一层模型的隐藏状态序列
    last_hidden_state: torch.FloatTensor = None
    # 过去的键/值，用于加速序列解码
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 模型每层的隐藏状态序列
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 每层的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithCrossAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """
    # 定义函数参数，描述函数的输入
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出的隐藏状态的序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组的 `torch.FloatTensor`（如果模型有嵌入层，则有一个用于嵌入输出的 `torch.FloatTensor`，加上每一层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。
            
            模型在每一层输出的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组的 `torch.FloatTensor`（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            元组的 `torch.FloatTensor`（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            
            解码器的交叉注意力层的注意力权重，注意力 softmax 后用于计算交叉注意力头中的加权平均值。
    """
    
    # 定义函数参数的类型和默认值
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义一个带有池化和交叉注意力的基础模型输出类
@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    """

    # 最后一个隐藏状态的张量
    last_hidden_state: torch.FloatTensor = None
    # 池化输出的张量
    pooler_output: torch.FloatTensor = None
    # 隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 过去键/值的元组，用于加速顺序解码
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 注意力的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义一个带有过去和交叉注意力的基础模型输出类
@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    # 初始化变量，用于存储模型输出的不同部分
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个包含过去状态的 MoE 因果语言模型输出类，继承自模型输出基类
@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs as well as Mixture of Expert's router hidden
    states terms, to train a MoE model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        z_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            z_loss for the sparse modules.
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """

    # 下面是类的各种属性
    loss: Optional[torch.FloatTensor] = None  # 语言建模损失（用于下一个标记的预测）
    logits: torch.FloatTensor = None  # 语言建模头的预测分数（SoftMax 前每个词汇标记的分数）
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 隐藏状态的预先计算值，用于加速顺序解码
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型在每一层输出的隐藏状态
    # 定义一个可选的元组类型变量 attentions，用于存储 torch.FloatTensor 类型的数据，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个 torch.FloatTensor 类型的变量 z_loss，默认为 None
    z_loss: torch.FloatTensor = None
    # 定义一个 torch.FloatTensor 类型的变量 aux_loss，默认为 None
    aux_loss: torch.FloatTensor = None
    # 定义一个可选的元组类型变量 router_logits，用于存储 torch.FloatTensor 类型的数据，默认为 None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储模型输出，包括隐藏状态和注意力信息
@dataclass
class MoEModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_probs (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router probabilities that are computed by MoE routers, these terms are used to compute the auxiliary
            loss and the z_loss for Mixture of Experts models.
    """

    # 最后一层模型的隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 每一层的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 每一层的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 每一层的路由器概率
    router_probs: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储带有过去信息的模型输出
@dataclass
class MoeModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
    """

    # 定义变量并初始化为 None，用于存储模型的输出
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    """
    causal language model (或者是自回归模型)带有混合专家输出的基类。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            语言模型损失（用于下一个标记的预测）。

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言模型头的预测分数（SoftMax之前每个词汇标记的分数）。

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            稀疏模块的辅助损失。

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            MoE路由器计算的原始路由器logtis（后SoftMax），这些术语用于计算混合专家模型的辅助损失。

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            长度为 `config.n_layers` 的元组 `tuple(torch.FloatTensor)`，每个元组包含 2 个形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量

            包含预先计算的隐藏状态（自注意力块中的键和值），可用于加速序列解码。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor`（如果模型具有嵌入层，则为输出的嵌入的输出，加上每个层的输出）的形状 `(batch_size, sequence_length, hidden_size)`。

            模型每层的隐藏状态加上可选的初始嵌入输出。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor`（每层一个）的形状 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的类型为 torch.FloatTensor 的元组变量 router_logits，并初始化为 None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class MoEModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding) as well as
    Mixture of Expert's router hidden states terms, to train a MoE model.

    """

    # 定义一个数据类，用于存储模型输出，可能包含过去的键/值（用于加速顺序解码），以及混合专家的路由器隐藏状态项，用于训练 MoE 模型

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_probs: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    # 定义一个数据类，用于存储模型编码器的输出，还包含：预先计算的隐藏状态，可以加速顺序解码

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqMoEModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    # 定义一个数据类，用于存储模型编码器的输出，还包含：预先计算的隐藏状态，可以加速顺序解码

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    # 定义函数参数及其类型注释

    # `loss` 参数：语言建模损失（用于下一个标记的预测），类型为 `torch.FloatTensor`，形状为 `(1,)`，可选参数，当提供 `labels` 时返回
    loss: Optional[torch.FloatTensor] = None

    # `logits` 参数：语言建模头的预测分数（SoftMax 之前的每个词汇标记的分数），类型为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, config.vocab_size)`
    logits: torch.FloatTensor = None

    # `hidden_states` 参数：模型在每一层输出的隐藏状态的元组，如果模型有嵌入层，则为一个 `torch.FloatTensor` 元组（一个用于嵌入层的输出，如果模型有的话，加上每一层的输出），形状为 `(batch_size, sequence_length, hidden_size)`，可选参数，在传递 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    # `attentions` 参数：自注意力机制的注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，可选参数，在传递 `output_attentions=True` 或者 `config.output_attentions=True` 时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None
```  
# 定义一个数据类，用于表示带有过去信息的因果语言模型输出
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    因果语言模型（或自回归模型）输出的基类。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            语言建模损失（用于下一个标记的预测）。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（SoftMax 前每个词汇标记的分数）。
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            长度为 `config.n_layers` 的 `tuple(torch.FloatTensor)` 的元组，每个元组有 2 个形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量

            包含预先计算的隐藏状态（注意力块中的键和值），可用于加速顺序解码。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor` 的元组（如果模型有嵌入层，则为嵌入层的输出 + 每个层的输出）。

            模型在每一层的输出的隐藏状态加上可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `torch.FloatTensor` 的元组（每个层一个）。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    因果语言模型（或自回归模型）输出的基类。
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True` is passed):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    # 定义变量并初始化为 None，用于存储不同类型的模型输出
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 导入必要的模块和类
@dataclass
# 定义一个带过去键值的序列分类器输出类，继承自模型输出基类
class SequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义损失，类型为 torch.FloatTensor，形状为 (1,)
    loss: Optional[torch.FloatTensor] = None
    # 定义逻辑输出，类型为 torch.FloatTensor，形状为 (batch_size, config.num_labels)
    logits: torch.FloatTensor = None
    # 定义过去键值，类型为 Tuple[Tuple[torch.FloatTensor]]，可选
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义隐藏状态，类型为 Tuple[torch.FloatTensor]，可选
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义注意力权重，类型为 Tuple[torch.FloatTensor]，可选
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# 定义一个掩码语言模型输出的基类
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
``` 
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
            掩码语言建模（MLM）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每层模型的隐藏状态加上可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力 softmax 之后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    # 损失值，默认为空
    logits: torch.FloatTensor = None
    # 预测的对数概率值，默认为空
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态，默认为空
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，默认为空
# 定义一个数据类，用于存储序列到序列语言模型的输出
@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 预测的logits，浮点张量
    logits: torch.FloatTensor = None
    # 过去的键值，可选的元组，包含元组的浮点张量
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的元组，包含浮点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力，可选的元组，包含浮点张量
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，可选的元组，包含浮点张量
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的元组，包含浮点张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力，可选的元组，包含浮点张量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储序列到序列MoE语言模型的输出
@dataclass
class Seq2SeqMoEOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 预测的logits，浮点张量
    logits: torch.FloatTensor = None
    # 编码器z损失，浮点张量
    encoder_z_loss: torch.FloatTensor = None
    # 解码器z损失，浮点张量
    decoder_z_loss: torch.FloatTensor = None
    # 编码器辅助损失，浮点张量
    encoder_aux_loss: torch.FloatTensor = None
    # 解码器辅助损失，浮点张量
    decoder_aux_loss: torch.FloatTensor = None
    # 过去的键值，可选的元组，包含元组的浮点张量
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的元组，包含浮点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力，可选的元组，包含浮点张量
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器路由器logits，可选的元组，包含浮点张量
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，可选的元组，包含浮点张量
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的元组，包含浮点张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力，可选的元组，包含浮点张量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器路由器logits，可选的元组，包含浮点张量
    encoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储预测两个句子是否连续的模型输出
@dataclass
class NextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `next_sentence_label` is provided):
            Next sequence prediction (classification) loss.
            下一序列预测（分类）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
            下一序列预测（分类）头部的预测分数（SoftMax 前的 True/False 连续性分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            模型在每一层输出处的隐藏状态加上可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均。
    """

    # 损失，如果提供了 `next_sentence_label` 则返回
    loss: Optional[torch.FloatTensor] = None
    # 下一序列预测头部的预测分数
    logits: torch.FloatTensor = None
    # 隐藏状态，如果 `output_hidden_states=True` 则返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，如果 `output_attentions=True` 则返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import ModelOutput

# 定义一个数据类，用于表示序列分类器的输出
@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 分类（或回归，如果 config.num_labels==1）的损失
    loss: Optional[torch.FloatTensor] = None
    # 分类（或回归，如果 config.num_labels==1）得分（SoftMax 之前）
    logits: torch.FloatTensor = None
    # 模型各层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于表示序列到序列的句子分类器的输出
@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    """

    # 分类（或回归，如果 config.num_labels==1）的损失
    loss: Optional[torch.FloatTensor] = None
    # 分类（或回归，如果 config.num_labels==1）得分（SoftMax 之前）
    logits: torch.FloatTensor = None
    # 过去的键值，形状为 `(num_layers, batch_size, num_heads, seq_length, dim_key)`
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器的隐藏状态，形状为 `(num_layers, batch_size, seq_length, hidden_size)`
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力权重，形状为 `(num_layers, batch_size, num_heads, seq_length, seq_length)`
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力权重，形状为 `(num_layers, batch_size, num_heads, seq_length, seq_length)`
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后一层的隐藏状态，形状为 `(batch_size, seq_length, hidden_size)`
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器各层的隐藏状态，形状为 `(num_layers, batch_size, seq_length, hidden_size)`
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力权重，形状为 `(num_layers, batch_size, num_heads, seq_length, seq_length)`
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于表示多项选择模型的输出
@dataclass
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.
    """
    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* 是输入张量的第二维度。(参见上面的 *input_ids*)。

            分类得分（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor` 元组。

            模型在每一层输出的隐藏状态，以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `torch.FloatTensor` 元组。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 导入必要的模块和类
@dataclass
# 定义一个数据类，用于表示标记分类器的输出
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 分类损失，当提供了 `labels` 参数时返回
    loss: Optional[torch.FloatTensor] = None
    # 分类分数（SoftMax 之前的得分）
    logits: torch.FloatTensor = None
    # 隐藏状态，当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# 定义一个数据类，用于表示问答模型的输出
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.


```  
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 总的跨度提取损失，是开始和结束位置的交叉熵之和，当提供了 `labels` 时返回
    loss: Optional[torch.FloatTensor] = None
    # 跨度开始得分（SoftMax 之前）
    start_logits: torch.FloatTensor = None
    # 跨度结束得分（SoftMax 之前）
    end_logits: torch.FloatTensor = None
    # 模型每层的隐藏状态，包括初始嵌入层的输出，当 `output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 自注意力头中的注意力权重，用于计算自注意力中的加权平均值，当 `output_attentions=True` 时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储序列到序列问答模型的输出结果
@dataclass
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 起始位置的logits，torch浮点张量
    start_logits: torch.FloatTensor = None
    # 结束位置的logits，torch浮点张量
    end_logits: torch.FloatTensor = None
    # 过去的键值，可选的元组
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的元组
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力，可选的元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，可选的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力，可选的元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储语义分割模型的输出结果
@dataclass
class SemanticSegmenterOutput(ModelOutput):
    """
    Base class for outputs of semantic segmentation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # logits，torch浮点张量
    logits: torch.FloatTensor = None
    # 隐藏状态，可选的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力，可选的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储图像分类模型的输出结果
@dataclass
class ImageClassifierOutput(ModelOutput):
    """
    Base class for outputs of image classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果config.num_labels==1）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果config.num_labels==1）得分（SoftMax之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组`torch.FloatTensor`（如果模型有嵌入层，则为嵌入输出，+每个阶段的输出）的形状为`(batch_size, sequence_length, hidden_size)`。模型在每个阶段输出的隐藏状态（也称为特征映射）。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组`torch.FloatTensor`（每个层一个）的形状为`(batch_size, num_heads, patch_size, sequence_length)`。

            在注意力SoftMax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储图像分类模型的输出结果，继承自ModelOutput类
@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果config.num_labels==1）损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果config.num_labels==1）得分（SoftMax之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含`torch.FloatTensor`（如果模型有嵌入层，则为嵌入层的输出，以及每个阶段的输出）的形状为`(batch_size, num_channels, height, width)`。
            模型在每个阶段输出的隐藏状态（也称为特征图）。

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储深度估计模型的输出结果，继承自ModelOutput类
@dataclass
class DepthEstimatorOutput(ModelOutput):
    """
    Base class for outputs of depth estimation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果config.num_labels==1）损失值。
        predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            每个像素的预测深度。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含`torch.FloatTensor`（如果模型有嵌入层，则为嵌入层的输出，以及每个层的输出）的形状为`(batch_size, num_channels, height, width)`。
            模型在每个层输出的隐藏状态，以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含`torch.FloatTensor`（每个层一个）的形状为`(batch_size, num_heads, patch_size, sequence_length)`。
            注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。

    """

    loss: Optional[torch.FloatTensor] = None
    predicted_depth: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储图像超分辨率模型的输出结果，继承自ModelOutput类
@dataclass
class ImageSuperResolutionOutput(ModelOutput):
    """
    Base class for outputs of image super resolution models.
    """
    # 定义函数的参数说明文档
    Args:
        # 如果提供了标签，则返回损失（一个形状为`(1,)`的浮点张量）
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Reconstruction loss.
        # 重构后的图像，可能是上采样的
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed images, possibly upscaled.
        # 如果传入了`output_hidden_states=True`或者`config.output_hidden_states=True`，则返回隐藏状态元组
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        # 如果传入了`output_attentions=True`或者`config.output_attentions=True`，则返回注意力元组
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义损失的类型为可选的浮点张量，默认为`None`
    loss: Optional[torch.FloatTensor] = None
    # 定义重构的张量为浮点张量，默认为`None`
    reconstruction: torch.FloatTensor = None
    # 定义隐藏状态的类型为可选的浮点张量元组，默认为`None`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义注意力的类型为可选的浮点张量元组，默认为`None`
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 @dataclass 装饰器定义了一个类，表示 Wav2Vec2 模型的基本输出
@dataclass
class Wav2Vec2BaseModelOutput(ModelOutput):
    """
    Base class for models that have been trained with the Wav2Vec2 loss objective.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
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

    # 定义类的属性，包括最后一层模型的隐藏状态、最后一个卷积层的特征向量、每层的隐藏状态和注意力权重
    last_hidden_state: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 @dataclass 装饰器定义了一个类，表示 XVector 模型的输出类型
@dataclass
class XVectorOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ForXVector`].
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`):
            AMSoftmax 之前的分类隐藏状态。
        embeddings (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`):
            用于基于向量相似性检索的话语嵌入。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor`（一个用于嵌入输出 + 每一层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `torch.FloatTensor`（每一层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个带有数据类装饰器的类，表示模型的输出，包含了模型的主干网络输出
@dataclass
class BackboneOutput(ModelOutput):
    """
    Base class for outputs of backbones.

    Args:
        feature_maps (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`):
            Feature maps of the stages.表示各个阶段的特征图
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
            depending on the backbone.表示模型在每个阶段的隐藏状态，以及初始嵌入输出
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Only applicable if the backbone uses attention.表示模型在每个层次的注意力权重，只有在主干网络使用注意力机制时才适用

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    feature_maps: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPoolingAndProjection(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    基类，包含了模型输出和最后隐藏状态的汇聚

    """
```  
    # 参数说明：
    # last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
    # 模型最后一层的隐藏状态的序列输出。
    # pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
    # 经过额外预训练任务的处理后，序列中第一个令牌（分类令牌）的最后一层隐藏状态。
    # hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
    # 元组的 `torch.FloatTensor`（如果模型有嵌入层，则包含嵌入层输出的一个元素，加上每一层的输出），
    # 形状为 `(batch_size, sequence_length, hidden_size)`。
    # 模型每一层的隐藏状态，以及可选的初始嵌入输出。
    # attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
    # 元组的 `torch.FloatTensor`（每一层一个元素），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    # 注意力 softmax 后的注意力权重，用于计算自注意力头的加权平均值。
    # projection_state (`tuple(torch.FloatTensor)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
    # 元组的 `torch.FloatTensor`，形状为 `(batch_size,config.project_dim)`。
    # 投影层之前的文本嵌入，用于模拟教师编码器的最后隐藏状态。

    # 初始化参数，默认为 None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    projection_state: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于表示序列到序列的频谱输出
class Seq2SeqSpectrogramOutput(ModelOutput):
    """
    Base class for sequence-to-sequence spectrogram outputs.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 频谱，浮点张量
    spectrogram: torch.FloatTensor = None
    # 过去的关键值，可选的元组，包含元组的浮点张量
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的元组，包含浮点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力，可选的元组，包含浮点张量
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，可选的元组，包含浮点张量
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的元组，包含浮点张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力，可选的元组，包含浮点张量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于表示序列到序列的时间序列模型输出
class Seq2SeqTSModelOutput(ModelOutput):
    """
    Base class for time series model's encoder outputs that also contains pre-computed hidden states that can speed up
    sequential decoding.

    """

    # 最后隐藏状态，浮点张量
    last_hidden_state: torch.FloatTensor = None
    # 过去的关键值，可选的元组，包含元组的浮点张量
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的元组，包含浮点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力，可选的元组，包含浮点张量
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，可选的元组，包含浮点张量
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的元组，包含浮点张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力，可选的元组，包含浮点张量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 位置，可选的浮点张量
    loc: Optional[torch.FloatTensor] = None
    # 尺度，可选的浮点张量
    scale: Optional[torch.FloatTensor] = None
    # 静态特征，可选的浮点张量
    static_features: Optional[torch.FloatTensor] = None


# 定义一个数据类，用于表示序列到序列的时间序列模型预测输出
class Seq2SeqTSPredictionOutput(ModelOutput):
    """
    Base class for time series model's decoder outputs that also contain the loss as well as the parameters of the
    chosen distribution.

    """

    # 损失值，可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 参数，可选的元组，包含浮点张量
    params: Optional[Tuple[torch.FloatTensor]] = None
    # 过去的关键值，可选的元组，包含元组的浮点张量
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的元组，包含浮点张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器注意力，可选的元组，包含浮点张量
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注���力，可选的元组��包含浮点张量
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器最后隐藏状态，可选的浮点张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的元组，包含浮点张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器注意力，可选的元组，包含浮点张量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 位置，可选的浮点张量
    loc: Optional[torch.FloatTensor] = None
    # 尺度，可选的浮点张量
    scale: Optional[torch.FloatTensor] = None
    # 静态特征，可选的浮点张量
    static_features: Optional[torch.FloatTensor] = None


# 定义一个数据类，用于表示样本时间序列模型的预测输出
class SampleTSPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length)` or `(batch_size, num_samples, prediction_length, input_size)`):
            Sampled values from the chosen distribution.
    """

    # 序列，浮点张量，形状为`(batch_size, num_samples, prediction_length)`或`(batch_size, num_samples, prediction_length, input_size)`
    sequences: torch.FloatTensor = None
# MaskedImageModelingOutput 类，用于表示遮罩图像补全 / 修复模型的输出结果
class MaskedImageModelingOutput(ModelOutput):
    """
    Base class for outputs of masked image completion / in-painting models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss. 重构损失，当提供 `bool_masked_pos` 参数时返回。
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed / completed images. 重建 / 完成的图像。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
            模型输出的隐藏状态（也称为特征图），形状为 `(batch_size, sequence_length, hidden_size)` 的元组，
            其中包含每个阶段的输出（如果模型有嵌入层，则包含嵌入层的输出）。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
            注意力权重，形状为 `(batch_size, num_heads, patch_size, sequence_length)` 的元组，
            其中包含每个层的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # Reconstruction loss. 重构损失。
    loss: Optional[torch.FloatTensor] = None
    # Reconstructed / completed images. 重建 / 完成的图像。
    reconstruction: torch.FloatTensor = None
    # Tuple of hidden-states (feature maps) of the model at the output of each stage. 模型输出的隐藏状态（特征图）元组。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Tuple of attention weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    # 注意力权重元组，用于计算自注意力头中的加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Deprecated attribute. Use the reconstruction attribute instead.
    # 不推荐使用的属性。请改用 reconstruction 属性。
    @property
    def logits(self):
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.reconstruction
```