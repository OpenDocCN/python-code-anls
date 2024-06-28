# `.\modeling_outputs.py`

```
# 导入警告模块，用于可能的警告消息
import warnings
# 导入 dataclass 模块，用于定义数据类
from dataclasses import dataclass
# 导入 Optional 和 Tuple 类型提示
from typing import Optional, Tuple

# 导入 PyTorch 模块
import torch

# 从当前包中导入 ModelOutput 类
from .utils import ModelOutput

# 定义 BaseModelOutput 数据类，继承自 ModelOutput
@dataclass
class BaseModelOutput(ModelOutput):
    """
    模型输出的基类，可能包含隐藏状态和注意力。
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            包含每层输出的元组 `torch.FloatTensor`（如果模型有嵌入层，则包含嵌入层输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每层的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            包含每层注意力权重的元组 `torch.FloatTensor`，
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力权重经过 softmax 后的结果，在自注意力头中用于计算加权平均值。
    """

    # 最后的隐藏状态，默认为 None
    last_hidden_state: torch.FloatTensor = None
    # 隐藏状态的元组，可选，默认为 None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 注意力权重的元组，可选，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义 BaseModelOutputWithNoAttention 数据类，继承自 ModelOutput
@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):
    """
    模型输出的基类，仅包含潜在的隐藏状态。

    继承自 ModelOutput。
    """
    # 定义函数的参数说明和类型注解。`last_hidden_state`参数是一个 `torch.FloatTensor` 类型的张量，
    # 其形状为 `(batch_size, num_channels, height, width)`，表示模型最后一层的隐藏状态序列。
    # `hidden_states`参数是一个可选的元组类型，包含 `torch.FloatTensor` 类型的张量，
    # 形状为 `(batch_size, num_channels, height, width)`。这个元组用于存储模型每一层的隐藏状态，
    # 如果模型具有嵌入层，则还包括初始嵌入层的输出。
    
    last_hidden_state: torch.FloatTensor = None
    # 初始化 `last_hidden_state` 变量为 `None`，表示这个参数可以不提供具体值。
    
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 初始化 `hidden_states` 变量为 `None`，表示这个参数也可以不提供具体值。
    # 它是一个可选的元组类型，元组中的每个元素都是 `torch.FloatTensor` 类型的张量。
    # 如果提供了 `output_hidden_states=True` 或 `config.output_hidden_states=True`，
    # 这个元组会包含模型每一层的隐藏状态和可能的初始嵌入层的输出。
# 使用 dataclass 装饰器定义一个基础模型输出类，包含池化后的最后隐藏状态等信息
@dataclass
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

    # 定义类变量，存储最后隐藏状态的张量
    last_hidden_state: torch.FloatTensor = None
    # 定义类变量，存储经过池化处理后的分类池化输出
    pooler_output: torch.FloatTensor = None
    # 定义类变量，存储模型隐藏状态的元组（每层的输出）
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义类变量，存储注意力权重的元组（每层的注意力权重）
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 dataclass 装饰器定义一个基础模型输出类，包含池化后的最后隐藏状态，但不包含注意力权重信息
@dataclass
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    """
    # `last_hidden_state`是模型最后一层的隐藏状态，形状为(batch_size, num_channels, height, width)
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
    
    # `pooler_output`是经过空间维度池化操作后的最后一层隐藏状态，形状为(batch_size, hidden_size)
    pooler_output: torch.FloatTensor = None
    
    # `hidden_states`是一个元组，包含了模型每一层的隐藏状态输出，如果模型有嵌入层，还包括初始嵌入输出。
    # 每个张量的形状为(batch_size, num_channels, height, width)。
    # 可选的返回值，当`output_hidden_states=True`被传递或者`config.output_hidden_states=True`时返回。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器声明一个数据类，表示带有过去键/值的模型输出
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

    # 定义类成员变量，表示模型输出中的最后隐藏状态、过去键/值、隐藏状态、注意力权重
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 dataclass 装饰器声明一个数据类，表示带有交叉注意力的模型输出
@dataclass
class BaseModelOutputWithCrossAttentions(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """
    # `last_hidden_state`是模型最后一层的输出隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    last_hidden_state: torch.FloatTensor = None
    # `hidden_states`是一个元组，包含模型每一层的隐藏状态（如果模型有嵌入层，则包含初始嵌入输出），
    # 其形状为(batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # `attentions`是一个元组，包含每一层的注意力权重张量，形状为(batch_size, num_heads, sequence_length, sequence_length)，
    # 用于计算自注意力头中的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # `cross_attentions`是一个元组，包含解码器交叉注意力层的注意力权重张量，形状为(batch_size, num_heads, sequence_length, sequence_length)，
    # 用于计算交叉注意力头中的加权平均值
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 带有池化和交叉注意力的基础模型输出类，继承自ModelOutput
@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    模型输出的基础类，还包含最后隐藏状态的池化。

    """

    # 最后隐藏状态，类型为 torch.FloatTensor
    last_hidden_state: torch.FloatTensor = None
    # 池化输出，类型为 torch.FloatTensor
    pooler_output: torch.FloatTensor = None
    # 隐藏状态的元组，可选类型为 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 过去键/值的元组，可选类型为 Tuple[Tuple[torch.FloatTensor]]
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 注意力权重的元组，可选类型为 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 交叉注意力权重的元组，可选类型为 torch.FloatTensor
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    模型输出的基础类，可能还包含过去的键/值（用于加速顺序解码）。

    """
    # 定义函数参数 `last_hidden_state`，表示模型最后一层的隐藏状态，是一个形状为 `(batch_size, sequence_length, hidden_size)` 的张量。
    # 如果使用了 `past_key_values`，则输出的是形状为 `(batch_size, 1, hidden_size)` 的序列最后隐藏状态。
    last_hidden_state: torch.FloatTensor = None
    
    # 定义函数可选参数 `past_key_values`，是一个元组，包含了预计算的隐藏状态键值对（在自注意力块中的键和值），长度为 `config.n_layers`，每个元组包含两个张量。
    # 如果 `config.is_encoder_decoder=True`，还包括两个形状为 `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)` 的张量。
    # 用于加速顺序解码过程。
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 定义函数可选参数 `hidden_states`，是一个元组，包含了模型每一层的隐藏状态。
    # 如果模型有嵌入层，第一个张量是嵌入层的输出；每个张量的形状为 `(batch_size, sequence_length, hidden_size)`。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义函数可选参数 `attentions`，是一个元组，包含了每一层的注意力权重。
    # 每个张量的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    # 这些权重在自注意力头中的注意力 softmax 之后，用于计算加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义函数可选参数 `cross_attentions`，是一个元组，包含了解码器交叉注意力层的注意力权重。
    # 每个张量的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    # 这些权重在交叉注意力头中的注意力 softmax 之后，用于计算加权平均值。
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
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

    # loss 属性，表示语言模型损失，用于下一个标记的预测
    loss: Optional[torch.FloatTensor] = None

    # logits 属性，表示语言模型头部的预测分数（SoftMax 前的每个词汇标记的分数）
    logits: torch.FloatTensor = None

    # past_key_values 属性，存储预先计算的隐藏状态（注意力机制的键值对），用于加速顺序解码
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    # hidden_states 属性，存储每一层的模型隐藏状态，包括可选的初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    # attentions 属性，存储每一层的注意力权重，用于计算自注意力头部的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # z_loss 属性，用于稀疏模块的 z_loss
    z_loss: Optional[torch.FloatTensor] = None

    # aux_loss 属性，用于稀疏模块的 aux_loss
    aux_loss: Optional[torch.FloatTensor] = None

    # router_logits 属性，存储编码器模型的路由器 logits，用于计算稀疏模块的辅助损失和 z_loss
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组 attentions，元组中包含了多个 torch.FloatTensor 类型的张量，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义一个 torch.FloatTensor 类型的张量 z_loss，初始值为 None
    z_loss: torch.FloatTensor = None
    
    # 定义一个 torch.FloatTensor 类型的张量 aux_loss，初始值为 None
    aux_loss: torch.FloatTensor = None
    
    # 定义一个可选的单元素元组 router_logits，元组中包含了一个 torch.FloatTensor 类型的张量，初始值为 None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
# 用于定义模型输出的数据类，继承自ModelOutput类
@dataclass
class MoEModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层的输出隐藏状态序列。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每一层模型的隐藏状态输出，包括初始嵌入层的输出（如果存在）。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            经过注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。

        router_probs (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router probabilities that are computed by MoE routers, these terms are used to compute the auxiliary
            loss and the z_loss for Mixture of Experts models.
            由MoE路由器计算得到的原始路由器概率，用于计算混合专家模型的辅助损失和z_loss。
    """

    # 定义最后一个隐藏状态，默认为None
    last_hidden_state: torch.FloatTensor = None
    # 定义隐藏状态的元组，可选，当output_hidden_states=True或config.output_hidden_states=True时返回
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义注意力的元组，可选，当output_attentions=True或config.output_attentions=True时返回
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义路由器概率的元组，可选，当output_router_probs=True和config.add_router_probs=True或config.output_router_probs=True时返回
    router_probs: Optional[Tuple[torch.FloatTensor]] = None


# 用于定义模型输出的数据类，继承自ModelOutput类，并且包含过去隐藏状态和注意力
@dataclass
class MoeModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    """
    # 定义输入参数 `last_hidden_state`，类型为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`
    last_hidden_state: torch.FloatTensor = None
    
    # 定义输入参数 `past_key_values`，类型为 `Optional[Tuple[Tuple[torch.FloatTensor]]]`，可选参数，
    # 当使用 `use_cache=True` 或 `config.use_cache=True` 时返回，包含预先计算的隐藏状态（在自注意力块中的键和值）
    # 如果 `config.is_encoder_decoder=True`，还包含交叉注意力块中的隐藏状态
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 定义输入参数 `hidden_states`，类型为 `Optional[Tuple[torch.FloatTensor, ...]]`，可选参数，
    # 当使用 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
    # 包含模型每一层的隐藏状态，以及可能的初始嵌入层输出
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义输入参数 `attentions`，类型为 `Optional[Tuple[torch.FloatTensor, ...]]`，可选参数，
    # 当使用 `output_attentions=True` 或 `config.output_attentions=True` 时返回，
    # 包含注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义输入参数 `router_logits`，类型为 `Optional[Tuple[torch.FloatTensor]]`，可选参数，
    # 当使用 `output_router_probs=True` 和 `config.add_router_probs=True` 时返回，
    # 包含 MoE 路由器计算的原始路由器 logits（经 softmax 处理后），用于混合专家模型的辅助损失计算
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) with mixture of experts outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            Auxiliary loss for the sparse modules.

        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logits (post-softmax) computed by MoE routers, used for computing auxiliary loss in Mixture of Experts models.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, each tuple containing 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.

            Pre-computed hidden states (keys and values in self-attention blocks) for speeding up sequential decoding.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for embedding layer output if present, plus one for each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden states of the model at each layer's output, including optional initial embedding outputs.

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

            Attention weights after the softmax operation, used for computing weighted averages in self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义了一个变量 router_logits，类型为 Optional[Tuple[torch.FloatTensor]]，初始值为 None
@dataclass
class MoEModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding) as well as
    Mixture of Expert's router hidden states terms, to train a MoE model.

    """

    last_hidden_state: torch.FloatTensor = None  # 最后一个隐藏状态的张量，默认为None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 可选的过去键/值对的元组，用于加速序列解码
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的隐藏状态的元组
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的注意力分布的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的交叉注意力分布的元组
    router_probs: Optional[Tuple[torch.FloatTensor]] = None  # 可选的路由器概率的元组


@dataclass
class Seq2SeqModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    last_hidden_state: torch.FloatTensor = None  # 最后一个隐藏状态的张量，默认为None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 可选的过去键/值对的元组，用于加速序列解码
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的解码器隐藏状态的元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的解码器注意力分布的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的交叉注意力分布的元组
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 可选的编码器最后一个隐藏状态的张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的编码器隐藏状态的元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的编码器注意力分布的元组


@dataclass
class Seq2SeqMoEModelOutput(ModelOutput):
    """
    Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
    decoding.

    """

    last_hidden_state: torch.FloatTensor = None  # 最后一个隐藏状态的张量，默认为None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 可选的过去键/值对的元组，用于加速序列解码
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的解码器隐藏状态的元组
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的解码器注意力分布的元组
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None  # 可选的解码器路由器逻辑概率的元组
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的交叉注意力分布的元组
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 可选的编码器最后一个隐藏状态的张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的编码器隐藏状态的元组
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 可选的编码器注意力分布的元组
    encoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None  # 可选的编码器路由器逻辑概率的元组


@dataclass
class CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """

    # 这里没有定义任何字段，但作为基类提供了一个用于因果语言模型输出的基础类
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
            语言建模损失（用于下一个标记预测），是一个形状为 `(1,)` 的 `torch.FloatTensor`，当提供 `labels` 时返回。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言建模头部的预测分数（在 SoftMax 之前每个词汇标记的得分），形状为 `(batch_size, sequence_length, config.vocab_size)` 的 `torch.FloatTensor`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            模型在每层输出的隐藏状态，以及可选的初始嵌入输出，形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor` 元组。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力 softmax 后的注意力权重，用于计算自注意力头部中的加权平均，形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `torch.FloatTensor` 元组。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器定义一个类，用于表示因果语言模型（或自回归模型）的输出结果，继承自 ModelOutput 类。
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    因果语言模型（或自回归模型）输出的基类。

    Args:
        loss (`torch.FloatTensor`，形状为 `(1,)`，*可选*，当提供 `labels` 参数时返回):
            语言建模的损失（用于下一个标记的预测）。
        logits (`torch.FloatTensor`，形状为 `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测分数（每个词汇标记的分数，在 SoftMax 之前）。
        past_key_values (`tuple(tuple(torch.FloatTensor))`，*可选*，当传递 `use_cache=True` 或 `config.use_cache=True` 时返回):
            包含预先计算的隐藏状态（自注意力块中的键和值），可用于加速顺序解码。
            是一个长度为 `config.n_layers` 的元组，每个元组包含 2 个形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量。
        hidden_states (`tuple(torch.FloatTensor)`，*可选*，当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            包含模型在每一层输出的隐藏状态张量的元组（如果模型有嵌入层，则包含初始嵌入输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`，*可选*，当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            自注意力头部中注意力 softmax 后的注意力权重张量的元组（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    # 以下是类的字段定义
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 dataclass 装饰器定义另一个类，用于表示具有交叉注意力的因果语言模型（或自回归模型）的输出结果，继承自 ModelOutput 类。
@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    因果语言模型（或自回归模型）输出的基类，具有交叉注意力。

    这个类继承自 ModelOutput。
    """

    # 注意：这里的类定义未完全提供，根据文档字符串需要添加额外的字段和解释。
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key,
            value states of the self-attention and the cross-attention layers if model is used in encoder-decoder
            setting. Only relevant if `config.is_decoder = True`.

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):
    """
    Base class for outputs of sequence classification models that also include past key values,
    hidden states, and attentions.

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

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for outputs of masked language models.

    This class inherits `ModelOutput`, indicating it provides standard output for models.

    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
            掩码语言建模（MLM）损失，当提供`labels`时返回此值。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer,
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            模型在每一层输出的隐藏状态，以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，经过注意力SoftMax后的值，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
@dataclass
class Seq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，可选的浮点张量，用于表示模型的损失
    loss: Optional[torch.FloatTensor] = None
    # 输出的对数概率值，用于表示模型生成的对数概率
    logits: torch.FloatTensor = None
    # 过去的键值，可选的嵌套元组，用于存储过去的键值
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的浮点张量元组，表示解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 解码器注意力权重，可选的浮点张量元组，表示解码器的注意力权重
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 交叉注意力权重，可选的浮点张量元组，表示编码器-解码器的交叉注意力权重
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器最后的隐藏状态，可选的浮点张量，表示编码器的最后隐藏状态
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的浮点张量元组，表示编码器的隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器注意力权重，可选的浮点张量元组，表示编码器的注意力权重
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Seq2SeqMoEOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    """

    # 损失值，可选的浮点张量，用于表示模型的损失
    loss: Optional[torch.FloatTensor] = None
    # 输出的对数概率值，用于表示模型生成的对数概率
    logits: torch.FloatTensor = None
    # 编码器 Z 损失，用于表示编码器的 Z 损失
    encoder_z_loss: torch.FloatTensor = None
    # 解码器 Z 损失，用于表示解码器的 Z 损失
    decoder_z_loss: torch.FloatTensor = None
    # 编码器辅助损失，用于表示编码器的辅助损失
    encoder_aux_loss: torch.FloatTensor = None
    # 解码器辅助损失，用于表示解码器的辅助损失
    decoder_aux_loss: torch.FloatTensor = None
    # 过去的键值，可选的嵌套元组，用于存储过去的键值
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器隐藏状态，可选的浮点张量元组，表示解码器的隐藏状态
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 解码器注意力权重，可选的浮点张量元组，表示解码器的注意力权重
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 解码器路由器对数概率，可选的浮点张量，表示解码器的路由器对数概率
    decoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力权重，可选的浮点张量元组，表示编码器-解码器的交叉注意力权重
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器最后的隐藏状态，可选的浮点张量，表示编码器的最后隐藏状态
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器隐藏状态，可选的浮点张量元组，表示编码器的隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器注意力权重，可选的浮点张量元组，表示编码器的注意力权重
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器路由器对数概率，可选的浮点张量，表示编码器的路由器对数概率
    encoder_router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class NextSentencePredictorOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    # 定义 loss 变量，用于存储下一个序列预测（分类）的损失值，类型为 torch.FloatTensor，可选参数，当提供 `next_sentence_label` 时返回。
    loss: Optional[torch.FloatTensor] = None
    # 定义 logits 变量，用于存储下一个序列预测（分类）头部的预测分数，形状为 `(batch_size, 2)` 的 torch.FloatTensor。
    logits: torch.FloatTensor = None
    # 定义 hidden_states 变量，用于存储模型每一层的隐藏状态输出，类型为元组 `Tuple[torch.FloatTensor, ...]`，可选参数，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义 attentions 变量，用于存储注意力权重输出，类型为元组 `Tuple[torch.FloatTensor, ...]`，可选参数，当 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 定义一个数据类，用于表示序列分类器模型的输出结果
@dataclass
class SequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
            分类模型的损失值（如果提供了`labels`）：一个形状为`(1,)`的`torch.FloatTensor`，在提供`labels`时返回。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
            分类（或回归，如果`config.num_labels==1`）得分（SoftMax之前）的`torch.FloatTensor`，形状为`(batch_size, config.num_labels)`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            模型每一层的输出的隐藏状态，以及可选的初始嵌入输出。形状为`(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重（经过注意力SoftMax后的）的元组，用于计算自注意力头中的加权平均值。形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义一个数据类，用于表示序列到序列的句子分类器模型的输出结果
@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence sentence classification models.

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义一个数据类，用于表示多选模型的输出结果
@dataclass
class MultipleChoiceModelOutput(ModelOutput):
    """
    Base class for outputs of multiple choice models.

    """
    """
    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            分类损失值。
            如果提供了`labels`，则返回此损失值。

        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            `num_choices` 是输入张量的第二个维度。
            分类分数（SoftMax 之前的值）。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor` 的张量。
            第一个张量是模型嵌入层的输出（如果存在），每一层输出的张量的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态，加上可选的初始嵌入输出。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `torch.FloatTensor` 的张量。
            每一层的注意力权重张量的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 用于描述令牌分类模型输出的基础类
@dataclass
class TokenClassifierOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
            分类损失，当提供 `labels` 参数时返回。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
            分类分数（SoftMax 之前的结果）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            模型在每一层输出的隐藏状态，以及可选的初始嵌入层输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，经过注意力 SoftMax 后的结果，用于计算自注意力头中的加权平均值。
    """

@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.
    用于描述问答模型输出的基础类。

    This class is currently empty but can be extended with specific outputs of QA models.
    该类目前为空，但可以通过扩展以包含问答模型的特定输出。
    """
    # 定义函数参数和返回值的文档字符串，描述了函数的输入和输出

    loss: Optional[torch.FloatTensor] = None
    # 可选的损失张量，当提供了 `labels` 参数时返回

    start_logits: torch.FloatTensor = None
    # 开始位置的得分张量，形状为 `(batch_size, sequence_length)`

    end_logits: torch.FloatTensor = None
    # 结束位置的得分张量，形状为 `(batch_size, sequence_length)`

    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的隐藏状态元组，包含每层输出的张量，形状为 `(batch_size, sequence_length, hidden_size)`
    # 如果模型有嵌入层，则还包含初始嵌入输出

    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选的注意力权重元组，包含每层注意力权重的张量
    # 形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    # 用于计算自注意力头中的加权平均值的注意力 softmax 后的注意力权重
# 定义了一个数据类，用于存储序列到序列问答模型的输出结果
@dataclass
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of sequence-to-sequence question answering models.

    """

    # 损失值，如果存在的话，类型为 torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 开始位置的预测 logits，类型为 torch.FloatTensor
    start_logits: torch.FloatTensor = None
    # 结束位置的预测 logits，类型为 torch.FloatTensor
    end_logits: torch.FloatTensor = None
    # 过去的键值，类型为可选的元组，包含了一系列 torch.FloatTensor
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 解码器的隐藏状态，类型为可选的元组，包含了一系列 torch.FloatTensor
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 解码器的注意力权重，类型为可选的元组，包含了一系列 torch.FloatTensor
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 交叉注意力权重，类型为可选的元组，包含了一系列 torch.FloatTensor
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器最后的隐藏状态，类型为可选的 torch.FloatTensor
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态，类型为可选的元组，包含了一系列 torch.FloatTensor
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 编码器的注意力权重，类型为可选的元组，包含了一系列 torch.FloatTensor
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义了一个数据类，用于存储语义分割模型的输出结果
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

    # 损失值，如果存在的话，类型为 torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 分类得分 logits，类型为 torch.FloatTensor，形状为 (batch_size, config.num_labels, logits_height, logits_width)
    logits: torch.FloatTensor = None
    # 隐藏状态，类型为可选的元组，包含了一系列 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 注意力权重，类型为可选的元组，包含了一系列 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 定义了一个数据类，用于存储图像分类模型的输出结果
@dataclass
class ImageClassifierOutput(ModelOutput):
    """
    Base class for outputs of image classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或者回归，如果 `config.num_labels==1`）的损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或者回归，如果 `config.num_labels==1`）的分数（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组类型的 `torch.FloatTensor`，包含模型在每个阶段输出的隐藏状态（特征映射），形状为 `(batch_size, sequence_length, hidden_size)`。如果模型包含嵌入层，则第一个张量表示嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组类型的 `torch.FloatTensor`，包含模型的注意力权重，形状为 `(batch_size, num_heads, patch_size, sequence_length)`。这些权重经过注意力 SoftMax 后得到，用于计算自注意力头中的加权平均值。
@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类模型的损失值（如果提供了`labels`参数）。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类模型的输出分数（在经过 SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每个阶段的隐藏状态（也称为特征图），形状为 `(batch_size, num_channels, height, width)`。

    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DepthEstimatorOutput(ModelOutput):
    """
    Base class for outputs of depth estimation models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            深度估计模型的损失值（如果提供了`labels`参数）。
        predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            每个像素预测的深度值。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每个层的隐藏状态（也称为特征图），形状为 `(batch_size, num_channels, height, width)`。

            每个层的输出以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            每个层的注意力权重，形状为 `(batch_size, num_heads, patch_size, sequence_length)`。

            注意力softmax后的权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_depth: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ImageSuperResolutionOutput(ModelOutput):
    """
    Base class for outputs of image super resolution models.
    """
    # 定义函数的参数和返回值的类型注解
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            重建损失，当提供`labels`时返回。
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            重建的图像，可能是上采样后的结果。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含`torch.FloatTensor`类型的张量：
            - 如果模型有嵌入层，则为形状为`(batch_size, sequence_length, hidden_size)`的张量；
            - 每个阶段输出的隐藏状态（也称为特征图）。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含`torch.FloatTensor`类型的张量：
            - 每层的注意力权重，形状为`(batch_size, num_heads, patch_size, sequence_length)`。
            
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    
    loss: Optional[torch.FloatTensor] = None
        # 默认值为 None 的可选项，类型为 torch.FloatTensor
    reconstruction: torch.FloatTensor = None
        # 默认值为 None 的 torch.FloatTensor 类型的变量
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
        # 默认值为 None 的可选项，类型为元组，包含 torch.FloatTensor 类型的张量
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
        # 默认值为 None 的可选项，类型为元组，包含 torch.FloatTensor 类型的张量
# 使用 `dataclass` 装饰器定义一个数据类，用于表示 Wav2Vec2 模型的基础输出。
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

    # 定义类变量 `last_hidden_state`，表示模型输出的最后一层隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 定义类变量 `extract_features`，表示模型输出的最后一个卷积层的特征向量序列
    extract_features: torch.FloatTensor = None
    # 定义类变量 `hidden_states`，表示模型每一层的隐藏状态的元组，可选返回项
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义类变量 `attentions`，表示注意力权重的元组，可选返回项，用于自注意力头中的加权平均计算
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 `dataclass` 装饰器定义一个数据类，表示 `Wav2Vec2ForXVector` 的输出类型。
@dataclass
class XVectorOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2ForXVector`].
    """

    # 此数据类暂未定义任何具体的输出内容，保留空的定义。
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类损失。
            如果提供了 `labels`，则返回分类损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`):
            AMSoftmax 前的分类隐藏状态。
            用于 AMSoftmax 前的分类隐藏状态。
        embeddings (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`):
            用于基于向量相似性检索的话语嵌入。
            用于基于向量相似性检索的话语嵌入。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每层输出的隐藏状态。
            当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
            元组包含了每层的 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`。
            包括每层的隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力权重。
            当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
            元组包含了每层的 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    embeddings: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器定义一个名为 `BackboneOutput` 的数据类，它继承自 `ModelOutput` 类
@dataclass
class BackboneOutput(ModelOutput):
    """
    Base class for outputs of backbones.

    Args:
        feature_maps (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`):
            Feature maps of the stages.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
            depending on the backbone.

            Hidden-states of the model at the output of each stage plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Only applicable if the backbone uses attention.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义特征图的属性，类型为元组，包含了每个阶段的特征图
    feature_maps: Tuple[torch.FloatTensor] = None
    # 定义隐藏状态的属性，类型为可选的元组，包含了每个阶段的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义注意力权重的属性，类型为可选的元组，包含了每个层的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 dataclass 装饰器定义一个名为 `BaseModelOutputWithPoolingAndProjection` 的数据类，它继承自 `ModelOutput` 类
@dataclass
class BaseModelOutputWithPoolingAndProjection(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    """
    # 定义函数参数和它们的类型注释，描述了函数所接收的不同类型的输入数据
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            经过附加预训练任务处理后的序列第一个标记（分类标记）的最后一层隐藏状态。
            例如，在BERT系列模型中，这是经过线性层和tanh激活函数处理后的分类标记。
            线性层的权重是从预训练过程中的下一句预测（分类）目标中训练得到的。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态序列的元组。
            每个元素的形状为 `(batch_size, sequence_length, hidden_size)`，包括可选的初始嵌入层输出。
            当 `output_hidden_states=True` 传递给模型或者 `config.output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            模型每一层的注意力权重的元组。
            每个元素的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，
            用于计算自注意力头中的加权平均值。
            当 `output_attentions=True` 传递给模型或者 `config.output_attentions=True` 时返回。
        projection_state (`tuple(torch.FloatTensor)`, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            投影层之前的文本嵌入的元组。
            每个元素的形状为 `(batch_size, config.project_dim)`，
            用于模拟教师编码器的最后隐藏状态。
@dataclass
class Seq2SeqSpectrogramOutput(ModelOutput):
    """
    Base class for sequence-to-sequence spectrogram outputs.

    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，用于存储模型输出的损失
    spectrogram: torch.FloatTensor = None  # 频谱图数据，存储模型生成的频谱图
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 过去的键值对，用于存储可加速顺序解码的隐藏状态
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 解码器的隐藏状态列表
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 解码器的注意力权重列表
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 交叉注意力权重列表
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 编码器的最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 编码器的隐藏状态列表
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 编码器的注意力权重列表


@dataclass
class Seq2SeqTSModelOutput(ModelOutput):
    """
    Base class for time series model's encoder outputs that also contains pre-computed hidden states that can speed up
    sequential decoding.

    """

    last_hidden_state: torch.FloatTensor = None  # 最后的隐藏状态，存储编码器最后的隐藏状态
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 过去的键值对，用于存储可加速顺序解码的隐藏状态
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 解码器的隐藏状态列表
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 解码器的注意力权重列表
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 交叉注意力权重列表
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 编码器的最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 编码器的隐藏状态列表
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 编码器的注意力权重列表
    loc: Optional[torch.FloatTensor] = None  # 位置参数，用于存储预测分布的位置参数
    scale: Optional[torch.FloatTensor] = None  # 尺度参数，用于存储预测分布的尺度参数
    static_features: Optional[torch.FloatTensor] = None  # 静态特征，用于存储与时间序列模型相关的静态特征


@dataclass
class Seq2SeqTSPredictionOutput(ModelOutput):
    """
    Base class for time series model's decoder outputs that also contain the loss as well as the parameters of the
    chosen distribution.

    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，用于存储模型输出的损失
    params: Optional[Tuple[torch.FloatTensor]] = None  # 参数，用于存储所选分布的参数
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 过去的键值对，用于存储可加速顺序解码的隐藏状态
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 解码器的隐藏状态列表
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 解码器的注意力权重列表
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 交叉注意力权重列表
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None  # 编码器的最后隐藏状态
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 编码器的隐藏状态列表
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 编码器的注意力权重列表
    loc: Optional[torch.FloatTensor] = None  # 位置参数，用于存储预测分布的位置参数
    scale: Optional[torch.FloatTensor] = None  # 尺度参数，用于存储预测分布的尺度参数
    static_features: Optional[torch.FloatTensor] = None  # 静态特征，用于存储与时间序列模型相关的静态特征


@dataclass
class SampleTSPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Args:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length)` or `(batch_size, num_samples, prediction_length, input_size)`):
            Sampled values from the chosen distribution.

    """

    # 该类用于存储时间序列模型的预测输出，包括从所选分布中采样得到的值
    # 声明一个变量 sequences，类型为 torch 的 FloatTensor，初始值为 None
    sequences: torch.FloatTensor = None
# 使用 dataclass 装饰器定义 MaskedImageModelingOutput 类，用于封装掩码图像完成/修补模型的输出结果
@dataclass
class MaskedImageModelingOutput(ModelOutput):
    """
    Base class for outputs of masked image completion / in-painting models.
    掩码图像完成/修补模型输出结果的基类。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Reconstruction loss.
            重建损失，当提供 `bool_masked_pos` 时返回。
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           Reconstructed / completed images.
           重建/完成的图像。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
        when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
            隐藏状态，模型在每个阶段输出的隐藏状态（特征图）元组。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
        `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
            注意力权重，经过注意力 softmax 后的权重，用于计算自注意力头中的加权平均值。
    """

    # 定义 loss 属性，类型为 torch.FloatTensor，可选，表示重建损失，默认为 None
    loss: Optional[torch.FloatTensor] = None

    # 定义 reconstruction 属性，类型为 torch.FloatTensor，表示重建/完成的图像
    reconstruction: torch.FloatTensor = None

    # 定义 hidden_states 属性，类型为 tuple(torch.FloatTensor)，可选，表示隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    # 定义 attentions 属性，类型为 tuple(torch.FloatTensor)，可选，表示注意力权重的元组
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    # @property 装饰器，定义 logits 属性，用于获取输出的最终结果
    @property
    def logits(self):
        # 发出警告，提醒 logits 属性在 Transformers 版本 5 中将被移除，请使用 reconstruction 属性获取最终输出
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回 reconstruction 属性作为最终输出
        return self.reconstruction
```