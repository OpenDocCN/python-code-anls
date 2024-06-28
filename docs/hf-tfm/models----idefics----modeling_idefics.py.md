# `.\models\idefics\modeling_idefics.py`

```py
# coding=utf-8
# 定义文件编码为UTF-8

# 版权声明和许可证信息，基于Apache License, Version 2.0
# 详细许可证信息可以在http://www.apache.org/licenses/LICENSE-2.0找到

""" PyTorch Idefics model. """
# 导入必要的库和模块
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义的模块和函数
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入IdeficsConfig配置文件
from .configuration_idefics import IdeficsConfig
# 导入IdeficsPerceiverResampler和IdeficsVisionTransformer模块
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置示例
_CONFIG_FOR_DOC = "IdeficsConfig"

# 预训练模型的存档列表
IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "HuggingFaceM4/idefics-9b",
    "HuggingFaceM4/idefics-80b",
    # 查看所有Idefics模型 https://huggingface.co/models?filter=idefics
]

@dataclass
# 带有过去键/值的Idefics模型输出的基类，用于加速顺序解码
class IdeficsBaseModelOutputWithPast(ModelOutput):
    """
    Base class for Idefics model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出隐藏状态序列。

            如果使用了 `past_key_values`，则只输出形状为 `(batch_size, 1, hidden_size)` 的每个序列的最后一个隐藏状态。
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *可选*, 当传入 `use_cache=True` 或 `config.use_cache=True` 时返回):
            长度为 `config.n_layers` 的元组，每个元组包含两个张量，形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)`。

            包含预计算的隐藏状态（自注意力块中的键和值，以及如果 `config.is_encoder_decoder=True` 在交叉注意力块中也包含），
            可用于加速序列解码。
        hidden_states (`tuple(torch.FloatTensor)`, *可选*, 当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            元组的 `torch.FloatTensor`（如果模型具有嵌入层，则为嵌入输出的张量 + 每层的输出张量），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态，加上可选的初始嵌入层输出。
        attentions (`tuple(torch.FloatTensor)`, *可选*, 当传入 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            元组的 `torch.FloatTensor`（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            自注意力头中注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        image_hidden_states (`tuple(torch.FloatTensor)`, *可选*):
            元组的 `torch.FloatTensor`（图像嵌入输出的一个，形状为 `(batch_size, num_images, sequence_length, hidden_size)`）。

            模型通过视觉编码器生成的图像隐藏状态，以及通过感知者生成的图像隐藏状态。
    """

    last_hidden_state: torch.FloatTensor = None  # 初始化最后一个隐藏状态
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 初始化预计算的键和值
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化所有层的隐藏状态
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 初始化注意力权重
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化图像隐藏状态
@dataclass
class IdeficsCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Idefics causal language model (or autoregressive) outputs.

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
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`).

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None  # 初始化为可选的 torch.FloatTensor，用于存储语言模型损失
    logits: torch.FloatTensor = None  # 初始化为 torch.FloatTensor，存储语言模型头部的预测分数（softmax之前）
    past_key_values: Optional[List[torch.FloatTensor]] = None  # 初始化为可选的列表，存储预计算的自注意力块中的键值对
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化为可选的元组，存储模型每层的隐藏状态输出
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 初始化为可选的元组，存储每层的注意力权重
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化为可选的元组，存储视觉编码器产生的图像隐藏状态


def expand_inputs_for_generation(
    input_ids,
    expand_size=1,
    is_encoder_decoder=False,
    attention_mask=None,
    encoder_outputs=None,
    **model_kwargs,
):
    """
    扩展输入以用于生成

    Args:
        input_ids: 输入的 token IDs
        expand_size: 扩展的大小，用于生成的副本数
        is_encoder_decoder: 是否是编码器-解码器结构
        attention_mask: 注意力掩码
        encoder_outputs: 编码器的输出，用于解码器的输入
        **model_kwargs: 其他模型的关键字参数
    """

    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    # 使用索引从输入张量中选择特定的行，更新 input_ids 变量
    input_ids = input_ids.index_select(0, expanded_return_idx)

    # 将像素值添加到模型关键字参数中，如果已存在则保持不变
    model_kwargs["pixel_values"] = model_kwargs.get("pixel_values", None)

    # 将图像编码器嵌入向量添加到模型关键字参数中，如果已存在则保持不变
    model_kwargs["image_encoder_embeddings"] = model_kwargs.get("image_encoder_embeddings", None)

    # 将感知器嵌入向量添加到模型关键字参数中，如果已存在则保持不变
    model_kwargs["perceiver_embeddings"] = model_kwargs.get("perceiver_embeddings", None)

    # 将图像注意力掩码添加到模型关键字参数中，如果已存在则保持不变
    model_kwargs["image_attention_mask"] = model_kwargs.get("image_attention_mask", None)

    # 如果模型关键字参数中存在 'token_type_ids'，则选择特定行更新其对应值
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

    # 如果存在注意力掩码，选择特定行更新模型关键字参数中的 'attention_mask'
    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

    # 如果模型关键字参数中的 'image_attention_mask' 不为 None，则选择特定行更新它
    if model_kwargs["image_attention_mask"] is not None:
        model_kwargs["image_attention_mask"] = model_kwargs["image_attention_mask"].index_select(
            0, expanded_return_idx
        )

    # 如果模型关键字参数中的 'pixel_values' 不为 None，则选择特定行更新它
    if model_kwargs["pixel_values"] is not None:
        model_kwargs["pixel_values"] = model_kwargs["pixel_values"].index_select(0, expanded_return_idx)

    # 否则，如果 'image_encoder_embeddings' 不为 None，则选择特定行更新它
    elif model_kwargs["image_encoder_embeddings"] is not None:
        model_kwargs["image_encoder_embeddings"] = model_kwargs["image_encoder_embeddings"].index_select(
            0, expanded_return_idx
        )

    # 否则，如果 'perceiver_embeddings' 不为 None，则选择特定行更新它
    elif model_kwargs["perceiver_embeddings"] is not None:
        model_kwargs["perceiver_embeddings"] = model_kwargs["perceiver_embeddings"].index_select(
            0, expanded_return_idx
        )

    # 返回更新后的 input_ids 和 model_kwargs
    return input_ids, model_kwargs
def prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # 如果 past_key_values 在 kwargs 中定义，则只使用 input_ids 的最后一个 token
    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # 为批量生成创建动态的 position_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)

    pixel_values = kwargs.get("pixel_values", None)
    image_encoder_embeddings = kwargs.get("image_encoder_embeddings", None)
    perceiver_embeddings = kwargs.get("perceiver_embeddings", None)
    image_attention_mask = kwargs.get("image_attention_mask", None)
    interpolate_pos_encoding = kwargs.get("interpolate_pos_encoding", False)

    # 返回包含所有输入准备数据的字典
    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "image_encoder_embeddings": image_encoder_embeddings,
        "perceiver_embeddings": perceiver_embeddings,
        "image_attention_mask": image_attention_mask,
        "interpolate_pos_encoding": interpolate_pos_encoding,
    }


def freeze_model(model, module_exceptions=[]):
    # 映射常见模块类型到 PyTorch 中对应的类
    mapping = {
        "LayerNorm": nn.LayerNorm,
        "Linear": nn.Linear,
        "Embedding": nn.Embedding,
    }
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    # 遍历模型的所有模块，冻结除了例外模块之外的所有参数
    for module in model.modules():
        if module_exceptions and any(isinstance(module, t) for t in module_exceptions_mapped):
            module.requires_grad_(True)  # 明确将其设置为 True，避免任何错误
        else:
            module.requires_grad_(False)
    return model


class IdeficsDecoupledEmbedding(nn.Embedding):
    # 源自 https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding
    """
    实现参数解耦以允许冻结（或不冻结）嵌入的子集。在实践中，regular `weight` 可以训练或冻结
    （即 `partially_freeze=True`），如果 `num_additional_embeddings > 0`，则会创建
    `num_additional_embeddings` 个额外的始终训练的参数。如果 `num_additional_embeddings=0`，
    则模块默认为 `nn.Embedding` 的常规行为。
    """
    # 初始化函数，用于创建一个新的嵌入层对象
    def __init__(
        self,
        num_embeddings,
        num_additional_embeddings,
        embedding_dim,
        partially_freeze: Optional[bool] = False,
        device=None,
        dtype=None,
        padding_idx=None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_embeddings (`int`):
                Size of the dictionary of embeddings
            num_additional_embeddings (`int`):
                Number of additional embeddings. Only useful when you `partially_freeze=True`.
            embedding_dim (`int`):
                The size of each embedding vector
            partially_freeze: (`bool`, *optional*, defaults to `False`):
                If `True`, the regular `weight` will be frozen. `additional_weight` is never frozen.
            padding_idx (`int`, *optional*):
                The padding index (needs to be less than num_embeddings)

        Note: there are a lot of other parameters to initialize a standard `nn.Embedding` such as `padding_idx`,
        `max_norm` or `norm_type`. We are not supporting these.
        """
        # 检查 padding_idx 是否有效，必须小于 num_embeddings
        if padding_idx is not None and padding_idx > num_embeddings:
            raise ValueError(f"padding_idx must be within num_embeddings. Got {padding_idx} and {num_embeddings}")
        
        # 调用父类 nn.Embedding 的初始化方法，传入大部分参数
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
            padding_idx=padding_idx,
            **kwargs,
        )
        
        # 初始化特定于当前类的成员变量
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.num_additional_embeddings = num_additional_embeddings
        self.partially_freeze = partially_freeze

        # 如果 partially_freeze 为 True，则冻结主要的 weight 参数
        if partially_freeze:
            self.weight.requires_grad_(False)

        # 如果有额外的嵌入向量需求，则创建额外的 nn.Embedding 对象
        if self.num_additional_embeddings > 0:
            self.additional_embedding = nn.Embedding(
                num_embeddings=self.num_additional_embeddings,
                embedding_dim=embedding_dim,
                device=device,
                dtype=dtype,
            )
    def forward(self, input_ids):
        """
        前向传播函数，用于模型的正向计算过程。

        we have 2 embeddings, with different indices - one pretrained self.weight and another
        self.additional_embedding.weight that is being trained.
        我们有两个嵌入层，它们有不同的索引范围：
        - 一个是预训练的 self.weight
        - 另一个是正在训练的 self.additional_embedding.weight

        in order to make a lookup of the input ids, we:
        为了查找输入的 id，我们执行以下步骤：
        1. find out the indices of the entries belonging to the 2nd embedding
        1. 找出属于第二个嵌入层的条目的索引
        2. extract those values while subtracting the size of the first embedding (num_embeddings), since the 2nd
           embedding starts from 0 and not num_embeddings
        2. 提取这些值，同时减去第一个嵌入层的大小（num_embeddings），因为第二个嵌入层的索引从0开始，而不是从num_embeddings开始
        3. perform the 2nd embedding lookup
        3. 执行第二个嵌入层的查找操作
        4. now we handle the 1st embedding, we overwrite indices belonging to the 2nd embedding with a padding index
        4. 现在处理第一个嵌入层，我们用填充索引覆盖属于第二个嵌入层的索引
        5. perform the 1st embedding lookup
        5. 执行第一个嵌入层的查找操作
        6. now we overwrite the values in the 1st embedding lookup with the values of the 2nd embedding lookup
        6. 现在我们用第二个嵌入层查找的值覆盖第一个嵌入层查找的值

        note: for the 1st embedding lookup we could have looked up only the low indices and not do the padding, but
        then we have to create a new tensor and populate it with 2 tensors that are spread out across various indices -
        i.e. not a simple concat - I haven't benchmarked the complex case if it's any faster, given that seqlens are
        usually relatively short it's probably not faster or if faster not by much - but might be a good idea to
        measure.
        注意：对于第一个嵌入层的查找，我们本可以只查找低索引而不进行填充，但那样我们就必须创建一个新的张量，并用两个分散在不同索引上的张量填充它 - 也就是不简单的连接操作 - 我还没有对复杂情况进行基准测试，如果更快的话，鉴于序列长度通常相对较短，可能并不更快，或者如果更快，提升也不会很大 - 但是测量一下可能是个好主意。

        """
        if self.num_additional_embeddings == 0:
            return F.embedding(input_ids, self.weight)

        # Clone so that we don't modify the original input_ids later on
        # 克隆 input_ids，以防后续修改原始输入
        input_ids = input_ids.clone()
        
        # Find indices where input_ids belong to the additional embedding
        # 找到 input_ids 中属于额外嵌入层的索引
        additional_vocab_indices = torch.where(input_ids >= self.num_embeddings)
        
        # Extract input_ids values that belong to the additional vocabulary
        # 提取属于额外词汇表的 input_ids 值
        input_ids_additional_vocab = input_ids[additional_vocab_indices]
        
        # Perform embedding lookup for additional embeddings
        # 执行额外嵌入层的查找
        additional_embeddings = self.additional_embedding(input_ids_additional_vocab - self.num_embeddings)

        # Set indices of additional vocabulary to 0, as these results will be discarded
        # 将额外词汇表的索引设置为0，因为这些结果将被丢弃
        input_ids[additional_vocab_indices] = 0
        
        # Perform embedding lookup for the main embedding (self.weight)
        # 执行主嵌入层（self.weight）的查找
        full_vector = F.embedding(input_ids, self.weight)

        # Overwrite the records with high indices with values from additional embeddings
        # 用额外嵌入层的值覆盖高索引位置的记录
        full_vector[additional_vocab_indices] = additional_embeddings

        return full_vector

    def extra_repr(self) -> str:
        """
        返回模型的额外信息，用于描述模型的属性。

        Returns:
        返回包含模型属性的字符串
        """
        return "num_embeddings={}, num_additional_embeddings={}, embedding_dim={}, partially_freeze={}".format(
            self.num_embeddings,
            self.num_additional_embeddings,
            self.embedding_dim,
            self.partially_freeze,
        )
class IdeficsDecoupledLinear(nn.Linear):
    # 从 https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear 派生而来的类，实现参数的解耦，允许部分参数冻结或训练。
    """
    Implements a decoupling of parameters to allow freezing (or not) a subset of the parameters. In practise, the
    regular `weight` can be trained or frozen (i.e. `partially_freeze=True`), and if `out_additional_features` > 0,
    then it will create `out_additional_features * in_features` additional parameters that are always trained. If
    `out_additional_features=0`, then the module defaults back to the regular behavior of `nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_additional_features: int = 0,
        bias: bool = True,
        partially_freeze: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        out_additional_features: int. Number of additional trainable dimensions. Only makes sense when
        `partially_freeze=True`. partially_freeze: bool. If True, the regular `weight` will be frozen and extra
        parameters (if any) will be trainable. If False, default to the regular behavior of nn.Linear.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        # 初始化自定义参数
        self.out_additional_features = out_additional_features
        self.partially_freeze = partially_freeze

        self.in_features = in_features
        self.out_features = out_features

        # 如果 partially_freeze 为 True，则冻结权重和偏置的梯度
        if partially_freeze:
            self.weight.requires_grad_(False)
            if bias:
                self.bias.requires_grad_(False)

        # 如果有额外的特征维度要训练，则创建额外的线性层 additional_fc
        if out_additional_features > 0:
            self.additional_fc = nn.Linear(
                in_features=in_features,
                out_features=out_additional_features,
                bias=bias,
                device=device,
                dtype=dtype,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行前向传播，计算线性层的输出
        output = F.linear(input, self.weight, self.bias)

        # 如果有额外的特征维度要处理，则将其连接到输出中
        if self.out_additional_features > 0:
            additional_features = self.additional_fc(input)
            output = torch.cat((output, additional_features), -1)

        return output

    def extra_repr(self) -> str:
        """Overwriting `nn.Linear.extra_repr` to include new parameters."""
        # 重写 `nn.Linear.extra_repr` 方法，以包含新的参数信息
        return "in_features={}, out_features={}, out_additional_features={}, bias={}, partially_freeze={}".format(
            self.in_features,
            self.out_features,
            self.out_additional_features,
            self.bias is not None,
            self.partially_freeze,
        )


# this was adapted from LlamaRMSNorm
class IdeficsRMSNorm(nn.Module):
    # 基于 LlamaRMSNorm 进行了适配
    def __init__(self, hidden_size, eps=1e-6):
        """
        IdeficsRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # 初始化权重参数为可训练的张量
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 设置方差 epsilon
        self.variance_epsilon = eps
    # 定义前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 计算每个隐藏状态的方差，并保持维度不变
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 根据方差对隐藏状态进行归一化处理
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是半精度浮点数（float16 或 bfloat16），则将隐藏状态转换为相应的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回加权后的归一化隐藏状态
        return self.weight * hidden_states
# 将 IdeficsRMSNorm 类型对象添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(IdeficsRMSNorm)


# 这是从 LlamaRotaryEmbedding 改编而来的类
class IdeficsEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算频率的倒数，用于位置编码
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使 `torch.jit.trace` 能够正常工作，在这里构建缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置余弦和正弦缓存
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 与论文不同，但使用了不同的排列顺序来获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    """将输入的隐藏维度的一半进行旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从 transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb 复制而来
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """将旋转位置编码应用到查询和键张量上。
    # 通过使用位置索引从余弦向量中选择对应的值，并在指定的维度上进行unsqueeze操作，以便与q和k的形状进行广播。
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    
    # 通过使用位置索引从正弦向量中选择对应的值，并在指定的维度上进行unsqueeze操作，以便与q和k的形状进行广播。
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # 将查询向量q与余弦向量cos相乘并加上查询向量q与正弦向量sin经过rotate_half函数后的乘积，生成旋转后的查询向量。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    # 将键向量k与余弦向量cos相乘并加上键向量k与正弦向量sin经过rotate_half函数后的乘积，生成旋转后的键向量。
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # 返回旋转后的查询向量和键向量组成的元组。
    return q_embed, k_embed
# 这段代码改编自 LlamaMLP
class IdeficsMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        # 定义一个线性层，用于门控投影，输入维度为 hidden_size，输出维度为 intermediate_size，无偏置
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 定义一个线性层，用于下游投影，输入维度为 intermediate_size，输出维度为 hidden_size，无偏置
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # 定义一个线性层，用于上游投影，输入维度为 hidden_size，输出维度为 intermediate_size，无偏置
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 激活函数为根据 hidden_act 参数选择的激活函数，从全局字典 ACT2FN 中获取
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        # 执行前向传播，结合门控投影、激活函数和上游投影，然后通过下游投影得到输出
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# 这段代码改编自 LlamaAttention
class IdeficsAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文中的多头注意力"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
        config: PretrainedConfig = None,
        qk_layer_norms: bool = False,
    ):
        super().__init__()
    ):
        super().__init__()  # 调用父类的初始化方法
        self.hidden_size = hidden_size  # 设置模型的隐藏层大小
        self.num_heads = num_heads  # 设置注意力头的数量
        self.head_dim = hidden_size // num_heads  # 计算每个注意力头的维度
        self.dropout = dropout  # 设置dropout的比例
        self.is_causal = True  # 设定是否是因果注意力机制

        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )  # 检查隐藏层大小是否能够被注意力头数量整除，如果不能则抛出数值错误异常

        self.is_cross_attention = is_cross_attention  # 标记是否是交叉注意力

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ValueError("this model requires pytorch 2.0 or higher")  # 检查是否支持所需的PyTorch版本

        if self.is_cross_attention:
            kv_input_dim = (
                self.hidden_size if not hasattr(config.vision_config, "embed_dim") else config.vision_config.embed_dim
            )
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )  # 创建查询投影层
            self.k_proj = nn.Linear(kv_input_dim, num_heads * self.head_dim, bias=False)  # 创建键投影层
            self.v_proj = nn.Linear(
                kv_input_dim,
                num_heads * self.head_dim,
                bias=False,
            )  # 创建值投影层
        else:
            self.q_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )  # 创建查询投影层
            self.k_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )  # 创建键投影层
            self.v_proj = nn.Linear(
                self.hidden_size,
                num_heads * self.head_dim,
                bias=False,
            )  # 创建值投影层
        self.o_proj = nn.Linear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )  # 创建输出投影层

        self.rotary_emb = IdeficsEmbedding(self.head_dim)  # 创建旋转嵌入层对象

        self.qk_layer_norms = qk_layer_norms  # 设置是否进行查询和键的层标准化

        if self.qk_layer_norms:
            self.q_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 创建查询层标准化对象
            self.k_layer_norm = IdeficsRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 创建键层标准化对象

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 将张量重塑为(batch_size, sequence_length, num_heads, head_dim)，并转置维度以符合注意力机制的需求

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
# this was adapted from LlamaDecoderLayer
# 定义一个名为 IdeficsDecoderLayer 的类，继承自 nn.Module
class IdeficsDecoderLayer(nn.Module):
    def __init__(self, config: IdeficsConfig):
        super().__init__()
        # 初始化隐藏层大小
        self.hidden_size = config.hidden_size
        # 创建自注意力层对象，使用配置中的参数进行初始化
        self.self_attn = IdeficsAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            config=config,
        )
        # 创建MLP对象，使用配置中的参数进行初始化
        self.mlp = IdeficsMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 创建输入层归一化对象，使用配置中的隐藏大小和RMS归一化的epsilon参数进行初始化
        self.input_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 创建注意力后归一化对象，使用配置中的隐藏大小和RMS归一化的epsilon参数进行初始化
        self.post_attention_layernorm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 设置Dropout概率，使用配置中的dropout参数
        self.dropout = config.dropout

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states  # 保留输入 hidden_states 的原始值，用于后续残差连接

        hidden_states = self.input_layernorm(hidden_states)  # 使用层归一化对输入进行归一化处理

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对输出进行 dropout 处理
        hidden_states = residual + hidden_states  # 残差连接：将归一化前的输入与经过 self attention 和 dropout 处理后的输出相加

        # Fully Connected
        residual = hidden_states  # 保留上一步操作后的值，用于后续残差连接
        hidden_states = self.post_attention_layernorm(hidden_states)  # 使用层归一化对输出进行归一化处理
        hidden_states = self.mlp(hidden_states)  # 使用全连接层进行线性变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对输出进行 dropout 处理
        hidden_states = residual + hidden_states  # 残差连接：将归一化后的输出与经过 MLP 和 dropout 处理后的输出相加

        outputs = (hidden_states,)  # 将处理后的 hidden_states 放入输出元组中

        if output_attentions:
            outputs += (self_attn_weights,)  # 如果需要输出 attention 权重，则将 self_attn_weights 放入输出元组中

        if use_cache:
            outputs += (present_key_value,)  # 如果需要使用缓存的过去键值状态，则将 present_key_value 放入输出元组中

        return outputs  # 返回包含处理后的结果的元组
# 定义自定义的 gated cross-attention 层，继承自 nn.Module
class IdeficsGatedCrossAttentionLayer(nn.Module):
    # 前向传播方法定义，接收多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力遮罩张量
        image_hidden_states: Optional[torch.Tensor] = None,  # 可选的图像隐藏状态张量
        image_attention_mask: Optional[torch.Tensor] = None,  # 可选的图像注意力遮罩张量
        cross_attention_gate: Optional[torch.Tensor] = None,  # 可选的交叉注意力门控张量
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重的标志
        use_cache: Optional[bool] = False,  # 是否使用缓存的标志
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的过去的键值对元组
LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`IdeficsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
# 定义一个预训练模型类，继承自 PreTrainedModel
class IdeficsPreTrainedModel(PreTrainedModel):
    config_class = IdeficsConfig  # 使用 IdeficsConfig 类作为配置类
    base_model_prefix = "model"  # 基础模型前缀为 "model"
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["IdeficsDecoderLayer", "IdeficsGatedCrossAttentionLayer"]  # 不拆分的模块列表
    _supports_sdpa = True  # 支持自动分配并行性加速（Self-Delegated Parallelism Acceleration, SDPA）

    def _init_weights(self, module):
        # 重要提示：这个 Idefics 的移植版本不适用于从头训练，只能用于推理和微调
        # 因此，初始化权重的正确代码已被删除。m4 代码库应该用于从头训练，并包含正确的代码。
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):  # 如果是线性层
            module.weight.data.normal_(mean=0.0, std=std)  # 权重初始化为正态分布
            if module.bias is not None:
                module.bias.data.zero_()  # 如果存在偏置，将其初始化为零
        elif isinstance(module, nn.Embedding):  # 如果是嵌入层
            module.weight.data.normal_(mean=0.0, std=std)  # 权重初始化为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果存在填充索引，将其初始化为零

    # 从 transformers.modeling_utils.PreTrainedModel._check_and_enable_sdpa 适配而来
    @classmethod
    # 定义一个类方法 `_check_and_enable_sdpa`，用于检查并启用 SDPA 注意力机制配置
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        # 检查是否启用了 `use_bettertransformer` 属性，用于决定是否返回原始配置
        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        # 如果不仅仅是进行硬性检查，设置注意力实现方式为 "sdpa"
        if not hard_check_only:
            config._attn_implementation = "sdpa"
        # 返回修改后的配置对象
        return config
# 定义一个多行字符串，用于文档化LLaMA输入的说明文档
LLAMA_INPUTS_DOCSTRING = r"""
"""

# 使用装饰器为IdeficsModel类添加文档字符串，在输出原始隐藏状态时不添加特定的顶部头信息
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
# 定义IdeficsModel类，继承自IdeficsPreTrainedModel类
class IdeficsModel(IdeficsPreTrainedModel):
    """
    Transformer解码器，由`config.num_hidden_layers`层组成。每一层是一个[`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    def __init__(self, config: IdeficsConfig):
        # 调用父类的构造函数进行初始化
        super().__init__(config)
        # 将config参数保存在实例变量中
        self.config = config
        # 设置填充索引为config中定义的pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置词汇表大小为config中定义的vocab_size
        self.vocab_size = config.vocab_size

        # 创建IdeficsDecoupledEmbedding实例，并保存在embed_tokens实例变量中
        self.embed_tokens = IdeficsDecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            padding_idx=self.padding_idx,
        )

        # 设置图像尺寸和视觉配置，从config参数中获取
        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        # 创建IdeficsVisionTransformer实例，并保存在vision_model实例变量中
        self.vision_model = IdeficsVisionTransformer(config.vision_config)

        # 如果config中设置了使用resampler，则创建IdeficsPerceiverResampler实例，并保存在perceiver_resampler实例变量中
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = IdeficsPerceiverResampler(
                config,
                config.vision_config.embed_dim,
                perceiver_config.resampler_depth,
                perceiver_config.resampler_n_heads,
                perceiver_config.resampler_head_dim,
                perceiver_config.resampler_n_latents,
            )

        # 创建包含config.num_hidden_layers个IdeficsDecoderLayer实例的模块列表，并保存在layers实例变量中
        self.layers = nn.ModuleList([IdeficsDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 设置跨层间隔为config中定义的cross_layer_interval
        self.cross_layer_interval = config.cross_layer_interval
        # 计算跨层注意力层的数量
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        # 创建包含num_cross_layers个IdeficsGatedCrossAttentionLayer实例的模块列表，并保存在gated_cross_attn_layers实例变量中
        self.gated_cross_attn_layers = nn.ModuleList(
            [IdeficsGatedCrossAttentionLayer(config) for _ in range(num_cross_layers)]
        )
        # 设置梯度检查点标志为False
        self.gradient_checkpointing = False

        # 创建IdeficsRMSNorm实例，并保存在norm实例变量中
        self.norm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 初始化权重并进行最终处理
        self.post_init()

        # 冻结相关参数
        self.freeze_relevant_params(config)

    # 方法：冻结相关参数
    def freeze_relevant_params(self, config=None):
        if config is None:
            config = self.config

        # 如果配置中指定冻结文本层，则调用freeze_text_layers方法冻结相关模块
        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)

        # 如果配置中指定冻结视觉层，则调用freeze_vision_layers方法冻结视觉模型
        if config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    # 方法：冻结文本层
    def freeze_text_layers(self, module_exceptions=[]):
        # 遍历self.layers和self.norm列表中的模块，调用freeze_model函数冻结指定模块
        for module in [self.layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    # 方法：冻结视觉层
    def freeze_vision_layers(self, module_exceptions=[]):
        # 调用freeze_model函数冻结self.vision_model中指定的模块
        freeze_model(self.vision_model, module_exceptions=module_exceptions)

    # 方法：获取输入嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens
    # 设置模型的输入嵌入表示
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 在模型前向传播过程中添加注释到模型文档的装饰器
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token IDs，类型为LongTensor
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选的Tensor类型
        position_ids: Optional[torch.LongTensor] = None,  # 位置IDs，可选的LongTensor类型
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，可选的浮点数张量列表
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，可选的浮点数张量
        pixel_values: Optional[torch.FloatTensor] = None,  # 像素值，可选的浮点数张量
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,  # 图像编码器嵌入，可选的浮点数张量
        perceiver_embeddings: Optional[torch.FloatTensor] = None,  # 感知器嵌入，可选的浮点数张量
        image_attention_mask: Optional[torch.Tensor] = None,  # 图像注意力遮罩，可选的Tensor类型
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔类型
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔类型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔类型
        interpolate_pos_encoding: Optional[bool] = False,  # 是否插值位置编码，布尔类型，默认为False
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选的布尔类型
    class IdeficsForVisionText2Text(IdeficsPreTrainedModel):
        # 在加载时需要忽略的键列表，用于处理缺失情况
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
        # 要绑定权重的键列表，指定需要共享权重的模型参数
        _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]

        def __init__(self, config, vision_model=None):
            # 调用父类的初始化方法，传入配置参数
            super().__init__(config)
            # 使用给定的配置参数初始化 IdeficsModel 模型
            self.model = IdeficsModel(config)

            # 使用 IdeficsDecoupledLinear 初始化 lm_head 层
            self.lm_head = IdeficsDecoupledLinear(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                out_additional_features=config.additional_vocab_size,
                bias=False,
                partially_freeze=config.freeze_lm_head,
            )

            # 执行初始化权重并进行最终处理
            self.post_init()

        def get_input_embeddings(self):
            # 返回模型的 embed_tokens 层，用于输入嵌入
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            # 设置模型的 embed_tokens 层，用于输入嵌入
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            # 返回 lm_head 层，用于输出嵌入
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            # 设置 lm_head 层，用于输出嵌入
            self.lm_head = new_embeddings

        def set_decoder(self, decoder):
            # 设置模型的 decoder 层
            self.model = decoder

        def get_decoder(self):
            # 返回模型的 decoder 层
            return self.model

        def tie_weights(self):
            """
            重写 `transformers.modeling_utils.PreTrainedModel.tie_weights` 方法，
            处理 IdeficsDecoupledLinear 和 IdeficsDecoupledEmbedding 的情况。
            """
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()

            # 如果配置允许绑定词嵌入，则将输出嵌入的权重设置为输入嵌入的权重
            if getattr(self.config, "tie_word_embeddings", True):
                output_embeddings.weight = input_embeddings.weight
                # 如果存在额外的嵌入，则也绑定额外的嵌入权重
                if input_embeddings.num_additional_embeddings > 0:
                    assert output_embeddings.out_additional_features == input_embeddings.num_additional_embeddings
                    output_embeddings.additional_fc.weight = input_embeddings.additional_embedding.weight

            # 更新输出嵌入的特征数和额外特征数，以匹配输入嵌入的数目
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings
                if hasattr(output_embeddings, "out_additional_features") and hasattr(
                    input_embeddings, "num_additional_embeddings"
                ):
                    output_embeddings.out_additional_features = input_embeddings.num_additional_embeddings

        @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=IdeficsCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于处理前向推断过程中的输入数据
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID序列，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量，默认为None
        position_ids: Optional[torch.LongTensor] = None,  # 可选的位置ID张量，默认为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 可选的过去键值对列表，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的嵌入输入张量，默认为None
        pixel_values: Optional[torch.FloatTensor] = None,  # 可选的像素值张量，默认为None
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,  # 可选的图像编码器嵌入张量，默认为None
        perceiver_embeddings: Optional[torch.FloatTensor] = None,  # 可选的感知器嵌入张量，默认为None
        image_attention_mask: Optional[torch.Tensor] = None,  # 可选的图像注意力掩码张量，默认为None
        labels: Optional[torch.LongTensor] = None,  # 可选的标签张量，默认为None
        use_cache: Optional[bool] = None,  # 可选的缓存使用标志，默认为None
        output_attentions: Optional[bool] = None,  # 可选的输出注意力张量，默认为None
        output_hidden_states: Optional[bool] = None,  # 可选的输出隐藏状态标志，默认为None
        interpolate_pos_encoding: Optional[bool] = False,  # 可选的位置编码插值标志，默认为False
        return_dict: Optional[bool] = None,  # 可选的返回字典标志，默认为None
    ):
        # 定义一个方法用于准备生成过程中的输入数据
        def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
            # 从kwargs中获取image_hidden_states参数，如果存在的话
            image_hidden_states = kwargs.pop("image_hidden_states", None)
            if image_hidden_states is not None:
                # 如果配置中使用resampler，则将perceiver_embeddings设置为image_hidden_states，否则设置为None
                if self.config.use_resampler:
                    kwargs["perceiver_embeddings"] = image_hidden_states
                else:
                    kwargs["image_encoder_embeddings"] = image_hidden_states
                kwargs["pixel_values"] = None  # 将像素值设置为None
            # 调用准备生成输入数据的函数，传递input_ids、past以及其他未处理的kwargs参数
            inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
            unwanted_kwargs = ["token_type_ids"]  # 定义一个不需要的kwargs参数列表
            for kwarg in unwanted_kwargs:
                inputs.pop(kwarg, None)  # 从inputs中移除不需要的kwargs参数
            return inputs  # 返回处理后的inputs字典

        @staticmethod
        def _expand_inputs_for_generation(
            *args,
            **model_kwargs,
        ):
            # 调用扩展生成输入数据的函数，传递args和model_kwargs参数
            return expand_inputs_for_generation(*args, **model_kwargs)

        # 定义一个方法，用于生成过程中更新模型关键字参数
        def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,  # 输出模型的结果
            model_kwargs: Dict[str, Any],  # 模型关键字参数的字典
            is_encoder_decoder: bool = False,  # 是否是编码器-解码器结构，默认为False
            standardize_cache_format: bool = False,  # 是否标准化缓存格式，默认为False
        ) -> Dict[str, Any]:  # 返回更新后的模型关键字参数的字典
            # 调用父类的更新模型关键字参数函数，传递outputs、model_kwargs、is_encoder_decoder和standardize_cache_format参数
            model_kwargs = super()._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder,
                standardize_cache_format,
            )

            # 如果model_kwargs中包含'image_attention_mask'键
            if "image_attention_mask" in model_kwargs:
                image_attention_mask = model_kwargs["image_attention_mask"]
                # 取图像注意力掩码的最后一个mask并添加一个维度
                last_mask = image_attention_mask[:, -1, :].unsqueeze(1)
                model_kwargs["image_attention_mask"] = last_mask  # 更新模型关键字参数中的'image_attention_mask'为最后一个mask

            # 获取预计算的image_hidden_states并添加到模型关键字参数中
            model_kwargs["image_hidden_states"] = outputs.image_hidden_states
            return model_kwargs  # 返回更新后的模型关键字参数的字典

        @staticmethod
        def _reorder_cache(past, beam_idx):
            reordered_past = ()
            # 遍历每一层的过去状态，并按beam_idx重新排序
            for layer_past in past:
                reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
            return reordered_past  # 返回重新排序后的过去状态
```