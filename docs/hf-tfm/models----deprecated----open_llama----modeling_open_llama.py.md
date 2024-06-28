# `.\models\deprecated\open_llama\modeling_open_llama.py`

```py
# coding=utf-8
# 版权 2023 年 EleutherAI 和 HuggingFace Inc. 团队保留所有权利。
#
# 此代码基于 EleutherAI 的 GPT-NeoX 库以及此库中的 GPT-NeoX 和 OPT 实现进行了修改，
# 以适应与 Meta AI 团队训练的模型相比的轻微架构差异。
#
# 根据 Apache 许可证 2.0 版（“许可证”）授权；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch Open-Llama 模型。"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ....activations import ACT2FN  # 导入激活函数映射
from ....modeling_attn_mask_utils import _prepare_4d_causal_attention_mask  # 导入注意力掩码相关的工具函数
from ....modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast  # 导入模型输出类
from ....modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 导入工具函数和日志模块
from .configuration_open_llama import OpenLlamaConfig  # 导入 OpenLlama 的配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

try:
    from xformers import ops as xops  # 尝试导入 xformers 库的操作模块
except ImportError:
    xops = None


_CONFIG_FOR_DOC = "OpenLlamaConfig"  # 文档中使用的配置名称


# 从 transformers.models.llama.modeling_llama.LlamaRMSNorm 复制而来，将 Llama 改为 OpenLlama
class OpenLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        OpenLlamaRMSNorm 等效于 T5LayerNorm。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化权重参数
        self.variance_epsilon = eps  # 方差的小值防止除零错误

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype  # 记录输入张量的数据类型
        hidden_states = hidden_states.to(torch.float32)  # 张量转换为 float32 类型
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 计算方差
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 标准化
        return self.weight * hidden_states.to(input_dtype)  # 返回加权标准化后的张量


# 从 transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding 复制而来，将 Mistral 改为 OpenLlama
class OpenLlamaRotaryEmbedding(nn.Module):
    # 初始化函数，设置模型的维度、最大位置嵌入长度、基础值和设备
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类初始化方法
        super().__init__()

        # 设置模型的维度
        self.dim = dim
        # 设置最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 设置基础值
        self.base = base

        # 计算频率的倒数，根据维度生成一维向量，然后转换为指定设备的数据类型
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        # 将频率的倒数注册为缓冲张量，不持久化保存
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建余弦和正弦缓存，以便 `torch.jit.trace` 正常工作。
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 记录已缓存的最大序列长度
        self.max_seq_len_cached = seq_len
        # 生成从0到最大序列长度的整数张量，类型与频率的倒数相同
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率张量的外积
        freqs = torch.outer(t, self.inv_freq)
        # 拼接余弦和正弦张量，沿着最后一个维度
        emb = torch.cat((freqs, freqs), dim=-1)
        # 将余弦和正弦张量注册为缓冲张量，并转换为指定数据类型
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播函数，输入张量 x 和序列长度 seq_len
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果输入的序列长度大于当前缓存的最大序列长度
        if seq_len > self.max_seq_len_cached:
            # 更新余弦和正弦缓存
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回更新后的余弦和正弦缓存，转换为输入张量 x 的数据类型
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从transformers.models.falcon.modeling_falcon.FalconLinearScalingRotaryEmbedding复制并将Falcon->OpenLlama
class OpenLlamaLinearScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    """OpenLlamaRotaryEmbedding扩展了线性缩放。鸣谢Reddit用户/u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # 创建一个torch tensor，其内容为从0到self.max_seq_len_cached-1的整数，设备为device，数据类型为torch.int64，并且根据self.scaling_factor缩放
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        # 创建一个形状为(seq_len, dim)的频率矩阵，外积操作
        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册缓冲区"cos_cached"和"sin_cached"，分别存储emb的余弦和正弦值，转换为dtype类型，并且是非持久性缓冲区
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从transformers.models.falcon.modeling_falcon.FalconDynamicNTKScalingRotaryEmbedding复制并将Falcon->OpenLlama
class OpenLlamaDynamicNTKScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    """OpenLlamaRotaryEmbedding扩展了动态NTK缩放。鸣谢Reddit用户/u/bloc97和/u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        # 如果seq_len大于max_position_embeddings，计算新的base和inv_freq
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 创建一个torch tensor，其内容为从0到self.max_seq_len_cached-1的整数，设备为device，数据类型为torch.int64
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 创建一个形状为(seq_len, dim)的频率矩阵，外积操作
        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但使用不同的排列方式以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册缓冲区"cos_cached"和"sin_cached"，分别存储emb的余弦和正弦值，转换为dtype类型，并且是非持久性缓冲区
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """旋转输入的一半隐藏维度。"""
    # 将输入张量x按最后一个维度切片为两部分，第一部分为前一半，第二部分为后一半，然后按最后一个维度连接起来
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb复制
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # Unsqueezes the cosine embeddings along the specified dimension to match the shape of q and k
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # Unsqueezes the sine embeddings along the specified dimension to match the shape of q and k
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # Applies rotary position embedding to the query tensor q
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # Applies rotary position embedding to the key tensor k
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OpenLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dropout_prob: float,
    ):
        """Initialize the OpenLlamaMLP module.

        Args:
            hidden_size (int): The size of the input and output hidden layers.
            intermediate_size (int): The size of the intermediate layer.
            hidden_act (str): The activation function to be used in the hidden layers.
            dropout_prob (float): The dropout probability for regularization.
        """
        super().__init__()
        # Linear transformation for the gating mechanism
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Linear transformation for the downsampling projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # Linear transformation for the upsampling projection
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Activation function for the hidden layers
        self.act_fn = ACT2FN[hidden_act]
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Applies gating projection, activation function, and upsampling projection to input x,
        # then applies downsampling projection and returns the result after dropout
        out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout(out)


class OpenLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化函数，接收一个OpenLlamaConfig类型的参数config
    def __init__(self, config: OpenLlamaConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量self.config中
        self.config = config
        # 从配置对象中获取隐藏层大小，并保存到self.hidden_size中
        self.hidden_size = config.hidden_size
        # 从配置对象中获取注意力头数，并保存到self.num_heads中
        self.num_heads = config.num_attention_heads
        # 根据隐藏层大小和注意力头数计算每个头的维度，并保存到self.head_dim中
        self.head_dim = self.hidden_size // self.num_heads
        # 从配置对象中获取最大位置嵌入的大小，并保存到self.max_position_embeddings中
        self.max_position_embeddings = config.max_position_embeddings
        # 从配置对象中获取注意力机制的dropout概率，并保存到self.dropout_prob中
        self.dropout_prob = config.attention_dropout_prob

        # 检查隐藏层大小是否能被注意力头数整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            # 若不能整除则抛出数值错误异常
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 初始化查询（query）、键（key）、值（value）、输出（output）的线性投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 调用内部初始化函数_init_rope()
        self._init_rope()

    # 从transformers.models.llama.modeling_llama.LlamaAttention._init_rope复制，并替换Llama为OpenLlama
    # 初始化RoPE（Rotary Positional Encoding，旋转位置编码）的嵌入
    def _init_rope(self):
        # 如果配置对象中的rope_scaling为None，则使用OpenLlamaRotaryEmbedding
        if self.config.rope_scaling is None:
            self.rotary_emb = OpenLlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 否则根据配置选择不同的RoPE嵌入方式
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = OpenLlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = OpenLlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 如果配置中的RoPE类型未知，则抛出值错误异常
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 将输入的tensor重新形状为(batch_size, seq_len, num_heads, head_dim)，并进行维度转置
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数定义，接收输入的隐藏状态张量和其他可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        ):
class OpenLlamaDecoderLayer(nn.Module):
    def __init__(self, config: OpenLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OpenLlamaAttention(config=config)
        self.mlp = OpenLlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dropout_prob=config.hidden_dropout_prob,
        )
        self.input_layernorm = OpenLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OpenLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

        residual = hidden_states

        # Layer normalization on the input states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention mechanism
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # Residual connection
        hidden_states = residual + hidden_states

        # Fully Connected Feedforward Network
        residual = hidden_states
        # Layer normalization after the attention mechanism
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Multilayer Perceptron transformation
        hidden_states = self.mlp(hidden_states)
        # Residual connection
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPEN_LLAMA_START_DOCSTRING = r"""
    # 这个模型继承自`PreTrainedModel`。查看超类文档可以了解到库实现的通用方法，例如下载或保存模型、调整输入嵌入大小、修剪头部等。

    # 这个模型也是一个PyTorch的`torch.nn.Module`子类。可以像使用常规PyTorch模块一样使用它，并参考PyTorch文档了解有关一般用法和行为的所有内容。

    # 参数：
    # config ([`OpenLlamaConfig`]):
    #     包含模型所有参数的模型配置类。使用配置文件初始化模型不会加载与模型关联的权重，仅加载配置。查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""
"""


@add_start_docstrings(
    "The bare Open-Llama Model outputting raw hidden-states without any specific head on top.",
    OPEN_LLAMA_START_DOCSTRING,
)
class OpenLlamaPreTrainedModel(PreTrainedModel):
    # 使用特定的文档字符串为类添加描述
    config_class = OpenLlamaConfig
    # 模型中基础模型的名称前缀
    base_model_prefix = "model"
    # 支持梯度检查点的标志
    supports_gradient_checkpointing = True
    # 不需要分割的模块名称列表
    _no_split_modules = ["OpenLlamaDecoderLayer"]

    def _init_weights(self, module):
        # 初始化模型权重的函数
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if self.config.use_stable_embedding:
                torch.nn.init.xavier_normal_(module.weight.data)
            else:
                module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OPEN_LLAMA_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare Open-Llama Model outputting raw hidden-states without any specific head on top.",
    OPEN_LLAMA_START_DOCSTRING,
)
class OpenLlamaModel(OpenLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OpenLlamaDecoderLayer`]

    Args:
        config: OpenLlamaConfig
    """

    def __init__(self, config: OpenLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 词嵌入层，根据配置初始化不同的稳定或非稳定嵌入方式
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.use_stable_embedding:
            self.embed_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.embed_layer_norm = None
        # 使用配置中的层数初始化解码器层列表
        self.layers = nn.ModuleList([OpenLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 使用自定义的 RMS 标准化器
        self.norm = OpenLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # OpenLlamaForCausalLM 类的 forward 方法尚未完全添加
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 使用配置对象创建一个 OpenLlamaModel 实例，并赋值给 self.model
        self.model = OpenLlamaModel(config)
        # 根据配置参数决定是否创建 lm_head 层
        if config.shared_input_output_embedding:
            self.lm_head = None  # 如果配置中设置共享输入输出嵌入，则 lm_head 为 None
        else:
            # 否则，创建一个线性层，将隐藏大小为 config.hidden_size，输出大小为 config.vocab_size，无偏置
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回模型的输入嵌入层
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置模型的输入嵌入层
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回模型的输出嵌入层（lm_head）
    def get_output_embeddings(self):
        return self.lm_head

    # 设置模型的输出嵌入层（lm_head）
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置模型的解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 返回模型的解码器
    def get_decoder(self):
        return self.model

    # 前向传播函数，用装饰器添加了文档字符串和返回值替换
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播函数具体实现在后续的方法和类中定义，此处不直接实现

    # 为生成准备输入的方法，接受多个参数，包括 input_ids, past_key_values, attention_mask, inputs_embeds
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 具体的输入准备过程在后续的方法和类中定义，此处不直接实现
    ):
        # 如果之前的键值对不为 None，则获取其长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                # 移除前缀的长度设为过去长度
                remove_prefix_length = past_length
            else:
                # 默认旧行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 更新输入 IDs 为移除前缀后的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 从 kwargs 中获取位置 IDs
        position_ids = kwargs.get("position_ids", None)
        # 如果存在注意力遮罩但没有位置 IDs
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建位置 IDs
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                # 如果存在过去的键值对，则仅保留与输入 IDs 形状相匹配的位置 IDs
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了 `inputs_embeds`，则仅在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 重新排序过去的键值对
        for layer_past in past_key_values:
            reordered_past += (
                # 对每个层的过去状态根据 beam_idx 重新排序
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 使用装饰器添加文档字符串到类上，描述LLaMa模型带有顶部序列分类头部的转换器
# 这里是使用了OpenLlamaForSequenceClassification类，它在顺序分类时使用最后一个token，类似于其他因果模型（如GPT-2）的做法。

# 根据配置的pad_token_id，确定最后一个非填充token的位置以进行分类。如果未定义pad_token_id，则直接取批次中每行的最后一个值。
# 当传入inputs_embeds而非input_ids时，无法猜测填充token，因此同样采用这种方式（取批次中每行的最后一个值）。

class OpenLlamaForSequenceClassification(OpenLlamaPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数，传入配置
        super().__init__(config)
        # 设置类别数目
        self.num_labels = config.num_labels
        # 初始化OpenLlama模型
        self.model = OpenLlamaModel(config)
        # 定义线性层，将隐藏状态映射到类别数目，无偏置
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 使用装饰器添加文档字符串到模型前向传播函数上，描述输入参数的文档字符串
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```