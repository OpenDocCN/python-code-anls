# `.\models\deprecated\open_llama\modeling_open_llama.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# - 2023 年 EleutherAI 和 HuggingFace Inc. 团队保留所有权利
# - 此代码基于 EleutherAI 的 GPT-NeoX 库和该库中的 GPT-NeoX 和 OPT 实现进行修改
# - 与 Meta AI 团队训练模型使用的 GPT-NeoX 和 OPT 相比，对原始形式进行了轻微的架构差异调整
# 根据 Apache 许可证 2.0 版本授权
# - 仅在符合许可证规定情况下可使用此文件
# - 可获取许可证副本
#    http://www.apache.org/licenses/LICENSE-2.0
# - 未在适用法律规定或书面同意的情况下，分发的软件是基于“按原样”分发的
# - 没有任何担保或条件，不管是明示的还是默示的
# - 请参阅许可证以获取特定语言对权限和限制的说明

""" PyTorch Open-Llama model."""
# 导入所需模块
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块和类
from ....activations import ACT2FN
from ....modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ....modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_open_llama import OpenLlamaConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 尝试导入第三方库 xformers 的 ops 模块，若导入失败则设置为 None
try:
    from xformers import ops as xops
except ImportError:
    xops = None

# 文档用的配置常量
_CONFIG_FOR_DOC = "OpenLlamaConfig"

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->OpenLlama
# 定义 OpenLlamaRMSNorm 类
class OpenLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        OpenLlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        # 定义权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 定义方差的最小值
        self.variance_epsilon = eps

    # 前向传播函数
    def forward(self, hidden_states):
        # 获取输入的数据类型
        input_dtype = hidden_states.dtype
        # 转换隐藏状态的数据类型为 float32
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 根据方差调整隐藏状态
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回调整后的隐藏状态，并根据输入的数据类型转换回原来的数据类��
        return self.weight * hidden_states.to(input_dtype)

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->OpenLlama
# 定义 OpenLlamaRotaryEmbedding 类
class OpenLlamaRotaryEmbedding(nn.Module):
    # 初始化函数，设置模块的维度、最大位置嵌入数，以及基数
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # 调用父类的初始化函数
        super().__init__()

        # 设置模块的维度
        self.dim = dim
        # 设置模块的最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 设置模块的基数
        self.base = base
        # 计算逆频率，注册为缓冲区，并且不进行持久化存储
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建余弦和正弦缓存，以便`torch.jit.trace`函数可以正常工作
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置余弦和正弦缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册余弦缓存，不进行持久化存储
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        # 注册正弦缓存，不进行持久化存储
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # 前向传播函数
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果指定的序列长度大于已缓存的最大序列长度，则重新设置余弦和正弦缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回截取后的余弦和正弦缓存
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
# 从 transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding 复制代码，并将 Llama 替换为 OpenLlama
class OpenLlamaLinearScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    """OpenLlamaRotaryEmbedding 扩展了线性缩放。感谢 Reddit 用户 /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但它使用了不同的置换，以获得相同的计算
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# 从 transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding 复制代码，并将 Llama 替换为 OpenLlama
class OpenLlamaDynamicNTKScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    """OpenLlamaRotaryEmbedding 扩展了动态 NTK 缩放。感谢 Reddit 用户 /u/bloc97 和 /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但它使用了不同的置换，以获得相同的计算
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """旋转输入的一半隐藏的维度"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从 transformers.models.llama.modeling_llama.apply_rotary_pos_emb 复制代码
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
    # 通过position_ids索引获取cos和sin的值，并在指定维度上进行扩展以与q和k的维度匹配
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 应用Rotary Position Embedding于查询和键张量，利用cos和sin作为旋转因子进行加权组合
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 返回应用Rotary Position Embedding后的查询和键张量
    return q_embed, k_embed
```  
# 创建一个名为OpenLlamaMLP的类，继承自nn.Module类
class OpenLlamaMLP(nn.Module):
    # 初始化函数，接受参数hidden_size, intermediate_size, hidden_act和dropout_prob
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dropout_prob: float,
    ):
        # 调用父类nn.Module的初始化函数
        super().__init__()
        # 创建一个线性层，输入大小为hidden_size，输出大小为intermediate_size，无偏置
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 创建一个线性层，输入大小为intermediate_size，输出大小为hidden_size，无偏置
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # 创建一个线性层，输入大小为hidden_size，输出大小为intermediate_size，无偏置
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 根据hidden_act选择激活函数，并将其保存为实例变量
        self.act_fn = ACT2FN[hidden_act]
        # 创建一个Dropout层，以概率dropout_prob丢弃输入
        self.dropout = nn.Dropout(dropout_prob)

    # 前向传播函数，接收输入x
    def forward(self, x):
        # 将输入x通过gate_proj、act_fn、up_proj进行线性变换和激活，再与输入x点乘
        # 然后通过down_proj进行线性变换，并经过dropout层
        out = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # 返回结果
        return self.dropout(out)


# 创建一个名为OpenLlamaAttention的类，继承自nn.Module类
class OpenLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，接受参数config
    def __init__(self, config: OpenLlamaConfig):
        # 调用父类nn.Module的初始化函数
        super().__init__()
        # 将config保存为实例变量
        self.config = config
        # 从config中获取hidden_size、num_attention_heads、max_position_embeddings、attention_dropout_prob
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout_prob = config.attention_dropout_prob

        # 检查hidden_size是否能被num_heads整除，如果不能则抛出错误
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 创建线性层q_proj，将输入大小为hidden_size，输出大小为num_heads * head_dim，无偏��
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # 创建线性层k_proj，将输入大小为hidden_size，输出大小为num_heads * head_dim，无偏置
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # 创建线性层v_proj，将输入大小为hidden_size，输出大小为num_heads * head_dim，无偏置
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # 创建线性层o_proj，将输入大小为num_heads * head_dim，输出大小为hidden_size，无偏置
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # 调用_init_rope函数
        self._init_rope()

    # 从transformers.models.llama.modeling_llama.LlamaAttention._init_rope函数复制而来，将Llama改为OpenLlama
    def _init_rope(self):
        ...
    # 初始化 RoPE（Rotary Positional Encoding）模块
    def _init_rope(self):
        # 如果配置文件中没有指定 RoPE 缩放参数
        if self.config.rope_scaling is None:
            # 使用 OpenLlamaRotaryEmbedding 创建 RoPE 模块
            self.rotary_emb = OpenLlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            # 获取 RoPE 缩放参数类型和缩放因子
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            # 根据 RoPE 缩放参数类型选择不同的 RoPE 模块创建方法
            if scaling_type == "linear":
                # 使用 OpenLlamaLinearScalingRotaryEmbedding 创建 RoPE 模块
                self.rotary_emb = OpenLlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                # 使用 OpenLlamaDynamicNTKScalingRotaryEmbedding 创建 RoPE 模块
                self.rotary_emb = OpenLlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                # 抛出异常，表示未知的 RoPE 缩放类型
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 重塑张量形状的方法，将其转换为适合多头注意力机制的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法，接受输入并完成模型的前向推断
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
class OpenLlamaDecoderLayer(nn.Module):
    def __init__(self, config: OpenLlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OpenLlamaAttention(config=config)  # 创建 OpenLlamaAttention 实例
        self.mlp = OpenLlamaMLP(  # 创建 OpenLlamaMLP 实例
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dropout_prob=config.hidden_dropout_prob,
        )
        self.input_layernorm = OpenLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 创建 OpenLlamaRMSNorm 实例
        self.post_attention_layernorm = OpenLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 创建 OpenLlamaRMSNorm 实例

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

        residual = hidden_states  # 保存输入hidden_states，用于残差连接

        hidden_states = self.input_layernorm(hidden_states)  # 对输入进行 Layer Normalization

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(  # 使用 self_attn 进行自注意力计算
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states  # 残差连接

        # Fully Connected
        residual = hidden_states  # 记录残差连接前的状态
        hidden_states = self.post_attention_layernorm(hidden_states)  # 对输入进行 Layer Normalization
        hidden_states = self.mlp(hidden_states)  # 使用 MLP 进行全连接
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 将 hidden_states 放入 outputs 元组中

        if output_attentions:  # 如果需要输出注意力权重
            outputs += (self_attn_weights,)  # 将注意力权重放入 outputs 元组中

        if use_cache:  # 如果使用缓存
            outputs += (present_key_value,)  # 将 present_key_value 放入 outputs 元组中

        return outputs  # 返回 outputs
    # 这个模型继承自 `PreTrainedModel`。查看超类文档以了解库实现的所有模型的通用方法（如下载或保存、调整输入嵌入、修剪头等）。
    
    # 这个模型也是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。
    # 可以像使用常规 PyTorch 模块一样使用它，并参考 PyTorch 文档了解与一般使用和行为相关的所有事项。
    
    # 参数：
    #     config ([`OpenLlamaConfig`]):
    #         模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只加载配置。
    #         请查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""
# 添加 OpenLlamaModel 的文档字符串
@add_start_docstrings(
    "The bare Open-Llama Model outputting raw hidden-states without any specific head on top.",
    OPEN_LLAMA_START_DOCSTRING,
)
# 定义 OpenLlamaPreTrainedModel 类，并继承 PreTrainedModel
class OpenLlamaPreTrainedModel(PreTrainedModel):
    # 设置 config_class 属性为 OpenLlamaConfig
    config_class = OpenLlamaConfig
    # 设置 base_model_prefix 属性为 "model"
    base_model_prefix = "model"
    # 设置 supports_gradient_checkpointing 属性为 True
    supports_gradient_checkpointing = True
    # 设置 _no_split_modules 属性为 ["OpenLlamaDecoderLayer"]
    _no_split_modules = ["OpenLlamaDecoderLayer"]

    # 定义 _init_weights 方法，用于初始化模型权重
    def _init_weights(self, module):
        # 获取配置中的初始化范围
        std = self.config.initializer_range
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置，则初始化为 zero
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 如果配置中使用了稳定的嵌入
            if self.config.use_stable_embedding:
                # 使用 xavier_normal 初始化权重
                torch.nn.init.xavier_normal_(module.weight.data)
            else:
                # 否则使用正态分布初始化权重
                module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在 padding 索引，则将对应位置的权重初始化为 zero
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# 添加 OpenLlamaModel 的文档字符串
@add_start_docstrings(
    "The bare Open-Llama Model outputting raw hidden-states without any specific head on top.",
    OPEN_LLAMA_START_DOCSTRING,
)
# 定义 OpenLlamaModel 类，并继承 OpenLlamaPreTrainedModel
class OpenLlamaModel(OpenLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OpenLlamaDecoderLayer`]

    Args:
        config: OpenLlamaConfig
    """
    # 定义 __init__ 方法
    def __init__(self, config: OpenLlamaConfig):
        # 调用父类的 __init__ 方法
        super().__init__(config)
        # 设置 padding_idx 属性为配置中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置 vocab_size 属性为配置中的 vocab_size
        self.vocab_size = config.vocab_size

        # 创建词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 如果配置中使用了稳定的嵌入
        if config.use_stable_embedding:
            # 创建 LayerNorm 层
            self.embed_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.embed_layer_norm = None
        # 创建一组 OpenLlamaDecoderLayer 层
        self.layers = nn.ModuleList([OpenLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建 RMSNorm 层
        self.norm = OpenLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 定义 get_input_embeddings 方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 定义 set_input_embeddings 方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 添加 OpenLlamaModel 的文档字符串到 forward 方法
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    # 定义 forward 方法
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
        # 省略了部分方法参数表示
class OpenLlamaForCausalLM(OpenLlamaPreTrainedModel):
    # 初始化模型对象，传入配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 根据配置创建 OpenLlamaModel 模型对象
        self.model = OpenLlamaModel(config)
        # 如果配置中指定共享输入输出嵌入，将语言模型头部置空
        if config.shared_input_output_embedding:
            self.lm_head = None
        else:
            # 否则，根据配置创建一个线性层作为语言模型头部
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用后处理方法，初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出嵌入（语言模型头部）
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入（语言模型头部）
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model

    # 前向传播方法装饰器，添加模型输入的文档字符串，并替换返回值的文档字符串
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
    # 为生成准备输入的方法，接受输入的 ID，过去的键值，注意力掩码，嵌入等作为输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    # 如果过去的键值不为空，则获取过去键值的长度
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[2]

        # 对于一些生成方法，可能只传入最后一个输入 ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # 默认操作是保留最后一个 ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]

    # 检查是否有传入 position_ids 参数，如果没有则根据 attention_mask 创建位置 ID
    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # 如果传入了 inputs_embeds，并且过去键值为空，则使用 inputs_embeds
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
    # 返回 model_inputs
    return model_inputs

# 静态方法：重新排序缓存中的过去键值
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 添加文档字符串和模型头
@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`OpenLlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPEN_LLAMA_START_DOCSTRING,
)
# 创建 OpenLlamaForSequenceClassification 类，继承自 OpenLlamaPreTrainedModel 类
class OpenLlamaForSequenceClassification(OpenLlamaPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 从配置中获取标签数
        self.num_labels = config.num_labels
        # 创建 OpenLlamaModel 模型
        self.model = OpenLlamaModel(config)
        # 创建一个线性层，用于分类
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入嵌入的函数
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入的函数
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 前向传播函数
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
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