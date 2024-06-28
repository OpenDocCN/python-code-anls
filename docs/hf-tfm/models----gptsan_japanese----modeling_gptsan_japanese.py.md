# `.\models\gptsan_japanese\modeling_gptsan_japanese.py`

```
# 编码声明，指定文件编码为UTF-8
# Copyright声明及许可信息，指明版权归属及许可协议
# 引入模块：copy模块用于复制对象；List, Optional, Tuple, Union用于类型提示

import copy  # 引入copy模块，用于对象复制
from typing import List, Optional, Tuple, Union  # 引入类型提示

import torch  # 引入PyTorch库
import torch.nn as nn  # 引入PyTorch的神经网络模块

# 引入相关模块和函数
from ...activations import ACT2FN  # 从本地模块引入ACT2FN函数
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions  # 从本地模块引入模型输出相关类
from ...modeling_utils import PreTrainedModel  # 从本地模块引入PreTrainedModel类
from ...utils import (  # 从本地模块引入多个函数和常量
    DUMMY_INPUTS,  # 引入DUMMY_INPUTS常量
    DUMMY_MASK,  # 引入DUMMY_MASK常量
    add_start_docstrings,  # 引入add_start_docstrings函数
    add_start_docstrings_to_model_forward,  # 引入add_start_docstrings_to_model_forward函数
    is_torch_fx_proxy,  # 引入is_torch_fx_proxy函数
    logging,  # 引入logging模块
)
from .configuration_gptsan_japanese import GPTSanJapaneseConfig  # 从本地模块引入GPTSanJapaneseConfig类

logger = logging.get_logger(__name__)  # 获取当前模块的logger对象

_CONFIG_FOR_DOC = "GPTSanJapaneseConfig"  # 模型配置文档的名称
_CHECKPOINT_FOR_DOC = "Tanrei/GPTSAN-japanese"  # 预训练模型的检查点名称

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
# 预训练模型的ID和关联的URL列表
GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Tanrei/GPTSAN-japanese",  # 预训练模型的名称和路径
    # 更多预训练模型详见https://huggingface.co/models?filter=gptsan-japanese
]


# 从transformers.models.switch_transformers.modeling_switch_transformers.router_z_loss_func中复制的函数
def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    计算PyTorch中实现的路由器z-loss。

    Args:
        router_logits (`float`):
            输入的logits张量，形状为 [batch_size, sequence_length, num_experts]

    Returns:
        标量路由器z-loss。
    """
    num_groups, tokens_per_group, _ = router_logits.shape  # 获取logits张量的形状信息
    log_z = torch.logsumexp(router_logits, dim=-1)  # 计算log-sum-exp并存储在log_z中
    z_loss = log_z**2  # 计算z-loss
    return torch.sum(z_loss) / (num_groups * tokens_per_group)  # 返回平均z-loss


# 从transformers.models.switch_transformers.modeling_switch_transformers.load_balancing_loss_func中复制的函数
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    计算PyTorch中实现的辅助负载平衡损失。

    Args:
        router_probs (`torch.Tensor`):
            路由概率张量，形状为 [batch_size, sequence_length, num_experts]
        expert_indices (`torch.Tensor`):
            专家索引张量，形状为 [batch_size, sequence_length]

    Returns:
        辅助负载平衡损失的标量。
    """
    num_groups, tokens_per_group, _ = router_probs.shape  # 获取路由概率张量的形状信息
    log_prob = torch.log(router_probs)  # 计算路由概率的对数并存储在log_prob中
    lb_loss = -torch.sum(log_prob * expert_indices) / (num_groups * tokens_per_group)  # 计算负载平衡损失
    return lb_loss  # 返回负载平衡损失值
    # 获取路由概率张量的最后一个维度大小，即专家数量
    num_experts = router_probs.shape[-1]

    # 将专家索引张量转换为 int64 类型，以便进行 one-hot 编码
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    # 如果专家索引张量的维度为 2，则添加一个维度来适应 one-hot 编码的需求
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    # 使用 one-hot 编码创建专家掩码张量
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # 对于每个 token，确定其是否被路由到特定的专家
    expert_mask = torch.max(expert_mask, axis=-2).values

    # 将专家掩码张量转换为 float32 类型，以便计算平均值
    expert_mask = expert_mask.to(torch.float32)

    # 计算每个组和专家的平均 token 数量
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    # 计算每个组和专家的路由概率的平均值
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)

    # 计算辅助损失，这是平均 token 数量和路由概率的乘积的平均值，乘以专家数量的平方
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)
class GPTSanJapaneseDenseActDense(nn.Module):
    """
    FFN Layer for Switch Transformer and Extra layers

    GPTSAN can mix Switch Transformer layers and normal Transformer layers This class is used as Expert in Switch
    Transformer layers and as FFN in regular Transformer layers. RELU is used in the Switch Transformer layer, and
    Swish is used in the normal Transformer layer, so there is a choice of which is used in the argument.

    """

    def __init__(self, config: GPTSanJapaneseConfig, ext_layer=False):
        super().__init__()
        # 根据是否是额外层选择不同的中间维度
        d_inter = config.d_ext if ext_layer else config.d_ff
        # 输入层到中间层的线性变换，不带偏置项
        self.wi = nn.Linear(config.d_model, d_inter, bias=ext_layer)
        # 中间层到输出层的线性变换，不带偏置项
        self.wo = nn.Linear(d_inter, config.d_model, bias=ext_layer)
        # 如果是额外层，使用恒等映射作为dropout，否则使用配置中的dropout率
        self.dropout = nn.Identity() if ext_layer else nn.Dropout(config.dropout_rate)
        # 根据是否是额外层选择激活函数
        self.act = ACT2FN["swish" if ext_layer else "relu"]

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        # 输入经过输入层到中间层的线性变换
        hidden_states = self.wi(hidden_states)
        # 应用选择的激活函数
        hidden_states = self.act(hidden_states)
        # 应用dropout或者恒等映射
        hidden_states = self.dropout(hidden_states)
        # 中间层到输出层的线性变换
        hidden_states = self.wo(hidden_states)
        return hidden_states


# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersTop1Router with SwitchTransformers->GPTSanJapanese
class GPTSanJapaneseTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__()
        # 专家数量
        self.num_experts = config.num_experts
        # 每个专家的容量
        self.expert_capacity = config.expert_capacity
        # 分类器层，将隐藏状态映射到专家数目的输出，带有偏置项
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        # 路由噪声
        self.jitter_noise = config.router_jitter_noise
        # 是否忽略填充标记
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        # 路由数据类型
        self.dtype = getattr(torch, config.router_dtype)
    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype
        # 存储当前输入 hidden_states 的数据类型，以备将输出重新转换回该数据类型
        self.input_dtype = hidden_states.dtype
        # 将 hidden_states 转换为 self.dtype，以确保稳定性和一致性
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # 如果在训练过程中，并且设置了 jitter_noise，则将输入的 token 乘以均匀分布的值，以添加一些噪音
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        # 调用 _cast_classifier 方法，确保分类器的数据类型与 self.dtype 一致
        self._cast_classifier()
        # 计算 router_logits，即路由器的原始逻辑回归结果
        router_logits = self.classifier(hidden_states)

        # 应用 Softmax 函数，并将结果转换回原始的数据类型 self.input_dtype
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        # 如果分类器不是 `Linear8bitLt` 类的实例（通过检查特定的属性），则将其转换为 self.dtype 类型
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)
    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        r"""
        Generic forward function for every Router class. Each Router expects to have the same input hidden states
        (`hidden_states`) corresponding to the hidden states for each token, the `expert_capacity` corresponding to the
        number of tokens the Router will send to each expert, some Routers can send up to few tokens to each expert.

        Each Router works as the following: it expects the hidden states for each token, gets the `router_probs` and
        `router_logits` from the `router_weights`. This will assign for each token, the raw probability to be assigned
        to an expert. Then each Router class will have to define its own `_compute_routing_instructions`.

        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`] Tuple containing the expert index, the router probs
            and the router logits. The router probabilities and logits are required to compute the loss.
        """
        # 计算路由概率和路由 logits
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        # 根据概率选择每个 token 被分配到的专家索引
        expert_index = torch.argmax(router_probs, dim=-1)
        # 将专家索引转换成 one-hot 格式
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # 计算每个专家接收的 token 数量的累积和
        token_priority = torch.cumsum(expert_index, dim=-2)
        # 创建专家接收容量的掩码，以限制 token 数量不超过 expert_capacity
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        # 取每个 token 的最大路由概率作为输出
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits
# Copied from transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersSparseMLP with SwitchTransformers->GPTSanJapanese
class GPTSanJapaneseSparseMLP(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: GPTSanJapaneseConfig, expert_class: nn.Module = GPTSanJapaneseDenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = GPTSanJapaneseTop1Router(config)

        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:

        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).

        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.

        """
        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.

        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)

        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)


class GPTSanJapaneseLayerSparseFF(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__()
        self.mlp = GPTSanJapaneseSparseMLP(config)  # 初始化稀疏多层感知机（MLP）模型
        self.soft_bypass_mlp = nn.Linear(config.d_model, config.d_model, bias=False)  # 创建线性层，用于软绕过MLP
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)  # 初始化层归一化，使用给定的epsilon值

    def forward(self, hidden_states, output_router_logits):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.  # 输入隐藏状态，形状为[num_groups, tokens_per_group, hidden_dim]，发送给专家
            output_router_logits (`bool`) :
                output experts router output.  # 输出专家的路由器输出
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]  # 返回形状为[num_groups, tokens_per_group, hidden_dim]的张量

        """
        forwarded_states, router_tuple = self.mlp(hidden_states)  # 使用MLP处理隐藏状态，获得前向状态和路由元组
        forwarded_states += torch.tanh(self.soft_bypass_mlp(hidden_states))  # 添加软绕过MLP的操作结果到前向状态
        output = hidden_states + self.norm(forwarded_states)  # 使用层归一化将前向状态与隐藏状态相加，得到最终输出

        if output_router_logits and router_tuple is not None:
            return output, router_tuple  # 如果需要输出路由器的输出且路由元组不为空，则返回输出和路由元组
        else:
            return output  # 否则只返回输出
class GPTSanJapaneseLayerDenseFF(nn.Module):
    r"""
    Extra Transformers Feed Forward layer module.

    Parameters:
        config : ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__()
        # 检查是否是稀疏层，如果不是则是密集层
        self.mlp = GPTSanJapaneseDenseActDense(config, ext_layer=True)
        # 使用 LayerNorm 对象进行归一化处理
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        r"""
        Args:
            hidden_states (`torch.Tensor`) :
                [num_groups, tokens_per_group, hidden_dim] inputs to send to experts.
        Returns:
            torch.Tensor[num_groups, tokens_per_group, hidden_dim]

        """
        # 通过 MLP 层处理隐藏状态
        forwarded_states = self.mlp(hidden_states)
        # 将处理后的状态与归一化后的隐藏状态相加作为输出
        output = hidden_states + self.norm(forwarded_states)
        return output


# 从 transformers.models.bart.modeling_bart.BartAttention 复制并修改为 GPTSanJapanese
class GPTSanJapaneseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[GPTSanJapaneseConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子设定为 head_dim 的负半径
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性变换层，用于计算 Q、K、V、输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重塑为 [bsz, num_heads, seq_len, head_dim] 的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义模型的前向传播方法，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    """
    Self Attention and Normalization Unit
    """

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层，使用配置中的模型维度和头数，同时作为解码器层
        self.self_attn = GPTSanJapaneseAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            is_decoder=True,
            bias=has_relative_attention_bias,
        )
        # 初始化层归一化，使用配置中的模型维度和层归一化的 epsilon 参数
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,



    """
    Self Attention and FFN Unit
    """

    def __init__(self, config, ext_layer=False):
        super().__init__()
        # 初始化自注意力和前馈网络单元，根据 ext_layer 参数决定使用稠密或稀疏的前馈网络层
        self.self_attn = GPTSanJapaneseLayerSelfAttention(config)
        self.feed_forward = GPTSanJapaneseLayerDenseFF(config) if ext_layer else GPTSanJapaneseLayerSparseFF(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_router_tuple: Optional[bool] = False,



    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTSanJapaneseConfig
    base_model_prefix = "gptsan_japanese"
    supports_gradient_checkpointing = False
    _no_split_modules = ["GPTSanJapaneseBlock"]
    _skip_keys_device_placement = "past_key_values"

    @property
    def dummy_inputs(self):
        # 创建一个包含虚拟输入的字典，包括输入标识符和注意力掩码
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right
    # 定义一个私有方法 `_shift_right`，接收参数 `input_ids`
    def _shift_right(self, input_ids):
        # 从模型配置中获取解码器起始标记的 ID
        decoder_start_token_id = self.config.decoder_start_token_id
        # 从模型配置中获取填充标记的 ID
        pad_token_id = self.config.pad_token_id

        # 如果解码器起始标记 ID 未定义，则引发数值错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # 将输入向右移动一位，以便为解码器准备输入
        if is_torch_fx_proxy(input_ids):
            # 对于 Torch FX 代理对象，不支持原生的项目赋值操作
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 创建一个与输入形状相同的零张量
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # 将输入向右移动一位，并在最左侧插入解码器起始标记 ID
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果填充标记 ID 未定义，则引发数值错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        
        # 将标签中可能存在的 -100 值替换为填充标记 ID
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 返回向右移位后的输入张量
        return shifted_input_ids
# 使用原始字符串字面值定义文档字符串，介绍了 GPTSAN-japanese 模型的概述和用途链接
GPTSAN_JAPANESE_START_DOCSTRING = r"""

    The [GPTSAN-japanese](https://github.com/tanreinama/GPTSAN) model was proposed in General-purpose Swich transformer
    based Japanese language model

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTSanJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 空字符串，准备用于描述模型输入的文档字符串
GPTSAN_JAPANESE_INPUTS_DOCSTRING = r"""
"""

# 添加描述信息到 GPTSanJapaneseModel 类的文档字符串中，指明它是不带任何特定头部的 GPTSAN-japanese 模型变压器，输出原始隐藏状态
@add_start_docstrings(
    "The bare GPTSAN-japanese Model transformer outputting raw hidden-states without any specific head on top.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSanJapaneseModel(GPTSanJapanesePreTrainedModel):
    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__(config)
        # 初始化位置嵌入，用于模型的位置编码
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        # 深拷贝模型配置
        self.config = copy.deepcopy(config)
        # 初始化词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        # 初始化最后一个投影层，用于将模型输出映射回特定维度
        self.last_project = nn.Linear(config.d_model, config.d_model, bias=True)
        # 设置激活函数为 Swish
        self.act = ACT2FN["swish"]

        # 初始化模型块列表
        self.blocks = torch.nn.ModuleList([])
        # 添加 switch 层
        for _ in range(config.num_switch_layers):
            self.blocks.append(GPTSanJapaneseBlock(config))
        # 添加 ext 层
        for _ in range(config.num_ext_layers):
            self.blocks.append(GPTSanJapaneseBlock(config, ext_layer=True))

        # 如果存在额外的 ext 层，初始化额外的位置嵌入
        if config.num_ext_layers > 0:
            self.extra_position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

        # 如果存在 d_spout，初始化 spout 层
        if config.d_spout:
            spouts = []
            for _ in range(8):
                spouts.append(nn.Linear(config.d_spout, config.d_spout, bias=False))
                spouts.append(nn.Tanh())
            spouts.append(nn.Linear(config.d_spout, config.num_layers * 2 * config.d_model, bias=False))
            self.spout = nn.Sequential(*spouts)

        # 执行初始化后的操作
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 添加描述信息到模型前向方法的文档字符串中，描述模型的输入
    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    # 定义模型的前向传播方法，接受多个可选的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token IDs，可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，可选的浮点数张量
        token_type_ids: Optional[torch.FloatTensor] = None,  # token 类型 IDs，可选的浮点数张量
        spout: Optional[torch.FloatTensor] = None,  # 特定应用的张量，可选的浮点数张量
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，可选的张量元组
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，可选的浮点数张量
        use_cache: Optional[bool] = False,  # 是否使用缓存，默认为False
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入张量，可选的浮点数张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入张量，可选的浮点数张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
        output_router_logits: Optional[bool] = None,  # 是否输出路由器日志，默认为None
        num_precontext: Optional[torch.LongTensor] = None,  # 前文上下文数目，可选的长整型张量
@add_start_docstrings(
    "The bare GPTSAN-japanese Model with a language modeling head.",
    GPTSAN_JAPANESE_START_DOCSTRING,
)
class GPTSanJapaneseForConditionalGeneration(GPTSanJapanesePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GPTSanJapaneseConfig):
        super().__init__(config)
        self.model = GPTSanJapaneseModel(config)  # 初始化一个GPTSanJapaneseModel模型
        self.register_buffer("final_logits_bias", torch.zeros([1, config.vocab_size]))  # 注册一个大小为1xvocab_size的零张量作为偏置项
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)  # 初始化一个线性层作为语言模型头部，无偏置项
        if not self.config.torchscript:
            self.lm_head.weight = self.model.embed_tokens.weight  # 如果不是torchscript模式，则将语言模型头部的权重与嵌入词嵌入权重绑定

    @add_start_docstrings_to_model_forward(GPTSAN_JAPANESE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.FloatTensor] = None,
        spout: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # 模型的前向传播方法，接受多种输入，返回生成的结果

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        token_type_ids: Optional[torch.FloatTensor] = None,
        spout: Optional[Union[List, torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ):
        if isinstance(spout, list):  # 如果spout是列表类型
            spout = torch.tensor(spout).float()  # 将其转换为torch张量并转换为float类型
            if input_ids is not None:  # 如果存在input_ids
                spout = spout.to(input_ids.device)  # 将spout移动到与input_ids相同的设备上
        if past_key_values is not None:  # 如果存在过去的键值
            return {
                "input_ids": input_ids[:, -1:] if input_ids is not None else None,  # 返回最后一个位置的input_ids
                "attention_mask": attention_mask,  # 返回attention_mask
                "token_type_ids": token_type_ids[:, -1:] if token_type_ids is not None else None,  # 返回最后一个位置的token_type_ids
                "spout": spout,  # 返回spout
                "past_key_values": past_key_values,  # 返回过去的键值
            }
        return {
            "input_ids": input_ids,  # 返回input_ids
            "attention_mask": attention_mask,  # 返回attention_mask
            "token_type_ids": token_type_ids,  # 返回token_type_ids
            "spout": spout,  # 返回spout
            "past_key_values": None,  # 返回空的过去的键值
        }

    # 从transformers.models.switch_transformers.modeling_switch_transformers.SwitchTransformersForConditionalGeneration.prepare_decoder_input_ids_from_labels复制而来，改为使用GPTSanJapanese
    # 根据标签张量生成解码器的输入序列（右移一位）
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    # 从父类中继承的方法，用于调整词嵌入矩阵的大小，支持可选的多重填充
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类方法调整词嵌入矩阵大小
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调用本类方法调整最终输出偏置的大小
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 返回调整后的词嵌入矩阵
        return new_embeddings

    # 调整最终输出偏置的大小，以适应新的标记数量
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的标记数量小于等于旧的数量，只截取现有偏置的一部分
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        # 如果新的标记数量大于旧的数量，则在偏置末尾填充零向量
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册更新后的偏置为缓冲区
        self.register_buffer("final_logits_bias", new_bias)

    # 获取模型的输入词嵌入
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    # 设置模型的输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.model.set_input_embeddings(new_embeddings)

    # 设置模型的输出词嵌入，用新的词嵌入替换语言模型头部
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取模型的输出词嵌入，即语言模型头部
    def get_output_embeddings(self):
        return self.lm_head

    # 将路由器的输出解压，返回总的路由器 logits 和专家索引
    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []
        total_expert_indexes = []
        for router_output in router_outputs:
            # 如果路由器输出的第一个张量维度大于1，表明有有效的路由器 logits 和专家索引
            if len(router_output[0].shape) > 1:
                router_logits, expert_indexes = router_output
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)
        # 沿着第一个维度拼接所有路由器 logits 和专家索引
        return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)
```