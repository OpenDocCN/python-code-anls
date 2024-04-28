# `.\models\gptj\modeling_gptj.py`

```
# 设置文件编码为 UTF-8
# 版权声明
#
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证的规定，否则不能使用此文件
# 您可以在以下地点获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据此许可证分发的软件
# 均为"按原样"分发，没有任何形式的担保或条件，无论是明示还是暗示的
# 有关特定目的作出的保证都不存在
# 请参阅许可证以获取有关权限和限制的特定语言，以及
# 许可证下授权的限制
""" PyTorch GPT-J 模型。"""

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.fx
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gptj import GPTJConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "hf-internal-testing/tiny-random-gptj"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-j-6B"
_CONFIG_FOR_DOC = "GPTJConfig"

# 预训练模型的存档列表
GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-j-6B",
    # 查看所有 GPT-J 模型：https://huggingface.co/models?filter=gptj
]

# 创建正弦位置编码
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

# 获取嵌入位置
@torch.fx.wrap
def get_embed_positions(embed_positions, position_ids):
    return embed_positions.to(position_ids.device).repeat(position_ids.shape[0], 1, 1)

# 对每两个元素进行旋转
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

# 应用旋转位置编码
def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

# GPT-J 注意力模块
class GPTJAttention(nn.Module):
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()

        # 从配置中获取最大位置嵌入
        max_positions = config.max_position_embeddings
        # 注册偏置缓冲区，用于注意力掩码
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册偏置缓冲区，用于注意力掩码
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        # 创建注意力丢弃层
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # 创建残差丢弃层
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 设置注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_attention_heads
        # 检查嵌入维度是否可以被注意力头数整除
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        # 计算缩放因子
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        # 创建键、值、查询投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim
        # 计算位置嵌入维度
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 创建正弦位置嵌入
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # 将隐藏维度分割成注意力头大小和注意力头数量
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            # 如果张量形状不符合预期，则抛出数值错误
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            # 如果张量形状不符合预期，则抛出数值错误
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        # 将注意力头大小维度和注意力头数量维度合并成隐藏维度
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)
    # 定义注意力计算函数，接受查询、键、值、注意力掩码和头掩码作为输入
    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # 从因果掩码缓冲区计算因果掩码
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    
        # 将注意力权重计算保持为fp32类型，避免溢出问题
        query = query.to(torch.float32)
        key = key.to(torch.float32)
    
        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
    
        # 创建mask_value以防止溢出
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    
        # 缩放注意力权重
        attn_weights = attn_weights / self.scale_attn
    
        if attention_mask is not None:
            # 应用注意力掩码
            attn_weights = attn_weights + attention_mask
    
        # 对注意力权重进行softmax操作
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
    
        # 如果需要，对头进行掩码处理
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
    
        # 求取注意力输出
        attn_output = torch.matmul(attn_weights, value)
    
        return attn_output, attn_weights
    
    # 获取嵌入位置信息
    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        return embed_positions.repeat(position_ids.shape[0], 1, 1)
    
    # 前向传播函数，接受隐藏状态、过去层、注意力掩码、位置ID、头掩码、使用缓存标志、输出注意力权重标志等作为输入
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
        # 使用 self.q_proj 方法生成查询向量
        query = self.q_proj(hidden_states)
        # 使用 self.k_proj 方法生成键向量
        key = self.k_proj(hidden_states)
        # 使用 self.v_proj 方法生成值向量
        value = self.v_proj(hidden_states)

        # 将查询向量分割成多个头
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        # 将键向量分割成多个头
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        # 将值向量分割成多个头
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        # 如果是 Torch FX 代理或者处于跟踪中
        if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
            # 在 Torch FX 情况下，无法追踪条件复制到 GPU 的逻辑，因此在 torch.fx 情况下每次都执行这个
            embed_positions = get_embed_positions(self.embed_positions, position_ids)
        else:
            embed_positions = self._get_embed_positions(position_ids)

        # 重复位置标识符
        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        # 从嵌入位置信息中收集 sin 和 cos
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        # 如果存在旋转维度
        if self.rotary_dim is not None:
            # 对键进行旋转位置编码
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            # 对查询进行旋转位置编码
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            # 拼接旋转后的键和值
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            # 对键和查询进行旋转位置编码
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        # 调换键的维度
        key = key.permute(0, 2, 1, 3)
        # 调换查询的维度
        query = query.permute(0, 2, 1, 3)

        # 如果存在过去的层
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            # 拼接过去的键和当前的键
            key = torch.cat((past_key, key), dim=-2)
            # 拼接过去的值和当前的值
            value = torch.cat((past_value, value), dim=-2)

        # 如果使用缓存
        if use_cache is True:
            # 注意这个强制转换非常丑，但在 ROPE 之前并不实现，因为原始代码库始终在计算过程中保留 key 为 float32
            present = (key.to(hidden_states.dtype), value)
        else:
            present = None

        # 计算自注意力
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多个头的输出
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        # 输出映射
        attn_output = self.out_proj(attn_output)
        # 通过残差连接后进行 dropout
        attn_output = self.resid_dropout(attn_output)

        # 输出结果
        outputs = (attn_output, present)
        # 如果需要输出注意力权重
        if output_attentions:
            # 添加注意力权重到输出
            outputs += (attn_weights,)

        # 返回结果
        return outputs  # a, present, (attentions)
class GPTJMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # 定义初始化方法，接受中间层大小和配置参数
        super().__init__()  # 调用父类的初始化方法
        embed_dim = config.n_embd  # 从配置参数中获取嵌入维度大小

        self.fc_in = nn.Linear(embed_dim, intermediate_size)  # 创建一个线性变换层，将输入维度转换为中间层维度
        self.fc_out = nn.Linear(intermediate_size, embed_dim)  # 创建一个线性变换层，将中间层维度转换为输出维度

        self.act = ACT2FN[config.activation_function]  # 根据配置中的激活函数名称获取对应的激活函数
        self.dropout = nn.Dropout(config.resid_pdrop)  # 创建一个Dropout层，用于防止过拟合

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:  # 定义前向传播方法，接受隐藏状态并返回隐藏状态
        hidden_states = self.fc_in(hidden_states)  # 将输入的隐藏状态进行线性变换
        hidden_states = self.act(hidden_states)  # 对变换后的隐藏状态应用激活函数
        hidden_states = self.fc_out(hidden_states)  # 再次进行线性变换
        hidden_states = self.dropout(hidden_states)  # 对结果进行Dropout处理
        return hidden_states  # 返回处理后的隐藏状态


class GPTJBlock(nn.Module):
    def __init__(self, config):  # 定义初始化方法，接受配置参数
        super().__init__()  # 调用父类的初始化方法
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd  # 根据配置参数确定内部维度

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)  # 创建LayerNorm层，用于对输入进行归一化
        self.attn = GPTJAttention(config)  # 创建GPTJAttention实例
        self.mlp = GPTJMLP(inner_dim, config)  # 创建GPTJMLP实例

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states  # 持有原始隐藏状态，用于残差连接
        hidden_states = self.ln_1(hidden_states)  # 对隐藏状态进行LayerNorm处理
        attn_outputs = self.attn(  # 调用GPTJAttention实例进行注意力计算
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # 从注意力输出中获取注意力结果
        outputs = attn_outputs[1:]  # 获取其余输出结果

        feed_forward_hidden_states = self.mlp(hidden_states)  # 将隐藏状态传入MLP进行前向传播
        hidden_states = attn_output + feed_forward_hidden_states + residual  # 将注意力输出、MLP输出和原始隐藏状态进行残差连接

  
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果是线性层
        if isinstance(module, (nn.Linear,)):
            # 使用正态分布初始化权重
            # 注意：这里与 Mesh Transformer JAX 有些不同，它使用截断正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将对应权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)
    # GPTJ_START_DOCSTRING 是模型的文档字符串的起始部分，用于描述模型的基本信息和参数说明
    GPTJ_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GPTJConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。详情见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免对填充标记索引执行注意力操作的掩码。
            # 掩码值在 `[0, 1]` 范围内选择：
            # - 1 表示 **不遮蔽** 的标记，
            # - 0 表示 **被遮蔽** 的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，指示输入的第一部分和第二部分。
            # 索引在 `[0, 1]` 范围内选择：
            # - 0 对应 *句子 A* 的标记，
            # - 1 对应 *句子 B* 的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 在范围 `[0, config.n_positions - 1]` 内选择。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            # 用于将自注意力模块中的选定头部置零的掩码。
            # 掩码值在 `[0, 1]` 范围内选择：
            # - 1 表示头部 **未被遮蔽**，
            # - 0 表示头部 **被遮蔽**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_dim)`, *optional*):
            # 可选地，您可以选择直接传递嵌入表示，而不是传递 `input_ids`。
            # 如果您希望更精确地控制如何将 *input_ids* 索引转换为关联向量，则此选项很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 有关更多详细信息，请参阅返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 有关更多详细信息，请参阅返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""

# PARALLELIZE_DOCSTRING
# 说明了并行化功能的实验性特性和使用方式
# 参数：
#   device_map：一个字典，将注意力模块映射到设备上，如果没有给出device_map，则会平均分配块到所有设备上
#   例子：
#   - 给出了一个在拥有4个GPU的机器上使用gpt-j-6B模型的设备映射示例

# DEPARALLELIZE_DOCSTRING
# 将模型从模型并行状态移动到CPU
# 例子：
# - 给出了一个使用gpt-j-6B在4个GPU机器上的示例

@add_start_docstrings(
    "The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.",
    GPTJ_START_DOCSTRING,
)
# 装饰器，为GPTJModel类增加文档字符串
class GPTJModel(GPTJPreTrainedModel):
    # GPTJModel类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化模型的属性
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPTJBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        # 模型并行化标志
        self.model_parallel = False
        # 设备映射
        self.device_map = None
        # 梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 装饰器，添加并行化文档字符串
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 发出警告，说明该方法即将被移除
    warnings.warn(
        "`GPTJModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
        " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
        " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
        " ...}",
        FutureWarning,
    )
    # 检查设备映射的有效性
    self.device_map = (
        get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
    )
    # 确保设备映射的正确性
    assert_device_map(self.device_map, len(self.h))
    # 标记模型为模型并行
    self.model_parallel = True
    # 确定第一个设备
    self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
    # 确定最后一个设备
    self.last_device = "cuda:" + str(max(self.device_map.keys()))
    # 将输入嵌入层移到第一个设备
    self.wte = self.wte.to(self.first_device)
    # 将模型块加载到对应设备
    for k, v in self.device_map.items():
        for block in v:
            cuda_device = "cuda:" + str(k)
            self.h[block] = self.h[block].to(cuda_device)
    # 将最终层移到最后一个设备
    self.ln_f = self.ln_f.to(self.last_device)

@add_start_docstrings(DEPARALLELIZE_DOCSTRING)
def deparallelize(self):
    # 发出警告，说明该方法即将被移除
    warnings.warn(
        "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
        FutureWarning,
    )
    # 标记模型为非模型并行
    self.model_parallel = False
    # 清空设备映射
    self.device_map = None
    # 将第一个设备和最后一个设备都设置为 CPU
    self.first_device = "cpu"
    self.last_device = "cpu"
    # 将输入嵌入层移到 CPU
    self.wte = self.wte.to("cpu")
    # 将模型块移到 CPU
    for index in range(len(self.h)):
        self.h[index] = self.h[index].to("cpu")
    # 将最终层移到 CPU
    self.ln_f = self.ln_f.to("cpu")
    # 清空 GPU 缓存
    torch.cuda.empty_cache()

# 获取输入嵌入层
def get_input_embeddings(self):
    return self.wte

# 设置输入嵌入层
def set_input_embeddings(self, new_embeddings):
    self.wte = new_embeddings

# 前向传播方法
@add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
@add_code_sample_docstrings(
    checkpoint=_CHECKPOINT_FOR_DOC,
    output_type=BaseModelOutputWithPast,
    config_class=_CONFIG_FOR_DOC,
    real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
)
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
# 以 GPT-J 模型为基础，增加语言模型头部的 GPT-J 模型变换器
@add_start_docstrings(
    """
    The GPT-J Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
class GPTJForCausalLM(GPTJPreTrainedModel):
    # 定义共享权重的关键字
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        super().__init__(config)
        # 创建 GPT-J 模型对象
        self.transformer = GPTJModel(config)
        # 创建线性层作为语言模型头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # Model parallel
        # 设置默认值为 False
        self.model_parallel = False
        # 初始化设备映射为 None
        self.device_map = None

        # 调用后初始化方法
        self.post_init()

    # 并行化方法的文档说明
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，说明方法即将被移除，建议在模型加载时使用 `device_map='balanced'`，或者提供自定义的 device_map
        warnings.warn(
            "`GPTJForCausalLM.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 如果未提供设备映射，则从配置中生成默认的设备映射
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射是否有效
        assert_device_map(self.device_map, len(self.transformer.h))
        # 并行化 GPT-J 模型
        self.transformer.parallelize(self.device_map)
        # 将语言模型头设置到模型的第一个设备
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        # 标记模型为并行
        self.model_parallel = True

    # 反并行化方法的文档说明
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，说明方法即将被移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 反并行化 GPT-J 模型
        self.transformer.deparallelize()
        # 将 GPT-J 模型和语言模型头移动到 CPU
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        # 取消标记模型为并行
        self.model_parallel = False
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 获取输出嵌入层的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 为生成准备输入数据
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 从参数中获取token_type_ids，如果没有的话默认为None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果存在过去的键值，则忽略被past_key_values覆盖的token
        if past_key_values:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入的ID数量大于过去键值的长度，获取要移除的前缀长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认使用旧的行为：只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 移除前缀长度
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                # 如果token_type_ids不为空，保留与输入ID数量对应的部分
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果存在注意力遮罩，并且位置ID为空
        if attention_mask is not None and position_ids is None:
            # 在批量生成时即时创建位置ID
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入了`inputs_embeds`，则仅在第一个生成步骤中使用它
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新model_inputs字典
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        # 返回model_inputs
        return model_inputs

    # 对于前向传播，需要添加模型参数的文档字符串和代码示例的文档字符串
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回一个元组，其中包含按顺序重排后的`past_key_values`缓存
        return tuple(
            # 遍历每一层的`past_key_values`
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            # 遍历所有`past_key_values`
            for layer_past in past_key_values
        )
# 导入所需的模块或函数
@add_start_docstrings(
    """
    The GPT-J Model transformer with a sequence classification head on top (linear layer).

    [`GPTJForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT, GPT-2, GPT-Neo) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPTJ_START_DOCSTRING,
)
# 定义用于序列分类的 GPTJForSequenceClassification 类
class GPTJForSequenceClassification(GPTJPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 获取配置中的标签数
        self.num_labels = config.num_labels
        # 创建 GPTJModel 模型
        self.transformer = GPTJModel(config)
        # 使用线性层进行打分
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/tiny-random-gptj-for-sequence-classification",
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    The GPT-J Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPTJ_START_DOCSTRING,
)
# 定义用于问答任务的 GPTJForQuestionAnswering 类
class GPTJForQuestionAnswering(GPTJPreTrainedModel):
    # 初始化函数，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 GPTJ 模型实例
        self.transformer = GPTJModel(config)
        # 创建线性层，输出维度为标签数量
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        # 是否使用模型并行
        self.model_parallel = False
        # 设备映射
        self.device_map = None

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写前向传播函数，接收输入参数，并返回结果
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
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
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer模块处理输入数据
        outputs = self.transformer(
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

        # 使用qa_outputs进行问答模型的logits计算
        logits = self.qa_outputs(sequence_output)
        # 将logits分割为开始和结束的位置
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除多余的维度
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上，添加一维
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            ignored_index = start_logits.size(1) # 忽略的索引
            start_positions = start_positions.clamp(0, ignored_index) # 对开始位置进行限制
            end_positions = end_positions.clamp(0, ignored_index) # 对结束位置进行限制

            # 定义交叉熵损失函数，忽略特定索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不需要返回字典，则返回输出结果
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
```