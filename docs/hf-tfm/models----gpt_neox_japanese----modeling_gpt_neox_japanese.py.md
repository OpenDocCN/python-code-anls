# `.\models\gpt_neox_japanese\modeling_gpt_neox_japanese.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
""" PyTorch GPTNeoX 模型。"""

# 导入必要的库
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging

# 导入 GPTNeoX 日语配置
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "abeja/gpt-neox-japanese-2.7b"
_CONFIG_FOR_DOC = "GPTNeoXJapaneseConfig"

# 预训练模型存档列表
GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = {
    "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json",
    # 查看所有 GPTNeoXJapanese 模型 https://huggingface.co/models?filter=gpt_neox_japanese
}

# GPTNeoXJapanese 预训练模型类
class GPTNeoXJapanesePreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化和下载加载预训练模型的抽象类。
    """

    config_class = GPTNeoXJapaneseConfig
    base_model_prefix = "gpt_neox_japanese"
    _no_split_modules = ["GPTNeoXJapaneseLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# GPTNeoXJapaneseAttention 类
class GPTNeoXJapaneseAttention(nn.Module):
    # 初始化函数，接受配置和是否使用偏置作为参数
    def __init__(self, config, use_bias=False):
        # 调用父类的初始化函数
        super().__init__()
        # 从配置中获取注意力头数和隐藏层大小
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        # 计算每个注意力头的大小
        self.head_size = self.hidden_size // self.num_attention_heads

        # 计算旋转嵌入的维度
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        # 创建旋转嵌入对象
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        # 获取最大位置编码
        self.max_positions = config.max_position_embeddings
        # 创建注意力丢弃层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # 计算归一化因子
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())

        # 创建查询、键、值的线性层
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        # 创建全连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # 如果是最后一层，则激活偏置
        self.use_bias = use_bias
        self.dense_bias = nn.Parameter(torch.zeros(config.hidden_size)) if use_bias else None

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
        ):
        # 检查是否存在先前的层信息，并且先前的层信息不为空
        has_layer_past = layer_past is not None and layer_past[0].numel() > 0

        # 计算 QKV
        # 注意力头 [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # 计算旋转嵌入在旋转维度上
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # 计算旋转嵌入的令牌偏移（用于解码）
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # 缓存 QKV 值
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # 计算注意力
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 重塑输出
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs, self.dense_bias

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        将隐藏维度分割为 attn_head_size 和 num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # 将张量的维度[num_attention_heads, seq_len, attn_head_size]重新排列为[bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # 重新组织后的张量形状为[bs, seq_len, num_attention_heads * attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # 返回合并后的张量形状为[bs, seq_len, hidden_size]
        return tensor

    def _create_causal_mask(self, key_length, query_length):
        # 创建一个下三角矩阵作为因果掩码，形状为[self.max_positions, self.max_positions]
        causal_mask = torch.tril(
            torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).view(
                1, 1, self.max_positions, self.max_positions
            )
        )
        # 返回从因果掩码中截取的部分，形状为[1, 1, key_length - query_length, key_length]
        return causal_mask[:, :, key_length - query_length : key_length, :key_length]
    # 定义注意力计算函数，接受查询、键、值以及注意力掩码等参数
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 获取查询、键、值的维度信息
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        # 创建因果掩码，用于自回归注意力
        causal_mask = self._create_causal_mask(key_length, query_length)

        # 重塑查询和键的形状以便计算
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        
        # 初始化注意力分数矩阵
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        
        # 计算注意力分数
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        
        # 将注意力分数矩阵恢复原始形状
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # 创建掩码值，用于处理掩码后的位置
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        causal_mask = causal_mask.to(attn_scores.device)
        
        # 根据因果掩码处理注意力分数
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        # 如果存在注意力掩码，则应用该掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # 对注意力分数进行 softmax 操作
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        attn_weights = attn_weights.to(value.dtype)

        # 如果存在头部掩码，则应用该掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights
# 从transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXRotaryEmbedding复制代码，并将GPTNeoXRotaryEmbedding->RotaryEmbedding
class RotaryEmbedding(nn.Module):
    # 从transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__中复制代码
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使`torch.jit.trace`正常工作，在此处构建
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # 与论文不同，但使用不同的排列顺序以获得相同的计算结果
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """旋转输入的一半隐藏维度。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def bias_dropout_add(x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    """为x添加偏置，应用dropout和残差连接

    Args:
        x (Tensor): 输出的主路径
        bias (Tensor): 最后一个注意力层的attn_bias或None
        residual (Optional[Tensor]): 残差值
        prob (float): dropout概率
        training (bool): 是否处于训练模式

    Returns:
        Tensor: dropout(x + bias) + residual
    """
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


class GPTNeoXJapaneseMLP(nn.Module):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 计算中间层的大小
        intermediate_size = int(config.hidden_size * config.intermediate_multiple_size)
        # 创建一个线性层，将隐藏状态映射到四倍大小的中间层
        self.dense_h_to_4h = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 将中间层映射回隐藏状态大小
        self.dense_4h_to_h = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 根据配置中的隐藏激活函数选择对应的激活函数
        self.act = ACT2FN[config.hidden_act]

    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 将隐藏状态映射到四倍大小的中间层
        intermediate = self.dense_h_to_4h(hidden_states)
        # 使用激活函数处理中间层
        intermediate = self.act(intermediate)
        # 将中间层映射回隐藏状态大小
        output = self.dense_4h_to_h(intermediate)
        # 返回输出
        return output
class GPTNeoXJapaneseLayer(nn.Module):
    # 定义 GPTNeoXJapaneseLayer 类，继承自 nn.Module
    def __init__(self, config, layer_number):
        # 初始化函数，接受配置和层编号作为参数
        super().__init__()
        # 调用父类的初始化函数
        self.layer_number = layer_number
        # 设置当前层的编号
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化输入层的 LayerNorm
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力后的 LayerNorm
        self.attention = GPTNeoXJapaneseAttention(config=config, use_bias=layer_number == config.num_hidden_layers - 1)
        # 初始化注意力机制
        self.mlp = GPTNeoXJapaneseMLP(config)
        # 初始化 MLP
        self.hidden_dropout = config.hidden_dropout
        # 设置隐藏层的 dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        # 前向传播函数
        residual = hidden_states
        # 保存输入的隐藏状态
        ln_out = self.input_layernorm(hidden_states)
        # 对输入进行 LayerNorm
        attention_layer_outputs, attn_bias = self.attention(
            ln_out,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力层的输出和注意力偏置
        attn_output = attention_layer_outputs[0]  # output_attn: a, present, (attentions)
        # 获取注意力输出
        outputs = attention_layer_outputs[1:]
        # 获取其他输出

        attn_output = bias_dropout_add(
            attn_output,
            bias=attn_bias.expand_as(residual) if attn_bias is not None else attn_bias,
            residual=residual,
            prob=self.hidden_dropout,
            training=self.training,
        )
        # 添加偏置和 dropout 到注意力输出
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
        # 对注意力输出进行 MLP 处理

        attn_output = bias_dropout_add(
            mlp_output, bias=None, residual=attn_output, prob=self.hidden_dropout, training=self.training
        )
        # 添加偏置和 dropout 到注意力输出

        if use_cache:
            outputs = (attn_output,) + outputs
        else:
            outputs = (attn_output,) + outputs[1:]
        # 根据是否使用缓存更新输出

        return outputs  # hidden_states, present, (attentions)
        # 返回输出结果

GPT_NEOX_JAPANESE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~GPTNeoXJapaneseConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# GPTNeoXJapaneseLayer 类的文档字符串

GPT_NEOX_JAPANESE_INPUTS_DOCSTRING = r"""
# GPTNeoXJapaneseLayer 类的输入文档字符串
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 来获取这些索引。

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 之间：
            # - 0 对应于*句子 A* 标记，
            # - 1 对应于*句子 B* 标记。

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]` 之间。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置零的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的头部，
            # - 0 表示**被掩码**的头部。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多控制如何将 *input_ids* 索引转换为相关向量，
            # 而不是使用模型的内部嵌入查找矩阵，这将非常有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~file_utils.ModelOutput`] 而不是一个普通的元组。
# 导入必要的库
import torch
import torch.nn as nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel, GPTNeoXLayer, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

# 定义 GPTNeoXJapaneseModel 类，继承自 GPTNeoXJapanesePreTrainedModel
@add_start_docstrings(
    "The bare GPTNeoXJapanese Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_NEOX_JAPANESE_START_DOCSTRING,
)
class GPTNeoXJapaneseModel(GPTNeoXJapanesePreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 定义输入的嵌入层
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        # 定义多层 GPTNeoXJapaneseLayer
        self.layers = nn.ModuleList(
            [GPTNeoXJapaneseLayer(config=config, layer_number=i) for i in range(config.num_hidden_layers)]
        )
        # 定义最终的 LayerNorm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embed_in

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embed_in = value

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(GPT_NEOX_JAPANESE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 定义 GPTNeoXJapaneseForCausalLM 类，继承自 GPTNeoXJapanesePreTrainedModel
@add_start_docstrings(
    """GPTNeoXJapanese Model with a `language modeling` head on top for Classifier Model fine-tuning.""",
    GPT_NEOX_JAPANESE_START_DOCSTRING,
)
class GPTNeoXJapaneseForCausalLM(GPTNeoXJapanesePreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    # 初始化函���
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 创建 GPTNeoXJapaneseModel 实例
        self.gpt_neox_japanese = GPTNeoXJapaneseModel(config)
        # 定义输出的线性层
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.embed_out

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(GPT_NEOX_JAPANESE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于前向传播，接受多个输入参数，包括输入的 token IDs、注意力掩码、输入的嵌入向量、头部掩码、过去的键值对、标签、是否使用缓存、是否输出注意力权重、是否输出隐藏状态、是否返回字典形式的结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    
    # 准备生成的输入，接受输入的 token IDs、过去的键值对、注意力掩码以及其他模型参数
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入 token IDs 的形状
        input_shape = input_ids.shape

        # 如果注意力掩码为空，则创建一个全为1的注意力掩码，用于decoder模型
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去的键值对，则截取输入 token IDs 的最后一个位置作为新的输入
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        # 返回包含输入 token IDs、注意力掩码和过去的键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存中的过去键值对，根据给定的 beam 索引
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空的重新排序后的过去键值对
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 将每一层的过去键值对根据给定的 beam 索引重新排序，并添加到重新排序后的过去键值对中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        # 返回重新排序后的过去键值对
        return reordered_past
```