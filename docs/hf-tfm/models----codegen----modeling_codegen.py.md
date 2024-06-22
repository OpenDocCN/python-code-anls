# `.\transformers\models\codegen\modeling_codegen.py`

```py
# coding=utf-8
# 版权声明和许可证信息
# 本模型代码基于 Apache License, Version 2.0 发布，详情可见：http://www.apache.org/licenses/LICENSE-2.0
# 该代码部分来自 Salesforce 作者、EleutherAI 团队和 HuggingFace 团队
#
# 注意：以下代码在 PyTorch 中实现 CodeGen 模型的注意力机制部分

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN  # 从模型库中导入激活函数映射
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast  # 从模型库中导入输出类
from ...modeling_utils import PreTrainedModel  # 从模型库中导入预训练模型基类
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 从模型库中导入辅助函数和工具类
from .configuration_codegen import CodeGenConfig  # 从当前目录中导入 CodeGen 模型配置类


logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "Salesforce/codegen-2B-mono"  # 用于文档的预训练模型检查点名称
_CONFIG_FOR_DOC = "CodeGenConfig"  # 用于文档的配置类名称

# 可用的预训练模型列表
CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/codegen-350M-nl",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-nl",
    "Salesforce/codegen-2B-multi",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-6B-multi",
    "Salesforce/codegen-6B-mono",
    "Salesforce/codegen-16B-nl",
    "Salesforce/codegen-16B-multi",
    "Salesforce/codegen-16B-mono",
    # 更多的 CodeGen 模型可在 https://huggingface.co/models?filter=codegen 查看
]

# 以下函数被从 GPT-J 模型中复制过来，用于生成正弦位置编码
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    # 计算正弦位置编码的频率
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    # 生成正弦位置编码
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

# 以下函数被从 GPT-J 模型中复制过来，用于对张量进行每两个元素一次旋转
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    # 按照每两个元素一次旋转的规则进行操作
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # 在 einsum 表示法中：rearrange(x, '... d j -> ... (d j)')

# 以下函数被从 GPT-J 模型中复制过来，用于应用旋转的位置编码到张量
def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # 对正弦和余弦位置编码进行扩展以匹配张量的形状
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    # 应用旋转的位置编码到张量
    return (tensor * cos) + (rotate_every_two(tensor) * sin)

# CodeGenAttention 类定义开始
class CodeGenAttention(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()

        # 获取最大位置嵌入数
        max_positions = config.max_position_embeddings
        # 注册一个缓冲区，用于存储因果掩码（上三角矩阵），形状为（1，1，max_positions，max_positions）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        # 定义注意力和残差的 dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 获取嵌入维度、注意力头数和头维度
        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        # 检查嵌入维度是否可以被注意力头数整除
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        # 计算缩放参数
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        # 创建线性层，用于计算查询、键和值的投影
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        # 创建输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # 获取旋转维度，并根据需要设置位置嵌入维度
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    # 将输入张量沿头维度和多处理器数量维度拆分
    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    # 合并注意力头维度和头数量维度到上下文维度
    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        # 调整张量的维度顺序
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        # 计算新的形状
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    # 注意力函数，接受查询、键、值、注意力掩码和头掩码作为参数
    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        # 从因果掩码缓冲区计算因果掩码
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # 将注意力权重计算保持在 fp32 中，以避免溢出问题
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        mask_value = torch.finfo(attn_weights.dtype).min
        # 需要是一个张量，否则会出现错误：`RuntimeError: expected scalar type float but found double`。
        # 需要在相同的设备上，否则会出现 `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用注意力掩码
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对头部进行掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
        qkv = self.qkv_proj(hidden_states)
        # 对隐藏状态进行查询、键、值的线性投影
        # TODO(enijkamp): 提取逻辑 TPU-v4 核心数量或使前向传播对其不可知
        mp_num = 4
        # 将查询、键、值张量重新形状为 [batch_size, sequence_length, mp_num, local_dim] 的张量
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        # 计算本地维度
        local_dim = self.head_dim * self.num_attention_heads // mp_num
        # 沿指定维度分割查询、键、值张量
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        # 将查询、键、值张量按照头数、头维度和本地维度分割
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        # 转置值张量的最后两个维度
        value = value.permute(0, 2, 1, 3)

        # 获取嵌入位置
        embed_positions = self.embed_positions
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions

        # 根据位置 ID 获取正弦和余弦值
        sincos = embed_positions[position_ids]
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            # 如果旋转维度不为 None，则分割键和查询，并应用旋转位置编码
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            # 否则，直接应用旋转位置编码到键和查询
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        # 转置键和查询张量的最后两个维度
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            # 如果存在过去的层，则将过去的键和值与当前的键和值拼接起来
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            # 如果使用缓存，则将键转换为隐藏状态的数据类型
            # 注意，此强制转换相当丑陋，但在 ROPE 之前尚未实现，
            # 原始代码库中的 k_rot 总是为 fp32。
            # 参考：https://github.com/salesforce/CodeGen/blob/f210c3bb1216c975ad858cd4132c0fdeabf4bfc2/codegen1/jaxformer/hf/codegen/modeling_codegen.py#L38
            present = (key.to(hidden_states.dtype), value)
        else:
            present = None

        # 计算自注意力：V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并注意力头
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        # 应用输出投影层
        attn_output = self.out_proj(attn_output)
        # 应用残差连接和 dropout
        attn_output = self.resid_dropout(attn_output)

        # 构建输出元组
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs  # a, present, (attentions)
# 从 transformers.models.gptj.modeling_gptj.GPTJMLP 复制并修改为 CodeGenMLP 类
class CodeGenMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        # 定义一个线性层，用于输入到中间层的线性变换，输入维度为 embed_dim，输出维度为 intermediate_size
        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        # 定义一个线性层，用于中间层到输出的线性变换，输入维度为 intermediate_size，输出维度为 embed_dim
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        # 激活函数，根据配置选择激活函数
        self.act = ACT2FN[config.activation_function]
        # Dropout 层，根据配置设置丢弃概率
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        # 输入到中间层的线性变换
        hidden_states = self.fc_in(hidden_states)
        # 中间层激活函数
        hidden_states = self.act(hidden_states)
        # 中间层到输出的线性变换
        hidden_states = self.fc_out(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从 transformers.models.gptj.modeling_gptj.GPTJBlock 复制并修改为 CodeGenBlock 类
class CodeGenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 内部维度，如果配置中未指定，则设为 4 * config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        # LayerNorm 层，用于归一化输入
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # 注意力机制，使用 CodeGenAttention
        self.attn = CodeGenAttention(config)
        # MLP 层，使用 CodeGenMLP
        self.mlp = CodeGenMLP(inner_dim, config)

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
        # 残差连接
        residual = hidden_states
        # LayerNorm 归一化输入
        hidden_states = self.ln_1(hidden_states)
        # 注意力机制的输出
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 提取注意力机制输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # 前馈神经网络的输出
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 将注意力机制输出、前馈神经网络输出和残差相加
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class CodeGenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 代码生成模型的配置类
    config_class = CodeGenConfig
    # 基本模型前缀
    base_model_prefix = "transformer"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块列表
    _no_split_modules = ["CodeGenBlock"]
    # 定义一个跳过设备放置的键名
    _skip_keys_device_placement = "past_key_values"

    # 初始化方法，接收任意数量的位置参数和关键字参数
    def __init__(self, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)

    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是线性层
        if isinstance(module, (nn.Linear,)):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将缩放系数初始化为 1
            module.weight.data.fill_(1.0)
CODEGEN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    
    Parameters:
        config ([`CodeGenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 代码生成器输入文档字符串，用于描述代码生成器的输入
CODEGEN_INPUTS_DOCSTRING = r"""
    # 此处应填写代码生成器的输入说明，但该注释为空白，没有提供额外信息
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用[`AutoProcenizer`]获取索引。参见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]了解详情。
            # [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。
            # 掩码值选择在`[0, 1]`之间:
            # - 对于**未被掩盖**的标记，为1，
            # - 对于**被掩盖**的标记，为0。
            # [什么是注意力掩码?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，指示输入的第一部分和第二部分。
            # 索引选择在`[0, 1]`之间:
            # - 0 对应于*句子 A*标记，
            # - 1 对应于*句子 B*标记。
            # [什么是分段标记 ID?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 选择在范围`[0, config.n_positions - 1]`内。
            # [什么是位置 ID?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            # 用于使自注意力模块中选择的头部失效的掩码。
            # 掩码值选择在`[0, 1]`之间:
            # - 1 表示头部**未被掩盖**，
            # - 0 表示头部**被掩盖**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_dim)`, *optional*):
            # 可选地，您可以选择直接传递嵌入表示，而不是传递`input_ids`。
            # 如果您想更精细地控制如何将*input_ids*索引转换为关联向量，而不是使用模型的内部嵌入查找矩阵，则这很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 有关更多详细信息，请参见返回张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 有关更多详细信息，请参见返回张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。
```  
"""
# 导入必要的库和模块
@add_start_docstrings(
    "The bare CodeGen Model transformer outputting raw hidden-states without any specific head on top.",
    CODEGEN_START_DOCSTRING,
)
# 定义 CodeGenModel 类，用于输出没有特定头部的原始隐藏状态
class CodeGenModel(CodeGenPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化模型参数
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # 词嵌入层
        self.drop = nn.Dropout(config.embd_pdrop)  # 丢弃层
        self.h = nn.ModuleList([CodeGenBlock(config) for _ in range(config.n_layer)])  # 多层 CodeGenBlock 模块
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)  # 层标准化
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)  # 旋转维度

        self.gradient_checkpointing = False  # 梯度检查点

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.wte

    # 设置输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(CODEGEN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
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
# 添加注释
@add_start_docstrings(
    """
    The CodeGen Model transformer with a language modeling head on top.
    """,
    CODEGEN_START_DOCSTRING,
)
# 定义 CodeGenForCausalLM 类，用于在顶部添加语言建模头的 CodeGen 模型
class CodeGenForCausalLM(CodeGenPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化模型参数
        self.transformer = CodeGenModel(config)  # 使用 CodeGenModel 类作为转换器
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # 语言建模头

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出词嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出词嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # 准备用于生成的输入数据，根据给定的参数进行处理
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # 获取token_type_ids参数，如果不存在则为None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果存在过去键值，则省略已被past_key_values覆盖的tokens
        if past_key_values:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经仅传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 移除前缀的长度为过去键值的长度
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 从输入IDs中移除前缀
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果存在token_type_ids，则同样裁剪它以匹配输入IDs的长度
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 获取attention_mask和position_ids参数
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果存在attention_mask但不存在position_ids，则在批量生成时动态创建position_ids
        if attention_mask is not None and position_ids is None:
            # 对于batch生成，动态创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 使用attention_mask的零值填充position_ids
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在过去键值，则仅保留position_ids的与输入IDs匹配的部分
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 返回处理后的输入数据字典
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    # 对模型进行前向传播
    @add_start_docstrings_to_model_forward(CODEGEN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
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
        # 检查是否应该返回字典格式的输出，若未指定则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用transformer模型进行前向传播
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
        # 获取transformer的输出中的hidden states
        hidden_states = transformer_outputs[0]

        # 确保在fp16中采样工作正常，并在fp32中计算损失，以与mesh-tf版本匹配
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        # 将hidden states通过lm_head转换为对应的logits，并转换为float32类型
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # 将labels移到正确的设备上以启用模型并行处理
            labels = labels.to(lm_logits.device)
            # 进行logits和labels的偏移以匹配训练时的预测与真实值
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 将tokens扁平化
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            # 如果不返回字典格式的输出，则组装并返回输出元组
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典格式的输出，则组装并返回CausalLMOutputWithPast对象
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
        # 返回一个元组，其中每个元素也是一个元组，用于重新排序`past_key_values`缓存，以匹配每一代步骤的正确beam_idx
        return tuple(
            # 对于past_key_values中的每个层的过去状态，根据beam_idx重新排序
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
```