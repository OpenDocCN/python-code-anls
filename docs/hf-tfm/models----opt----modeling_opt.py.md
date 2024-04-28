# `.\transformers\models\opt\modeling_opt.py`

```
# 定义编码为 UTF-8
# 版权声明，版权所有
# 根据 Apache 许可证 2.0 版本使用和分发此代码
# 可以从以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非根据适用法律或书面同意要求，否则在“AS IS”基础上分发，不提供任何明示或暗示的担保或条件
# 查看许可证以获取特定语言的具体权限和限制
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
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
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig

# 检测是否具有 flash attention 2
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 日志记录
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # 查看所有 OPT 模型 https://huggingface.co/models?filter=opt
]


# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    # 初始化函数，初始化 Embedding 层对象
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 如果指定了 padding_idx，则设置偏移量为2，并相应调整 num_embeddings
        # 这个设置主要是为了处理填充索引，其他模型不需要这个偏移
        self.offset = 2
        # 调用父类的初始化方法来初始化 Embedding 层对象
        super().__init__(num_embeddings + self.offset, embedding_dim)

    # 前向传播函数，接收注意力掩码和过去键值对的长度作为参数
    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 将注意力掩码转换为长整型
        attention_mask = attention_mask.long()

        # 根据注意力掩码创建位置张量
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # 如果 past_key_values_length 大于 0，则截取位置张量
        positions = positions[:, past_key_values_length:]

        # 调用父类的前向传播方法，传入偏移量修正后的位置张量
        return super().forward(positions + self.offset)
class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        # 初始化 OPTAttention 类
        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 是否为自回归模型（decoder）
        self.is_causal = True

        # 检查 embed_dim 是否能被 num_heads 整除
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        # 设置缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新组织张量形状，用于多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
class OptFlashAttention2(OPTAttention):
    """
    OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of flash
    # 处理填充令牌的注意事项，以防输入包含任何填充令牌。
    
    # 从transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__复制过来的
    # 初始化方法，继承自父类
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
    
        # TODO: 一旦RoCm Flash Attention升级到2.1，这里应该被移除。
        # flash_attn<2.1生成左上对齐的因果蒙版，而这里需要右下对齐，默认的flash_attn>=2.1中已经实现了这一特性。这个属性用于处理这种差异。参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0。
        # 请注意，对于flash_attn<2.1，除了q_seqlen == 1的情况之外，使用q_seqlen != k_seqlen会生成一个错误的蒙版（左上）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    # 从transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward复制过来的
    # flash_attention的前向传播方法
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        """
        如果输入的隐藏状态至少包含一个填充标记，则调用 Flash Attention 的前向方法。
        首先对输入进行去填充，然后计算注意力分数并填充最终的注意力分数。

        参数：
            query_states (`torch.Tensor`)：
                要传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`)：
                要传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`)：
                要传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`)：
                填充遮罩 - 对应于大小为 `(batch_size, seq_len)` 的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*)：
                注意力的丢弃率
            softmax_scale (`float`, *optional*)：
                在应用 softmax 前的 QK^T 的缩放。默认为 1 / sqrt(head_dim)
        """
        # 如果 Flash Attention 不使用左上角遮罩
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦 Flash Attention for RoCm 升级到 2.1，请删除 `query_length != 1` 检查。有关详细信息，请参阅 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # 去填充输入
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Flash Attention 的可变长度功能
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            # 填充输入
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # 使用 Flash Attention 的默认功能
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制
    # 定义内部函数_upad_input，用于处理输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 调用_get_unpad_data函数获取未填充数据的索引、当前序列长度和批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取输入张量的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据未填充数据的索引重新排列key_layer和value_layer的形状
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 如果query_length等于kv_seq_len，则按照indices_k对query_layer进行重新排列
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果query_length等于1，则将query_layer的形状调整为(batch_size, head_dim)
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 根据query_length截取attention_mask的右侧部分
            attention_mask = attention_mask[:, -query_length:]
            # 调用unpad_input函数处理输入数据
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的query_layer、key_layer、value_layer、indices_q、cu_seqlens、max_seqlen_in_batch元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义了一个字典，将字符串映射到对应的 OPTAttention 类
OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,  # 对应字符串 "eager" 的类是 OPTAttention
    "flash_attention_2": OptFlashAttention2,  # 对应字符串 "flash_attention_2" 的类是 OptFlashAttention2
}

# 定义了一个 OPTDecoderLayer 类，继承自 nn.Module
class OPTDecoderLayer(nn.Module):
    # 初始化方法，接受一个 OPTConfig 实例作为参数
    def __init__(self, config: OPTConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将隐藏层大小作为嵌入维度
        self.embed_dim = config.hidden_size

        # 根据配置选择自注意力机制类，并实例化
        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)

        # 是否在层规范化之前执行 dropout
        self.do_layer_norm_before = config.do_layer_norm_before
        # dropout 概率
        self.dropout = config.dropout
        # 激活函数
        self.activation_fn = ACT2FN[config.activation_function]

        # 自注意力层规范化
        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        # 第一个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        # 第二个全连接层
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        # 最终层规范化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        # ...
        pass

# 定义了一个原始的 OPT 模型类，继承自 PreTrainedModel
@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",  # 添加文档字符串
    OPT_START_DOCSTRING,  # 添加参数文档字符串
)
class OPTPreTrainedModel(PreTrainedModel):
    # 模型的配置类
    config_class = OPTConfig
    # 模型的基本名称前缀
    base_model_prefix = "model"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块名称列表
    _no_split_modules = ["OPTDecoderLayer"]
    # 是否支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 从配置中获取初始化标准差
        std = self.config.init_std
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，将对应权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 定义一个文档字符串，用于说明模型的输入
OPT_INPUTS_DOCSTRING = r"""
"""

# 定义 OPTDecoder 类，继承自 OPTPreTrainedModel 类
class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer 解码器，由 *config.num_hidden_layers* 层组成。每一层都是一个 [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    # 初始化方法
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        # 设置 dropout 概率
        self.dropout = config.dropout
        # 设置层间 dropout 概率
        self.layerdrop = config.layerdrop
        # 设置填充标记的索引
        self.padding_idx = config.pad_token_id
        # 设置目标位置的最大数
        self.max_target_positions = config.max_position_embeddings
        # 设置词汇表大小
        self.vocab_size = config.vocab_size

        # 创建词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        # 创建位置嵌入层
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        # 如果词嵌入维度不等于隐藏层维度，则创建线性投影层
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        # 如果词嵌入维度不等于隐藏层维度，则创建线性投影层
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # 如果在层规范化之前进行规范化且不移除最终层规范化，则创建最终层规范化层
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        # 创建解码器层列表
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 根据配置选择是否使用 FLASH2 注意力机制
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 初始化梯度检查点标志
        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化方法，接受一个配置参数config，调用父类的初始化方法
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        # 初始化解码器，传入配置参数config
        self.decoder = OPTDecoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 前向传播方法，接受多个参数，并返回模型输出或元组
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 如果未指定output_attentions，则使用config中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用config中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定use_cache，则使用config中的值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果未指定return_dict，则使用config中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器输出包括(dec_features, past_key_value, dec_hidden, dec_attn)
        # 调用解码器的forward方法进行前向传播
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果return_dict为False，则返回解码器输出
        if not return_dict:
            return decoder_outputs

        # 返回包含last_hidden_state、past_key_values、hidden_states和attentions的BaseModelOutputWithPast对象
        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )
```  
# OPTForCausalLM 模型继承自 OPTPreTrainedModel，实现了因果语言模型的功能
class OPTForCausalLM(OPTPreTrainedModel):
    # 绑定模型头部与嵌入层的权重
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 OPTModel 模型对象
        self.model = OPTModel(config)

        # 创建线性层，将隐藏状态映射到词汇表大小的logits
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播函数，实现因果语言建模任务
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        ...

    # 准备用于生成的输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 如果输入长度大于过去的长度，则去掉前缀部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 如果提供了输入嵌入，则在第一个生成步骤使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    # 重新排序缓存中的过去键-值对
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去键-值对
        reordered_past = ()
        # 遍历每一层的过去键-值对
        for layer_past in past_key_values:
            # 对每个过去状态按照beam_idx进行重新排序，并封装成元组
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键-值对
        return reordered_past
# 引入必要的库和模块，包括 add_start_docstrings, OPT_START_DOCSTRING, OPTPreTrainedModel, OPTConfig, nn, torch 等
@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
# 定义 OPTForSequenceClassification 类，继承自 OPTPreTrainedModel
class OPTForSequenceClassification(OPTPreTrainedModel):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config: OPTConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类属性 num_labels
        self.num_labels = config.num_labels
        # 实例化一个 OPTModel 对象，并设置为类属性
        self.model = OPTModel(config)
        # 实例化一个线性层对象，并设置为类属性
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # 调用初始化权重和应用最终处理的方法
        self.post_init()

    # 定义 forward 方法
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 定义设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
# 定义 OPTForQuestionAnswering 类，继承自 OPTPreTrainedModel
class OPTForQuestionAnswering(OPTPreTrainedModel):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config: OPTConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 实例化一个 OPTModel 对象，并设置为类属性
        self.model = OPTModel(config)
        # 实例化一个线性层对象，并设置为类属性
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # 调用初始化权重和应用最终处理的方法
        self.post_init()

    # 定义 forward 方法
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    # 用于替换返回值的文档字符串，指定输出类型和配置类
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键数值
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        start_positions: Optional[torch.LongTensor] = None,  # 起始位置
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens  # 获取输入嵌入

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value  # 设置输入嵌入
```