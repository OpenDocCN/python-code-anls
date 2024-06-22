# `.\transformers\models\xmod\modeling_xmod.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# Meta AI 团队和 HuggingFace Inc. 团队，2023年
#
# 根据 Apache 许可证 2.0 版（"许可证"）授权；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"按原样"的基础分发的，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 有关特定语言的权限，请参阅许可证。
"""PyTorch X-MOD 模型。"""

# 导入模块
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xmod import XmodConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练模型存档列表
XMOD_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xmod-base",
    "facebook/xmod-large-prenorm",
    "facebook/xmod-base-13-125k",
    "facebook/xmod-base-30-125k",
    "facebook/xmod-base-30-195k",
    "facebook/xmod-base-60-125k",
    "facebook/xmod-base-60-265k",
    "facebook/xmod-base-75-125k",
    "facebook/xmod-base-75-269k",
    # 查看所有 X-MOD 模型 https://huggingface.co/models?filter=xmod
]


# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制，将 Roberta 改为 Xmod
class XmodEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制
```  
    # 初始化函数，接受配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # word_embeddings为词嵌入层，根据配置参数创建嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # position_embeddings为位置嵌入层，根据配置参数创建嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # token_type_embeddings为标记类型嵌入层，根据配置参数创建嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
        # LayerNorm命名不符合snake命名规范，为了与TensorFlow的模型变量名保持一致以便加载任何TensorFlow的checkpoint文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout为丢弃层，根据配置参数创建丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # position_embedding_type为位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置ID张量，持久化为False
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册标记类型ID张量，持久化为False
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    
        # 结束复制
        # 设置padding_idx为配置中的pad_token_id
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层，使用padding_idx作为填充索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果未提供位置标识，则根据输入的标记标识创建位置标识。任何填充的标记仍然保持填充状态。
        if position_ids is None:
            if input_ids is not None:
                # 根据输入的标记标识创建位置标识。任何填充的标记保持填充状态。
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 从输入的嵌入数据创建位置标识
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 将 token_type_ids 设置为在构造函数中注册的缓冲区，其中所有值为零，这通常发生在自动生成时，通过注册缓冲区可以在不传递 token_type_ids 的情况下帮助用户跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 使用 word_embeddings 对输入标识进行嵌入处理
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据 token_type_ids 获取 token 类型的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将嵌入数据与 token 类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型是“绝对”的，则获取位置嵌入
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序位置标识
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.roberta.modeling_roberta.RobertaSelfAttention复制并修改为XmodSelfAttention
class XmodSelfAttention(nn.Module):
    # 初始化函数
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 验证hidden_size是否能被num_attention_heads整除，同时验证是否有embedding_size属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化参数
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值参数
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为absolute
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为relative_key或relative_key_query，初始化distance_embedding
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量维度重排成(样本数，头数，每头维度)，用于计算注意力分数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从transformers.models.roberta.modeling_roberta.RobertaSelfOutput.__init__复制
class XmodSelfOutput(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm层、Dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states)
        # 跳连接
        hidden_states = hidden_states + input_tensor
        return hidden_states


class XmodAttention(nn.Module):
    # 初始化方法，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 self-attention 层
        self.self = XmodSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 self-attention 输出层
        self.output = XmodSelfOutput(config)
        # 初始化一个空集合，用于存储需要剪枝的注意力头
        self.pruned_heads = set()
        # 获取是否使用预层归一化的配置
        self.pre_norm = config.pre_norm

    # 从 transformers.models.roberta.modeling_roberta.RobertaAttention.prune_heads 复制的方法
    def prune_heads(self, heads):
        # 如果要剪枝的头部数量为0，则直接返回
        if len(heads) == 0:
            return
        # 查找可剪枝的头部以及它们的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 将隐藏状态保存为残差
        residual = hidden_states
        # 如果使用预层归一化，则对隐藏状态进行归一化
        if self.pre_norm:
            hidden_states = self.output.LayerNorm(hidden_states)
        # 使用 self-attention 层进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将注意力输出与残差相加，并通过输出层进行处理
        attention_output = self.output(self_outputs[0], residual)
        # 如果不使用预层归一化，则对注意力输出进行归一化
        if not self.pre_norm:
            attention_output = self.output.LayerNorm(attention_output)
        # 将注意力输出与可能的额外输出合并并返回
        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力
        return outputs
# 定义一个包含多个隐藏层的神经网络模块
class XmodIntermediate(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为隐藏层大小，输出大小为中间层大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断配置中的隐藏激活函数是字符串还是函数，设置中间层激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受隐藏状态张量，返回中间层输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层计算中间层输出
        hidden_states = self.dense(hidden_states)
        # 应用中间层激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义适配器模块
class XmodAdapter(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        super().__init__()
        # 计算适配器的压缩大小
        self.bottleneck_size = config.hidden_size // config.adapter_reduction_factor
        # 创建两个全连接层，用于压缩和恢复大小
        self.dense1 = nn.Linear(config.hidden_size, self.bottleneck_size)
        self.dense2 = nn.Linear(self.bottleneck_size, config.hidden_size)
        # 判断配置中的隐藏激活函数是字符串还是函数，设置适配器激活函数
        if isinstance(config.hidden_act, str):
            self.adapter_act_fn = ACT2FN[config.hidden_act]
        else:
            self.adapter_act_fn = config.hidden_act

    # 前向传播函数，接受隐藏状态张量，返回适配器输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过第一个全连接层进行压缩
        hidden_states = self.dense1(hidden_states)
        # 应用适配器激活函数
        hidden_states = self.adapter_act_fn(hidden_states)
        # 通过第二个全连接层进行大小恢复
        hidden_states = self.dense2(hidden_states)
        return hidden_states


# 定义输出模块
class XmodOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        super().__init__()
        # 创建全连接层和 LayerNorm 层
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_before_adapter = config.ln_before_adapter
        # 创建一个 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 根据配置创建适配器的 LayerNorm 层
        if config.adapter_layer_norm:
            self.adapter_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.adapter_layer_norm = None
        # 判断是否重用适配器的 LayerNorm 层
        self.adapter_reuse_layer_norm = config.adapter_reuse_layer_norm
        # 空的适配器模块字典
        self.adapter_modules = nn.ModuleDict({})
        # 针对每种语言，创建一个适配器模块，并存储在适配器模块字典中
        for language in config.languages:
            self.adapter_modules[str(language)] = XmodAdapter(config)

    # 前向传播函数，接受隐藏状态张量、输入张量、语言 id 张量，返回输出张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        # 通过全连接层计算输出
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层
        hidden_states = self.dropout(hidden_states)
        # 将输入张量加到隐藏状态张量上
        hidden_states = hidden_states + input_tensor
        # 通过语言适配器模块进行适配
        hidden_states = self.lang_adapter(lang_ids, hidden_states)
        return hidden_states
    # 对不同语言的样本进行并行处理
    lang_ids, lang_lengths = torch.unique_consecutive(lang_ids, return_counts=True)

    # 如果在适配器之前不使用 LayerNorm，则将隐藏状态保存为残差
    if not self.ln_before_adapter:
        residual = hidden_states

    # 如果存在适配器 LayerNorm，则对隐藏状态进行规范化
    if self.adapter_layer_norm is not None:
        hidden_states = self.adapter_layer_norm(hidden_states)
    # 如果适配器复用 LayerNorm，则使用默认的 LayerNorm 对隐藏状态进行规范化
    elif self.adapter_reuse_layer_norm:
        hidden_states = self.LayerNorm(hidden_states)

    # 如果在适配器之前使用 LayerNorm，则将隐藏状态保存为残差
    if self.ln_before_adapter:
        residual = hidden_states

    # 将隐藏状态按照语言长度进行分割
    split_hidden_states = torch.split(hidden_states, lang_lengths.tolist(), 0)
    lang_wise_outputs = []
    # 对每种语言的隐藏状态应用对应的适配器模块
    for i, (lang_id, split_hidden_state) in enumerate(zip(lang_ids, split_hidden_states)):
        lang = list(self.adapter_modules.keys())[int(lang_id.item())]
        lang_wise_outputs.append(self.adapter_modules[lang](split_hidden_state))
    # 将不同语言的适配器输出拼接在一起
    hidden_states = torch.cat(lang_wise_outputs, 0)

    # 对适配器输出进行 dropout
    hidden_states = self.dropout(hidden_states)
    # 将残差添加到适配器输出上
    hidden_states += residual
    return hidden_states
# 定义一个名为 XmodLayer 的类，继承自 nn.Module
class XmodLayer(nn.Module):
    # 定义初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 chunk_size_feed_forward 属性为传入的 config 的 chunk_size_feed_forward 属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置 seq_len_dim 属性为 1
        self.seq_len_dim = 1
        # 初始化 attention 属性为一个 XmodAttention 对象，使用传入的 config
        self.attention = XmodAttention(config)
        # 设置 is_decoder 属性为传入的 config 的 is_decoder 属性
        self.is_decoder = config.is_decoder
        # 设置 add_cross_attention 属性为传入的 config 的 add_cross_attention 属性
        self.add_cross_attention = config.add_cross_attention
        # 如果 add_cross_attention 为 True
        if self.add_cross_attention:
            # 如果 is_decoder 为 False
            if not self.is_decoder:
                # 抛出值错误，提示 cross attention 只能在 decoder 模型中使用
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化 crossattention 属性为一个 XmodAttention 对象，使用传入的 config，设置 position_embedding_type 为 "absolute"
            self.crossattention = XmodAttention(config, position_embedding_type="absolute")
        # 初始化 intermediate 属性为一个 XmodIntermediate 对象，使用传入的 config
        self.intermediate = XmodIntermediate(config)
        # 初始化 output 属性为一个 XmodOutput 对象，使用传入的 config
        self.output = XmodOutput(config)
        # 设置 pre_norm 属性为传入的 config 的 pre_norm 属性

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键值对不为None，则提取前两个作为decoder单向自注意力缓存的键/值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # 如果是decoder，则最后一个输出是self-attn缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果要输出注意力权重，则加入自注意力计算结果

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # 如果过去的键值对不为None，则提取倒数第二和最后一个作为跨注意力缓存的键/值对
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行跨注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果要输出注意力权重，则加入跨注意力计算结果

            # 将跨注意力缓存加到present_key_value元组的第三和第四位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        residual = attention_output
        if self.pre_norm:
            attention_output = self.output.LayerNorm(attention_output)
        intermediate_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        layer_output = self.output(intermediate_output, residual, lang_ids)
        if not self.pre_norm:
            layer_output = self.output.LayerNorm(layer_output)
        outputs = (layer_output,) + outputs

        # 如果是decoder，则将注意力键/值对作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        return self.intermediate(attention_output)
# XmodEncoder 类继承自 nn.Module，表示一个基于 Transformer 的编码器模块
class XmodEncoder(nn.Module):
    def __init__(self, config):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建一个 nn.ModuleList 容器，包含 config.num_hidden_layers 个 XmodLayer 模块
        self.layer = nn.ModuleList([XmodLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用预归一化 (pre-norm) 结构
        self.is_pre_norm = config.pre_norm
        # 如果使用预归一化，创建一个 nn.LayerNorm 层
        if self.is_pre_norm:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 是否使用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        # 执行前向传播计算

# XmodPooler 类继承自 nn.Module，表示一个用于pooling的模块
class XmodPooler(nn.Module):
    def __init__(self, config):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 创建一个全连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活层
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出第一个token的隐状态向量
        first_token_tensor = hidden_states[:, 0]
        # 经过全连接层和 Tanh 激活得到 pooled 输出
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# XmodPreTrainedModel 类继承自 PreTrainedModel，提供预训练模型的初始化和加载功能
class XmodPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 XmodConfig
    config_class = XmodConfig
    # 基础模型前缀为 "roberta"
    base_model_prefix = "roberta"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的辅助函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，标准差为 config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，标准差为 config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充 token，将其对应的向量初始化为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为 0，将权重项初始化为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def set_default_language(self, language: str):
        """
        Set the default language code for the model. This is used when the language is not specified in the input.

        Args:
            language (`str`): The language code, such as `"en_XX"` or `"de_DE"`.
        """
        # 检查输入的语言代码是否在模型配置的支持语言列表中
        if language not in self.config.languages:
            raise ValueError(
                f"{self} does not have an adapter for {language}. Supported languages: {list(self.config.languages)}"
            )
        # 设置模型的默认语言为输入的语言代码
        self.config.default_language = language

    def freeze_embeddings_and_language_adapters(self):
        """
        Freeze the embeddings and language adapters of the model. Usually, this is applied before the model is
        fine-tuned on a downstream task.
        """
        # 输出日志信息，表示开始冻结嵌入层
        logger.info("Freezing embeddings")
        # 循环遍历模型中的嵌入层参数，设置其不需要进行梯度计算
        for parameter in self.roberta.embeddings.parameters():
            parameter.requires_grad = False
        # 输出日志信息，表示开始冻结语言适配器
        logger.info("Freezing adapters")
        # 循环遍历模型中的每个编码层，设置其适配器层参数和适配器模块参数不需要进行梯度计算
        for layer in self.roberta.encoder.layer:
            if layer.output.adapter_layer_norm is not None:
                for parameter in layer.output.adapter_layer_norm.parameters():
                    parameter.requires_grad = False
            for parameter in layer.output.adapter_modules.parameters():
                parameter.requires_grad = False
# XMOD_START_DOCSTRING 是一个包含模型文档字符串的常量，提供关于模型的详细信息。
# 该模型继承自 PreTrainedModel，可以查看超类文档以了解库实现的通用方法。
# 该模型也是 PyTorch 的 torch.nn.Module 子类，可以像普通的 PyTorch 模块一样使用。
# 请参考 PyTorch 文档以了解一般使用和行为。
# 参数:
#   config (XmodConfig): 包含模型所有参数配置的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 from_pretrained 方法以加载模型权重。



# XMOD_INPUTS_DOCSTRING 是一个包含输入文档字符串的常量，提供有关输入的描述信息。
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
            # [什么是输入 ID？](../glossary#input-ids)
        lang_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个样本应激活的语言适配器的索引。默认为 `self.config.default_language` 对应的索引。
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖填充标记索引，避免在填充位置进行注意力计算。遮盖值选择在 `[0, 1]` 之间：
            # - 1 表示**未被遮盖**的标记
            # - 0 表示**被遮盖**的标记
            # [什么是注意力蒙版？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的分段标记索引。索引选择在 `[0, 1]` 之间：
            # - 0 对应一个*句子A*标记
            # - 1 对应一个*句子B*标记
            # [什么是分段 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中的部分头部失效的遮盖。遮盖值选择在 `[0, 1]` 之间：
            # - 1 表示**未被遮盖**的头部
            # - 0 表示**被遮盖**的头部
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 选择直接传递嵌入表示而不是传递 `input_ids`。如果您想对如何将 `input_ids` 索引转换为相关向量拥有更多控制，那么这将很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 以获得更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 以获得更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    "The bare X-MOD Model transformer outputting raw hidden-states without any specific head on top.",
    XMOD_START_DOCSTRING,
)
class XmodModel(XmodPreTrainedModel):
    """
    XmodModel是XmodPreTrainedModel的子类，它是一个裸的X-MOD模型变压器，在顶部没有特定的头部输出原始的隐藏状态。
    """

    # 从transformers.models.bert.modeling_bert.BertModel.__init__中复制而来，用于初始化XmodModel实例
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化XmodEmbeddings实例
        self.embeddings = XmodEmbeddings(config)
        # 初始化XmodEncoder实例
        self.encoder = XmodEncoder(config)

        # 如果add_pooling_layer为True，则初始化XmodPooler实例并赋值给self.pooler，否则为None
        self.pooler = XmodPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.roberta.modeling_roberta.RobertaModel.get_input_embeddings复制而来，用于获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 从transformers.models.roberta.modeling_roberta.RobertaModel.set_input_embeddings中复制而来，用于设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 从transformers.models.roberta.modeling_roberta.RobertaModel._prune_heads中复制而来，用于修剪模型的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可选参数，默认为 None
        lang_ids: Optional[torch.LongTensor] = None,  # 输入的语言 IDs，可选参数，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 自注意力机制的掩码，可选参数，默认为 None
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选参数，默认为 None
        position_ids: Optional[torch.Tensor] = None,  # token 的位置 IDs，可选参数，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选参数，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，可选参数，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，可选参数，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的自注意力机制掩码，可选参数，默认为 None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 上下文键值对列表，可选参数，默认为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选参数，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的结果，可选参数，默认为 None
# 为 XMOD Model 添加文档字符串，描述其在 CLM fine-tuning 中带有 `language modeling` 头部的作用
@add_start_docstrings(
    "X-MOD Model with a `language modeling` head on top for CLM fine-tuning.",
    XMOD_START_DOCSTRING,
)
class XmodForCausalLM(XmodPreTrainedModel):
    # 定义 tied_weights_keys 属性
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.__init__ 复制并修改得到的方法
    def __init__(self, config):
        # 调用父类 XmodPreTrainedModel 的构造函数
        super().__init__(config)

        # 如果不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `XmodLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 XmodModel 和 XmodLMHead
        self.roberta = XmodModel(config, add_pooling_layer=False)
        self.lm_head = XmodLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.get_output_embeddings 复制得到的方法
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.set_output_embeddings 复制得到的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 为 model_forward 方法添加文档字符串
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM.prepare_inputs_for_generation
    # 为生成准备输入数据，处理输入的参数以及生成注意力掩码
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入数据的形状
        input_shape = input_ids.shape
        # 如果未提供注意力掩码，则创建一个全为1的注意力掩码，形状与输入数据相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果存在过去的键值，需要对decoder输入进行修剪
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 如果输入ID的长度大于过去键值的长度，则保留部分输入ID
                remove_prefix_length = past_length
            else:
                # 否则，保留仅最后一个输入ID的旧行为
                remove_prefix_length = input_ids.shape[1] - 1

            # 对输入ID进行修剪
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回准备好的输入数据和注意力掩码，以及可能存在的过去键值
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 从transformers库中的models.roberta.modeling_roberta.RobertaForCausalLM._reorder_cache方法复制而来
    # 重新排序过去的键值，以适应Beam Search
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排序后的过去键值
        reordered_past = ()
        # 遍历每一层的过去键值
        for layer_past in past_key_values:
            # 对每一层的过去状态按照beam_idx重新排序，并添加到reordered_past中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值
        return reordered_past
# 基于 `XMOD` 模型，在其顶部添加了一个 `语言建模` 头部的模型类
@add_start_docstrings(
    """X-MOD Model with a `language modeling` head on top.""",  # 描述这个模型是在 X-MOD 模型基础上添加了语言建模头部
    XMOD_START_DOCSTRING,  # 使用预定义的起始文档字符串
)
class XmodForMaskedLM(XmodPreTrainedModel):  # 定义一个 XMOD 语言模型类，继承自 XmodPreTrainedModel

    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]  # 定义权重绑定的键列表

    # 从 `transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.__init__` 复制而来，将 `Roberta` 改为 `Xmod`
    def __init__(self, config):
        super().__init__(config)  # 调用父类构造函数初始化

        if config.is_decoder:
            logger.warning(
                "If you want to use `XmodForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = XmodModel(config, add_pooling_layer=False)  # 使用 XmodModel 初始化 `self.roberta`
        self.lm_head = XmodLMHead(config)  # 使用 XmodLMHead 初始化 `self.lm_head`

        # 初始化权重并应用最终处理
        self.post_init()  # 调用 `post_init` 完成后续初始化步骤

    # 从 `transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.get_output_embeddings` 复制而来
    def get_output_embeddings(self):
        return self.lm_head.decoder  # 返回语言模型头部的输出嵌入层（decoder）

    # 从 `transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.set_output_embeddings` 复制而来
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings  # 设置新的输出嵌入层（decoder）

    # 将 `XMOD_INPUTS_DOCSTRING` 格式化为模型前向传播的文档字符串并添加注释
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 定义模型的前向传播逻辑，接受一系列输入参数并进行计算
    # 定义方法，返回类型为 Tuple[torch.Tensor] 或 MaskedLMOutput
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        lang_id: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 roberta 模型的 forward 方法，得到输出结果
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 通过 lm_head 模型得到预测得分
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 计算损失函数
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回 output，否则返回 MaskedLMOutput
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            # 返回损失、预测得分、隐藏状态和注意力
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.roberta.modeling_roberta.RobertaLMHead 复制了 XmodLMHead 类
class XmodLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 线性变换层，将输入特征映射到隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 层归一化，对输入的特征进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 线性变换层，将隐藏状态映射到词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 偏置项，用于词汇表映射
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    # 前向传播函数，对给定特征进行处理
    def forward(self, features, **kwargs):
        # 线性变换层
        x = self.dense(features)
        # GELU激活函数
        x = gelu(x)
        # 层归一化
        x = self.layer_norm(x)

        # 词汇表映射
        x = self.decoder(x)

        return x

    # 如果权重断开连接，则绑定这两个权重（在TPU上或偏置项大小发生变化时）
    def _tie_weights(self):
        # 为了加速兼容性并且不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    X-MOD Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    XMOD_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.__init__ 复制了 XmodForSequenceClassification 类
class XmodForSequenceClassification(XmodPreTrainedModel):
    # Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.__init__ with Roberta->Xmod
    def __init__(self, config):
        super().__init__(config)
        # 序列分类/回归任务的标签数量
        self.num_labels = config.num_labels
        self.config = config

        # XmodModel 类用于生成模型的隐藏状态，同时不输出汇聚结果
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # XmodClassificationHead 类用于对隐藏状态进行分类
        self.classifier = XmodClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 在前向传播中接收输入数据，并选出关键字参数用于计算
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化是否返回字典的标志，默认为模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Roberta 模型进行处理
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型自动设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不返回字典
        if not return_dict:
            # 将损失和输出组合成元组返回
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回以序列分类器输出为结果的对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加多选分类头部的 X-MOD 模型（在汇总输出之上添加一个线性层和一个 softmax 层），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    X-MOD Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    XMOD_START_DOCSTRING,
)
class XmodForMultipleChoice(XmodPreTrainedModel):
    # 从 transformers.models.roberta.modeling_roberta.RobertaForMultipleChoice.__init__ 复制而来，并替换 'Roberta' 为 'Xmod'
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)

        # 创建 XmodModel 对象
        self.roberta = XmodModel(config)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数，接受输入 ID、语言 ID、位置 ID、token 类型 ID 和注意力掩码等张量作为输入
    # 返回多选择分类任务的输出，包括损失、逻辑输出、隐状态和注意力权重
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        # 如果 return_dict 为 None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算输入的选择数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        
        # 将输入 ID、语言 ID、位置 ID、token 类型 ID 和注意力掩码展平为 2D 张量
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_lang_ids = lang_ids.repeat(input_ids.size(0) * input_ids.size(1)) if lang_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果存在输入 embedding，也将其展平为 3D 张量
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
    
        # 将展平后的输入传入 RoBERTa 模型，获取输出
        outputs = self.roberta(
            flat_input_ids,
            lang_ids=flat_lang_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从输出中获取池化输出
        pooled_output = outputs[1]
    
        # 对池化输出进行 dropout 操作
        pooled_output = self.dropout(pooled_output)
        # 将池化输出传入分类器，获取逻辑输出
        logits = self.classifier(pooled_output)
        # 根据选择数重塑逻辑输出
        reshaped_logits = logits.view(-1, num_choices)
    
        # 如果存在标签，计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
    
        # 根据 return_dict 参数返回不同的输出格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 add_start_docstrings 装饰器添加类的文档字符串，描述该类在顶部具有一个用于标记分类的头部的 X-MOD 模型
# 这样的头部通常是基于隐藏状态输出的线性层，用于命名实体识别 (NER) 任务
class XmodForTokenClassification(XmodPreTrainedModel):
    # 从 transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.__init__ 复制代码，并将 Roberta->Xmod
    # 初始化方法，接收一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设定类别数量
        self.num_labels = config.num_labels

        # 使用 XmodModel 创建一个 Xmod 模型
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 如果配置中指定分类器的 dropout，则使用配置中的值，否则使用隐藏状态的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器添加模型前向传播的文档字符串
    # 描述模型前向传播的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 该函数接受一些输入参数并返回一个 TokenClassifierOutput 对象
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        # 如果 return_dict 为 None，则使用配置中设置的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用 self.roberta 函数，传入各种输入参数，获得输出结果
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取 sequence_output 作为后续的输入
        sequence_output = outputs[0]
    
        # 对 sequence_output 应用 dropout 操作
        sequence_output = self.dropout(sequence_output)
    
        # 将 sequence_output 送入分类器得到 logits
        logits = self.classifier(sequence_output)
    
        # 初始化 loss 为 None
        loss = None
    
        # 如果传入了 labels，则计算分类损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 如果不需要返回字典，则返回一个 tuple
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 否则返回一个 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制得到的类
class XmodClassificationHead(nn.Module):
    """用于句子级分类任务的头部部分。"""

    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将hidden_size的输入转换为hidden_size的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果配置中指定了分类器的dropout，则使用该值，否则使用hidden_dropout_prob的值
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用指定的dropout值创建一个dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个全连接层，将hidden_size的输入转换为num_labels的输出
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取特征的第一个token作为输入（等同于取[CLS]）
        x = features[:, 0, :]
        # 对输入进行dropout处理
        x = self.dropout(x)
        # 使用全连接层进行线性变换
        x = self.dense(x)
        # 将线性变换的结果应用tanh函数
        x = torch.tanh(x)
        # 对结果再进行一次dropout处理
        x = self.dropout(x)
        # 再使用全连接层进行线性变换
        x = self.out_proj(x)
        # 返回输出结果
        return x


@add_start_docstrings(
    """
    用于抽取式问答任务的X-MOD模型。在隐藏状态的输出上使用一个线性层来计算“span start logits”和“span end logits”。
    """,
    XMOD_START_DOCSTRING,
)
class XmodForQuestionAnswering(XmodPreTrainedModel):
    # 从transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.__init__复制得到的方法，将Roberta->Xmod
    def __init__(self, config):
        super().__init__(config)
        # 初始化num_labels属性
        self.num_labels = config.num_labels

        # 创建XmodModel对象，并将pooling_layer设置为False
        self.roberta = XmodModel(config, add_pooling_layer=False)
        # 创建一个全连接层，将hidden_size的输入转换为num_labels的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最后的处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
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
        # 定义模型的预测函数，输入参数包括input_ids, attention_mask等，返回预测的起始位置和结束位置
        ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        # 是否返回字典类型的结果，如果return_dict参数为None，则使用模型配置中配置的属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型进行预测，输出包括序列输出和其他可选的返回结果
        outputs = self.roberta(
            input_ids,
            lang_ids=lang_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入 QA 输出层得到预测的起始位置和结束位置的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 初始化总损失为None
        total_loss = None
        # 如果给定了起始位置和结束位置的标签
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU环境下，对start_positions和end_positions添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出了模型的输入范围，这些部分的损失被忽略
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 利用交叉熵损失函数计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不返回字典类型的结果
        if not return_dict:
            # 返回包括总损失在内的输出结果
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回包括总损失、起始位置logits、结束位置logits、隐藏状态和注意力矩阵在内的结果
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的输入 IDs 创建位置 IDs，替换非填充符号为它们的位置数字。位置数字从填充索引加一开始。填充符号被忽略。这是从 fairseq 的 `utils.make_positions` 修改而来。

# 输入参数：
# - input_ids: 输入的 ID 序列
# - padding_idx: 填充符号的索引
# - past_key_values_length: 过去键值长度，用于计算增量索引

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个 mask，标记非填充符号为 1，填充符号为 0
    mask = input_ids.ne(padding_idx).int()
    # 计算每个位置的增量索引，并加上过去键值长度，然后将其乘以 mask，以忽略填充符号
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回最终的位置 IDs，加上填充索引以保持填充符号不变
    return incremental_indices.long() + padding_idx
```