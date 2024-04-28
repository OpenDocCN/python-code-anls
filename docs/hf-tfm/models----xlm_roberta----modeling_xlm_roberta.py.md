# `.\transformers\models\xlm_roberta\modeling_xlm_roberta.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括 Facebook AI Research 和 HuggingFace Inc. 团队的版权信息
# 版权声明，包括 NVIDIA 公司的版权信息
# Apache 许可证，版本 2.0，允许在符合许可证的情况下使用该文件
# 你可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则本软件按"原样"分发，不提供任何形式的明示或暗示担保
# 请参阅许可证以了解详细信息
"""PyTorch XLM-RoBERTa model."""

# 导入模块
import math
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关模块和类
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
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_xlm_roberta import XLMRobertaConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点标记
_CHECKPOINT_FOR_DOC = "xlm-roberta-base"
_CONFIG_FOR_DOC = "XLMRobertaConfig"

# XLM-RoBERTa 的预训练模型存档列表
XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    # 在 https://huggingface.co/models?filter=xlm-roberta 查看所有 XLM-RoBERTa 模型
]

# XLM-RoBERTa 的嵌入层模块
# 与 BertEmbeddings 相同，但对于位置嵌入的索引进行了微小的调整
class XLMRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    
    # 与 BertEmbeddings.__init__ 相同
    def __init__(self, config):
        # 初始化函数，接受配置参数，并调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层对象，将词汇量、隐藏层大小和填充索引作为参数
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层对象，将最大位置嵌入数和隐藏层大小作为参数
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层对象，将类型词汇量和隐藏层大小作为参数
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 将LayerNorm重命名为LayerNorm，以保持与TensorFlow模型的变量名称一致，并能够加载任何TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout对象，将隐藏层的丢弃概率作为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 是内存中连续的，并在序列化时导出
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区position_ids，由torch.arange生成，表示位置嵌入的位置索引，persistent为False表示不持久化
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区token_type_ids，由torch.zeros生成，表示标记类型索引，persistent为False表示不持久化
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 将填充索引设置为配置中的填充标记ID
        self.padding_idx = config.pad_token_id
        # 重新创建位置嵌入层对象，将最大位置嵌入数、隐藏层大小和填充索引作为参数
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果未提供位置 id，则根据输入的 token id 创建位置 id。任何填充的 token 保持填充状态。
        if position_ids is None:
            # 如果输入的 token id 不为空，则从输入 token id 创建位置 id
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            # 否则，从输入嵌入中创建位置 id
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入的 token id 不为空，则获取输入 shape
        if input_ids is not None:
            input_shape = input_ids.size()
        # 否则，获取输入嵌入的 shape
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区中的值，该值全为零，通常在自动生成时出现。注册的缓冲区可帮助用户在不传递 token_type_ids 的模型追踪时解决问题 #5664
        if token_type_ids is None:
            # 如果存在 token_type_ids 属性
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则，将 token_type_ids 设置为全零的 tensor
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为空，则使用 word_embeddings 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算嵌入
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为 "absolute"
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 经过 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 经过 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings

    # 从输入嵌入中创建位置 id
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序的位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.roberta.modeling_roberta.RobertaSelfAttention复制代码，将Roberta替换为XLMRoberta
class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不是注意力头数的倍数且没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 查询、键和值的线性变换
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则创建距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器
        self.is_decoder = config.is_decoder

    # 调整形状以便计算得分
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从transformers.models.roberta.modeling_roberta.RobertaSelfOutput复制代码，将Roberta替换为XLMRoberta
class XLMRobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从transformers.models.roberta.modeling_roberta.RobertaAttention复制到XLMRobertaAttention
class XLMRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化self属性为XLMRobertaSelfAttention对象
        self.self = XLMRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化output属性为XLMRobertaSelfOutput对象
        self.output = XLMRobertaSelfOutput(config)
        # 初始化pruned_heads属性为空集合
        self.pruned_heads = set()

    # 精简头部
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可精简头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 精简线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已精简的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播
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
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力，则添加到输出中
        return outputs


# 从transformers.models.roberta.modeling_roberta.RobertaIntermediate复制到XLMRobertaIntermediate
class XLMRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化dense属性为Linear对象
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act为字符串，则使用ACT2FN中对应的激活函数；否则使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.roberta.modeling_roberta.RobertaOutput复制到XLMRobertaOutput
class XLMRobertaOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，对隐藏层的结果进行归一化，eps为配置中的层归一化eps值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，按照配置中的hidden_dropout_prob丢弃部分隐藏层结果
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受隐藏状态和输入张量，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用dropout层对处理后的隐藏状态进行丢弃
        hidden_states = self.dropout(hidden_states)
        # 使用LayerNorm层对丢弃后的隐藏状态进行归一化，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回最终的隐藏状态
        return hidden_states
# 从transformers.models.roberta.modeling_roberta.RobertaLayer复制代码，并将Roberta改为XLMRoberta
class XLMRobertaLayer(nn.Module):
    def __init__(self, config):
        # 初始化XLMRobertaLayer对象
        super().__init__()
        # 设置前向传播中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 初始化注意力机制
        self.attention = XLMRobertaAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器则引发错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力机制
            self.crossattention = XLMRobertaAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = XLMRobertaIntermediate(config)
        # 初始化输出层
        self.output = XLMRobertaOutput(config)

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
        # 如果过去的键/值存在，则解码器的单向自注意力缓存键/值元组位于位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力模型计算自注意力
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后的输出是自注意力缓存的元组
        if self.is_decoder:
            # 解码器的输出不包括最后一个元组，它是自注意力缓存
            outputs = self_attention_outputs[1:-1]
            # 获取当前的键/值
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果需要输出注意力权重，则添加自注意力
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 如果传入了`encoder_hidden_states`，则必须通过设置`config.add_cross_attention=True`来实例化具有交叉注意力层的模型
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的缓存键/值元组位于过去键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模型计算交叉注意力
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力到输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前的键/值元组中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用前馈网络分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 前馈网络的中间输出
        intermediate_output = self.intermediate(attention_output)
        # 前馈网络的输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.roberta.modeling_roberta.RobertaEncoder中复制的代码，将Roberta->XLMRoberta
class XLMRobertaEncoder(nn.Module):
    # 初始化函数，接受config参数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个包含多个XLMRobertaLayer对象的模块列表，列表长度为config.num_hidden_layers
        self.layer = nn.ModuleList([XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor, # 隐藏状态
        attention_mask: Optional[torch.FloatTensor] = None, # 注意力掩码
        head_mask: Optional[torch.FloatTensor] = None, # 头部掩码
        encoder_hidden_states: Optional[torch.FloatTensor] = None, # 编码器隐藏状态
        encoder_attention_mask: Optional[torch.FloatTensor] = None, # 编码器注意力掩码
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, # 过去键值
        use_cache: Optional[bool] = None, # 使用缓存
        output_attentions: Optional[bool] = False, # 输出注意力
        output_hidden_states: Optional[bool] = False, # 输出隐藏状态
        return_dict: Optional[bool] = True, # 返回字典
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，则初始化一个空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组用于存储所有自注意力权重
        all_self_attentions = () if output_attentions else None
        # 如果输出注意力权重且模型配置中包含交叉注意力，则初始化一个空元组用于存储所有交叉注意力权重
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且处于训练状态，则处理缓存参数的使用情况
        if self.gradient_checkpointing and self.training:
            # 如果同时设置了使用缓存和梯度检查点，则发出警告并将使用缓存设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果设置了使用缓存，则初始化一个空元组用于存储下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码（如果有）
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值（如果有）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练状态，则使用梯度检查点功能调用当前层的前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            # 否则，直接调用当前层的前向传播
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的自注意力权重添加到所有自注意力权重中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中包含交叉注意力，则将当前层的交叉注意力权重添加到所有交叉注意力权重中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回一个元组，包含需要返回的所有值
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则，返回一个字典对象，包含需要返回的所有值
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaPooler中拷贝，将Roberta替换为XLMRoberta
class XLMRobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 使用线性层进行全连接
        self.activation = nn.Tanh()  # tanh激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过获取隐藏状态的第一个标记来对模型进行“池化”
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)  # 进行全连接操作
        pooled_output = self.activation(pooled_output)  # 使用激活函数
        return pooled_output  # 返回池化输出


# 从transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel中拷贝，将Roberta替换为XLMRoberta
class XLMRobertaPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和下载加载预训练模型的抽象类。
    """

    config_class = XLMRobertaConfig  # 使用XLMRobertaConfig类的配置
    base_model_prefix = "roberta"  # 基础模型前缀为roberta
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["XLMRobertaEmbeddings", "XLMRobertaSelfAttention"]  # 不进行分割的模块名称列表


    # 从transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights中拷贝
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与TF版本略有不同，使用标准正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 使用标准正态分布进行初始化
            if module.bias is not None:
                module.bias.data.zero_()  # 将偏置项置零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 使用标准正态分布进行初始化
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 将填充索引处的权重置零
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()  # 将偏置项置零
            module.weight.data.fill_(1.0)  # 将权重填充为1.0


XLM_ROBERTA_START_DOCSTRING = r"""

    该模型继承自[`PreTrainedModel`]。查看超类文档以了解库实现的通用方法（如下载或保存，调整输入嵌入大小，修剪头等）。

    该模型还是PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。
    将其视为常规的PyTorch Module，并参考PyTorch文档以了解一切有关一般用法和行为的相关事宜。

    参数：
        config ([`XLMRobertaConfig`]): 模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只加载配置。
            检查[`~PreTrainedModel.from_pretrained`]方法以载入模型权重。
"""

XLM_ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中词汇标记的索引。
            # 可以使用[`AutoTokenizer`]获得索引。参见[`PreTrainedTokenizer.encode`]和[`PreTrainedTokenizer.__call__`]获取详细信息。
            # [什么是输入ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。
            # 掩码的值在`[0, 1]`之间选取：
            # - 1表示**未屏蔽**的标记，
            # - 0表示**已屏蔽**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引。索引在`[0, 1]`中选择：
            # - 0对应于*句子A*的标记，
            # - 1对应于*句子B*的标记。
            # [什么是标记类型ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。在范围`[0, config.max_position_embeddings - 1]`中选择。
            # [什么是位置ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块的特定头部失效的掩码。掩码值在`[0, 1]`之间选取：
            # - 1表示头部**未被屏蔽**，
            # - 0表示头部**被屏蔽**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示，而不是传递`input_ids`。如果希望更多控制将`input_ids`索引转换为相关向量的方式，则可以使用此选项，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请查看返回张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请查看返回张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通的元组。
``` 
"""
定义 XLMRobertaModel 类，继承自 XLMRobertaPreTrainedModel 类
"""
@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
# 从 transformers.models.roberta.modeling_roberta.RobertaModel 中复制而来，将 Roberta 替换为 XLMRoberta, ROBERTA 替换为 XLM_ROBERTA
class XLMRobertaModel(XLMRobertaPreTrainedModel):
    """
    模型可以作为编码器（仅具有自注意力）或解码器的角色，此时在自注意力层之间添加了一个交叉注意力层，遵循 *Attention is all you need* 中描述的架构，作者为 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin.

    要使模型成为解码器，需使用配置设置的 `is_decoder` 参数初始化为 `True`。要用于 Seq2Seq 模型，需将模型初始化为 `is_decoder` 参数和 `add_cross_attention` 参数设置为 `True`；然后期望在前向传递中作为输入的是 `encoder_hidden_states`。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 中复制而来，将 Bert 替换为 XLMRoberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 使用配置初始化 XLMRobertaEmbeddings、XLMRobertaEncoder
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)

        # 如有需要，初始化 XLMRobertaPooler
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的注意力头
    # heads_to_prune：层数到要剪枝的注意力头列表的字典
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 为模型前向传递添加文档字符串
    # XLM_ROBERTA_INPUTS_DOCSTRING 格式是 (batch_size, sequence_length)
    # add_code_sample_docstrings 从 _CHECKPOINT_FOR_DOC、BaseModelOutputWithPoolingAndCrossAttentions、_CONFIG_FOR_DOC 获取文档字符串
    # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制而来
    # 定义一个函数 forward，用于模型的前向传播
    def forward(
        # 输入的 token IDs，是一个可选的 PyTorch 张量
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，用于指定模型需要关注的 token，是一个可选的 PyTorch 张量
        attention_mask: Optional[torch.Tensor] = None,
        # 标记 token 类型的张量，是一个可选的 PyTorch 张量
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID 的张量，用于指定输入 token 在序列中的位置，是一个可选的 PyTorch 张量
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码，用于控制特定头部在注意力计算中的作用，是一个可选的 PyTorch 张量
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入的张量，是一个可选的 PyTorch 张量
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器隐藏状态的张量，是一个可选的 PyTorch 张量
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器注意力掩码，是一个可选的 PyTorch 张量
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 过去的键值对，是一个可选的 PyTorch 浮点数列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否使用缓存，是一个可选的布尔值
        use_cache: Optional[bool] = None,
        # 是否输出注意力值，是一个可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，是一个可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，是一个可选的布尔值
        return_dict: Optional[bool] = None,
# 使用 `add_start_docstrings` 装饰器为模型添加文档字符串，说明了这是一个在 CLM 微调中使用的带有 `language modeling` 头部的 XLM-RoBERTa 模型
# 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM 复制而来，并将 Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForCausalLM(XLMRobertaPreTrainedModel):
    # 定义权重共享的键列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)

        # 如果不是解码器，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `XLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 XLM-RoBERTa 模型和语言建模头部
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 重写前向传播方法，添加文档字符串，替换输出文档字符串的类型为 CausalLMOutputWithCrossAttentions，配置类为 _CONFIG_FOR_DOC
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
    # 为生成准备输入。这个方法准备用于模型生成的输入。
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入张量的形状
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为1的张量作为注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果过去的键值对被使用，则截断decoder输入ID
        if past_key_values is not None:
            # 获取过去键值对的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截断输入ID
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含输入ID、注意力掩码和过去键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存。这个方法用于重新排序过去的键值对缓存。
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空的元组用于存储重新排序后的过去键值对
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 对每一个过去状态，根据beam_idx重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 为XLM-RoBERTa Model添加文档字符串，并参考XLM_ROBERTA_START_DOCSTRING以及transformers.models.roberta.modeling_roberta.RobertaForMaskedLM
class XLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    # 被绑定权重的键值对列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 如果config.is_decoder为True时，则发出警告
        if config.is_decoder:
            logger.warning(
                "如果要使用`XLMRobertaForMaskedLM`，请确保`config.is_decoder=False`，以便进行双向自注意力。"
            )

        # 初始化XLMRobertaModel和XLMRobertaLMHead
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 将XLM_ROBERTA_INPUTS_DOCSTRING和add_code_sample_docstrings应用到模型的前向方法上
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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

注意：由于代码太长，部分内容已被省略。
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 检查是否返回字典，默认使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过 RoBERTa 模型处理输入
        outputs = self.roberta(
            input_ids,
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
        # 获得 RoBERTa 模型输出的序列结果
        sequence_output = outputs[0]
        # 对序列输出进行预测得到的分数
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 将标签移到正确的设备以启用模型并行化
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            # 计算损失，将预测分数和标签展平成1维
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 根据返回字典标志决定输出内容
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回带有损失、预测logits、隐藏状态和注意力的 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaLMHead中复制代码
class XLMRobertaLMHead(nn.Module):
    """用于遮蔽语言建模的Roberta头部。"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建全连接层，输入和输出维度都是hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建LayerNorm层，参数为hidden_size

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)  # 创建全连接层，输入维度是hidden_size，输出维度是vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个偏置参数，维度为vocab_size
        self.decoder.bias = self.bias  # 将全连接层的偏置参数设置为刚刚创建的偏置参数

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 使用全连接层处理输入features
        x = gelu(x)  # 使用gelu激活函数处理x
        x = self.layer_norm(x)  # 使用LayerNorm处理x

        # 通过带有偏置的全连接层将x映射回词汇表的大小
        x = self.decoder(x)

        return x  # 返回处理后的结果

    def _tie_weights(self):
        # 当这两个权重断开连接时（在TPU上或偏置被调整大小时），将这两个权重连接起来
        # 为了加速兼容性和不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    XLM-RoBERTa模型变换器，在顶部具有序列分类/回归头（在池化输出顶部的线性层），例如用于GLUE任务。
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification复制代码，将Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 获取配置中的标签数量
        self.config = config  # 存储配置信息

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)  # 创建XLM-RoBERTa模型，不添加池化层
        self.classifier = XLMRobertaClassificationHead(config)  # 创建XLM-RoBERTa的分类头部

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="cardiffnlp/twitter-roberta-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'optimism'",
        expected_loss=0.08,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,  # 是否返回字典格式的输出
        )
        sequence_output = outputs[0]  # 获取 RoBERTa 模型的输出序列
        logits = self.classifier(sequence_output)  # 用全连接层对输出序列进行分类

        loss = None  # 初始化损失为 None
        if labels is not None:  # 如果有标签
            # 将标签移动到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            if self.config.problem_type is None:  # 如果问题类型未指定
                if self.num_labels == 1:  # 如果标签数量为 1
                    self.config.problem_type = "regression"  # 设定问题类型为回归
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):  # 如果标签数量大于 1 并且类型是长整型或整型
                    self.config.problem_type = "single_label_classification"  # 设定问题类型为单标签分类
                else:  # 其他情况
                    self.config.problem_type = "multi_label_classification"  # 设定问题类型为多标签分类

            if self.config.problem_type == "regression":  # 如果问题类型是回归
                loss_fct = MSELoss()  # 使用均方误差作为损失函数
                if self.num_labels == 1:  # 如果标签数量为 1
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # 计算损失
                else:  # 其他情况
                    loss = loss_fct(logits, labels)  # 计算损失
            elif self.config.problem_type == "single_label_classification":  # 如果问题类型是单标签分类
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失作为损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失
            elif self.config.problem_type == "multi_label_classification":  # 如果问题类型是多标签分类
                loss_fct = BCEWithLogitsLoss()  # 使用带 logits 的二分类交叉熵损失作为损失函数
                loss = loss_fct(logits, labels)  # 计算损失

        if not return_dict:  # 如果不返回字典格式的输出
            output = (logits,) + outputs[2:]  # 模型输出结果
            return ((loss,) + output) if loss is not None else output  # 返回损失和模型输出结果或者仅返回模型输出结果

        # 返回序列分类器的输出对象，包括损失、逻辑回归输出、隐藏状态和注意力
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 该类是 XLM-RoBERTa 模型的多项选择分类头版本
@add_start_docstrings(
    """
    XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultipleChoice(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造方法
        super().__init__(config)
        # 创建 XLM-RoBERTa 模型的实例
        self.roberta = XLMRobertaModel(config)
        # 创建一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层用于分类
        self.classifier = nn.Linear(config.hidden_size, 1)
        # 进行权重初始化和其他后续处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 代码省略
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置返回字典，如果不为空则使用输入值，否则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选择个数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的IDS扁平化，以便处理
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用roberta模型进行推理
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取池化后的输出
        pooled_output = outputs[1]

        # 通过dropout层处理池化后的输出
        pooled_output = self.dropout(pooled_output)
        # 通过分类器获取logits
        logits = self.classifier(pooled_output)
        # 调整logits的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 计算损失
        loss = None
        if labels is not None:
            # 将标签转移到正确的设备上，以实现模型并行计算
            labels = labels.to(reshaped_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典，则以元组形式返回损失和输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则以多��模型输出对象的形式返回损失、logits、隐藏状态和注意力权重
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 XLM-RoBERTa 模型添加了一个标记分类头部，用于例如命名实体识别（NER）任务
# 这个头部是在隐藏状态输出的基础上添加了一个线性层
@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForTokenClassification复制并修改而来，将Roberta->XLMRoberta，ROBERTA->XLM_ROBERTA
class XLMRobertaForTokenClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 初始化 XLM-RoBERTa 模型
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # 获取分类器的丢弃率，如果没有指定则使用配置中的隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 初始化丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 对模型进行前向传播
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 设置默认返回字典，如果未指定则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用RoBERTa模型进行前向传播
        outputs = self.roberta(
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

        # 获取RoBERTa模型输出的序列表示
        sequence_output = outputs[0]

        # 对序列表示进行dropout处理
        sequence_output = self.dropout(sequence_output)
        # 通过分类器获取预测的logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备上以启用模型并行处理
            labels = labels.to(logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # 如果不返回字典, 则返回logits以及其他模型输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回Token分类器输出对象，包括损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制代码，并将Roberta->XLMRoberta
class XLMRobertaClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果config.classifier_dropout为None，则使用config.hidden_dropout_prob进行dropout，否则使用config.classifier_dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取features的第一个元素，即<\s>标记（等同于[CLS]）
        x = features[:, 0, :]
        # 对x进行dropout
        x = self.dropout(x)
        # 使用全连接层dense
        x = self.dense(x)
        # 对x进行tanh激活函数
        x = torch.tanh(x)
        # 再次进行dropout
        x = self.dropout(x)
        # 使用全连接层out_proj
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    顶部带有用于类似SQuAD的抽取式问答任务的跨度分类头的XLM-RoBERTa模型
    （在隐藏状态输出的线性层上计算`span start logits`和`span end logits`）。
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
# 从transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering复制代码，将Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForQuestionAnswering(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 创建XLMRobertaModel，并关闭添加池化层选项
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="deepset/roberta-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="' puppet'",
        expected_loss=0.86,
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
    # 这段代码定义了一个 Question Answering 模型的前向传播过程，包括了损失函数的计算
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        # 检查是否使用 return_dict，如果没有则使用默认配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过 ROBERTA 模型得到输出序列
        outputs = self.roberta(
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
    
        # 取出序列输出
        sequence_output = outputs[0]
    
        # 通过 qa_outputs 层得到 start 和 end logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    
        # 如果提供了 start_positions 和 end_positions，计算损失
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 和 end_positions 的 size 大于 1，压缩最后一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时 start/end positions 会超出模型输入序列长度，我们忽略这些值
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
    
            # 使用交叉熵损失计算 start 和 end 的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
    
        # 如果 return_dict 为 False，返回一个包含 start_logits、end_logits 和其他输出的元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        # 否则返回一个 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的 input_ids 中创建位置标识符，用 padding_idx 标识填充位置，过去的键值长度为 past_key_values_length
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # 创建一个与 input_ids 形状相同的张量，其中非填充符号用其位置数替换，位置编号从 padding_idx+1 开始
    mask = input_ids.ne(padding_idx).int()
    # 计算累积和，表示每个位置的位置编号，再加上过去的键值长度，然后乘以 mask，使填充位置保持为0
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回位置编号张量，加上填充索引，保持填充位置不变
    return incremental_indices.long() + padding_idx
```