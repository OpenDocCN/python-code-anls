# `.\models\data2vec\modeling_data2vec_text.py`

```
# 设置代码文件编码为 UTF-8

# 版权声明
# 版权归属于 2022 年 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版本获得授权，除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则根据许可证分发的软件
# 按"原样"分发，不提供任何明示或暗示的保证或条件。
# 请参阅许可证了解具体语言授权限制和不适用担保

# 引入 PyTorch 和相关定义
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引入相关的输出类型
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

# 引入模型相关的工具函数
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_data2vec_text import Data2VecTextConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 2

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "facebook/data2vec-text-base"
_CONFIG_FOR_DOC = "Data2VecTextConfig"

# 预训练模型存档列表
DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-text-base",
    # 查看所有的 Data2Vec 模型列表: https://huggingface.co/models?filter=data2vec-text
]

# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制到 Data2VecTextForTextEmbeddings
class Data2VecTextForTextEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制过来的
    # 初始化函数，接受配置参数，并调用父类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，根据词汇量、隐藏层大小和填充标记创建词嵌入矩阵
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，根据最大位置嵌入数量和隐藏层大小创建位置嵌入矩阵
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，根据标记类型词汇量和隐藏层大小创建标记类型嵌入矩阵
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 不使用蛇形命名法以保持与 TensorFlow 模型变量名的一致性，并能够加载任何 TensorFlow 检查点文件
        # 创建层归一化层，根据隐藏层大小和层归一化参数创建层归一化层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，根据隐藏层 dropout 概率创建 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) 在序列化时是连续的，并在序列化时导出
        # 根据配置中的位置嵌入类型创建位置嵌入类型标志
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置嵌入矩阵，持久性为 False
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册标记类型嵌入矩阵，持久性为 False
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置填充标记 ID
        self.padding_idx = config.pad_token_id
        # 创建位置嵌入层，根据最大位置嵌入数量、隐藏层大小和填充标记创建位置嵌入矩阵
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果未提供位置 id，则根据输入 token id 创建位置 id。任何填充的 token 保持填充状态。
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果提供了输入 token id，则获取其形状
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            # 获取序列长度
            seq_length = input_shape[1]

            # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中全部为零。这通常发生在自动生成时，
            # 注册的缓冲区可以在不传递 token_type_ids 的情况下帮助用户跟踪模型，解决了问题 #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    # 获取注册的 token_type_ids 缓冲区，并截取与序列长度相同的部分
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    # 扩展为与输入形状相同的张量
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    # 如果未注册 token_type_ids，则创建全部为零的张量
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # 如果未提供输入嵌入，则使用 word_embeddings 层对输入 token id 进行嵌入
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # 使用 token_type_embeddings 层对 token_type_ids 进行嵌入
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            # 将输入嵌入和 token 类型嵌入相加
            embeddings = inputs_embeds + token_type_embeddings
            # 如果位置嵌入类型为 "absolute"，则使用位置嵌入层对位置 id 进行嵌入，并将结果加到 embeddings 上
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            # 对 embeddings 进行 LayerNorm 处理
            embeddings = self.LayerNorm(embeddings)
            # 对 embeddings 进行 dropout 处理
            embeddings = self.dropout(embeddings)
            # 返回 embeddings
            return embeddings

        def create_position_ids_from_inputs_embeds(self, inputs_embeds):
            """
            We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

            Args:
                inputs_embeds: torch.Tensor

            Returns: torch.Tensor
            """
            # 获取输入嵌入的形状
            input_shape = inputs_embeds.size()[:-1]
            # 获取序列长度
            sequence_length = input_shape[1]

            # 生成从 padding_idx + 1 到 sequence_length + padding_idx + 1 的顺序位置 id
            position_ids = torch.arange(
                self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
            )
            # 将位置 id 扩展为与输入形状相同的张量
            return position_ids.unsqueeze(0).expand(input_shape)
# 从 transformers.models.roberta.modeling_roberta.RobertaSelfAttention 复制代码，修改类名为 Data2VecTextSelfAttention
class Data2VecTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查 hidden_size 是否是 num_attention_heads 的倍数，如果不是则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的 Linear 层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置嵌入，创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力得分所需要的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制代码，修改类名为 Data2VecTextSelfOutput
class Data2VecTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建全连接层 dense
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertAttention复制并修改的Data2VecTextAttention类
class Data2VecTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化self属性为Data2VecTextSelfAttention类对象
        self.self = Data2VecTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化output属性为Data2VecTextSelfOutput类对象
        self.output = Data2VecTextSelfOutput(config)
        # 初始化pruned_heads属性为空集合
        self.pruned_heads = set()

    # 剪枝头部
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
    ) -> Tuple[torch.Tensor]:
        # 调用self属性的前向传播函数
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 通过output属性，将self_outputs[0]和hidden_states传入得到attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 输出结果
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了attention，将其添加到结果中
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制的Data2VecTextIntermediate类
class Data2VecTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化dense属性为一个线性层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断config.hidden_act的类型并选取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过dense属性对hidden_states进行线性变换
        hidden_states = self.dense(hidden_states)
        # 通过intermediate_act_fn对hidden_states进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput复制的Data2VecTextOutput类
class Data2VecTextOutput(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入维度为配置对象中的中间尺寸，输出维度为配置对象中的隐藏尺寸
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入维度为配置对象中的隐藏尺寸，epsilon 参数为配置对象中的层归一化 epsilon
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，丢弃概率为配置对象中的隐藏层 Dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受两个张量作为输入，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 对 Dropout 后的隐藏状态进行层归一化，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer中复制代码，并将Bert->Data2VecText
class Data2VecTextLayer(nn.Module):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 设置前向传播中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度的索引
        self.seq_len_dim = 1
        # 初始化注意力机制
        self.attention = Data2VecTextAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力
        if self.add_cross_attention:
            # 如果不是解码器则引发值错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨注意力机制
            self.crossattention = Data2VecTextAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = Data2VecTextIntermediate(config)
        # 初始化输出层
        self.output = Data2VecTextOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果有过去的键/值缓存，则从中提取自注意力部分的键/值，否则为None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力模块处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 提取自注意力模块的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重则加入自注意力部分

        cross_attn_present_key_value = None
        # 如果是解码器并且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有交叉注意力层，则抛出异常
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 在过去的键/值缓存元组的位置3,4上提取交叉注意力的键/值缓存
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模块处理自注意力模块的输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 提取交叉注意力模块的输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重则加入交叉注意力部分

            # 将交叉注意力的缓存加入到现在的键/值缓存元组的位置3,4上
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对注意力输出应用分块处理，并返回结果
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力的键/值作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 处理前馈部分的函数
    def feed_forward_chunk(self, attention_output):
        # 通过中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 通过输出层处理中间层输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理结果
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制并修改了类名为Data2VecTextEncoder
class Data2VecTextEncoder(nn.Module):
    # 初始化Data2VecTextEncoder类
    def __init__(self, config):
        super().__init__()
        # 保存配置信息到self.config
        self.config = config
        # 创建一个包含多个Data2VecTextLayer对象的ModuleList，数量为config.num_hidden_layers
        self.layer = nn.ModuleList([Data2VecTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        # 隐藏状态张量
        hidden_states: torch.Tensor,
        # 注意力掩码
        attention_mask: Optional[torch.FloatTensor] = None,
        # 头掩码
        head_mask: Optional[torch.FloatTensor] = None,
        # 编码器隐藏状态
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 编码器注意力掩码
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 过去的键值
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 输出注意力
        output_attentions: Optional[bool] = False,
        # 输出隐藏状态
        output_hidden_states: Optional[bool] = False,
        # 是否返回字典
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不输出隐藏状态，那么将其设置为一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力，那么将其设置为一个空元组
        all_self_attentions = () if output_attentions else None
        # 如果不输出交叉注意力，或者配置中不包含交叉注意力，那么将其设置为一个空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了渐变检查点且处于训练状态下
        if self.gradient_checkpointing and self.training:
            # 如果使用缓存，则发出警告并设置 use_cache=False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不使用缓存，则将 next_decoder_cache 设置为一个空元组
        next_decoder_cache = () if use_cache else None
        # 遍历所有层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果 head_mask 存在，那么将其设置为 layer_head_mask
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果 past_key_values 存在，那么将其设置为 past_key_value
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了渐变检查点且处于训练状态下
            if self.gradient_checkpointing and self.training:
                # 使用渐变检查点的函数来计算层的输出
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
            else:
                # 否则，直接调用层模块来计算输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力，将当前层的自注意力信息添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中包含交叉注意力，则将当前层的交叉注意力信息添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，将最后一个隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，就返回一个包含指定内容的元组
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
        # 否则，返回一个包含指定内容的 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 定义一个自定义的模型 `Data2VecTextPooler` 用于文本池化。

class Data2VecTextPooler(nn.Module):
  
  # 初始化函数，接收配置信息作为参数
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出大小均为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个激活函数
        self.activation = nn.Tanh()

    # 前向传播函数，接收隐藏状态的张量作为输入，返回经过池化处理后的输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将第一个标记对应的隐藏状态作为池化后的输出
        first_token_tensor = hidden_states[:, 0]
        # 经过全连接层处理
        pooled_output = self.dense(first_token_tensor)
        # 经过激活函数处理
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 定义一个预训练模型类 `Data2VecTextPreTrainedModel`，并继承自 `PreTrainedModel`。

class Data2VecTextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
  
    # 模型的配置类为 `Data2VecTextConfig`
    config_class = Data2VecTextConfig
    # 基础模型的前缀为 "data2vec_text"
    base_model_prefix = "data2vec_text"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["Data2VecTextForTextEmbeddings", "Data2VecTextLayer"]

    # 初始化模型权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 权重初始化为标准正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 偏置初始化为0
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 权重初始化为标准正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在padding_idx，则将其对应的权重初始化为0
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, "bias") and module.bias is not None:
                # 如果存在偏置，则将其初始化为0
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                # 如果存在权重，则将其初始化为1
                module.weight.data.fill_(1.0)


DATA2VECTEXT_START_DOCSTRING = r"""
    Data2VecText was proposed in [data2vec: A General Framework for Self-supervised Learning in Speech, Vision and
    Language](https://arxiv.org/pdf/2202.03555) by Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu and
    Michael Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    # 参数：
    # config ([`Data2VecTextConfig`]): 模型配置类，包含模型的所有参数。
    # 用配置文件初始化不会加载与模型关联的权重，只加载配置信息。
    # 查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# DATA2VECTEXT_INPUTS_DOCSTRING: Data2VecText模型的输入参数说明文档字符串
DATA2VECTEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            输入序列标记在词汇表中的索引。

            可以使用[`AutoTokenizer`]获取索引。详情请见[`PreTrainedTokenizer.encode`]和
            [`PreTrainedTokenizer.__call__`]。

            [input IDs是什么?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            避免对填充标记索引执行注意力计算的掩码。掩码值选在 `[0, 1]` 之间：

            - 1表示**未屏蔽**的标记,
            - 0表示**屏蔽**的标记。

            [注意力掩码是什么?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            分段标记索引，指示输入的第一部分和第二部分。索引选在 `[0,
            1]` 之间：

            - 0 对应*句子A*的标记,
            - 1 对应*句子B*的标记。

            [标记类型ID是什么?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            输入序列标记在位置嵌入中的位置索引。选在范围 `[0,
            config.max_position_embeddings - 1]` 之间。

            [位置ID是什么?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            空值化自注意力模块的选择头部的掩码。掩码值选在 `[0, 1]` 之间:

            - 1表示头部**未屏蔽**,
            - 0表示头部**屏蔽**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            选择直接传递嵌入表示代替 `input_ids`。如果需要对如何将 `input_ids`索引转换为相关向量有更多控制权，那么这是很有用的，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。更多详情请见返回张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。更多详情请见返回张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回[`~utils.ModelOutput`]而不是普通元组。

"""


@add_start_docstrings(
    "The bare Data2VecText Model for text transformer outputting raw hidden-states without any specific head on top.",
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextModel(Data2VecTextPreTrainedModel):
    """
    # 这个模型可以表现为编码器（只有自注意力）或解码器,在后者情况下,在自注意力层之间添加一层交叉注意力,遵循 Attention is all you need 论文中描述的架构。
        The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
        cross-attention is added between the self-attention layers, following the architecture described in *Attention is
        all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomex, Lukasz
        Kaiser and Illia Polosukhin.
    
        # 如果要作为解码器使用,需要在配置中将 is_decoder 参数设置为 True。
        # 如果要在 Seq2Seq 模型中使用,需要将 is_decoder 和 add_cross_attention 参数都设置为 True,并在前向传播中输入 encoder_hidden_states。
        To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
        to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
        `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    
        .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    
        """
    
        # 初始化模型,包括文本嵌入、编码器和池化层
        def __init__(self, config, add_pooling_layer=True):
            super().__init__(config)
            self.config = config
    
            self.embeddings = Data2VecTextForTextEmbeddings(config)
            self.encoder = Data2VecTextEncoder(config)
    
            self.pooler = Data2VecTextPooler(config) if add_pooling_layer else None
    
            # 初始化权重并应用最终处理
            self.post_init()
    
        # 获取输入嵌入层
        def get_input_embeddings(self):
            return self.embeddings.word_embeddings
    
        # 设置输入嵌入层
        def set_input_embeddings(self, value):
            self.embeddings.word_embeddings = value
    
        # 修剪模型头部
        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)
    
        # 前向传播函数
        @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=BaseModelOutputWithPoolingAndCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
        # 复制自 transformers.models.bert.modeling_bert.BertModel.forward
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
# 创建一个带有LM头部的Data2VecText模型，用于CLM微调
@add_start_docstrings(
    """Data2VecText Model with a `language modeling` head on top for CLM fine-tuning.""", DATA2VECTEXT_START_DOCSTRING
)
class Data2VecTextForCausalLM(Data2VecTextPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `Data2VecTextLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化Data2VecText模型和LM头部
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        self.lm_head = Data2VecTextLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成准备输入数据的方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果模型用作编码器-解码器模型的解码器，会动态创建解码器注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用过去的键值，会截取输入的解码器id
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：保留最终ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排列缓存中的过去键和值，以适应beam搜索后的顺序
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排列后的过去键和值
        reordered_past = ()
        # 遍历每一层的过去键和值
        for layer_past in past_key_values:
            # 将过去键和值按照beam_idx中的索引重新排列，并添加到reordered_past中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排列后的过去键和值
        return reordered_past
# 为 Data2VecTextForMaskedLM 类添加文档字符串，描述该类具有一个在其顶部的语言建模头部
class Data2VecTextForMaskedLM(Data2VecTextPreTrainedModel):
    # 定义需要共享参数的键值对
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化函数，接受 config 参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)

        # 如果配置中指定为解码器，则发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `Data2VecTextForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 Data2VecTextModel 对象和 Data2VecTextLMHead 对象
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        self.lm_head = Data2VecTextLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回 lm_head.decoder 作为输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置 lm_head.decoder 为新的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
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
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 确定是否返回字典类型的输出结果，若未指定则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入序列转换为向量表示
        outputs = self.data2vec_text(
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 预测下一个词的概率分布
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果存在标签，则计算掩码语言模型损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()

            # 将标签转移到与预测得分相同的设备上
            labels = labels.to(prediction_scores.device)
            # 计算掩码语言模型损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回字典类型的输出结果，则组装输出元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回掩码语言模型的输出结果
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaLMHead复制而来，将Roberta替换为Data2VecText
class Data2VecTextLMHead(nn.Module):
    """用于数据2文本头部的遮盖语言建模。"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建一个LayerNormalization层

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)  # 创建一个全连接层
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个参数张量
        self.decoder.bias = self.bias  # 设置解码器的偏置

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 将输入通过全连接层
        x = gelu(x)  # 将输入应用GELU激活函数
        x = self.layer_norm(x)  # 将输入通过LayerNormalization层

        # 通过偏置将其投影回词汇表的大小
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 当这两个权重被断开连接时把它们捆绑在一起（在TPU上或者当偏置被重新调整大小时）
        # 为了加速兼容性和不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    用于顶部有序列分类/回归头部的Data2VecText模型转换器（在池化输出之上有一个线性层），例如用于GLUE任务。
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextForSequenceClassification(Data2VecTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 设置标签数量
        self.config = config

        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)  # 创建Data2VecText模型
        self.classifier = Data2VecTextClassificationHead(config)  # 创建Data2VecText的分类头部

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        定义函数的输入参数和返回类型，该函数用于进行序列分类或回归任务的计算。labels参数表示用于计算损失的标签，如果config.num_labels为1，则进行回归任务计算（使用均方差损失），如果config.num_labels大于1，则进行分类任务计算（使用交叉熵损失）。

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        初始化return_dict变量，检查return_dict是否为None，如果是则使用self.config.use_return_dict的值。

        outputs = self.data2vec_text(
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
        调用self.data2vec_text方法，对输入进行文本向量化。

        sequence_output = outputs[0]
        获取outputs的第一个元素，作为序列的输出。

        logits = self.classifier(sequence_output)
        使用self.classifier方法对sequence_output进行分类得到预测结果logits。

        loss = None
        初始化loss变量，用于表示损失值，初始值为None。

        if labels is not None:
            判断labels是否为空，如果不为空进行下面的操作。

            labels = labels.to(logits.device)
            将labels转移到与logits相同的设备上，以便后续计算。

            if self.config.problem_type is None:
                判断self.config.problem_type是否为空，如果为空进行下面的操作。

                if self.num_labels == 1:
                    判断类别数是否为1，如果是进行下面的操作。

                    self.config.problem_type = "regression"
                    将问题类型设为回归任务。

                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    判断类别数是否大于1且labels的数据类型是否为torch.long或torch.int，如果是进行下面的操作。

                    self.config.problem_type = "single_label_classification"
                    将问题类型设为单标签分���任务。

                else:
                    进行下面的操作。

                    self.config.problem_type = "multi_label_classification"
                    将问题类型设为多标签分类任务。

            if self.config.problem_type == "regression":
                判断问题类型是否为回归任务，如果是进行下面的操作。

                loss_fct = MSELoss()
                初始化loss_fct为均方差损失函数。

                if self.num_labels == 1:
                    判断类别数是否为1，如果是进行下面的操作。

                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                    计算损失值，使用logits.squeeze()和labels.squeeze()作为参数。

                else:
                    进行下面的操作。

                    loss = loss_fct(logits, labels)
                    计算损失值，使用logits和labels作为参数。

            elif self.config.problem_type == "single_label_classification":
                判断问题类型是否为单标签分类任务，如果是进行下面的操作。

                loss_fct = CrossEntropyLoss()
                初始化loss_fct为交叉熵损失函数。

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                计算损失值，使用logits.view(-1, self.num_labels)和labels.view(-1)作为参数。

            elif self.config.problem_type == "multi_label_classification":
                判断问题类型是否为多标签分类任务，如果是进行下面的操作。

                loss_fct = BCEWithLogitsLoss()
                初始化loss_fct为带Logits的二元交叉熵损失函数。

                loss = loss_fct(logits, labels)
                计算损失值，使用logits和labels作为参数。

        if not return_dict:
            判断return_dict是否为False，如果是进行下面的操作。

            output = (logits,) + outputs[2:]
            将logits和outputs的后两个元素组成元组output。

            return ((loss,) + output) if loss is not None else output
            如果loss不为空，将loss和output组成元组返回，否则返回output。

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        返回SequenceClassifierOutput对象，包含损失值loss、预测结果logits、隐藏状态hidden_states和注意力attentions。
# 使用多项选择分类头 (线性层 + softmax) 的 Data2VecText 模型，用于 RocStories/SWAG 任务
class Data2VecTextForMultipleChoice(Data2VecTextPreTrainedModel):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建 Data2VecText 模型
        self.data2vec_text = Data2VecTextModel(config)
        # 创建一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，用于多项选择分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # forward 函数
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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 return_dict 参数判断是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取 input_ids 张量的第二个维度，即选择个数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 根据是否存在 input_ids，确定是否需要展平 input_ids 张量
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 根据是否存在 position_ids，确定是否需要展平 position_ids 张量
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 根据是否存在 token_type_ids，确定是否需要展平 token_type_ids 张量
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 根据是否存在 attention_mask，确定是否需要展平 attention_mask 张量
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 根据是否存在 inputs_embeds，确定是否需要展平 inputs_embeds 张量
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 data2vec_text 方法进行计算
        outputs = self.data2vec_text(
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
        # 获取计算后的 pooled_output
        pooled_output = outputs[1]

        # 对 pooled_output 应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 通过 classifier 获取 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重塑为形状为 (-1, num_choices) 的张量
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 判断是否存在 labels
        if labels is not None:
            # 定义 CrossEntropyLoss 作为损失函数
            loss_fct = CrossEntropyLoss()

            # 将 labels 移到 reshaped_logits 设备上
            labels = labels.to(reshaped_logits.device)
            # 计算损失值
            loss = loss_fct(reshaped_logits, labels)

        # 判断是否使用返回字典
        if not return_dict:
            # 如果不使用返回字典，则将输出重新组合，并返回
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则返回 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Data2VecText Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextForTokenClassification(Data2VecTextPreTrainedModel):
    # 初始化 Token Classification 模型，继承自 Data2VecTextPreTrainedModel
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签类别数量
        self.num_labels = config.num_labels

        # 实例化 Data2VecTextModel 模型，不添加池化层
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        # 获取分类器 dropout 参数，如果为空则使用隐藏层 dropout 参数
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型前向传播方法
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
    # 此函数定义用于计算token分类任务的前向计算和损失
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        # 根据输入参数 return_dict 确定是否使用 return_dict 机制，如果为 None 则使用配置文件中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 调用 data2vec_text 函数进行前向计算，获取输出结果
        outputs = self.data2vec_text(
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
        
        # 取出序列输出结果
        sequence_output = outputs[0]
        
        # 对序列输出进行 dropout 操作
        sequence_output = self.dropout(sequence_output)
        
        # 将 dropout 后的序列输出送入分类器得到 logits
        logits = self.classifier(sequence_output)
        
        # 如果提供了 labels，则计算分类损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # 根据 return_dict 机制返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制并改名为Data2VecTextClassificationHead
class Data2VecTextClassificationHead(nn.Module):
    """用于句子级分类任务的头部。"""

    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入的特征维度映射到隐藏层维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 计算分类器的丢弃率，如果没有指定，则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个全连接层，将隐藏层维度映射到类别数量维度
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 取出features中每个句子的第一个token对应的特征作为输入
        x = features[:, 0, :]  # 拿到<s> token（等同于[CLS]）
        x = self.dropout(x)
        # 将输入特征映射到隐藏层维度
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        # 将隐藏层映射到类别数量维度
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    Data2VecText模型的一个span分类头部，用于提取式问答任务（例如SQuAD），在隐藏状态输出的基础上计算`span start logits`和`span end logits`。
    """,
    DATA2VECTEXT_START_DOCSTRING,
)
class Data2VecTextForQuestionAnswering(Data2VecTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 创建一个Data2VecTextModel，不添加池化层
        self.data2vec_text = Data2VecTextModel(config, add_pooling_layer=False)
        # 创建一个全连接层，将隐藏层维度映射到类别数量维度
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DATA2VECTEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
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
    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
            r"""
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            """
            # 是否返回字典形式的结果，如果未指定，则使用配置中的默认设置
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            # 将输入数据传入模型中进行前向传播
            outputs = self.data2vec_text(
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
    
            # 从模型输出中获取序列输出
            sequence_output = outputs[0]
    
            # 通过序列输出计算答案开始和结束的对数概率
            logits = self.qa_outputs(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()
    
            total_loss = None
            # 如果提供了答案的开始和结束位置，则计算损失
            if start_positions is not None and end_positions is not None:
                # 如果是多 GPU 运行，添加一个维度
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # 有时开始/结束位置超出了模型输入，这些位置被忽略
                ignored_index = start_logits.size(1)
                start_positions = start_positions.clamp(0, ignored_index)
                end_positions = end_positions.clamp(0, ignored_index)
    
                # 定义交叉熵损失函数，忽略指定的索引位置
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                # 计算总损失
                total_loss = (start_loss + end_loss) / 2
    
            # 如果不返回字典形式的结果，则返回损失和其他输出
            if not return_dict:
                output = (start_logits, end_logits) + outputs[2:]
                return ((total_loss,) + output) if total_loss is not None else output
    
            # 返回 QuestionAnsweringModelOutput 类的实例，其中包含损失、答案开始和结束的对数概率，以及其他输出
            return QuestionAnsweringModelOutput(
                loss=total_loss,
                start_logits=start_logits,
                end_logits=end_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
# 从输入的 input_ids 中创建位置 ID。非填充符号被替换为它们的位置数字。位置数字从 padding_idx+1 开始。填充符号被忽略。这是修改自 fairseq 的 `utils.make_positions`。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 连串的类型转换和转型在这里被精心平衡，以便同时适用于 ONNX 导出和 XLA。
    # 根据 input_ids 中是否是填充符号创建一个遮罩 mask
    mask = input_ids.ne(padding_idx).int()
    # 在遮罩上累加，得到递增的位置索引，然后再加上过去的键值长度，最后再乘以遮罩
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 得到位置索引后，加上 padding_idx 即可得到最终位置 ID
    return incremental_indices.long() + padding_idx
```