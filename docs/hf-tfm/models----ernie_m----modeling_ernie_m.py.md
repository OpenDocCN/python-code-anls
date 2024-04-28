# `.\models\ernie_m\modeling_ernie_m.py`

```py
# 设置文件编码为utf-8
# 声明版权信息
# 根据Apache许可证2.0版规定，权限受限，仅在遵循许可证的情况下可使用此文件
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则分发的软件将基于"原样"分发，没有任何明示或暗示的担保或条件。
# 详见许可证以了解详细的权限和限制

# 导入所需的库和模块
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig

# 获取logger实例并初始化
logger = logging.get_logger(__name__)

# 文档中使用的模型检查点
_CHECKPOINT_FOR_DOC = "susnato/ernie-m-base_pytorch"
# 文档中使用的配置
_CONFIG_FOR_DOC = "ErnieMConfig"
# 文档中使用的分词器
_TOKENIZER_FOR_DOC = "ErnieMTokenizer"

# 可用的ErnieM预训练模型列表
ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "susnato/ernie-m-base_pytorch",
    "susnato/ernie-m-large_pytorch",
    # 在https://huggingface.co/models?filter=ernie_m 查看所有ErnieM模型
]


# 从paddlenlp.transformers.ernie_m.modeling.ErnieEmbeddings调整得到
class ErnieMEmbeddings(nn.Module):
    """从词嵌入和位置嵌入构建嵌入层"""

    def __init__(self, config):
        # 初始化函数
        super().__init__()
        self.hidden_size = config.hidden_size
        # 词嵌入层，将词索引映射为向量表示
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，将位置索引映射为向量表示
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
        )
        # LayerNorm层，用于归一化隐藏维度
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层，用于随机失活
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # 记录填充标记的索引
        self.padding_idx = config.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
        ) -> torch.Tensor:
            # 如果输入的嵌入是空的，则使用词嵌入模型对输入的词进行嵌入
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # 如果位置id是空的
            if position_ids is None:
                # 获取输入嵌入的形状
                input_shape = inputs_embeds.size()[:-1]
                # 创建大小为input_shape的全1张量，并转换为int64类型
                ones = torch.ones(input_shape, dtype=torch.int64, device=inputs_embeds.device)
                # 使用torch.cumsum函数获取序列长度
                seq_length = torch.cumsum(ones, dim=1)
                # position_ids为seq_length减去全1的张量得到
                position_ids = seq_length - ones

                # 如果有过去关键值存在，则更新position_ids
                if past_key_values_length > 0:
                    position_ids = position_ids + past_key_values_length
            # 为了模仿paddlenlp的实现，对position_ids进行加2操作
            position_ids += 2
            # 使用位置嵌入模型获取位置嵌入
            position_embeddings = self.position_embeddings(position_ids)
            # 将输入嵌入和位置嵌入相加
            embeddings = inputs_embeds + position_embeddings
            # 对结果进行layer normalization
            embeddings = self.layer_norm(embeddings)
            # 对结果进行dropout
            embeddings = self.dropout(embeddings)

            # 返回结果张量
            return embeddings
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制代码，并将Bert->ErnieM, self.value->self.v_proj, self.key->self.k_proj, self.query->self.q_proj
class ErnieMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q_proj = nn.Linear(config.hidden_size, self.all_head_size)  # 创建一个线性层，将输入特征映射到所有注意力头的大小
        self.k_proj = nn.Linear(config.hidden_size, self.all_head_size)  # 创建一个线性层，将输入特征映射到所有注意力头的大小
        self.v_proj = nn.Linear(config.hidden_size, self.all_head_size)  # 创建一个线性层，将输入特征映射到所有注意力头的大小

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 创建一个Dropout层
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)  # 创建一个Embedding层

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)  # 改变张量的形状
        return x.permute(0, 2, 1, 3)  # 交换张量的维度顺序

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
class ErnieMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self_attn = ErnieMSelfAttention(config, position_embedding_type=position_embedding_type)  # 创建一个ErnieMSelfAttention的实例
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个线性层，将输入特征映射到相同大小的输出特征
        self.pruned_heads = set()  # 创建一个空集合作为剪枝头的标记
    # 修剪多头注意力模型中的头部
    def prune_heads(self, heads):
        # 如果待修剪的头部为空，则直接返回
        if len(heads) == 0:
            return
        # 找到可修剪的头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self_attn.num_attention_heads, self.self_attn.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self_attn.q_proj = prune_linear_layer(self.self_attn.q_proj, index)
        self.self_attn.k_proj = prune_linear_layer(self.self_attn.k_proj, index)
        self.self_attn.v_proj = prune_linear_layer(self.self_attn.v_proj, index)
        self.out_proj = prune_linear_layer(self.out_proj, index, dim=1)

        # 更新超参数并储存已修剪的头部
        self.self_attn.num_attention_heads = self.self_attn.num_attention_heads - len(heads)
        self.self_attn.all_head_size = self.self_attn.attention_head_size * self.self_attn.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 使用Transformer模型进行前向传播
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
        # 通过自注意力机制进行前向传播
        self_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 通过输出投影层处理注意力输出
        attention_output = self.out_proj(self_outputs[0])
        # 构建输出结果，如果需要输出注意力权重则添加进来
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
class ErnieMEncoderLayer(nn.Module):
    # ErnieM 编码器层的定义
    def __init__(self, config):
        super().__init__()
        # 为了模仿 PaddleNLP 的实现，设置默认的 dropout 值
        dropout = 0.1 if config.hidden_dropout_prob is None else config.hidden_dropout_prob
        # 如果未指定激活函数的 dropout，则使用隐藏层 dropout
        act_dropout = config.hidden_dropout_prob if config.act_dropout is None else config.act_dropout

        # 定义自注意力机制
        self.self_attn = ErnieMAttention(config)
        # 第一个线性层
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # dropout 操作
        self.dropout = nn.Dropout(act_dropout)
        # 第二个线性层
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNormalization 层
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout 操作
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 激活函数
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = True,
    ):
        # 保存残差连接
        residual = hidden_states
        # 如果需要输出注意力权重，则返回注意力权重
        if output_attentions:
            hidden_states, attention_opt_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        # 否则，只进行自注意力计算
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        # 残差连接和 dropout 操作
        hidden_states = residual + self.dropout1(hidden_states)
        # LayerNormalization 层
        hidden_states = self.norm1(hidden_states)
        # 保存残差连接
        residual = hidden_states

        # 第一个线性层和激活函数
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 第二个线性层和残差连接
        hidden_states = self.linear2(hidden_states)
        hidden_states = residual + self.dropout2(hidden_states)
        # LayerNormalization 层
        hidden_states = self.norm2(hidden_states)

        # 如果需要输出注意力权重，则返回注意力权重
        if output_attentions:
            return hidden_states, attention_opt_weights
        # 否则，只返回隐藏状态
        else:
            return hidden_states


class ErnieMEncoder(nn.Module):
    # ErnieM 编码器的定义
    def __init__(self, config):
        super().__init__()
        # 设置配置
        self.config = config
        # 创建多层 ErnieM 编码器层
        self.layers = nn.ModuleList([ErnieMEncoderLayer(config) for _ in range(config.num_hidden_layers)])
```  
    # 前向传播函数，用于模型的前向计算
    def forward(
        self,
        input_embeds: torch.Tensor,  # 输入的嵌入张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 上一次的键值对，默认为 None
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: Optional[bool] = False,  # 是否输出隐藏状态，默认为 False
        return_dict: Optional[bool] = True,  # 是否返回字典，默认为 True
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        hidden_states = () if output_hidden_states else None  # 如果需要输出隐藏状态，则初始化隐藏状态为一个空元组，否则为 None
        attentions = () if output_attentions else None  # 如果需要输出注意力，则初始化注意力为一个空元组，否则为 None

        output = input_embeds  # 将输入的嵌入作为初始输出
        if output_hidden_states:
            hidden_states = hidden_states + (output,)  # 如果需要输出隐藏状态，则将当前输出加入隐藏状态中
        for i, layer in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None  # 获取当前层的头部掩码
            past_key_value = past_key_values[i] if past_key_values is not None else None  # 获取上一次的键值对

            # 调用当前层的前向传播函数
            output, opt_attn_weights = layer(
                hidden_states=output,  # 输入隐藏状态
                attention_mask=attention_mask,  # 注意力掩码
                head_mask=layer_head_mask,  # 头部掩码
                past_key_value=past_key_value,  # 上一次的键值对
            )

            if output_hidden_states:
                hidden_states = hidden_states + (output,)  # 如果需要输出隐藏状态，则将当前输出加入隐藏状态中
            if output_attentions:
                attentions = attentions + (opt_attn_weights,)  # 如果需要输出注意力，则将当前注意力加入注意力中

        last_hidden_state = output  # 最终的隐藏状态即为最后一次输出
        if not return_dict:  # 如果不返回字典
            return tuple(v for v in [last_hidden_state, hidden_states, attentions] if v is not None)  # 返回非 None 的值

        # 返回包含最终隐藏状态、隐藏状态序列和注意力权重的字典
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,  # 最终隐藏状态
            hidden_states=hidden_states,  # 隐藏状态序列
            attentions=attentions  # 注意力权重
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制并修改为ErnieMPooler
class ErnieMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个token的隐藏状态来对模型做“池化”
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieMPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和下载加载预训练模型的抽象类。

    """

    config_class = ErnieMConfig
    base_model_prefix = "ernie_m"

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与 TF 版本稍有不同，这里使用正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
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


ERNIE_M_START_DOCSTRING = r"""

    该模型继承自 [`PreTrainedModel`]。请查阅父类文档了解库为所有模型提供的通用方法（如下载或保存、调整输入嵌入、剪枝头等）。

    该模型是 PyTorch 的 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其视为常规的 PyTorch 模块，并请参阅 PyTorch 文档了解一般用法和行为。

    参数:
        config ([`ErnieMConfig`]): 模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载模型相关的权重，只会加载配置。请查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

ERNIE_M_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列 tokens 在词汇表中的索引
            # 可以使用 `ErnieMTokenizer` 获取索引，参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充令牌索引上执行注意力的掩码
            # 选择在 `[0, 1]` 范围内的掩码值:
            # - 1 表示 **未屏蔽** 的令牌
            # - 0 表示 **已屏蔽** 的令牌

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列令牌在位置嵌入中的位置索引
            # 选择范围为 `[0, config.max_position_embeddings - 1]`

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于设置自注意力模块中选择的头部的掩码
            # 选择在 `[0, 1]` 范围内的掩码值:
            # - 1 表示头部 **未屏蔽**
            # - 0 表示头部 **已屏蔽**

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，可以直接传递嵌入表示，而不是传递 `input_ids`
            # 如果想要更多控制如何将 *input_ids* 索引转换为关联向量，可以使用此选项

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量
            # 查看返回的张量中的 `attentions` 以获取更多细节

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态
            # 查看返回的张量中的 `hidden_states` 以获取更多细节

        return_dict (`bool`, *optional*):
            # 是否返回一个包含 `~utils.ModelOutput` 的对象，而不是普通元组
# 导入模块
import torch
import torch.nn as nn
from .modeling_utils import PreTrainedModel
from .configuration_ernie_m import ErnieMConfig

# 定义 ErnieMModel，继承自 ErnieMPreTrainedModel
class ErnieMModel(ErnieMPreTrainedModel):
    # 初始化函数
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化函数
        super(ErnieMModel, self).__init__(config)
        # 设置初始化范围
        self.initializer_range = config.initializer_range
        # 创建 ErnieMEmbeddings 对象
        self.embeddings = ErnieMEmbeddings(config)
        # 创建 ErnieMEncoder 对象
        self.encoder = ErnieMEncoder(config)
        # 是否添加 pooling 层
        self.pooler = ErnieMPooler(config) if add_pooling_layer else None
        # 执行后续初始化
        self.post_init()

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 裁剪模型的头部
    def _prune_heads(self, heads_to_prune):
        # 循环遍历要裁剪的头部
        for layer, heads in heads_to_prune.items():
            self.encoder.layers[layer].self_attn.prune_heads(heads)

    # 前向传播函数
    def forward(
        # 输入的 token id
        self,
        input_ids: Optional[tensor] = None,
        position_ids: Optional[tensor] = None,
        attention_mask: Optional[tensor] = None,
        head_mask: Optional[tensor] = None,
        inputs_embeds: Optional[tensor] = None,
        past_key_values: Optional[Tuple[Tuple[tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处省略具体的前向传播逻辑

# 定义 ErnieMForSequenceClassification，继承自 ErnieMPreTrainedModel
class ErnieMForSequenceClassification(ErnieMPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数
        self.num_labels = config.num_labels
        self.config = config
        # 创建 ErnieMModel 对象
        self.ernie_m = ErnieMModel(config)
        # 分类器的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 分类器的线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 执行后续初始化
        self.post_init()
    # 添加代码示例的文档字符串，包括处理器类、检查点、输出类型和配置类
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,  # 用于文档的处理器类
        checkpoint=_CHECKPOINT_FOR_DOC,      # 用于文档的检查点
        output_type=SequenceClassifierOutput,  # 输出类型为序列分类器输出
        config_class=_CONFIG_FOR_DOC,         # 用于文档的配置类
    )
    # 前向传播函数，接收多个参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,           # 输入的标识符张量
        attention_mask: Optional[torch.Tensor] = None,      # 注意力遮罩张量
        position_ids: Optional[torch.Tensor] = None,        # 位置标识符张量
        head_mask: Optional[torch.Tensor] = None,           # 头部遮罩张量
        inputs_embeds: Optional[torch.Tensor] = None,       # 输入嵌入张量
        past_key_values: Optional[List[torch.Tensor]] = None,  # 过去的键-值张量列表
        use_cache: Optional[bool] = None,                   # 是否使用缓存的布尔值
        output_hidden_states: Optional[bool] = None,        # 输出隐藏状态的布尔值
        output_attentions: Optional[bool] = None,           # 输出注意力权重的布尔值
        return_dict: Optional[bool] = True,                 # 返回字典的布尔值，默认为 True
        labels: Optional[torch.Tensor] = None,              # 标签张量
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Ernie 模型进行前向传播
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器输出 logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            # 如果不返回字典形式的结果，则返回 logits 和其它相关输出
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器的输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串，描述 ErnieM 模型及其在多项选择分类任务中的应用
@add_start_docstrings(
    """ErnieM Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    ERNIE_M_START_DOCSTRING,
)
class ErnieMForMultipleChoice(ErnieMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ 复制代码，并对其中的 Bert->ErnieM, bert->ernie_m 进行修改
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建 ErnieM 模型
        self.ernie_m = ErnieMModel(config)
        # 设置分类器的 dropout，若未设置则使用隐藏层 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.FloatTensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入张量的第二个维度大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入张量
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用 ERNIE 模型处理输入数据
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类
        logits = self.classifier(pooled_output)
        # 重新塑形分类结果
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            # 返回结果
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选题模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串描述 ErnieM 模型和其在标记分类任务中的应用
@add_start_docstrings(
    """ErnieM Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    ERNIE_M_START_DOCSTRING,
)
# 定义 ErnieMForTokenClassification 类，继承自 ErnieMPreTrainedModel 类
class ErnieMForTokenClassification(ErnieMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制而来，修改 Bert 为 ErnieM，bert 为 ernie_m
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 初始化 ErnieMModel 对象，不添加池化层
        self.ernie_m = ErnieMModel(config, add_pooling_layer=False)
        # 获取分类器的丢弃率并进行设置
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义线性层，将隐藏层的输出映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 为 forward 方法添加模型前向传播的文档字符串描述和代码示例
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确保返回的结果字典是否为空，若为空则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ERNIE 模型进行前向传播
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 应用 dropout 层
        sequence_output = self.dropout(sequence_output)
        # 应用分类器层
        logits = self.classifier(sequence_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回结果字典，则按照非字典格式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回结果字典格式的 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为提取型问答任务设计的 ErnieM 模型，包括一个用于分类的 span 头部（在隐藏状态输出顶部的线性层，用于计算“跨度起始标记”和“跨度结束标记”）。

# 继承自 ErnieMPreTrainedModel 类
class ErnieMForQuestionAnswering(ErnieMPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 中复制而来, 将 Bert->ErnieM, bert->ernie_m
    def __init__(self, config):
        # 调用父类的初始化
        super().__init__(config)
        # 获取类别数
        self.num_labels = config.num_labels

        # 创建 ErnieMModel 对象
        self.ernie_m = ErnieMModel(config, add_pooling_layer=False)
        # 创建线性层，连接到分类输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
# Optional[torch.Tensor] 表示参数为可选
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], QuestionAnsweringModelOutput]:
        r"""
        定义函数签名，指定参数和返回类型。此函数是用于问答任务的前向传播。

        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            起始位置标签，用于计算标记分类损失的起始位置（索引）。
            位置被限制在序列长度内(`sequence_length`)。超出序列范围的位置不会被考虑在损失计算中。

        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            结束位置标签，用于计算标记分类损失的结束位置（索引）。
            位置被限制在序列长度内(`sequence_length`)。超出序列范围的位置不会被考虑在损失计算中。
        """
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给ERNIE模型以获取输出
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 通过QA输出层生成logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出了模型输入范围，我们忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不返回字典，将输出整理为元组
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回QuestionAnsweringModelOutput对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 ErnieMForInformationExtraction 类添加文档字符串，说明其设计用途和结构
@add_start_docstrings(
    """ErnieMForInformationExtraction is a Ernie-M Model with two linear layer on top of the hidden-states output to
    compute `start_prob` and `end_prob`, designed for Universal Information Extraction.""",
    ERNIE_M_START_DOCSTRING,
)
# 定义一个用于信息抽取的 Ernie-M 模型，该模型在隐藏状态输出之上有两个线性层，用于计算 `start_prob` 和 `end_prob`
# 用于通用信息抽取任务
class ErnieMForInformationExtraction(ErnieMPreTrainedModel):
    def __init__(self, config):
        # 调用父类 ErnieMPreTrainedModel 的初始化方法
        super(ErnieMForInformationExtraction, self).__init__(config)
        # 初始化 Ernie-M 模型
        self.ernie_m = ErnieMModel(config)
        # 定义线性层，用于预测起始位置的概率
        self.linear_start = nn.Linear(config.hidden_size, 1)
        # 定义线性层，用于预测终止位置的概率
        self.linear_end = nn.Linear(config.hidden_size, 1)
        # 定义 sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()
        # 调用后续初始化方法
        self.post_init()

    # 为 forward 方法添加输入文档字符串，描述输入参数和返回值
    @add_start_docstrings_to_model_forward(ERNIE_M_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.FloatTensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for position (index) for computing the start_positions loss. Position outside of the sequence are
            not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) for computing the end_positions loss. Position outside of the sequence are not
            taken into account for computing the loss.
        """

        # 使用 ERNIE 模型处理输入
        result = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 如果需要返回字典形式的结果
        if return_dict:
            # 获取最后一层的隐藏状态作为序列输出
            sequence_output = result.last_hidden_state
        # 如果不需要返回字典形式的结果
        elif not return_dict:
            # 获取第一个元素作为序列输出
            sequence_output = result[0]

        # 计算开始位置的 logits
        start_logits = self.linear_start(sequence_output)
        start_logits = start_logits.squeeze(-1)
        # 计算结束位置的 logits
        end_logits = self.linear_end(sequence_output)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # 如果有提供开始和结束位置
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略那些超出模型输入的开始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = BCEWithLogitsLoss()
            # 计算开始位置的损失
            start_loss = loss_fct(start_logits, start_positions)
            # 计算结束位置的损失
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 返回包含总���失、开始位置 logits、结束位置 logits、隐藏状态和注意力权重的结果元组
            return tuple(
                i
                for i in [total_loss, start_logits, end_logits, result.hidden_states, result.attentions]
                if i is not None
            )

        # 返回包含总损失、开始位置 logits、结束位置 logits、隐藏状态和注意力权重的输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
        )
```