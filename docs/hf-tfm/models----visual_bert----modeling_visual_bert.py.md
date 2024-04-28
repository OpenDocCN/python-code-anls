# `.\transformers\models\visual_bert\modeling_visual_bert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 UCLA NLP 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
# PyTorch VisualBERT 模型

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_visual_bert import VisualBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点
_CONFIG_FOR_DOC = "VisualBertConfig"
_CHECKPOINT_FOR_DOC = "uclanlp/visualbert-vqa-coco-pre"

# VisualBERT 预训练模型存档列表
VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uclanlp/visualbert-vqa",
    "uclanlp/visualbert-vqa-pre",
    "uclanlp/visualbert-vqa-coco-pre",
    "uclanlp/visualbert-vcr",
    "uclanlp/visualbert-vcr-pre",
    "uclanlp/visualbert-vcr-coco-pre",
    "uclanlp/visualbert-nlvr2",
    "uclanlp/visualbert-nlvr2-pre",
    "uclanlp/visualbert-nlvr2-coco-pre",
    # 查看所有 VisualBERT 模型：https://huggingface.co/models?filter=visual_bert
]

# VisualBertEmbeddings 类，用于构建来自单词、位置和标记类型嵌入以及视觉嵌入的嵌入
class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建词嵌入层，根据词汇大小、隐藏层大小和填充标识来初始化
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，根据最大位置嵌入大小和隐藏层大小来初始化
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，根据类型词汇大小和隐藏层大小来初始化
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 不使用蛇形命名以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建丢弃层，根据隐藏层丢弃概率来初始化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 位置 ID（1，长度位置嵌入）在内存中是连续的，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # 对于视觉特征
        # 图像特征的标记类型和位置嵌入
        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 如果配置参数中有特殊的视觉初始化
        if config.special_visual_initialize:
            # 将视觉标记类型嵌入层的权重数据设置为与标记类型嵌入层相同，并设置为可训练
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            # 将视觉位置嵌入层的权重数据设置为与位置嵌入层相同，并设置为可训练
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

        # 创建视觉投影层，根据视觉嵌入维度和隐藏层大小来初始化
        self.visual_projection = nn.Linear(config.visual_embedding_dim, config.hidden_size)

    # 前向传播函数
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        visual_embeds=None,
        visual_token_type_ids=None,
        image_text_alignment=None,
class VisualBertSelfAttention(nn.Module):
    # 定义 VisualBertSelfAttention 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化函数

        # 检查隐藏层大小是否是注意力头数的倍数，如果不是则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 调整张量形状以便计算注意力分数
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键和值的转置形式
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算查询的转置形式
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # 如果存在注意力掩码，则应用
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 层
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            # 如果存在头掩码，则应用
            attention_probs = attention_probs * head_mask

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并将Bert->VisualBert
class VisualBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建LayerNorm层，输入维度是config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行Dropout
        hidden_states = self.dropout(hidden_states)
        # 对Dropout后的输出进行LayerNorm，并与input_tensor相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的hidden_states
        return hidden_states


class VisualBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建VisualBertSelfAttention对象
        self.self = VisualBertSelfAttention(config)
        # 创建VisualBertSelfOutput对象
        self.output = VisualBertSelfOutput(config)
        # 初始化pruned_heads为一个空集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头部和索引
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

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 调用self.self进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 将自注意力输出传递给self.output进行处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，将注意力信息添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回outputs
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制并将Bert->VisualBert
class VisualBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度是config.hidden_size，输出维度是config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串，则使用ACT2FN字典中对应的激活函数，否则使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对全连接层的输出进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput复制代码，并将Bert->VisualBert
class VisualBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入维度转换为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建LayerNorm层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层转换
        hidden_states = self.dense(hidden_states)
        # 对转换后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 对随机失活后的隐藏状态进行LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


class VisualBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置feed forward的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建VisualBertAttention层
        self.attention = VisualBertAttention(config)
        # 创建VisualBertIntermediate层
        self.intermediate = VisualBertIntermediate(config)
        # 创建VisualBertOutput层
        self.output = VisualBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 使用VisualBertAttention层处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取注意力输出
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，则将自注意力加入到输出中
        outputs = self_attention_outputs[1:]

        # 将注意力输出应用于前向传播的chunking处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将处理后的输出加入到输出中
        outputs = (layer_output,) + outputs

        # 返回处理后的输出
        return outputs

    def feed_forward_chunk(self, attention_output):
        # 使用VisualBertIntermediate层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用VisualBertOutput层处理中间输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的输出
        return layer_output


class VisualBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建多个VisualBertLayer层，根据config.num_hidden_layers的数量
        self.layer = nn.ModuleList([VisualBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        # 初始化所有隐藏状态为空元组，如果不需要输出隐藏状态则为None
        all_hidden_states = () if output_hidden_states else None
        # 初始化所有自注意力为空元组，如果不需要输出注意力则为None
        all_self_attentions = () if output_attentions else None

        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点并且处于训练模式，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力，则将当前层的注意力添加到所有自注意力中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回元组形式的结果
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        # 返回基础模型输出对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制代码，并将Bert->VisualBert
class VisualBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个token的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制代码，并将Bert->VisualBert
class VisualBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据config中的hidden_act选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm层，输入维度为config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制代码，并将Bert->VisualBert
class VisualBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个VisualBertPredictionHeadTransform对象
        self.transform = VisualBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个token有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接变量，以便偏置在`resize_token_embeddings`时正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertPreTrainingHeads复制代码，并将Bert->VisualBert
class VisualBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个VisualBertLMPredictionHead对象
        self.predictions = VisualBertLMPredictionHead(config)
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class VisualBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义一个抽象类，用于处理权重初始化和一个简单的接口用于下载和加载预训练模型

    config_class = VisualBertConfig
    # 设置配置类为VisualBertConfig
    base_model_prefix = "visual_bert"
    # 设置基础模型前缀为"visual_bert"
    supports_gradient_checkpointing = True
    # 支持梯度检查点

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重

        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 如果module是nn.Linear或nn.Embedding类型
            # 与TF版本略有不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 使用正态分布初始化权重

        elif isinstance(module, nn.LayerNorm):
            # 如果module是nn.LayerNorm类型
            module.bias.data.zero_()
            # 将偏置项数据置零
            module.weight.data.fill_(1.0)
            # 将权重数据填充为1.0

        if isinstance(module, nn.Linear) and module.bias is not None:
            # 如果module是nn.Linear类型且存在偏置项
            module.bias.data.zero_()
            # 将偏置项数据置零
# 定义一个数据类，用于存储VisualBertForPreTraining模型的输出结果
class VisualBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`VisualBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the sentence-image prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义输出结果的属性，包括loss、prediction_logits、seq_relationship_logits、hidden_states和attentions
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义VisualBert模型的文档字符串起始部分
VISUAL_BERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VisualBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义VisualBert模型的输入文档字符串起始部分
VISUAL_BERT_INPUTS_DOCSTRING = r"""
"""

# 添加文档字符串的装饰器
@add_start_docstrings(
    # 描述 VisualBert 模型的基本特性，输出原始隐藏状态而不带任何特定的头部
    # VISUAL_BERT_START_DOCSTRING 是一个文档字符串的起始标记
# 定义 VisualBertModel 类，继承自 VisualBertPreTrainedModel
class VisualBertModel(VisualBertPreTrainedModel):
    """
    模型可以作为一个编码器（只有自注意力），遵循 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中描述的架构
    作者为 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin
    """

    # 初始化方法
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置配置
        self.config = config

        # 初始化嵌入层
        self.embeddings = VisualBertEmbeddings(config)
        # 初始化编码器
        self.encoder = VisualBertEncoder(config)

        # 如果需要添加池化层，则初始化池化层，否则为 None
        self.pooler = VisualBertPooler(config) if add_pooling_layer else None

        # 是否绕过 transformer
        self.bypass_transformer = config.bypass_transformer

        # 如果绕过 transformer，则初始化额外的层
        if self.bypass_transformer:
            self.additional_layer = VisualBertLayer(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,


# 添加模型前向传播的文档字符串
@add_start_docstrings(
    """
    VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence-image prediction (classification)` head.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
# 定义 VisualBertForPreTraining 类，继承自 VisualBertPreTrainedModel
class VisualBertForPreTraining(VisualBertPreTrainedModel):
    # 共享权重的键
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]
    # 初始化函数，接受配置参数，并调用父类的初始化函数
    def __init__(self, config):
        super().__init__(config)

        # 创建 VisualBertModel 对象
        self.visual_bert = VisualBertModel(config)
        # 创建 VisualBertPreTrainingHeads 对象
        self.cls = VisualBertPreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=VisualBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        sentence_image_labels: Optional[torch.LongTensor] = None,
# 定义一个带有多选分类头部的 VisualBert 模型（在池化输出的顶部有一个线性层和 softmax），例如用于 VCR 任务
class VisualBertForMultipleChoice(VisualBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 VisualBert 模型
        self.visual_bert = VisualBertModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加线性层
        self.cls = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(
        VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,



# 定义一个带有分类/回归头部的 VisualBert 模型（在池化输出的顶部有一个 dropout 和线性层），用于 VQA
class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 VisualBert 模型
        self.visual_bert = VisualBertModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加线性层
        self.cls = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个名为forward的方法，用于执行前向传播
    def forward(
        # 输入的token的id，数据类型为可选的长整型张量，可以为None
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，数据类型为可选的长整型张量，可以为None
        attention_mask: Optional[torch.LongTensor] = None,
        # token类型id，数据类型为可选的长整型张量，可以为None
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置id，数据类型为可选的长整型张量，可以为None
        position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，数据类型为可选的长整型张量，可以为None
        head_mask: Optional[torch.LongTensor] = None,
        # 输入的嵌入张量，数据类型为可选的浮点型张量，可以为None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 视觉嵌入张量，数据类型为可选的浮点型张量，可以为None
        visual_embeds: Optional[torch.FloatTensor] = None,
        # 视觉注意力掩码，数据类型为可选的长整型张量，可以为None
        visual_attention_mask: Optional[torch.LongTensor] = None,
        # 视觉token类型id，数据类型为可选的长整型张量，可以为None
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        # 图像文本对齐信息，数据类型为可选的长整型张量，可以为None
        image_text_alignment: Optional[torch.LongTensor] = None,
        # 是否输出注意力信息，数据类型为可选的布尔型，可以为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态信息，数据类型为可选的布尔型，可以为None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典类型结果，数据类型为可选的布尔型，可以为None
        return_dict: Optional[bool] = None,
        # 标签，数据类型为可选的长整型张量，可以为None
        labels: Optional[torch.LongTensor] = None,
# 用于添加起始文档字符串的装饰器，描述了 VisualBert 模型及其在视觉推理领域的应用
@add_start_docstrings(
    """
    VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
    output) for Visual Reasoning e.g. for NLVR task.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForVisualReasoning(VisualBertPreTrainedModel):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数
        self.num_labels = config.num_labels

        # 初始化 VisualBert 模型
        self.visual_bert = VisualBertModel(config)
        # 使用配置的隐藏神经元丢弃概率创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层用于分类，输入维度为隐藏大小，输出维度为标签数
        self.cls = nn.Linear(config.hidden_size, config.num_labels)  # 2

        # 初始化权重并进行最终的处理
        self.post_init()

    # 前向传播方法，接收多个输入参数
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
# 用于实现 VisualBert 区域到短语的注意力机制
class VisualBertRegionToPhraseAttention(nn.Module):
    # 初始化方法，接收一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏大小不能整除注意力头数，则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 获取注意力头数和注意力头大小
        self.num_attention_heads = 1  # config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入重新排列以便计算得分
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    # 将注意力掩码转换为query张量的数据类型
    attention_mask = attention_mask.to(query.dtype)
    # 将注意力掩码进行维度扩展，以匹配query张量的形状
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    # 将未被掩码的位置替换为一个极小的负数，使得在softmax计算中不产生影响
    attention_mask = (1.0 - attention_mask) * torch.finfo(query.dtype).min

    # 使用self.query对query进行线性变换
    mixed_query_layer = self.query(query)
    # 使用self.key对key进行线性变换
    mixed_key_layer = self.key(key)

    # 将混合后的query层张量转置，为计算注意力分数做准备
    query_layer = self.transpose_for_scores(mixed_query_layer)
    # 将混合后的key层张量转置，为计算注意力分数做准备
    key_layer = self.transpose_for_scores(mixed_key_layer)

    # 计算query与key之间的注意力分数
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    # 对注意力分数进行缩放，以避免计算过程中的梯度爆炸或梯度消失问题
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # 将注意力掩码加到注意力分数上，以屏蔽无效的位置
    attention_scores = attention_scores + attention_mask

    # 去除维度为1的维度，以匹配后续计算的维度
    attention_scores = attention_scores.squeeze(1)
    # 返回计算得到的注意力分数
    return attention_scores
# 使用 VisualBert 模型实现 Region-to-Phrase Alignment 的预训练模型
@add_start_docstrings(
    # 添加模型描述的 docstring
    """
    VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment
    e.g. for Flickr30 Entities task.
    """,
    VISUAL_BERT_START_DOCSTRING,
)
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    # 需要绑定权重的键
    _tied_weights_keys = ["cls.predictions.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 VisualBert 模型
        self.visual_bert = VisualBertModel(config)
        # 添加一个Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个VisualBertPreTrainingHeads层
        self.cls = VisualBertPreTrainingHeads(config)
        # 添加一个VisualBertRegionToPhraseAttention层
        self.attention = VisualBertRegionToPhraseAttention(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型前向传播添加 docstring
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        visual_embeds: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.LongTensor] = None,
        visual_token_type_ids: Optional[torch.LongTensor] = None,
        image_text_alignment: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        region_to_phrase_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
```