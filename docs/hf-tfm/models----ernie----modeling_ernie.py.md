# `.\models\ernie\modeling_ernie.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 2022年版权归The HuggingFace Inc.团队所有
# 根据Apache License，Version 2.0（“许可证”）授权
# 您不得使用此文件，除非遵守许可证的规定
# 您可以在以下获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则不得将许可证下的软件分发为“原样”，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 请查看许可证以了解具体语言的权限和限制
"""PyTorch ERNIE model."""


import math  # 导入数学库
import warnings  # 导入警告库
from dataclasses import dataclass  # 导入数据类
from typing import List, Optional, Tuple, Union  # 导入类型标注

import torch  # 导入PyTorch
import torch.utils.checkpoint  # 导入PyTorch的checkpoint
from torch import nn  # 从PyTorch中导入神经网络
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从PyTorch中导入损失函数

from ...activations import ACT2FN  # 从激活函数库中导入函数
from ...modeling_outputs import (  # 从输出模型函数库中导入输出模型
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 从模型工具库中导入预训练模型
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer  # 从PyTorch工具库中导入函数
from ...utils import (  # 从工具库中导入函数
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_ernie import ErnieConfig  # 从ERNIE配置文件中导入配置

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "nghuyong/ernie-1.0-base-zh"  # 设定文档的检查点
_CONFIG_FOR_DOC = "ErnieConfig"  # 设定文档的配置

ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = [  # ERNIE预训练模型存档列表
    "nghuyong/ernie-1.0-base-zh",
    "nghuyong/ernie-2.0-base-en",
    "nghuyong/ernie-2.0-large-en",
    "nghuyong/ernie-3.0-base-zh",
    "nghuyong/ernie-3.0-medium-zh",
    "nghuyong/ernie-3.0-mini-zh",
    "nghuyong/ernie-3.0-micro-zh",
    "nghuyong/ernie-3.0-nano-zh",
    "nghuyong/ernie-gram-zh",
    "nghuyong/ernie-health-zh",
    # 查看所有ERNIE模型 https://huggingface.co/models?filter=ernie
]


class ErnieEmbeddings(nn.Module):  # ERNIE嵌入层类
    """Construct the embeddings from word, position and token_type embeddings."""  # 从词嵌入、位置和令牌类型嵌入中构建嵌入
    # 初始化函数，配置模型参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 创建词嵌入层，将词汇 ID 映射成隐藏状态的向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，将位置信息映射成隐藏状态的向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建类型嵌入层，将类型信息映射成隐藏状态的向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 配置是否使用任务类型 ID
        self.use_task_id = config.use_task_id
        if config.use_task_id:
            # 如果使用任务类型 ID，则创建任务类型嵌入层
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)
    
        # 不使用蛇式命名的变量，以保持与 TensorFlow 模型变量名称一致，并能加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 随机丢弃法层，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置 ID 张量，用于序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册类型 ID 张量，用于序列化时导出
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    
    # 正向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        ) -> torch.Tensor:
        # 如果传入了input_ids，则获取其大小作为input_shape；否则获取inputs_embeds的倒数第二维作为input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取输入序列的长度
        seq_length = input_shape[1]

        # 如果position_ids为None，则从self.position_ids中获取相应位置的值
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果token_type_ids为None，则根据条件进行处理
        if token_type_ids is None:
            # 如果self中存在"token_type_ids"属性，则从中获取值并进行扩展
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则，创建全零tensor作为token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果inputs_embeds为None，则根据input_ids获取word_embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据token_type_ids获取token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将inputs_embeds和token_type_embeddings相加得到embeddings
        embeddings = inputs_embeds + token_type_embeddings
        # 如果position_embedding_type是"absolute"，则根据position_ids获取position_embeddings，并加到embeddings上
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 如果use_task_id为True，且task_type_ids为None，则创建全零tensor作为task_type_ids
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            # 根据task_type_ids获取task_type_embeddings，并加到embeddings上
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings += task_type_embeddings

        # 对embeddings进行LayerNorm操作
        embeddings = self.LayerNorm(embeddings)
        # 对embeddings进行dropout操作
        embeddings = self.dropout(embeddings)
        # 返回embeddings
        return embeddings
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制并将Bert->Ernie
class ErnieSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小能否被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数、注意力头大小和所有头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对键或相对键查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量转换为分数张量
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
```  


# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并将Bert->Ernie
class ErnieSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```  
# 定义 ErnieAttention 类，用于实现 Ernie 模型中的注意力机制
class ErnieAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层和输出层
        self.self = ErnieSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = ErnieSelfOutput(config)
        # 存储被修剪的注意力头索引的集合
        self.pruned_heads = set()

    # 修剪注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的头
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
        # 使用自注意力层进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力层的输出传递给输出层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则将其添加到输出中
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 ErnieIntermediate 类
class ErnieIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义线性层，将隐藏状态映射到中间尺寸
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 设置激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行映射
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 ErnieOutput 类
class ErnieOutput(nn.Module):
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，将输入特征的维度从config.intermediate_size映射到config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，用于对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于随机丢弃隐藏状态中的一部分神经元，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，用于执行神经网络的前向传播计算
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行随机丢弃一部分神经元
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的隐藏状态与输入的张量相加，并经过LayerNorm层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer中复制代码，并将Bert改为Ernie
class ErnieLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化ErnieLayer对象，设置config中的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ErnieAttention(config)  # 创建ErnieAttention对象
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果需要跨层attention，且不是解码器，则抛出异常
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ErnieAttention(config, position_embedding_type="absolute")  # 创建ErnieAttention对象
        self.intermediate = ErnieIntermediate(config)  # 创建ErnieIntermediate对象
        self.output = ErnieOutput(config)  # 创建ErnieOutput对象

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    # 定义函数feed_forward_chunk，用于处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层输出和注意力输出，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层输出
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制并修改为 ErnieEncoder 类
class ErnieEncoder(nn.Module):
    # 初始化 ErnieEncoder 类
    def __init__(self, config):
        super().__init__()
        # 存储传入的配置信息
        self.config = config
        # 创建一系列 ErnieLayer 层，层数由配置信息指定
        self.layer = nn.ModuleList([ErnieLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数的返回值类型，是一个元组，包含torch.Tensor类型的元素或BaseModelOutputWithPastAndCrossAttentions类型
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果需要输出隐藏状态，则创建一个空元组，否则设置为None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则创建一个空元组，否则设置为None
        all_self_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力权重，并且配置中包含交叉注意力，则创建一个空元组，否则设置为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用梯度检查点并且处于训练模式，则进行判断和警告
        if self.gradient_checkpointing and self.training:
            if use_cache:
                # 输出警告信息，指出`use_cache=True`与梯度检查点是不兼容的，将`use_cache=False`
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                # 将use_cache设置为False
                use_cache = False

        # 如果不需要缓存，创建一个空元组，否则设置为None
        next_decoder_cache = () if use_cache else None
        # 遍历self.layer中的每一个元素
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取头部遮罩，如果head_mask为非空，则取出第i个元素，否则设置为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取过去的键值对，如果past_key_values为非空，则取出第i个元素，否则设置为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用梯度检查点并且处于训练模式，调用_gradient_checkpointing_func，否则调用layer_module
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

            # 更新隐藏状态
            hidden_states = layer_outputs[0]
            # 如果需要缓存，将当前层的缓存值添加到next_decoder_cache中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，将当前层的注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中包含交叉注意力，则将当前层的交叉注意力权重添加到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回一个元组，包含需要输出的值
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
        # 返回BaseModelOutputWithPastAndCrossAttentions类型的结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制而来，将Bert->Ernie
# 定义了一个ErniePooler类，继承自nn.Module
class ErniePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 池化模型，通过取第一个token对应的隐藏状态作为汇总
        first_token_tensor = hidden_states[:, 0]
        # 将汇总后的隐藏状态通过全连接层
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制而来，将Bert->Ernie
# 定义了一个ErniePredictionHeadTransform类，继承自nn.Module
class ErniePredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置文件中的激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 初始化LayerNorm层，输入维度为config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用LayerNorm层
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制而来，将Bert->Ernie
# 定义了一个ErnieLMPredictionHead类，继承自nn.Module
class ErnieLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个ErniePredictionHeadTransform实例
        self.transform = ErniePredictionHeadTransform(config)

        # 输出权重与输入embedding相同，但是对于每个token有一个仅有输出的偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接以确保偏置项能够正确地调整大小以适应`resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 使用ErniePredictionHeadTransform对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        # 使用decoder层对变换后的隐藏状态进行线性变换
        hidden_states = self.decoder(hidden_states)
        # 返回线性变换后的结果
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制而来，将Bert->Ernie
# 定义了一个ErnieOnlyMLMHead类，继承自nn.Module
class ErnieOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个ErnieLMPredictionHead实例
        self.predictions = ErnieLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 对序列输出进行MLM预测
        prediction_scores = self.predictions(sequence_output)
        # 返回预测得分
        return prediction_scores


# 从transformers.models.bert.modeling_bert.BertOnlyNSPHead复制而来，将Bert->Ernie
# 定义了一个ErnieOnlyNSPHead类，继承自nn.Module
class ErnieOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入维度为config.hidden_size，输出维度为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 定义一个方法，用于计算序列关系得分
    def forward(self, pooled_output):
        # 使用池化后的输出计算序列关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回序列关系得分
        return seq_relationship_score
# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制并将 Bert 改为 Ernie
class ErniePreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测头部
        self.predictions = ErnieLMPredictionHead(config)
        # 初始化序列关系
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 通过预测头部计算预测分数
        prediction_scores = self.predictions(sequence_output)
        # 计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class ErniePreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和一个简单的接口，用于下载和加载预训练模型。
    """

    # Ernie 配置类
    config_class = ErnieConfig
    # 基础模型前缀
    base_model_prefix = "ernie"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
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


@dataclass
# 从 transformers.models.bert.modeling_bert.BertForPreTrainingOutput 复制并将 Bert 改为 Ernie
class ErnieForPreTrainingOutput(ModelOutput):
    """
    [`ErnieForPreTraining`] 的输出类型。
    """
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            总损失，由掩码语言建模损失和下一个序列预测（分类）损失相加得到。
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头部的预测得分（SoftMax之前每个词汇标记的分数）。
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            下一个序列预测（分类）头部的预测得分（SoftMax之前的True/False连续性的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为`(batch_size, sequence_length, hidden_size)`的`torch.FloatTensor`组成的元组，包含模型每层的隐藏状态（嵌入输出和每一层的输出）。

            每层的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`torch.FloatTensor`组成的元组，包含注意力softmax之后的注意力权重，
            用于计算自注意力头部的加权平均。
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# ERNIE 模型起始文档字符串，描述模型继承自 PreTrainedModel 类，并提供有关使用方法和参数配置的文档说明
ERNIE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ErnieConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ERNIE 模型输入文档字符串
ERNIE_INPUTS_DOCSTRING = r"""
"""

# 添加起始文档字符串注释
@add_start_docstrings(
    "The bare Ernie Model transformer outputting raw hidden-states without any specific head on top.",
    ERNIE_START_DOCSTRING,
)
# ERNIE 模型类定义
class ErnieModel(ErniePreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    # 定义初始化方法
    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 中复制，并将 Bert 替换为 Ernie
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置配置信息
        self.config = config
        # 初始化嵌入层
        self.embeddings = ErnieEmbeddings(config)
        # 初始化编码器
        self.encoder = ErnieEncoder(config)
        # 如果设置了添加池化层为真，则初始化池化层
        self.pooler = ErniePooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝头部方法，从 transformers.models.bert.modeling_bert.BertModel._prune_heads 复制而来
    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        对模型的注意力头进行剪枝。heads_to_prune: {层编号: 需要在此层剪枝的注意力头列表}
        请参阅基类 PreTrainedModel
        """
        # 遍历每个需要剪枝的层和其对应的需要剪枝的注意力头列表
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数，接收各种输入参数
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的词索引列表
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，标记哪些位置的词需要被忽略
        token_type_ids: Optional[torch.Tensor] = None,  # 用于区分不同句子的标记
        task_type_ids: Optional[torch.Tensor] = None,  # 任务类型标记
        position_ids: Optional[torch.Tensor] = None,  # 位置索引
        head_mask: Optional[torch.Tensor] = None,  # 注意力头的掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入向量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 存储历史的键值对
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典类型的输出
# 使用给定的文档字符串为Ernie模型添加头部，在预训练期间，该头部包含一个“掩码语言建模”头部和一个“下一个句子预测（分类）”头部。
@add_start_docstrings(
    """
    Ernie Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    ERNIE_START_DOCSTRING,
)
class ErnieForPreTraining(ErniePreTrainedModel):
    # 定义与权重相关的键，这些键被绑定在一起
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 从transformers.models.bert.modeling_bert.BertForPreTraining.__init__复制而来，将Bert->Ernie,bert->ernie
    def __init__(self, config):
        super().__init__(config)

        # 初始化Ernie模型和Ernie的预训练头部
        self.ernie = ErnieModel(config)
        self.cls = ErniePreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.bert.modeling_bert.BertForPreTraining.get_output_embeddings复制而来
    def get_output_embeddings(self):
        # 返回预测头部的解码器权重
        return self.cls.predictions.decoder

    # 从transformers.models.bert.modeling_bert.BertForPreTraining.set_output_embeddings复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置预测头部的解码器权重为新的嵌入
        self.cls.predictions.decoder = new_embeddings

    # 对模型的前向传播进行注释
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ErnieForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加对Ernie模型进行条件语言建模微调的文档字符串
@add_start_docstrings(
    """Ernie Model with a `language modeling` head on top for CLM fine-tuning.""", ERNIE_START_DOCSTRING
)
class ErnieForCausalLM(ErniePreTrainedModel):
    # 定义与权重相关的键，这些键被绑定在一起
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.__init__复制而来，将BertLMHeadModel->ErnieForCausalLM,Bert->Ernie,bert->ernie
    def __init__(self, config):
        super().__init__(config)

        # 如果不是解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `ErnieForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化Ernie模型和Ernie的仅MLM头部
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        self.cls = ErnieOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings复制而来
    def get_output_embeddings(self):
        # 返回仅MLM头部的解码器权重
        return self.cls.predictions.decoder
    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings复制而来，用于设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 添加模型前向传播的文档字符串和示例代码文档字符串，参考ERNIE_INPUTS_DOCSTRING格式，并指定检查点、输出类型、配置类
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.prepare_inputs_for_generation复制而来，用于生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        # 获取输入形状
        input_shape = input_ids.shape
        # 如果模型作为编码器-解码器模型的解码器使用，解码器注意力掩码会在运行时创建
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用过去的键值对，则截取解码器输入ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留仅最后一个ID的旧行为
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel._reorder_cache复制而来，用于重新排序缓存
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 导入函数装饰器，用于添加文档字符串
@add_start_docstrings("""Ernie Model with a `language modeling` head on top.""", ERNIE_START_DOCSTRING)
# 定义 ErnieForMaskedLM 类，继承自 ErniePreTrainedModel 类
class ErnieForMaskedLM(ErniePreTrainedModel):
    # 定义私有属性，存储需要共享权重的键名列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置指定为解码器，发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `ErnieForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 ErnieModel 实例，不添加池化层
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        # 创建 ErnieOnlyMLMHead 实例
        self.cls = ErnieOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层的函数
    def get_output_embeddings(self):
        # 返回预测的解码器权重
        return self.cls.predictions.decoder

    # 设置输出嵌入层的函数
    def set_output_embeddings(self, new_embeddings):
        # 设置预测的解码器权重为新的嵌入层权重
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接收多个输入参数
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 参数为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ERNIE 模型进行预测
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取预测的序列输出
        sequence_output = outputs[0]
        # 使用分类层预测下一个词的分数
        prediction_scores = self.cls(sequence_output)

        # 计算掩码语言模型损失
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 计算损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典，则组合输出并返回
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回掩码语言模型输出对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 从 transformers.models.bert.modeling_bert.BertForMaskedLM.prepare_inputs_for_generation 复制而来
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 添加一个虚拟的 token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 拼接 attention_mask，增加一个虚拟 token 对应的 mask
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全为 PAD token 的虚拟 token
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 将虚拟 token 拼接到输入中
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回输入字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 添加 Ernie 模型的下一个句子预测（分类）头部
@add_start_docstrings(
    """Ernie Model with a `next sentence prediction (classification)` head on top.""",
    ERNIE_START_DOCSTRING,
)

# 创建 Ernie 的下一个句子预测模型类，继承自 ErniePreTrainedModel
class ErnieForNextSentencePrediction(ErniePreTrainedModel):
    
    # 从 transformers.models.bert.modeling_bert.BertForNextSentencePrediction.__init__ 复制而来，做了 Bert->Ernie, bert->ernie 的替换
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 ErnieModel 实例
        self.ernie = ErnieModel(config)
        # 创建 ErnieOnlyNSPHead 实例
        self.cls = ErnieOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 方法，添加注释
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
        ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
        标签(`torch.LongTensor`格式，形状为`(batch_size,)`，*可选*):
        Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
        (see `input_ids` docstring). Indices should be in `[0, 1]`:

        - 0 indicates sequence B is a continuation of sequence A,
        - 1 indicates sequence B is a random sequence.

        Returns:
        返回：

        Example:
        例子：

        ```python
        >>> from transformers import AutoTokenizer, ErnieForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
        >>> model = ErnieForNextSentencePrediction.from_pretrained("nghuyong/ernie-1.0-base-zh")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
        如果`return_dict`不为真：
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output
            返回((如果`next_sentence_loss`不为空则加上`output`的`next_sentence_loss`,)否则为`output`)

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入函数装饰器，用于添加模型文档字符串
@add_start_docstrings(
    """
    Ernie Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ERNIE_START_DOCSTRING,
)
# 定义一个 Ernie 序列分类模型，继承自 ErniePreTrainedModel 类
class ErnieForSequenceClassification(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ 复制而来，将 Bert 替换为 Ernie
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置模型的标签数量
        self.num_labels = config.num_labels
        # 保存模型配置
        self.config = config

        # 使用 ErnieModel 创建 Ernie 模型
        self.ernie = ErnieModel(config)
        # 设置分类器的 dropout 概率，如果未设置，则使用配置中的隐藏层 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用模型文档字符串装饰器，添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个函数，用于执行序列分类任务，并返回一个元组或SequenceClassifierOutput
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 将未指定的返回值赋值为self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 使用Ernie模型来处理输入，获取输出
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 通过处理后的输出获取池化输出
        pooled_output = outputs[1]
    
        # 对池化输出应用dropout
        pooled_output = self.dropout(pooled_output)
        # 将处理后的结果输入到分类器中，获取logits
        logits = self.classifier(pooled_output)
    
        # 初始化loss为空
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果问题类型未指定
            if self.config.problem_type is None:
                # 判断标签数量确定问题类型
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
    
            # 根据问题类型计算loss
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
            # 将输出结果组成元组
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回SequenceClassifierOutput类型的结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class ErnieForMultipleChoice(ErniePreTrainedModel):
    # 多选分类头部的 Ernie 模型（在汇总输出的基础上加一个线性层和 softmax），例如用于 RocStories/SWAG 任务。
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Ernie 模型
        self.ernie = ErnieModel(config)
        # 分类器的丢弃率取决于配置，如果未设置，则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用丢弃来避免过拟合
        self.dropout = nn.Dropout(classifier_dropout)
        # 分类器线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 此处的注释包含了 model_forward 方法的文档字符串和代码示例的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token ids
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 ids
        task_type_ids: Optional[torch.Tensor] = None,  # 任务类型 ids
        position_ids: Optional[torch.Tensor] = None,  # 位置 ids
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入向量
        labels: Optional[torch.Tensor] = None,  # 标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置返回值类型，可以是一个元组或者MultipleChoiceModelOutput对象
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择的数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果input_ids不为空，则将其形状变为(-1, input_ids的最后一个维度)，否则为None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果attention_mask不为空，则将其形状变为(-1, attention_mask的最后一个维度)，否则为None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果token_type_ids不为空，则将其形状变为(-1, token_type_ids的最后一个维度)，否则为None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果position_ids不为空，则将其形状变为(-1, position_ids的最后一个维度)，否则为None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果inputs_embeds不为空，则将其形状变为(-1, inputs_embeds的倒数第二个维度, inputs_embeds的最后一个维度)，否则为None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用self.ernie()方法，传入相关参数，获取outputs
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从outputs中获取pooled_output
        pooled_output = outputs[1]

        # 对pooled_output进行dropout
        pooled_output = self.dropout(pooled_output)
        # 通过分类器获取logits
        logits = self.classifier(pooled_output)
        # 将logits进行形状变换，变为(-1, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果labels不为空，计算loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果return_dict为False，则返回output
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回MultipleChoiceModelOutput对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入必要的模块或函数
@add_start_docstrings(
    """
    Ernie Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ERNIE_START_DOCSTRING,
)
# 定义一个 Ernie 用于标记分类任务的模型，继承自 ErniePreTrainedModel 类
class ErnieForTokenClassification(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 中复制而来，将 Bert 替换为 Ernie，bert 替换为 ernie
    def __init__(self, config):
        # 调用父类 ErniePreTrainedModel 的初始化方法
        super().__init__(config)
        # 设置模型的标签数量
        self.num_labels = config.num_labels

        # 实例化 ErnieModel 类，并且设置不添加池化层
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        # 设置分类器的 dropout，如果未指定则使用隐藏层的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个线性层，将隐藏状态映射到标签数量上
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 对模型的前向传播进行注释，根据输入生成输出
    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        # 定义函数返回值类型为 torch.Tensor 元组或 TokenClassifierOutput 类型
        r"""
        # 定义 labels 参数的注释，表示形状为 (batch_size, sequence_length) 的 LongTensor，用于计算 token 分类的损失
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 解释 labels 参数的作用和范围
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否使用返回字典，若 return_dict 未设置则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用给定的参数调用 ernie 模型，返回一个 outputs 元组
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 outputs 中获取序列输出结果，即第一个元素
        sequence_output = outputs[0]

        # 应用 dropout 层在序列输出结果上
        sequence_output = self.dropout(sequence_output)
        # 使用分类器将序列输出转换为 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了 labels 参数，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 使用 logits 和 labels 计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典
        if not return_dict:
            # 将 logits 和 outputs 的其余部分组合成一个元组并返回
            output = (logits,) + outputs[2:]
            # 如果存在损失，则在返回的元组中包含损失
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包括损失、logits、隐藏状态和注意力信息
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 add_start_docstrings 函数添加模型介绍文档，包括用于提取问题答案任务的 Span 分类头
class ErnieForQuestionAnswering(ErniePreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 复制代码，将 Bert 替换为 Ernie
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 为模型设置标签数量
        self.num_labels = config.num_labels

        # 创建 Ernie 模型，不添加池化层
        self.ernie = ErnieModel(config, add_pooling_layer=False)
        # 创建用于问题答案输出的线性层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 函数添加模型前向传播的介绍文档
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        """
        # 检查是否需要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用ERNIE模型进行预测
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 通过QA输出层计算logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # 计算损失
        if start_positions is not None and end_positions is not None:
            # 如果是多GPU，添加一维
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时候起始/结束位置超出模型输入，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典，则返回输出
        if not return_dict:
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
```