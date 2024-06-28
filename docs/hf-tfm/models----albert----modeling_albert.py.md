# `D:\Python310\Lib\site-packages\transformers\models\albert\modeling_albert.py`

```py
# coding=utf-8
# 上面的注释指定了源代码文件的编码格式为UTF-8

# 版权声明和许可证信息，这里使用Apache License 2.0
# 允许根据此许可证使用和分发代码
# 详细信息可参考 http://www.apache.org/licenses/LICENSE-2.0

"""PyTorch ALBERT model."""
# 引入标准库和第三方库
import math  # 导入数学库
import os  # 导入操作系统相关的功能
from dataclasses import dataclass  # 从标准库导入 dataclass 类型装饰器
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的功能

import torch  # 导入PyTorch库
from torch import nn  # 导入PyTorch的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数类

# 导入子模块和函数
from ...activations import ACT2FN  # 从上层包中导入 ACT2FN 激活函数
from ...modeling_outputs import (  # 导入模型输出相关的类
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import (  # 导入PyTorch工具函数
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (  # 导入实用函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_albert import AlbertConfig  # 导入ALBERT配置类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档用的模型检查点和配置文件路径
_CHECKPOINT_FOR_DOC = "albert/albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"

# 预训练模型存档列表，包含多个ALBERT模型
ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert/albert-base-v1",
    "albert/albert-large-v1",
    "albert/albert-xlarge-v1",
    "albert/albert-xxlarge-v1",
    "albert/albert-base-v2",
    "albert/albert-large-v2",
    "albert/albert-xlarge-v2",
    "albert/albert-xxlarge-v2",
    # 可以在 https://huggingface.co/models?filter=albert 查看所有的ALBERT模型
]

# 加载TensorFlow模型权重到PyTorch模型
def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入NumPy库
        import tensorflow as tf  # 导入TensorFlow库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    # 获取TensorFlow模型检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")

    # 从TensorFlow模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    # 遍历TensorFlow模型的所有变量
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # 打印加载的变量名
    for name, array in zip(names, arrays):
        print(name)

    return model
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: AlbertConfig):
        super().__init__()
        # 初始化词嵌入层，将词汇表大小、嵌入维度和填充标记索引传入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，将最大位置嵌入数和嵌入维度传入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # 初始化标记类型嵌入层，将标记类型词汇表大小和嵌入维度传入
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 初始化 LayerNorm 层，传入嵌入维度和层归一化 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，传入隐藏单元的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 注册 position_ids 缓冲区，这是一个持久化的张量，表示位置 ID 序列
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册 token_type_ids 缓冲区，表示标记类型 ID 序列，初始化为全零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 定义函数，输入参数为input_ids（输入的标识符序列）、inputs_embeds（嵌入的输入序列）、position_ids（位置标识符序列）、
    # token_type_ids（标记类型序列）以及past_key_values_length（过去的键值对长度），返回torch.Tensor类型的张量
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果给定input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取inputs_embeds的形状，但不包括最后一维（通常是批处理维度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，假设为输入形状的第二个维度
        seq_length = input_shape[1]

        # 如果未提供position_ids，则从self.position_ids中获取一个子集，其范围从past_key_values_length到seq_length + past_key_values_length
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置token_type_ids为在构造函数中注册的缓冲区，通常是全零。这在模型跟踪时帮助用户，避免传递token_type_ids时的问题
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果没有已注册的token_type_ids，则创建一个全零的张量，形状与输入形状相同，类型为torch.long，设备为self.position_ids的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供inputs_embeds，则使用self.word_embeddings对input_ids进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 根据token_type_ids获取token类型的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和token类型嵌入相加得到最终嵌入
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置嵌入类型为"absolute"，则加上位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对嵌入进行LayerNorm归一化
        embeddings = self.LayerNorm(embeddings)
        
        # 对嵌入进行dropout处理
        embeddings = self.dropout(embeddings)
        
        # 返回处理后的嵌入张量作为结果
        return embeddings
    # 定义 AlbertAttention 类，继承自 nn.Module
    class AlbertAttention(nn.Module):
        def __init__(self, config: AlbertConfig):
            # 调用父类构造函数
            super().__init__()
            # 检查隐藏层大小是否是注意力头数的整数倍，如果不是且没有嵌入大小的属性，则抛出错误
            if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({config.num_attention_heads}"
                )

            # 初始化类的属性
            self.num_attention_heads = config.num_attention_heads
            self.hidden_size = config.hidden_size
            self.attention_head_size = config.hidden_size // config.num_attention_heads
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # 定义查询、键、值的线性层
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

            # 定义注意力机制的 dropout
            self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
            # 定义输出的 dropout
            self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
            # 定义最终输出的线性层
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            # 定义 LayerNorm 层
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            # 初始化被修剪的注意力头集合
            self.pruned_heads = set()

            # 设置位置嵌入的类型，默认为绝对位置嵌入
            self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
            # 如果位置嵌入类型是相对键或相对键查询，则需要额外的距离嵌入层
            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                self.max_position_embeddings = config.max_position_embeddings
                self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 定义转置操作函数，用于将输入张量 x 转置成适合多头注意力的形状
        def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        # 定义修剪注意力头的方法
        def prune_heads(self, heads: List[int]) -> None:
            if len(heads) == 0:
                return
            # 查找可修剪的注意力头并获取其索引
            heads, index = find_pruneable_heads_and_indices(
                heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
            )

            # 对线性层进行修剪
            self.query = prune_linear_layer(self.query, index)
            self.key = prune_linear_layer(self.key, index)
            self.value = prune_linear_layer(self.value, index)
            self.dense = prune_linear_layer(self.dense, index, dim=1)

            # 更新超参数并记录修剪的头
            self.num_attention_heads = self.num_attention_heads - len(heads)
            self.all_head_size = self.attention_head_size * self.num_attention_heads
            self.pruned_heads = self.pruned_heads.union(heads)

        # 定义前向传播函数
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: bool = False,
# 定义一个 Albert 模型的层
class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        # 初始化层的配置信息
        self.config = config
        # 设置前馈网络分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # 全连接层的 LayerNorm
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 注意力机制模块
        self.attention = AlbertAttention(config)
        # 前馈网络的线性层
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        # 前馈网络输出层
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 使用注意力机制模块处理隐藏状态
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        # 应用分块机制到前馈网络
        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        # 对前馈网络的输出进行 LayerNorm 处理
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        # 返回处理后的隐藏状态和注意力机制输出
        return (hidden_states,) + attention_output[1:]  # 如果有输出注意力机制，则返回它们

    # 前馈网络分块处理函数定义
    def ff_chunk(self, attention_output: torch.Tensor) -> torch.Tensor:
        # 前馈网络的线性层
        ffn_output = self.ffn(attention_output)
        # 应用激活函数
        ffn_output = self.activation(ffn_output)
        # 前馈网络输出层
        ffn_output = self.ffn_output(ffn_output)
        # 返回前馈网络的输出
        return ffn_output


# 定义 Albert 模型的层组
class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        # 创建多个 AlbertLayer 组成的层列表
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    # 层组的前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 对每一层的 AlbertLayer 进行前向传播
        for layer_module in self.albert_layers:
            # 使用当前层的前向传播函数处理隐藏状态
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )[0]

        # 返回处理后的隐藏状态和可能的输出
        return hidden_states,  # 注意返回值是一个元组
    # 定义函数签名，指定返回类型为元组，元组中包含 torch.Tensor 或元组的联合类型
    -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # 初始化空元组用于存储每个层的隐藏状态
        layer_hidden_states = ()
        # 初始化空元组用于存储每个层的注意力权重
        layer_attentions = ()

        # 遍历 self.albert_layers 中的每一层
        for layer_index, albert_layer in enumerate(self.albert_layers):
            # 调用当前层的前向传播方法，计算当前层的输出
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            # 更新 hidden_states 为当前层的输出
            hidden_states = layer_output[0]

            # 如果需要输出注意力权重
            if output_attentions:
                # 将当前层的注意力权重添加到 layer_attentions 中
                layer_attentions = layer_attentions + (layer_output[1],)

            # 如果需要输出每层的隐藏状态
            if output_hidden_states:
                # 将当前层的隐藏状态添加到 layer_hidden_states 中
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        # 构建最终的输出元组，包含最后一层的隐藏状态
        outputs = (hidden_states,)
        # 如果需要输出每层的隐藏状态，则将其添加到 outputs 中
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        # 如果需要输出每层的注意力权重，则将其添加到 outputs 中
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        # 返回最终的 outputs，包括最后一层的隐藏状态、每层的隐藏状态和每层的注意力权重（根据需要）
        return outputs  # 最后一层的隐藏状态，(每层隐藏状态)，(每层注意力权重)
class AlbertTransformer(nn.Module):
    # AlbertTransformer 类定义，继承自 nn.Module
    def __init__(self, config: AlbertConfig):
        # 初始化方法，接收 AlbertConfig 类型的 config 参数
        super().__init__()

        # 将传入的配置信息保存到对象的 config 属性中
        self.config = config
        # 创建一个线性层，将输入的 embedding_size 维度映射到 hidden_size 维度
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        # 创建包含多个 AlbertLayerGroup 对象的 ModuleList，数量为 config.num_hidden_groups
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        # 将输入的 hidden_states 通过线性映射层 embedding_hidden_mapping_in 进行维度映射
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        # 初始化用于存储所有隐藏状态和注意力的空元组
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 如果未提供头部遮罩 head_mask，则创建一个与层数相同长度的 None 列表
        head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask

        # 遍历每个隐藏层
        for i in range(self.config.num_hidden_layers):
            # 计算每个隐藏组中的层数
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)
            # 确定当前隐藏层所在的隐藏组索引
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            # 调用对应隐藏组的 AlbertLayerGroup 进行前向传播
            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            # 更新 hidden_states 为当前隐藏组的输出的第一个元素（通常是隐藏状态）
            hidden_states = layer_group_output[0]

            # 如果需要输出注意力，将当前隐藏组的注意力信息添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            # 如果需要输出隐藏状态，将当前隐藏组的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则按顺序返回 hidden_states, all_hidden_states, all_attentions 中非 None 的元素
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 否则，以 BaseModelOutput 形式返回结果，包括最后的隐藏状态、所有隐藏状态和所有注意力信息
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class AlbertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # AlbertPreTrainedModel 类，继承自 PreTrainedModel，是用于处理权重初始化和预训练模型下载加载的抽象类

    # 配置类的引用
    config_class = AlbertConfig
    # 加载 TensorFlow 权重的方法引用
    load_tf_weights = load_tf_weights_in_albert
    # 基础模型前缀
    base_model_prefix = "albert"
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0.0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0.0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
@dataclass
class AlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`AlbertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
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

    loss: Optional[torch.FloatTensor] = None  # 损失值，当提供`labels`时返回，为浮点数张量，形状为`(1,)`
    prediction_logits: torch.FloatTensor = None  # 语言建模头部的预测分数，softmax之前每个词汇标记的分数，形状为`(batch_size, sequence_length, config.vocab_size)`
    sop_logits: torch.FloatTensor = None  # 下一个序列预测（分类）头部的预测分数，softmax之前True/False延续的分数，形状为`(batch_size, 2)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型每层输出的隐藏状态，包括初始嵌入输出，形状为`(batch_size, sequence_length, hidden_size)`的浮点数张量元组，当`output_hidden_states=True`时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，用于计算自注意力头部的加权平均值，形状为`(batch_size, num_heads, sequence_length, sequence_length)`的浮点数张量元组，当`output_attentions=True`时返回


ALBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`AlbertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ALBERT_INPUTS_DOCSTRING = r"""
"""
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记在词汇表中的索引

                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.__call__`] and
                [`PreTrainedTokenizer.encode`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 注意力掩码，避免在填充标记上执行注意力计算

                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 分段标记索引，指示输入的第一部分和第二部分

                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
                1]`:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                [What are token type IDs?](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 每个输入序列标记在位置嵌入中的位置索引

                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.

                [What are position IDs?](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 自注意力模块中选择性屏蔽的头部掩码

                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选参数，直接传入嵌入表示而不是输入标记索引

                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量

                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态

                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                # 是否返回一个 `~utils.ModelOutput` 而不是一个普通元组

                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
AlbertModel 类定义了 ALBERT 模型，用于处理文本数据的转换器模型。

@add_start_docstrings(
    "The bare ALBERT Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class AlbertModel(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)  # 初始化 ALBERT 的嵌入层
        self.encoder = AlbertTransformer(config)  # 初始化 ALBERT 的 transformer 编码器

        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)  # 添加池化层的线性变换
            self.pooler_activation = nn.Tanh()  # 池化层的激活函数为 Tanh
        else:
            self.pooler = None
            self.pooler_activation = None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings  # 返回嵌入层的词嵌入

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embeddings.word_embeddings = value  # 设置嵌入层的词嵌入为指定值

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            # 剪枝指定层的注意力头部
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[BaseModelOutputWithPooling, Tuple]:
        # 如果未显式指定，使用配置中的值来确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未显式指定，使用配置中的值来确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未显式指定，使用配置中的值来确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查输入参数，确保不同时指定 input_ids 和 inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果指定了 input_ids，则检查是否需要警告关于填充和注意力蒙版的使用
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 如果指定了 inputs_embeds，则获取其形状（除最后一个维度）
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取批次大小和序列长度
        batch_size, seq_length = input_shape
        # 获取设备信息，用于在 GPU 或 CPU 上执行操作
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供 attention_mask，则创建一个全为1的张量
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果未提供 token_type_ids，则根据嵌入层的特性来决定是否需要创建
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 扩展注意力蒙版的维度以匹配编码器的期望形状
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # 为了fp16的兼容性
        # 对扩展的注意力蒙版进行填充，使未考虑部分的权重为负无穷
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        # 获取头部遮罩，用于确定哪些层的注意力应被屏蔽
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 通过嵌入层获取嵌入输出
        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # 通过编码器处理嵌入输出，获取编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器的输出中获取序列输出
        sequence_output = encoder_outputs[0]

        # 如果存在池化器，则对序列输出的首个位置进行池化操作并激活
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        # 如果不返回字典格式的输出，则返回元组形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回字典格式的输出，包括序列输出、池化输出、隐藏状态和注意力权重
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
"""
Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
`sentence order prediction (classification)` head.
"""
# 导入所需的模块和库
from transformers import AlbertPreTrainedModel, AlbertModel, AlbertConfig
from transformers.modeling_outputs import AlbertForPreTrainingOutput
from transformers.activations import ACT2FN
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
import torch
import torch.nn as nn

# ALBERT_START_DOCSTRING 定义在引入的模块中，这里假设是常量或全局变量

@add_start_docstrings(
    """
    Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence order prediction (classification)` head.
    """,
    ALBERT_START_DOCSTRING,
)
class AlbertForPreTraining(AlbertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig):
        super().__init__(config)

        # 初始化 Albert 模型和两个头部
        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=AlbertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sentence_order_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
        # 省略了 forward 方法的实现，由 AlbertPreTrainedModel 提供

class AlbertMLMHead(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()

        # 初始化 MLM 头部的各个组件
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.decoder.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MLM 头部的前向传播
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores

    def _tie_weights(self) -> None:
        # 如果这两个权重被分离（在 TPU 上或当偏置被调整大小时），将它们绑定起来
        self.bias = self.decoder.bias


class AlbertSOPHead(nn.Module):
    # 初始化方法，接受一个 AlbertConfig 类型的参数 config
    def __init__(self, config: AlbertConfig):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()

        # 创建一个 dropout 层，使用 config 中的 classifier_dropout_prob 参数作为丢弃概率
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        
        # 创建一个全连接层（线性变换），输入大小为 config.hidden_size，输出大小为 config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    # 前向传播方法，输入一个 torch.Tensor 类型的 pooled_output，返回一个 torch.Tensor 类型的 logits
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        # 对 pooled_output 应用 dropout 操作
        dropout_pooled_output = self.dropout(pooled_output)
        
        # 将 dropout 后的 pooled_output 输入到全连接层中进行线性变换，得到 logits
        logits = self.classifier(dropout_pooled_output)
        
        # 返回 logits 作为输出
        return logits
# 使用装饰器添加文档字符串，描述该类是带有顶层语言建模头的Albert模型。
@add_start_docstrings(
    "Albert Model with a `language modeling` head on top.",
    ALBERT_START_DOCSTRING,
)
# 定义Albert模型的具体实现类，继承自AlbertPreTrainedModel。
class AlbertForMaskedLM(AlbertPreTrainedModel):
    # 定义需要共享权重的关键键列表。
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    # 初始化方法，接收配置参数config并调用父类的初始化方法。
    def __init__(self, config):
        super().__init__(config)

        # 创建Albert模型实例，不添加池化层。
        self.albert = AlbertModel(config, add_pooling_layer=False)
        # 创建AlbertMLMHead实例，用于预测。
        self.predictions = AlbertMLMHead(config)

        # 初始化权重并进行最终处理。
        self.post_init()

    # 获取输出嵌入层的方法，返回预测层的解码器。
    def get_output_embeddings(self) -> nn.Linear:
        return self.predictions.decoder

    # 设置输出嵌入层的方法，用新的线性层替换预测层的解码器。
    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.predictions.decoder = new_embeddings

    # 获取输入嵌入层的方法，返回Albert模型的词嵌入。
    def get_input_embeddings(self) -> nn.Embedding:
        return self.albert.embeddings.word_embeddings

    # 前向传播方法，接收多个输入参数，返回掩码语言模型的输出。
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
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

        # 方法的具体实现在AlbertPreTrainedModel中被覆盖。
        # 根据输入参数执行Albert模型的前向传播，返回预测的掩码语言模型输出。
        pass  # 这里只是声明方法，实现在父类中


这样的注释能够清晰地解释每个方法和类的作用及其关键细节，帮助他人理解代码的功能和实现方式。
        ) -> Union[MaskedLMOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 根据 return_dict 是否为 None 决定是否使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Albert 模型，传入各种参数，获取模型输出
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从模型输出中取出序列输出
        sequence_outputs = outputs[0]

        # 使用预测器预测序列输出中的预测分数
        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        # 如果 labels 不为 None，则计算 masked language modeling 的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回完整的输出，包括预测分数和其他输出状态
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则返回 MaskedLMOutput 对象，包括损失、预测分数、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有顶部序列分类/回归头部的 Albert 模型转换器，例如用于 GLUE 任务（该头部是在汇总输出之上的线性层）。
# 这是一个使用 ALBERT 的特定文档字符串的装饰器。
@add_start_docstrings(
    """
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ALBERT_START_DOCSTRING,  # 引入 ALBERT 模型的通用文档字符串
)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量
        self.config = config

        self.albert = AlbertModel(config)  # 创建 ALBERT 模型
        self.dropout = nn.Dropout(config.classifier_dropout_prob)  # 应用分类器的 dropout
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)  # 创建分类器的线性层

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="textattack/albert-base-v2-imdb",  # 提供了一个代码示例的检查点
        output_type=SequenceClassifierOutput,  # 预期的输出类型
        config_class=_CONFIG_FOR_DOC,  # 用于文档的配置类
        expected_output="'LABEL_1'",  # 预期的输出
        expected_loss=0.12,  # 预期的损失
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
        # 上述参数用于模型前向传播，描述了每个参数的类型和作用

        # 以下是参数的描述：
        # - input_ids: 输入的token IDs
        # - attention_mask: 注意力掩码，用于指示哪些元素需要被注意力机制忽略
        # - token_type_ids: 标识不同序列的token类型，如segment A/B用于BERT等模型
        # - position_ids: 标识每个token在序列中的位置ID
        # - head_mask: 用于控制多头注意力机制中每个头部的掩码
        # - inputs_embeds: 可选的嵌入表示输入
        # - labels: 对应于每个输入的标签，用于训练
        # - output_attentions: 是否返回注意力权重
        # - output_hidden_states: 是否返回所有隐藏状态
        # - return_dict: 是否返回字典类型的输出

        # 下面的装饰器为模型的前向方法添加了文档字符串，描述了输入参数的具体形状和含义
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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否使用返回字典，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ALBERT 模型进行前向传播计算
        outputs = self.albert(
            input_ids=input_ids,                      # 输入的词索引序列
            attention_mask=attention_mask,            # 输入的注意力掩码，指示哪些标记是实际输入，哪些是填充
            token_type_ids=token_type_ids,            # 输入的标记类型 IDs，用于区分不同句子或片段
            position_ids=position_ids,                # 输入的位置 IDs，指示每个位置在输入中的位置
            head_mask=head_mask,                      # 多头注意力机制的掩码，用于控制每个头的权重
            inputs_embeds=inputs_embeds,              # 替代输入的嵌入表示
            output_attentions=output_attentions,      # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,                  # 是否返回字典格式的输出
        )

        # 获取 ALBERT 模型的汇聚输出
        pooled_output = outputs[1]

        # 对汇聚输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        
        # 使用分类器对汇聚输出进行分类得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型和标签数量推断问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"  # 如果标签数量为 1，则为回归问题
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"  # 如果标签数量大于 1 且标签类型为 long 或 int，则为单标签分类问题
                else:
                    self.config.problem_type = "multi_label_classification"  # 否则为多标签分类问题

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()  # 均方误差损失函数
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # 对于回归问题，计算损失
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()  # 交叉熵损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 对于单标签分类问题，计算损失
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()  # 二元交叉熵损失函数
                loss = loss_fct(logits, labels)  # 对于多标签分类问题，计算损失

        # 如果不需要返回字典格式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]  # 输出结果包括 logits 和额外的输出信息
            return ((loss,) + output) if loss is not None else output  # 如果有损失，则加入到输出结果中

        # 如果需要返回字典格式的输出，构造 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,                               # 损失值
            logits=logits,                           # 模型输出的 logits
            hidden_states=outputs.hidden_states,     # 隐藏状态
            attentions=outputs.attentions,           # 注意力权重
        )
"""
Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""  # 描述 Albert 模型，在隐藏状态输出之上添加了一个用于标记分类（例如命名实体识别）的线性层的头部。

# 使用 ALBERT_START_DOCSTRING 和额外提供的描述来为类添加文档字符串
@add_start_docstrings(
    ALBERT_START_DOCSTRING,
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """
)
class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 ALBERT 模型，不包括池化层
        self.albert = AlbertModel(config, add_pooling_layer=False)

        # 根据配置设置分类器的 dropout 概率
        classifier_dropout_prob = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout_prob)

        # 线性层，将 ALBERT 隐藏层的输出映射到标签数量的空间
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为 forward 方法添加文档字符串，描述输入和输出的格式
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，指定了模型使用的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
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
):
        ) -> Union[TokenClassifierOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则根据配置决定是否使用 return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ALBERT 模型进行前向传播，获取输出结果
        outputs = self.albert(
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

        # 从 ALBERT 模型输出中获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        
        # 将 dropout 后的输出送入分类器得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用 return_dict，则返回完整的输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 类构建返回结果，包括损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
@add_start_docstrings(
    """
    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ALBERT_START_DOCSTRING,  # 添加 Albert 模型的文档字符串和 Albert 的开始文档字符串
)
class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config: AlbertConfig):
        super().__init__(config)  # 调用父类的初始化方法
        self.num_labels = config.num_labels  # 设置标签数量

        self.albert = AlbertModel(config, add_pooling_layer=False)  # 初始化 Albert 模型，不加汇聚层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 使用线性层进行输出

        # Initialize weights and apply final processing
        self.post_init()  # 调用自定义的初始化方法

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="twmkn9/albert-base-v2-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=12,
        qa_target_end_index=13,
        expected_output="'a nice puppet'",
        expected_loss=7.36,
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
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
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
        # Determine whether to use the return_dict from function arguments or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to the Albert model and obtain outputs
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract sequence output from the model outputs
        sequence_output = outputs[0]

        # Generate logits by passing sequence output through the QA output layer
        logits: torch.Tensor = self.qa_outputs(sequence_output)

        # Split logits into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Ensure start_positions and end_positions are properly shaped for processing
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # Clamp positions within valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define loss function and compute start_loss and end_loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # If return_dict is False, return output as tuple
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # If return_dict is True, return structured output
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个 Albert 模型，该模型在顶部具有用于多选分类任务的分类头部（一个线性层叠加在汇总输出和 softmax 上），例如用于 RocStories/SWAG 任务。
@add_start_docstrings(
    """
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ALBERT_START_DOCSTRING,  # 添加起始注释，描述 Albert 模型的概述
)
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    
    def __init__(self, config: AlbertConfig):
        super().__init__(config)

        # 初始化 Albert 模型
        self.albert = AlbertModel(config)
        # 使用配置中的 dropout 概率初始化 dropout 层
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 使用隐藏层大小初始化分类器线性层，输出维度为1（用于多选任务）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()  # 执行后期初始化操作，可能包括权重初始化等

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
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
        # 定义前向传播函数，接受一系列输入参数，并输出多选模型的预测结果
        ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
            *input_ids* above)
        """
        # 根据是否指定返回字典的选项来确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选项数目
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果传入了input_ids，则将其视作(batch_size, num_choices, sequence_length)的形状
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果传入了attention_mask，则将其视作(batch_size, num_choices, sequence_length)的形状
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果传入了token_type_ids，则将其视作(batch_size, num_choices, sequence_length)的形状
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果传入了position_ids，则将其视作(batch_size, num_choices, sequence_length)的形状
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果传入了inputs_embeds，则将其视作(batch_size, num_choices, sequence_length, hidden_size)的形状
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        # 调用Albert模型进行前向传播，获取输出
        outputs = self.albert(
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

        # 从模型输出中获取汇聚的输出表示
        pooled_output = outputs[1]

        # 对汇聚的输出表示应用dropout
        pooled_output = self.dropout(pooled_output)
        # 将处理后的汇聚输出表示输入分类器，得到logits
        logits: torch.Tensor = self.classifier(pooled_output)
        # 将logits变形为(batch_size * num_choices, -1)的形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果给定了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典形式的输出，则按照元组的形式返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出，则构建MultipleChoiceModelOutput对象并返回
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```