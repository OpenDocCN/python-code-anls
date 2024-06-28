# `.\models\git\modeling_git.py`

```
# coding=utf-8
# 定义文件编码格式为 UTF-8

# 版权声明和许可证信息，声明版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
# 受 Apache 许可证第 2.0 版的限制，除非遵守许可证，否则不得使用此文件
# 可在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本

# 引入 math 模块，用于数学运算
import math
# 引入 dataclass 用于创建数据类，引入 List、Optional、Tuple 和 Union 用于类型注解
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# 引入 PyTorch 深度学习框架
import torch
# 引入 PyTorch 中的检查点功能
import torch.utils.checkpoint
# 从 torch 模块中导入 nn 模块，用于神经网络构建
from torch import nn
# 从 nn 模块导入交叉熵损失函数
from torch.nn import CrossEntropyLoss

# 引入相对路径下的模块和函数
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
# 从 modeling_utils 模块导入预训练模型的基类
from ...modeling_utils import PreTrainedModel
# 从 pytorch_utils 模块导入一些辅助函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 从 utils 模块导入添加文档字符串、模型前向方法的文档字符串、日志记录和替换返回文档字符串的函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从当前目录的 configuration_git 模块中导入 GitConfig 和 GitVisionConfig 类
from .configuration_git import GitConfig, GitVisionConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "microsoft/git-base"
# 用于文档的配置名称
_CONFIG_FOR_DOC = "GitConfig"

# 预训练的 Git 模型存档列表
GIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/git-base",
    # 可以在 https://huggingface.co/models?filter=git 查看所有 GIT 模型
]

# 数据类，用于描述 Git 视觉模型的输出
@dataclass
# 继承自 ModelOutput 类
# 与 CLIP 模型中的 CLIPVisionModelOutput 类似，但适用于 Git 模型
class GitVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    """
    """
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 可选参数：图像嵌入向量，形状为 `(batch_size, output_dim)`，在模型初始化时如果使用 `with_projection=True` 会返回
    image_embeds: Optional[torch.FloatTensor] = None

    # 必需参数：最后一层模型的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`
    last_hidden_state: torch.FloatTensor = None

    # 可选参数：模型隐藏状态的元组，包含每层的输出，如果设置了 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    # 可选参数：注意力权重的元组，包含每层的注意力权重，如果设置了 `output_attentions=True` 或 `config.output_attentions=True` 时返回
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
class GitEmbeddings(nn.Module):
    """构建从词嵌入和位置嵌入到最终嵌入的模块。"""

    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，根据配置参数指定词汇量大小、隐藏层大小，并设置填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，根据配置参数指定最大位置嵌入数量和隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 使用非蛇形命名以保持与 TensorFlow 模型变量名的一致性，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，根据配置参数指定隐藏层的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置 id 缓冲区，是一个 1x最大位置嵌入数量的张量，用于序列化时持久化存储
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 如果未提供位置 ids，则使用预注册的位置 ids，并根据序列长度截取所需部分
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供输入嵌入向量，则根据输入 ids 获取词嵌入
        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)
        else:
            embeddings = inputs_embeds

        # 如果位置嵌入类型为绝对，则添加位置嵌入到词嵌入中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对嵌入向量进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 Dropout 处理
        embeddings = self.dropout(embeddings)
        return embeddings


class GitSelfAttention(nn.Module):
    # 在这里开始编写 GitSelfAttention 类的注释
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否可以被注意力头的数量整除，同时没有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 计算图像块标记的数量
        self.image_patch_tokens = int((config.vision_config.image_size / config.vision_config.patch_size) ** 2 + 1)
        if config.num_image_with_embedding is not None:
            self.image_patch_tokens *= config.num_image_with_embedding

        # 定义查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 确定位置嵌入的类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则设置最大位置嵌入数和距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 将张量重新形状以适应注意力头的排列
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class GitSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入和输出大小都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个LayerNorm层，对隐藏状态进行归一化，设置eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，使用config.hidden_dropout_prob概率进行随机丢弃
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过线性层dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的结果与输入张量input_tensor相加，并通过LayerNorm层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states


class GitAttention(nn.Module):
    # Copied from transformers.models.bert.modeling_bert.BertAttention.__init__ with Bert->Git
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建GitSelfAttention对象，传入config和position_embedding_type参数
        self.self = GitSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建GitSelfOutput对象，传入config参数
        self.output = GitSelfOutput(config)
        # 初始化一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
        # 如果heads列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用find_pruneable_heads_and_indices函数，找到可以修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对self中的query、key、value和output.dense属性进行修剪线性层操作
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用self.self的forward方法，进行自注意力机制计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            past_key_value,
            output_attentions,
            pixel_values_present,
        )
        # 将self_outputs的第一个元素作为输入，通过self.output进行输出层的处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其加入到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回输出元组
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class GitIntermediate(nn.Module):
    # 初始化函数，用于创建一个新的神经网络层对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，将输入维度为 config.hidden_size 的向量映射到 config.intermediate_size 的向量
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置文件中的激活函数类型，选择合适的激活函数
        if isinstance(config.hidden_act, str):
            # 如果配置的激活函数是字符串类型，则从预定义的字典 ACT2FN 中获取对应的函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则，直接使用配置中指定的激活函数
            self.intermediate_act_fn = config.hidden_act
    
    # 前向传播函数，接收一个张量 hidden_states 作为输入，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层 self.dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的张量通过选择的激活函数 self.intermediate_act_fn 进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过激活函数处理后的张量作为输出
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertOutput
# 从 transformers 库中的 BertOutput 类复制而来，用于定义 Git 模型的输出层
class GitOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将中间隐藏层的大小映射到最终隐藏层的大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，以一定概率随机将输入置零，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层映射操作，将中间隐藏状态映射到最终隐藏状态
        hidden_states = self.dense(hidden_states)
        # Dropout 操作，对全连接层输出进行随机置零
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 操作，对加上输入张量后的隐藏状态进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为输出
        return hidden_states


class GitLayer(nn.Module):
    # 从 transformers.models.bert.modeling_bert.BertLayer 复制而来，用于定义 Git 模型的层
    def __init__(self, config):
        super().__init__()
        # 定义一个块大小用于前馈传播的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度，默认为1
        self.seq_len_dim = 1
        # Git 模型中的注意力机制层
        self.attention = GitAttention(config)
        # Git 模型中的中间层
        self.intermediate = GitIntermediate(config)
        # Git 模型中的输出层
        self.output = GitOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 如果过去键/值存在，则从中提取自注意力的缓存键/值元组
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力机制层的前向传播，输出包含注意力输出和其他可能的输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            pixel_values_present=pixel_values_present,
        )
        # 提取自注意力层的注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后的输出是自注意力缓存元组
        outputs = self_attention_outputs[1:-1]
        # 提取当前键/值
        present_key_value = self_attention_outputs[-1]

        # 将前馈传播函数应用于注意力输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将层输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后的输出添加到元组中
        outputs = outputs + (present_key_value,)

        # 返回所有输出
        return outputs

    def feed_forward_chunk(self, attention_output):
        # 中间层的前馈传播函数，先通过中间层处理注意力输出，然后通过输出层得到层输出
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output


class GitEncoder(nn.Module):
    # 从 transformers.models.bert.modeling_bert.BertEncoder.__init__ 复制而来，用于定义 Git 模型的编码器层
    # 暂时没有提供具体实现
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置参数保存到实例变量中
        self.config = config
        # 创建一个包含多个 GitLayer 实例的模块列表，数量由配置参数决定
        self.layer = nn.ModuleList([GitLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法，定义了模型的正向计算过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        pixel_values_present: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        # 如果启用了梯度检查点并且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 参数为 True，则发出警告并强制将其设为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化用于存储所有隐藏状态和自注意力权重的变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果不使用缓存，初始化存储下一步解码器缓存的变量
        next_decoder_cache = () if use_cache else None

        # 遍历所有的 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码（如果有）
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 获取过去的键值对（如果有）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且处于训练模式下，调用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的 forward 方法
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    past_key_value,
                    output_attentions,
                    pixel_values_present,
                )

            # 更新当前隐藏状态为当前层输出的隐藏状态
            hidden_states = layer_outputs[0]

            # 如果使用缓存，将当前层的缓存信息添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            # 如果输出自注意力权重，则将当前层的自注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，将各个部分组合成元组返回
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        # 否则返回一个包含多个属性的对象，表示模型的输出
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# GitPreTrainedModel 类继承自 PreTrainedModel 类，用于处理权重初始化和预训练模型的下载和加载接口。
class GitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类，指定为 GitConfig
    config_class = GitConfig
    # 基础模型前缀为 "git"
    base_model_prefix = "git"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 GitVisionEmbeddings 类的实例
        if isinstance(module, GitVisionEmbeddings):
            # 初始化 class_embedding 层的权重为正态分布，均值为 0.0，标准差为 self.config.initializer_range
            nn.init.normal_(module.class_embedding, mean=0.0, std=self.config.initializer_range)
            # 初始化 patch_embedding 层的权重为正态分布，标准差为 self.config.initializer_range
            nn.init.normal_(module.patch_embedding.weight, std=self.config.initializer_range)
            # 初始化 position_embedding 层的权重为正态分布，标准差为 self.config.initializer_range
            nn.init.normal_(module.position_embedding.weight, std=self.config.initializer_range)
        # 如果 module 是 nn.Linear 类的实例
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0.0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类的实例
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0.0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有 padding_idx，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类的实例
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)


GIT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GIT_INPUTS_DOCSTRING = r"""
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记在词汇表中的索引

                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 注意力遮罩，避免在填充标记上执行注意力操作

                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 输入序列标记在位置嵌入中的索引

                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.

                [What are position IDs?](../glossary#position-ids)

            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                # 像素值，可以使用 [`AutoImageProcessor`] 获取

                Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
                [`CLIPImageProcessor.__call__`] for details.

            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 自注意力模块中选择性屏蔽的头部掩码

                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 直接传入嵌入表示而不是 `input_ids` 的选择性参数

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
                # 是否返回 [`~utils.ModelOutput`] 而不是普通元组

                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 从 transformers.models.clip.modeling_clip.CLIPVisionEmbeddings 复制而来的 GitVisionEmbeddings 类定义
class GitVisionEmbeddings(nn.Module):
    # 初始化函数，接收一个 GitVisionConfig 类型的参数 config
    def __init__(self, config: GitVisionConfig):
        super().__init__()
        # 将 config 参数保存到对象的 config 属性中
        self.config = config
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 设置图像大小为配置中的图像大小
        self.image_size = config.image_size
        # 设置patch大小为配置中的patch大小
        self.patch_size = config.patch_size

        # 创建一个 nn.Parameter 类型的类别嵌入，维度为隐藏大小
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 创建一个二维卷积层，输入通道数为配置中的通道数，输出通道数为隐藏大小，核大小为patch大小，步长为patch大小，无偏置项
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像中patch的数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 计算位置嵌入的数量为patch的数量加1
        self.num_positions = self.num_patches + 1
        # 创建一个位置嵌入，大小为(num_positions, embed_dim)，用于嵌入位置信息
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 注册一个缓冲区，存储位置ID的张量，形状为(1, num_positions)，不持久保存
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播函数，接收像素值张量作为输入，返回嵌入的张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入张量的批量大小
        batch_size = pixel_values.shape[0]
        # 设置目标数据类型为patch_embedding权重的数据类型
        target_dtype = self.patch_embedding.weight.dtype
        # 对输入像素值进行patch嵌入，形状为[*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        # 将patch_embeds扁平化，并转置维度1和2
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 将类别嵌入扩展到与批量大小匹配的维度
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 拼接类别嵌入和patch嵌入，维度为[batch_size, num_patches + 1, embed_dim]
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 将位置嵌入加到嵌入张量上，形状为[batch_size, num_patches + 1, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # 返回嵌入张量
        return embeddings


# 从 transformers.models.clip.modeling_clip.CLIPMLP 复制而来的 GitVisionMLP 类定义
class GitVisionMLP(nn.Module):
    # 初始化函数，接收一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 将 config 参数保存到对象的 config 属性中
        self.config = config
        # 设置激活函数为配置中指定的隐藏层激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 创建一个全连接层，输入大小为隐藏大小，输出大小为中间大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建一个全连接层，输入大小为中间大小，输出大小为隐藏大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # 前向传播函数，接收隐藏状态张量作为输入，返回变换后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过第一个全连接层和激活函数的变换
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        # 经过第二个全连接层的变换
        hidden_states = self.fc2(hidden_states)
        # 返回变换后的张量
        return hidden_states


# 从 transformers.models.clip.modeling_clip.CLIPAttention 复制而来的 GitVisionAttention 类定义
class GitVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存为属性
        self.config = config
        # 设置嵌入维度为配置中的隐藏大小
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量为配置中的注意力头数
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查嵌入维度是否能整除注意力头数，若不能则抛出异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_heads})."
            )
        # 计算缩放因子，用于注意力计算
        self.scale = self.head_dim**-0.5
        # 设置注意力中的 dropout 率
        self.dropout = config.attention_dropout

        # 初始化线性变换层，用于查询、键、值和输出的投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 定义一个函数用于重塑张量形状，用于多头注意力
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，接受隐藏状态张量和可选的注意力掩码作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->GitVision
class GitVisionEncoderLayer(nn.Module):
    def __init__(self, config: GitVisionConfig):
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 使用 GitVisionAttention 类创建自注意力机制对象
        self.self_attn = GitVisionAttention(config)
        # 第一层归一化，使用 LayerNorm
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # MLP 层，使用 GitVisionMLP 创建多层感知机
        self.mlp = GitVisionMLP(config)
        # 第二层归一化，使用 LayerNorm
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 注意力掩码张量，形状为
                `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值指示。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions`。

        Returns:
            outputs: 包含处理结果的元组，其中元素为 `torch.FloatTensor` 张量
        """
        # 保存残差连接
        residual = hidden_states

        # 应用第一层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力机制计算注意力权重和新的隐藏状态
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 添加残差连接到新的隐藏状态
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 应用第二层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 应用 MLP 层
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接到最终输出
        hidden_states = residual + hidden_states

        # 构建输出元组，包含最终隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->GitVision, CLIPConfig
class GitVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`GitVisionEncoderLayer`].

    Args:
        config: GitVisionConfig
    """

    def __init__(self, config: GitVisionConfig):
        super().__init__()
        # 保存配置对象
        self.config = config
        # 使用 GitVisionEncoderLayer 创建多个编码层，并组成层列表
        self.layers = nn.ModuleList([GitVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点默认关闭
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Args:
            inputs_embeds: 输入的嵌入张量
            attention_mask: 注意力掩码张量
            causal_attention_mask: 因果注意力掩码张量
            output_attentions: 是否返回注意力张量
            output_hidden_states: 是否返回隐藏状态张量
            return_dict: 是否以字典形式返回结果

        Returns:
            depending on `return_dict`, a tuple of shape `(last_hidden_state, (attentions))` where each
            element is a tensor
        """
        # 留空，等待实现具体的前向传播逻辑
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values input to the model. This tensor represents the images to be processed, organized as batches
            with specified channels, height, and width.
            Padding in the input will be ignored by default.
            Pixel values can be obtained using `AutoImageProcessor`. See `CLIPImageProcessor.__call__` for more details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attention tensors from all attention layers. If set to `True`, the returned
            tensors will include the attentions for each layer.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states from all layers. If set to `True`, the returned tensors will
            include the hidden states for each layer.

        return_dict (`bool`, *optional*):
            Whether or not to return a `utils.ModelOutput` object instead of a plain tuple. If `True`, the returned
            output will be a structured object containing various model outputs such as logits, hidden states,
            attentions, etc.
"""
@add_start_docstrings_to_model_forward(GIT_VISION_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutput, config_class=GitVisionConfig)
# 覆盖了 forward 方法的 docstring，指定了输入和输出的详细描述

class GitVisionTransformer(nn.Module):
    # 从 transformers.models.clip.modeling_clip.CLIPVisionTransformer.__init__ 复制而来，将 CLIPEncoder 改为 GitVisionEncoder，CLIP 改为 Git
    # 初始化 GitVisionTransformer 类
    def __init__(self, config: GitVisionConfig):
        super().__init__()
        # 设置配置信息
        self.config = config
        # 设定嵌入维度为隐藏大小
        embed_dim = config.hidden_size

        # 初始化嵌入层
        self.embeddings = GitVisionEmbeddings(config)
        # 初始化前层归一化层
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化编码器
        self.encoder = GitVisionEncoder(config)
        # 初始化后层归一化层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 覆盖了 forward 方法的 docstring，指定了返回值的详细描述
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Returns:
            Either a tuple or a BaseModelOutput depending on `return_dict`.
        """
        # 如果 output_attentions 未指定，则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 未指定，则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 未指定，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 对输入的像素值进行嵌入处理
        hidden_states = self.embeddings(pixel_values)
        # 应用前层归一化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 将处理后的隐藏状态传入编码器进行处理
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 应用后层归一化
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 如果 return_dict 为 False，则返回一个包含最后隐藏状态和其他编码器输出的元组
        if not return_dict:
            return (last_hidden_state,) + encoder_outputs[1:]

        # 否则，返回一个包含所有输出的 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """The vision model from CLIP, used in GIT, without any head or projection on top.""",
    GIT_START_DOCSTRING,
)
# 覆盖了类 GitVisionModel 的 docstring，提供了关于这个视觉模型的描述以及 GIT 的起始文档字符串
class GitVisionModel(GitPreTrainedModel):
    # 指定配置类为 GitVisionConfig
    config_class = GitVisionConfig
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 从 transformers.models.clip.modeling_clip.CLIPVisionModel.__init__ 复制而来，将 CLIP 改为 Git
    # 初始化 GitVisionModel 类
    def __init__(self, config: GitVisionConfig):
        super().__init__(config)
        # 初始化视觉模型
        self.vision_model = GitVisionTransformer(config)
        # 执行初始化权重和应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding
    @add_start_docstrings_to_model_forward(GIT_VISION_INPUTS_DOCSTRING)
    # 调用装饰器，添加模型前向传播方法的文档字符串，使用给定的输入文档字符串GIT_VISION_INPUTS_DOCSTRING
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=GitVisionConfig)
    # 调用装饰器，替换返回值的文档字符串，指定输出类型为BaseModelOutput，配置类为GitVisionConfig

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        模型的前向传播方法。

        Returns:
            返回值是一个元组或BaseModelOutput对象。

        Examples:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict不为None，则使用它；否则使用self.config.use_return_dict的值

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用self.vision_model进行模型的前向传播，传入相关参数和返回值设置
# 定义一个 GitModel 类，继承自 GitPreTrainedModel，表示一个 GIT 模型
@add_start_docstrings(
    "The bare GIT Model transformer consisting of a CLIP image encoder and text decoder outputting raw hidden-states"
    " without any specific head on top.",
    GIT_START_DOCSTRING,
)
class GitModel(GitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 初始化 GitModel 对象时执行以下操作：

        # 实例化 GitEmbeddings 类，用于处理模型的嵌入层
        self.embeddings = GitEmbeddings(config)

        # 实例化 GitVisionModel 类，用于处理视觉编码器部分
        self.image_encoder = GitVisionModel(config.vision_config)

        # 实例化 GitEncoder 类，用于处理文本编码器部分
        self.encoder = GitEncoder(config)

        # 实例化 GitProjection 类，用于定义视觉投影层
        self.visual_projection = GitProjection(config)

        # 如果配置中指定了 num_image_with_embedding，创建对应数量的图像嵌入参数列表
        if config.num_image_with_embedding is not None:
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, config.vision_config.hidden_size))
                for _ in range(config.num_image_with_embedding)
            )

        # 调用 post_init 方法，用于初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型的输入嵌入层对象
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置模型的输入嵌入层对象
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历待修剪的头信息，并在相应层中执行修剪操作
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(self, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # 生成一个未来的遮罩，用于自注意力机制
        # 默认遮罩适用于正向方向，如果需要反向遮罩，需要将其翻转
        mask = torch.triu(torch.ones(size, size, device=device, dtype=dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))  # 将遮罩中的所有 1 替换为负无穷
        return mask
    # 创建注意力遮罩，用于Transformer模型的self-attention机制，生成一个三维张量
    def create_attention_mask(self, tgt, memory, tgt_mask, past_key_values_length, memory_key_padding_mask=None):
        # 获取目标（target）序列的长度
        num_tgt = tgt.shape[1]
        # 获取记忆（memory）序列的长度
        num_memory = memory.shape[1]
        # 获取目标（target）张量所在设备
        device = tgt.device
        # 获取目标（target）张量的数据类型
        dtype = tgt.dtype

        # 创建左上角的全零张量
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        # 创建右上角的全负无穷张量，用于填充注意力矩阵的右上部分
        top_right = torch.full(
            (num_memory, num_tgt + past_key_values_length),
            float("-inf"),
            device=tgt.device,
            dtype=dtype,
        )
        # 创建左下角的全零张量，用于填充注意力矩阵的左下部分
        bottom_left = torch.zeros(
            (num_tgt, num_memory),
            dtype=dtype,
            device=tgt_mask.device,
        )

        # 如果存在过去的键值长度大于零，则需要重新定义目标掩码
        if past_key_values_length > 0:
            tgt_mask = torch.zeros(
                (tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length),
                dtype=dtype,
                device=tgt_mask.device,
            )

        # 将左上角、左下角组合成左侧部分张量
        left = torch.cat((top_left, bottom_left), dim=0)
        # 将右上角和目标掩码组合成右侧部分张量
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        # 将左侧部分张量和右侧部分张量连接起来，形成完整的注意力矩阵
        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        # 如果记忆序列的键值填充掩码为None，则创建全假掩码张量
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # 如果记忆序列的键值填充掩码不是布尔类型，则抛出值错误异常
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError("Memory key padding mask must be a boolean tensor.")
        # 创建与记忆序列形状相同的全零张量
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        # 将填充的位置替换为负无穷
        zero_negative_infinity[memory_key_padding_mask] = float("-inf")
        # 将完整的注意力矩阵张量扩展为指定形状
        full_attention_mask = full_attention_mask.expand(
            (memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + past_key_values_length + num_tgt)
        )
        # 克隆扩展后的完整注意力矩阵张量
        full_attention_mask = full_attention_mask.clone()
        # 获取注意力矩阵的左侧原始部分
        origin_left = full_attention_mask[:, :, :num_memory]
        # 执行更新操作，将负无穷的张量加到原始左侧部分
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        # 为多头注意力添加额外的维度
        full_attention_mask = full_attention_mask[:, None, :, :]

        # 返回完整的注意力矩阵
        return full_attention_mask

    # 用于Transformer模型的前向传播，包含各种输入参数和输出控制标志
    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
GIT Model with a `language modeling` head on top for autoregressive language modeling.
"""
@add_start_docstrings(
    GIT_START_DOCSTRING
)
class GitForCausalLM(GitPreTrainedModel):
    # List of keys whose weights are tied
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        """
        Initializes the GitForCausalLM model.

        Args:
            config (:class:`~transformers.GitConfig`):
                The configuration object that defines the model architecture.
        """
        super().__init__(config)

        # Initialize the base GitModel with the provided configuration
        self.git = GitModel(config)
        # Linear layer for output
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output layer.

        Returns:
            :obj:`torch.nn.Linear`: The output layer of the model.
        """
        return self.output

    def set_output_embeddings(self, new_embeddings):
        """
        Sets new output embeddings.

        Args:
            new_embeddings (:obj:`torch.nn.Linear`):
                New embeddings to set for the output layer.
        """
        self.output = new_embeddings

    @add_start_docstrings_to_model_forward(GIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the GitForCausalLM model.

        Args:
            input_ids (:obj:`torch.Tensor`, optional):
                Input tensor of token indices.
            attention_mask (:obj:`torch.Tensor`, optional):
                Mask to avoid performing attention on padding tokens.
            position_ids (:obj:`torch.Tensor`, optional):
                Indices of positions of each input sequence tokens in the position embeddings.
            pixel_values (:obj:`torch.Tensor`, optional):
                Pixel values if the model is a vision model.
            head_mask (:obj:`torch.Tensor`, optional):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (:obj:`torch.Tensor`, optional):
                Optional tensor to override the input embeddings.
            labels (:obj:`torch.Tensor`, optional):
                Labels for computing the cross entropy classification loss.
            past_key_values (:obj:`List[torch.Tensor]`, optional):
                List of tensors containing cached keys and values.
            use_cache (:obj:`bool`, optional):
                Whether to use the cache for faster decoding.
            output_attentions (:obj:`bool`, optional):
                Whether to output the attentions weights.
            output_hidden_states (:obj:`bool`, optional):
                Whether to output the hidden states.
            return_dict (:obj:`bool`, optional):
                Whether to return a dictionary as the output instead of a tuple.

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutputWithPast`:
                The output of the model, potentially with past states.
        """
        # Actual forward logic implementation is within the transformers library

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        """
        Prepares input for generation.

        Args:
            input_ids (:obj:`torch.Tensor`):
                Input tensor of token indices.
            past_key_values (:obj:`List[torch.Tensor]`, optional):
                List of tensors containing cached keys and values.
            attention_mask (:obj:`torch.Tensor`, optional):
                Mask to avoid performing attention on padding tokens.
            use_cache (:obj:`bool`, optional):
                Whether to use the cache for faster decoding.
            **kwargs:
                Additional keyword arguments.

        Returns:
            :obj:`Dict[str, torch.Tensor]`:
                Dictionary containing the prepared inputs for generation.
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": kwargs.get("pixel_values", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cached states for beam search.

        Args:
            past_key_values (:obj:`List[torch.Tensor]`):
                List of tensors containing cached keys and values.
            beam_idx (:obj:`torch.Tensor`):
                Indices of beams to reorder the past states.

        Returns:
            Tuple[List[torch.Tensor]]:
                Reordered past states for beam search.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```