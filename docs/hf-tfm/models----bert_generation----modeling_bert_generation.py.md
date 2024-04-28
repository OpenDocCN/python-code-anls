# `.\transformers\models\bert_generation\modeling_bert_generation.py`

```
# 设定文件编码为 UTF-8
# 版权声明
# 版权所有 2020 年 Google AI 语言团队作者和 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版（“许可证”）许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不提供任何明示或暗示的保证或条件。
# 请参阅许可证了解特定语言的管理权限和限制。
"""用于生成任务的 PyTorch BERT 模型。"""

# 导入必要的库
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义的激活函数映射和模型输出
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入生成任务的配置类
from .configuration_bert_generation import BertGenerationConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "google/bert_for_seq_generation_L-24_bbc_encoder"
_CONFIG_FOR_DOC = "BertGenerationConfig"


# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制而来，将 Bert->BertGeneration
class BertGenerationSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建全连接层，用于调整隐藏状态的尺寸
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建层归一化层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，用于随机丢弃部分神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态传入全连接层
        hidden_states = self.dense(hidden_states)
        # 在全连接层输出上应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 将全连接层输出与输入张量相加，并通过层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制而来，将 Bert->BertGeneration
class BertGenerationSelfAttention(nn.Module):
    # 初始化方法，用于初始化一个多头注意力层
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除且配置中没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
    
        # 设置多头注意力的头数
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
    
        # 创建查询、键、值的线性映射层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
    
        # 创建用于丢弃的Dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对位置编码，则创建相应的距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
    
        # 设置是否为解码器
        self.is_decoder = config.is_decoder
    
    # 将张量变换为适合计算注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新调整张量形状
        x = x.view(new_x_shape)
        # 将维度重新排列以适应注意力计算所需的顺序
        return x.permute(0, 2, 1, 3)
    
    # 前向传播方法，用于计算多头注意力
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从 transformers.models.bert.modeling_bert.BertAttention 复制并修改为 BertGenerationAttention 类
class BertGenerationAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 BertGenerationSelfAttention 和 BertGenerationSelfOutput 类
        self.self = BertGenerationSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertGenerationSelfOutput(config)
        self.pruned_heads = set()  # 存储被剪枝的注意力头的集合

    # 剪枝指定的注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
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
        # 调用 BertGenerationSelfAttention 的前向传播函数
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 BertGenerationSelfOutput 的前向传播函数
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力，则添加到输出中
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 BertGenerationIntermediate 类
class BertGenerationIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层和激活函数
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性层前向传播
        hidden_states = self.dense(hidden_states)
        # 激活函数前向传播
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 BertGenerationOutput 类
class BertGenerationOutput(nn.Module):
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为中间大小，输出大小为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于对隐藏状态进行归一化，设置 epsilon 为 config 中的值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，用于在训练时随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，用于计算模型的输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行随机丢弃一部分神经元
        hidden_states = self.dropout(hidden_states)
        # 对丢弃后的隐藏状态进行 LayerNorm 归一化，并将输入张量加到其中
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为模型的输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制代码，并将Bert->BertGeneration
class BertGenerationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度为1
        self.seq_len_dim = 1
        # 创建BertGenerationAttention对象
        self.attention = BertGenerationAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建BertGenerationAttention对象，使用绝对位置嵌入
            self.crossattention = BertGenerationAttention(config, position_embedding_type="absolute")
        # 创建BertGenerationIntermediate对象
        self.intermediate = BertGenerationIntermediate(config)
        # 创建BertGenerationOutput对象
        self.output = BertGenerationOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键/值不为空，则将decoder单向自注意力的缓存键/值元组放在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self.attention进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
          
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组在过去键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用crossattention进行交叉注意力计算
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
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到现在的键/值元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块技术对前向传播进行处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 使用中间层进行前向传播
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层进行前向传播
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制代码，修改为 Bert->BertGeneration
class BertEncoder(nn.Module):
    def __init__(self, config):
        # 初始化方法，接受一个配置对象
        super().__init__()
        # 将配置对象存储到实例属性中
        self.config = config
        # 创建一个由多个 BertGenerationLayer 组成的层列表，层数由配置中的隐藏层数确定
        self.layer = nn.ModuleList([BertGenerationLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认情况下关闭梯度检查点
        self.gradient_checkpointing = False

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
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，则初始化一个空元组，否则设置为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组，否则设置为None
        all_self_attentions = () if output_attentions else None
        # 如果输出交叉注意力权重且模型配置中包含交叉注意力，则初始化一个空元组，否则设置为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点并且处于训练模式下，则检查缓存的使用
        if self.gradient_checkpointing and self.training:
            # 如果use_cache为True，则发出警告并设置为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果use_cache为True，则初始化下一个解码器缓存为空元组，否则设置为None
        next_decoder_cache = () if use_cache else None
        # 遍历解码器的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果有头部掩码，则获取当前层的头部掩码，否则设置为None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果有过去的键值，则获取当前层的过去键值，否则设置为None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且处于训练模式下，则使用梯度检查点函数计算当前层的输出
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
            # 否则，直接调用当前层的前向传播计算当前层的输出
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

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果use_cache为True，则将当前层的输出的最后一个元素添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的自注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中包含交叉注意力，则将当前层的交叉注意力权重添加到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回包含隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重和所有交叉注意力权重的元组
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
        # 否则，返回一个包含最终隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重和所有交叉注意力权重的字典
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 导入所需的库
def load_tf_weights_in_bert_generation(
    model, tf_hub_path, model_class, is_encoder_named_decoder=False, is_encoder=False
):
    # 尝试导入 TensorFlow 相关库
    try:
        import numpy as np
        import tensorflow.compat.v1 as tf
        import tensorflow_hub as hub
        import tensorflow_text  # noqa: F401

        # 禁用 TensorFlow 的即时执行模式
        tf.disable_eager_execution()
    # 如果导入失败，输出错误信息并抛出异常
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 从 TensorFlow Hub 加载模型
    tf_model = hub.Module(tf_hub_path)
    # 初始化 TensorFlow 全局变量
    init = tf.global_variables_initializer()

# BertGenerationEmbeddings 类，用于构建词嵌入和位置嵌入
class BertGenerationEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 创建词嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建 LayerNorm 层
        # self.LayerNorm 不采用蛇形命名以保持与 TensorFlow 模型变量名称的一致性，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在序列化时是内存连续的，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播函数
    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        # 如果传入的是 input_ids，则获取其形状；否则，获取 inputs_embeds 的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 如果未提供 position_ids，则根据序列长度和过去键值长度创建新的 position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供 inputs_embeds，则通过 word_embeddings 层获取
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 将词嵌入和位置嵌入相加
        embeddings = inputs_embeds + position_embeddings
        # 经过 LayerNorm 层
        embeddings = self.LayerNorm(embeddings)
        # 经过 Dropout 层
        embeddings = self.dropout(embeddings)
        # 返回嵌入结果
        return embeddings


# BertGenerationPreTrainedModel 类，用于处理权重初始化、下载和加载预训练模型
class BertGenerationPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为 BertGenerationConfig
    config_class = BertGenerationConfig
    # 基础模型前缀为 "bert"
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
# BERT_GENERATION_START_DOCSTRING 是一个原始字符串，包含了关于 BERT 生成模型的文档字符串
BERT_GENERATION_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertGenerationConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BERT_GENERATION_INPUTS_DOCSTRING 是一个原始字符串，包含了关于 BERT 生成模型输入的文档字符串
BERT_GENERATION_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.__call__`] 和 [`PreTrainedTokenizer.encode`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置零的掩码。掩码值在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的头部，
            # - 0 表示**被掩码**的头部。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多控制如何将 `input_ids` 索引转换为相关向量，这很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
# 使用特定文档字符串初始化 BertGenerationEncoder 类
@add_start_docstrings(
    "The bare BertGeneration model transformer outputting raw hidden-states without any specific head on top.",  # 初始化 BertGenerationEncoder 类的描述
    BERT_GENERATION_START_DOCSTRING,  # 引用 BertGenerationStart 的文档字符串
)
class BertGenerationEncoder(BertGenerationPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    This model should be used when leveraging Bert or Roberta checkpoints for the [`EncoderDecoderModel`] class as
    described in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
    by Sascha Rothe, Shashi Narayan, and Aliaksei Severyn.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    # 初始化 BertGenerationEncoder 类
    def __init__(self, config):
        super().__init__(config)  # 调用父类 BertGenerationPreTrainedModel 的初始化方法
        self.config = config  # 配置初始化

        # 初始化嵌入层和编码器
        self.embeddings = BertGenerationEmbeddings(config)  # BertGenerationEmbeddings 类的实例化
        self.encoder = BertEncoder(config)  # BertEncoder 类的实例化

        # 初始化权重并应用最终处理
        self.post_init()  # 调用自身的 post_init 方法

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 获取输入嵌入

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入嵌入

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():  # 遍历需要修剪的头部
            self.encoder.layer[layer].attention.prune_heads(heads)  # 修剪对应的头部

    @add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.format("batch_size, sequence_length"))  # 将特定文档字符串添加到模型前向方法
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 样例代码的检查点
        output_type=BaseModelOutputWithPastAndCrossAttentions,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 配置类
    )
    # 实现模型的前向传播
    def forward(
        self,
        # 输入的标识符（token IDs），可选的张量，默认为 None
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，可选的张量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 位置标识符，可选的张量，默认为 None
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码，可选的张量，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 输入的嵌入式张量，可选的张量，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器隐藏状态，可选的张量，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器注意力掩码，可选的张量，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 过去的键值对，可选的张量的元组，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 是否使用缓存，可选的布尔值，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力，可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,
class BertGenerationOnlyLMHead(nn.Module):
    # 定义一个类，继承自 nn.Module，用于生成 BERT 模型的仅有语言模型头部的部分
    def __init__(self, config):
        # 初始化方法
        super().__init__()
        # 创建一个线性层，用于从隐藏状态映射到词汇表大小的输出
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 创建一个参数，用于偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将线性层的偏置设置为上述参数
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 前向传播方法，将隐藏状态映射为词汇表大小的输出
        logits = self.decoder(hidden_states)
        return logits

    def _tie_weights(self):
        # 如果权重断开连接（在TPU上或偏置被调整大小时），则将这两个权重连接起来
        self.bias = self.decoder.bias


@add_start_docstrings(
    """BertGeneration Model with a `language modeling` head on top for CLM fine-tuning.""",
    BERT_GENERATION_START_DOCSTRING,
)
class BertGenerationDecoder(BertGenerationPreTrainedModel):
    # BertGenerationDecoder 类，用于在 CLM 微调时在顶部添加一个语言建模头部
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 初始化方法
        super().__init__(config)

        if not config.is_decoder:
            # 如果不是解码器，则发出警告
            logger.warning("If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`")

        # 创建一个 BertGenerationEncoder 对象
        self.bert = BertGenerationEncoder(config)
        # 创建一个 BertGenerationOnlyLMHead 对象
        self.lm_head = BertGenerationOnlyLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 获取输出嵌入层
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置输出嵌入层
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_GENERATION_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成准备输入，在生成序列时的输入处理函数
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入的形状
        input_shape = input_ids.shape
        # 如果没有给出注意力掩码，则创建一个全为1的注意力掩码，与输入形状相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果传入了过去的键值，表示模型用作编码器-解码器中的解码器，动态创建解码器的注意力掩码
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入的长度大于过去键值的长度，则需要截断输入
            if input_ids.shape[1] > past_length:
                # 截断长度为过去键值的长度
                remove_prefix_length = past_length
            else:
                # 否则，只保留最后一个输入的 ID
                # 默认保留行为：只保留最后一个输入的 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 对输入进行截断，仅保留需要用于生成的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回输入字典，包括输入的 ID、注意力掩码和过去的键值
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排列缓存函数
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排列后的过去键值元组
        reordered_past = ()
        # 对每一层的过去键值进行重新排列
        for layer_past in past_key_values:
            # 对于每一层的过去状态，根据 beam_idx 重新排列
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排列后的过去键值元组
        return reordered_past
```