# `.\models\roberta\modeling_roberta.py`

```
# coding=utf-8
# 定义编码方式为 UTF-8，确保支持中文等多种字符集
# 版权声明，包括版权归属信息和许可协议
# 此部分代码的版权归 Google AI Language Team 和 HuggingFace Inc. 团队所有
# 版权归 NVIDIA CORPORATION 所有，保留所有权利
#
# 根据 Apache 许可协议版本 2.0 使用本文件
# 除非法律要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可协议的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 本代码基于 "按原样提供" 基础分发，不附带任何明示或暗示的保证或条件
# 查看许可协议了解具体条款和条件
"""PyTorch RoBERTa model."""
# 引入数学库，用于数学运算
import math
# 引入类型注解，用于声明变量、函数参数和返回值的类型
from typing import List, Optional, Tuple, Union
# 引入 PyTorch 框架
import torch
# 引入 PyTorch 的模块
import torch.utils.checkpoint
# 引入 PyTorch 的神经网络模块
from torch import nn
# 引入 PyTorch 的损失函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 引入激活函数及相关函数
from ...activations import ACT2FN, gelu
# 引入模型输出类
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
# 引入模型工具类
from ...modeling_utils import PreTrainedModel
# 引入 PyTorch 工具函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 引入常用工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入 RoBERTa 的配置类
from .configuration_roberta import RobertaConfig

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "FacebookAI/roberta-base"
# 用于文档的配置信息
_CONFIG_FOR_DOC = "RobertaConfig"

# RoBERTa 预训练模型的存档列表
ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "FacebookAI/roberta-large-mnli",
    "distilbert/distilroberta-base",
    "openai-community/roberta-base-openai-detector",
    "openai-community/roberta-large-openai-detector",
    # 查看所有 RoBERTa 模型：https://huggingface.co/models?filter=roberta
]

# RoBERTaEmbeddings 类，继承自 nn.Module，用于定义 RoBERTa 的嵌入层
class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 与 BertEmbeddings 相同，稍作调整以支持位置嵌入的索引
    # 初始化函数，用于初始化模型参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化词嵌入层，vocab_size表示词汇表大小，hidden_size表示隐藏层大小，padding_idx指定填充的token ID
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，max_position_embeddings表示最大位置编码数量，hidden_size表示隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化token类型嵌入层，type_vocab_size表示token类型的数量，hidden_size表示隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm命名不使用蛇形命名法以保持与TensorFlow模型变量名的一致性，并能够加载任何TensorFlow检查点文件
        # 初始化LayerNorm层，hidden_size表示层的大小，eps为LayerNorm层的epsilon值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout层，hidden_dropout_prob表示隐藏层的dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_embedding_type表示位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册position_ids张量为缓冲区，表示位置编码，形状为(1, max_position_embeddings)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册token_type_ids张量为缓冲区，表示token类型编码，形状与position_ids相同，类型为长整型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置padding_idx，用于嵌入层中的填充
        self.padding_idx = config.pad_token_id
        # 重新初始化位置嵌入层，使用与之前不同的方式，padding_idx指定填充的token ID
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        ):
            # 如果未提供位置id，则根据输入token id创建位置id，保留任何填充的token的填充状态。
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果提供了input_ids，则获取其形状；否则，获取inputs_embeds的形状但不包括最后一维。
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            # 将token_type_ids设置为构造函数中注册的缓冲区，通常为全零。这在模型跟踪时有帮助，而不需要传递token_type_ids，解决问题＃5664。
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # 如果未提供inputs_embeds，则使用input_ids获取词嵌入。
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            embeddings = inputs_embeds + token_type_embeddings
            # 如果使用绝对位置嵌入，则添加位置嵌入。
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

        # 从inputs_embeds直接生成位置id。无法推断哪些是填充的，因此仅生成顺序位置id。
        def create_position_ids_from_inputs_embeds(self, inputs_embeds):
            """
            We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

            Args:
                inputs_embeds: torch.Tensor

            Returns: torch.Tensor
            """
            input_shape = inputs_embeds.size()[:-1]
            sequence_length = input_shape[1]

            position_ids = torch.arange(
                self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
            )
            return position_ids.unsqueeze(0).expand(input_shape)
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制并将 Bert 替换为 Roberta
class RobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏大小不是注意力头数的整数倍且配置中没有嵌入大小属性，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 重新排列张量形状以准备进行注意力得分计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,



# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并将 Bert 替换为 Roberta
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集层：输入和输出大小都为隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 密集层前向传播
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # 层归一化并添加输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 定义 RobertaAttention 类，继承自 nn.Module
class RobertaAttention(nn.Module):
    # 初始化方法
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建 RobertaSelfAttention 对象，并传入 config 和 position_embedding_type 参数
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 RobertaSelfOutput 对象，传入 config 参数
        self.output = RobertaSelfOutput(config)
        # 初始化一个空集合，用于存储要剪枝的注意力头的索引
        self.pruned_heads = set()

    # 剪枝注意力头的方法
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数获取可剪枝的头部索引及其所在的层级索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部索引
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
        # 调用 self 对象的 forward 方法进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力输出传入 self.output 对象，得到最终的注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出 attentions，则将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，添加 attentions
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制得到的类
class RobertaIntermediate(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将隐藏状态的尺寸转换为中间尺寸
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则使用对应的激活函数，否则使用 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行尺寸转换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制得到的类
class RobertaOutput(nn.Module):
    # 在这里没有提供该类的完整定义和方法，所以这里不添加具体注释
    pass
    # 初始化函数，用于创建一个新的神经网络层
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入大小为config中的intermediate_size，输出大小为config中的hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，对输入进行归一化，归一化的维度为config中的hidden_size，eps为归一化过程中的小数值防止除零错误
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，以config中的hidden_dropout_prob的概率对输入进行随机置零，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了数据从输入到输出的流动方式
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换，将输入hidden_states映射到新的空间
        hidden_states = self.dense(hidden_states)
        # 对变换后的结果进行随机置零，防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将置零后的结果与输入张量input_tensor相加，并进行LayerNorm归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的结果张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制代码，并将Bert->Roberta
class RobertaLayer(nn.Module):
    # 初始化函数，用于设置层的参数和子模块
    def __init__(self, config):
        super().__init__()
        # 设置前馈传播的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度的索引
        self.seq_len_dim = 1
        # 初始化自注意力机制
        self.attention = RobertaAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨注意力，且不是解码器，则引发错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨注意力机制，使用绝对位置编码
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = RobertaIntermediate(config)
        # 初始化输出层
        self.output = RobertaOutput(config)

    # 前向传播函数，定义层的计算逻辑
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
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用 self_attn_past_key_value 来缓存解码器自注意力机制的键/值对，位置在 1 和 2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后的输出是自注意力缓存的元组
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

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用 cross_attn_past_key_value 来缓存跨注意力机制的键/值对，位置在 past_key_value 元组的 3 和 4
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
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加跨注意力

            # 将跨注意力的缓存添加到 present_key_value 元组的位置 3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制并替换为使用 Roberta 模型的编码器类
class RobertaEncoder(nn.Module):
    # 初始化方法，接受一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 将配置参数保存到实例中
        self.config = config
        # 创建一个由多个 RobertaLayer 组成的层列表，层数由配置参数中的 num_hidden_layers 决定
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点（gradient checkpointing）默认关闭
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个输入参数，并返回输出结果
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
        all_hidden_states = () if output_hidden_states else None
        # 初始化一个空元组用于存储所有隐藏状态，如果不需要输出隐藏状态则置为None
        all_self_attentions = () if output_attentions else None
        # 初始化一个空元组用于存储所有自注意力权重，如果不需要输出注意力权重则置为None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 初始化一个空元组用于存储所有跨注意力权重，如果不需要输出跨注意力权重或模型未配置跨注意力则置为None

        if self.gradient_checkpointing and self.training:
            # 如果启用了梯度检查点且处于训练模式
            if use_cache:
                # 如果设置了使用缓存，则发出警告并禁用缓存
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        # 如果使用缓存，则初始化一个空元组用于存储下一个解码器缓存，否则置为None
        for i, layer_module in enumerate(self.layer):
            # 遍历解码器层列表
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的头部掩码，如果未提供头部掩码则置为None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            # 获取当前层的过去键值对，如果未提供则置为None

            if self.gradient_checkpointing and self.training:
                # 如果启用了梯度检查点且处于训练模式
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
                # 否则直接调用解码器层模块计算输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            # 更新当前隐藏状态为当前层的输出的第一个元素（通常是隐藏状态）
            if use_cache:
                # 如果使用缓存，则将当前层的缓存输出添加到下一个解码器缓存元组中
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                # 如果需要输出注意力权重
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 将当前层的自注意力权重输出添加到所有自注意力权重元组中
                if self.config.add_cross_attention:
                    # 如果模型配置中包括跨注意力机制
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
                    # 将当前层的跨注意力权重输出添加到所有跨注意力权重元组中

        if output_hidden_states:
            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到所有隐藏状态元组中
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不要求返回字典形式的结果
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
            # 返回包含所有结果元组的元组，过滤掉为None的值
        return BaseModelOutputWithPastAndCrossAttentions(
            # 否则返回一个包含所有结果的字典对象
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler中复制而来的类，用于RoBERTa模型的池化层
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Tanh激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过仅仅使用第一个token对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个token的隐藏状态通过全连接层dense进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 应用Tanh激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


class RobertaPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化以及一个简单的接口，用于下载和加载预训练模型。
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RobertaEmbeddings", "RobertaSelfAttention"]

    # 从transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights中复制而来的函数
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 和TF版本稍有不同，这里使用正态分布初始化权重
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm层的偏置初始化为0
            module.bias.data.zero_()
            # LayerNorm层的权重初始化为1
            module.weight.data.fill_(1.0)


ROBERTA_START_DOCSTRING = r"""
    此模型继承自[`PreTrainedModel`]。请查看其超类文档以了解库实现的所有模型的通用方法（例如下载或保存模型、调整输入嵌入、修剪头等）。

    此模型也是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。
    您可以像使用常规PyTorch模块一样使用它，并参考PyTorch文档以获取有关一般使用和行为的所有相关信息。

    参数:
        config ([`RobertaConfig`]): 包含模型所有参数的配置类。使用配置文件初始化模型不会加载模型的权重，只会加载配置。
        请查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    ```
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记的索引，对应于词汇表中的标记

                # 可以使用 [`AutoTokenizer`] 获取这些索引。参见 [`PreTrainedTokenizer.encode`] 和
                # [`PreTrainedTokenizer.__call__`] 获取更多详情。

                # [什么是输入 ID？](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 遮罩，用于在填充的标记索引上避免执行注意力计算。遮罩的值选在 `[0, 1]`：

                # - 对于 **未遮罩的** 标记，值为 1，
                # - 对于 **遮罩的** 标记，值为 0。

                # [什么是注意力遮罩？](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 段标记索引，指示输入的第一部分和第二部分。索引值选在 `[0,1]`：

                # - 0 对应 *句子 A* 的标记，
                # - 1 对应 *句子 B* 的标记。
                # 当模型用 `type_vocab_size` 参数初始化且值 >= 2 时才能使用此参数。此张量中的所有值应始终 < type_vocab_size。

                # [什么是标记类型 ID？](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]`。

                # [什么是位置 ID？](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 用于屏蔽自注意力模块中选定头部的遮罩。遮罩的值选在 `[0, 1]`：

                # - 1 表示头部 **未被屏蔽**，
                # - 0 表示头部 **被屏蔽**。

            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选地，您可以直接传递嵌入表示而不是传递 `input_ids`。这在您想要更精确地控制如何将 `input_ids` 索引转换为相关向量时很有用，而不是使用模型内部的嵌入查找矩阵。

            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。

            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量中的 `hidden_states`。

            return_dict (`bool`, *optional*):
                # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
    """
    # 使用指定的文档字符串作为 RoBERTa 模型的描述
    @add_start_docstrings(
        "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
        ROBERTA_START_DOCSTRING,
    )
    """
    """
    # RoBERTaModel 类的定义，继承自 RoBERTaPreTrainedModel
    class RobertaModel(RobertaPreTrainedModel):
        """

        RoBERTa 模型可以作为编码器（只有自注意力）或解码器使用，后者在自注意力层之间增加了一个交叉注意力层，
        遵循 *Attention is all you need*_ 中描述的架构，由 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、
        Llion Jones、Aidan N. Gomez、Lukasz Kaiser 和 Illia Polosukhin 提出。

        若要作为解码器使用，模型需要使用配置中设置 `is_decoder` 参数为 `True` 进行初始化。
        若要在 Seq2Seq 模型中使用，模型需要同时设置 `is_decoder` 参数和 `add_cross_attention` 参数为 `True`，
        并期望在前向传播中输入 `encoder_hidden_states`。

        .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

        """

        # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制过来，将 Bert 改为 RoBERTa
        def __init__(self, config, add_pooling_layer=True):
            super().__init__(config)
            # 初始化模型配置
            self.config = config

            # 初始化 RoBERTaEmbeddings
            self.embeddings = RobertaEmbeddings(config)
            # 初始化 RoBERTaEncoder
            self.encoder = RobertaEncoder(config)

            # 如果指定添加池化层，则初始化 RoBERTaPooler
            self.pooler = RobertaPooler(config) if add_pooling_layer else None

            # 初始化权重并应用最终处理
            self.post_init()

        # 获取输入嵌入层（词嵌入）
        def get_input_embeddings(self):
            return self.embeddings.word_embeddings

        # 设置输入嵌入层（词嵌入）
        def set_input_embeddings(self, value):
            self.embeddings.word_embeddings = value

        # 剪枝模型的注意力头
        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        # 为 RoBERTaModel 的前向传播方法添加文档字符串
        @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=BaseModelOutputWithPoolingAndCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
        # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制过来
        """
    # 定义 Transformer 模型的前向传播方法，接受多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,   # 输入的 token IDs 张量，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs 张量，可选
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs 张量，可选
        head_mask: Optional[torch.Tensor] = None,   # 头部掩码张量，可选
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入张量，可选
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态张量，可选
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码张量，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选
# 定义 RoBERTa 语言模型，用于条件语言建模 fine-tuning
@add_start_docstrings(
    """RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.""", ROBERTA_START_DOCSTRING
)
class RobertaForCausalLM(RobertaPreTrainedModel):
    # 共享权重的键名列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中指定不是解码器，则记录警告日志
        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化 RoBERTa 模型，不包含池化层
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 初始化 RoBERTa 语言建模头部
        self.lm_head = RobertaLMHead(config)

        # 执行初始化权重并应用最终处理
        self.post_init()

    # 返回语言建模头部的输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置语言建模头部的输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数并返回预测结果
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ):
        # 准备生成的输入数据，根据需要动态创建解码器的注意力遮罩
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            input_shape = input_ids.shape
            # 如果未提供注意力遮罩，则使用全为1的遮罩
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果存在过去键值，则截取解码器输入的 ID
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法可能只传递最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认行为：保留最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排序缓存中的过去键值对，以适应束搜索的索引顺序
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空的重排序后的过去键值对元组
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 对于每个层的过去状态，根据束搜索的索引重新选择对应的过去状态
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对元组
        return reordered_past
# 使用自定义的文档字符串注释 RoBERTa 模型，包含了一个顶部的语言建模头部
@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top.""", ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(RobertaPreTrainedModel):
    # 定义一个列表，包含了需要共享权重的键名
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中指定了 is_decoder 为 True，发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建一个 RoBERTa 模型，不包含池化层
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 创建一个 RoBERTa 语言建模头部
        self.lm_head = RobertaLMHead(config)

        # 执行额外的初始化操作和最终处理
        self.post_init()

    # 返回语言建模头部的输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置新的输出嵌入到语言建模头部
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 使用文档字符串和代码示例的注释来注释 forward 方法
    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.1,
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
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # Determine if the output should be returned as a dictionary or not
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the Roberta model
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
        # Extract the sequence output from the model outputs
        sequence_output = outputs[0]
        # Generate prediction scores using the language model head
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # Move labels tensor to the device used for prediction_scores
            labels = labels.to(prediction_scores.device)
            # Define the loss function for masked language modeling
            loss_fct = CrossEntropyLoss()
            # Compute masked language modeling loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # Prepare output tuple without returning a dictionary
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return MaskedLMOutput named tuple if return_dict is True
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个用于 RoBERTa 的语言模型头部的类
class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入大小映射到隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个层归一化层，用于标准化隐藏状态
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建一个线性层，将隐藏大小映射到词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 创建一个偏置参数，用于解码器线性层
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将解码器的偏置设置为自定义的偏置参数
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 输入特征经过线性层变换
        x = self.dense(features)
        # 使用 GELU 激活函数进行非线性变换
        x = gelu(x)
        # 输入经过层归一化处理
        x = self.layer_norm(x)

        # 将隐藏状态映射回词汇表大小，并加上偏置
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果解码器的偏置参数设备类型是 "meta"，则将其与自定义偏置参数绑定
        # 这是为了加速兼容性和保持向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            # 否则，将自定义偏置参数与解码器的偏置参数绑定
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化 RoBERTa 模型的分类/回归头部
        self.num_labels = config.num_labels
        self.config = config

        # 创建 RoBERTa 模型，不包含池化层
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 创建 RoBERTa 分类头部
        self.classifier = RobertaClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 RoBERTa 模型进行处理，并获取输出结果
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
        # 从 RoBERTa 模型的输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给分类器模型获取 logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 将 labels 移动到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            # 根据问题类型自动推断配置中的 problem_type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据 problem_type 计算损失函数
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

        # 如果 return_dict 为 False，则返回 logits 和可能的额外输出
        if not return_dict:
            output = (logits,) + outputs[2:]  # 保留 logits 和额外的 hidden states
            return ((loss,) + output) if loss is not None else output

        # 返回一个 SequenceClassifierOutput 对象，包含 loss、logits、hidden states 和 attentions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用多项选择分类头部的 RoBERTa 模型（在汇总输出之上有一个线性层和 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForMultipleChoice(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 RoBERTa 模型
        self.roberta = RobertaModel(config)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器线性层，输出维度为1（用于多项选择任务）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据返回值字典是否为空来确定是否使用预设的返回值设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算输入张量的第二维大小，即选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果存在input_ids，则将其展平为二维张量，否则为None
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果存在position_ids，则将其展平为二维张量，否则为None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果存在token_type_ids，则将其展平为二维张量，否则为None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果存在attention_mask，则将其展平为二维张量，否则为None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果存在inputs_embeds，则将其展平为三维张量，否则为None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用RoBERTa模型进行前向传播
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
        # 提取汇总输出
        pooled_output = outputs[1]

        # 对汇总输出应用dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器得到logits
        logits = self.classifier(pooled_output)
        # 将logits重新调整形状为(batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            # 将标签移动到正确的设备上以实现模型并行计算
            labels = labels.to(reshaped_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回字典，则返回输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回多选模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有标记分类头部的 RoBERTa 模型类，用于例如命名实体识别（NER）任务
@add_start_docstrings(
    """
    Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 模型，不包括汇聚层
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # 根据配置设定分类器的 dropout 率，若未设置则使用隐藏层的 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # 创建一个线性层，将 RoBERTa 隐藏层的输出转换为分类标签
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Jean-Baptiste/roberta-large-ner-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'LOC', 'O', 'LOC', 'LOC']",
        expected_loss=0.01,
    )
    # 前向传播函数，接受 RoBERTa 的输入并输出分类结果
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
        # RoBERTa 输入的详细说明文档
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果没有指定 return_dict，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoBERTa 模型处理输入数据，并获取输出结果
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

        # 从 RoBERTa 模型输出中获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        
        # 将 dropout 后的序列输出送入分类器得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值
        loss = None
        # 如果给定了标签，则计算交叉熵损失
        if labels is not None:
            # 将标签移到正确的设备上以支持模型并行计算
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            # 计算损失值
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典，则组织输出结果
        if not return_dict:
            output = (logits,) + outputs[2:]  # 这里的 outputs[2:] 包含额外的隐藏状态
            return ((loss,) + output) if loss is not None else output

        # 构造 TokenClassifierOutput 对象用于返回结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 确定分类器的 dropout 率，如果未提供，则使用隐藏层 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层，应用上述确定的 dropout 率
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个全连接层，将隐藏状态映射到类别数 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 从 features 中选择第一个 token 的隐藏状态作为输出，类似于取 [CLS] token
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)  # 应用 dropout
        x = self.dense(x)  # 应用全连接层
        x = torch.tanh(x)  # 应用 tanh 激活函数
        x = self.dropout(x)  # 再次应用 dropout
        x = self.out_proj(x)  # 应用最终的全连接层映射到输出类别数
        return x


@add_start_docstrings(
    """
    Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 RoBERTa 模型，不包含池化层
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # 定义一个全连接层，将 RoBERTa 的隐藏状态映射到类别数 config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
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
        # Determine whether to use return_dict or default based on configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input data to the Roberta model for processing
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

        # Extract the sequence output from the Roberta model's outputs
        sequence_output = outputs[0]

        # Compute logits for question answering start and end positions
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Handle multi-GPU scenarios by adjusting tensor dimensions
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # Clamp positions to ensure they are within valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define loss function and compute start/end losses
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # Prepare output tuple without using return_dict
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # Return structured output using QuestionAnsweringModelOutput class
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: torch.Tensor, input tensor containing token ids
        padding_idx: int, index of padding token in the vocabulary
        past_key_values_length: int, length of past key values to consider for incremental indexing

    Returns:
        torch.Tensor, tensor of position ids corresponding to input_ids
    """
    # 创建一个掩码，标记非填充符号的位置为1，填充符号的位置为0
    mask = input_ids.ne(padding_idx).int()
    # 根据掩码累积计算位置索引，加上过去关键值长度，然后乘以掩码以忽略填充符号
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回最终的位置 ids，加上填充索引以确保填充符号仍然为填充索引
    return incremental_indices.long() + padding_idx
```