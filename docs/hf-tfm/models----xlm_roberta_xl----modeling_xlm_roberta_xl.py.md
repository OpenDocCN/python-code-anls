# `.\models\xlm_roberta_xl\modeling_xlm_roberta_xl.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，指明版权归 HuggingFace Inc. 团队所有，使用 Apache License, Version 2.0 许可
# 详细许可信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
# 根据适用法律或书面同意，本软件是基于“原样”分发，不提供任何明示或暗示的保证或条件
# 请查阅许可证，了解具体的法律条款和限制条件

"""PyTorch XLM RoBERTa xl,xxl model."""
# 导入所需模块和类型注解
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数及模型输出类
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
# 导入模型工具函数及预训练模型基类
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入工具函数和日志记录函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 XLM-Roberta XL 配置文件
from .configuration_xlm_roberta_xl import XLMRobertaXLConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/xlm-roberta-xl"
_CONFIG_FOR_DOC = "XLMRobertaXLConfig"

# XLM-RoBERTa XL 预训练模型存档列表
XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xlm-roberta-xl",
    "facebook/xlm-roberta-xxl",
    # 更多的 RoBERTa 模型可在 https://huggingface.co/models?filter=xlm-roberta-xl 查看
]

# XLM-RoBERTa XL 嵌入层定义
class XLMRobertaXLEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 初始化函数，用于初始化一个新的实例对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建词嵌入层，vocab_size表示词汇表大小，hidden_size表示隐藏单元的大小，
        # padding_idx表示填充标记的索引位置，用于处理变长序列
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 创建位置嵌入层，max_position_embeddings表示最大的位置编码数，
        # hidden_size表示隐藏单元的大小，用于表示单词在句子中的位置信息
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # 创建token类型嵌入层，type_vocab_size表示token类型的数量，
        # hidden_size表示隐藏单元的大小，用于区分不同类型的token
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建Dropout层，用于随机将一部分元素置为0，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # position_embedding_type用于指定位置编码的类型，默认为"absolute"
        # 将position_ids张量注册为模型的缓冲区，包含从0到max_position_embeddings-1的位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        
        # 将token_type_ids张量注册为模型的缓冲区，初始化为全0的张量，形状与position_ids相同
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 设置padding_idx属性为config.pad_token_id，用于词嵌入层和位置嵌入层
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数，定义了模型的数据流向
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果位置 ID 为 None
        if position_ids is None:
            # 如果输入的 token IDs 不为 None，则从输入的 token IDs 创建位置 IDs。任何填充的 token 仍然保持填充状态。
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            # 否则，从输入嵌入创建位置 IDs
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入的 token IDs 不为 None
        if input_ids is not None:
            # 获取输入 token IDs 的形状
            input_shape = input_ids.size()
        else:
            # 否则，获取输入嵌入的形状，去掉最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即输入的第二维度的大小
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，通常为全零。这通常在自动生成时发生，
        # 注册的缓冲区有助于在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            # 如果 self 中有 "token_type_ids" 属性
            if hasattr(self, "token_type_ids"):
                # 从 self.token_type_ids 中获取缓冲的 token_type_ids，并截取到序列长度的部分
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 扩展 buffered_token_type_ids 以匹配输入形状的第一维和序列长度
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则，创建全零的 token_type_ids，dtype 为 long 类型，设备为 self.position_ids 的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为 None
        if inputs_embeds is None:
            # 使用 self.word_embeddings 对输入 token IDs 进行嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的嵌入表示，包括输入嵌入和 token_type_embeddings
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型为 "absolute"
        if self.position_embedding_type == "absolute":
            # 计算位置嵌入并添加到 embeddings 中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对 embeddings 应用 dropout
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入表示
        return embeddings

    # 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings.create_position_ids_from_inputs_embeds 复制而来
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状，去掉最后一维
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成从 padding_idx + 1 到 sequence_length + padding_idx + 1 的位置 IDs
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将 position_ids 扩展为与 input_shape 相同的形状
        return position_ids.unsqueeze(0).expand(input_shape)
# 从 transformers.models.bert.modeling_bert.BertSelfAttention 复制并修改为 XLMRobertaXLSelfAttention 类
class XLMRobertaXLSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除且配置中没有嵌入大小属性，则引发异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 注意力概率的dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对键（relative_key）或相对键查询（relative_key_query），则创建距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量重新排列为注意力分数的形状
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
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->XLMRobertaXL
class XLMRobertaXLSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集层，输入和输出大小为隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states, input_tensor):
        # 通过密集层
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # 添加输入张量并返回
        hidden_states = hidden_states + input_tensor
        return hidden_states


class XLMRobertaXLAttention(nn.Module):
    # 初始化函数，用于创建一个新的自注意力模型实例
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 创建自注意力层的 LayerNorm 层，用于归一化隐藏状态
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建自注意力层实例，传入配置和位置嵌入类型
        self.self = XLMRobertaXLSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建自注意力输出层实例
        self.output = XLMRobertaXLSelfOutput(config)
        # 存储需要被剪枝的注意力头的索引
        self.pruned_heads = set()

    # 头部剪枝函数，用于剪枝自注意力模型的某些注意力头
    def prune_heads(self, heads):
        # 若剪枝头部列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用辅助函数找到可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层中的查询、键、值和输出层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接收隐藏状态和多种注意力相关参数，并输出模型的前向传播结果
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 对输入的隐藏状态进行 LayerNorm 归一化处理
        intermediate = self.self_attn_layer_norm(hidden_states)
        # 调用自注意力层进行前向传播，得到自注意力层的输出
        self_outputs = self.self(
            intermediate,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用自注意力输出层处理自注意力层的输出和输入的隐藏状态，得到最终的注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回模型的输出结果
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate复制而来
class XLMRobertaXLIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数，如果config.hidden_act是字符串，则从预定义的映射ACT2FN中选择对应的函数；否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states传入全连接层self.dense，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果传入激活函数self.intermediate_act_fn进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class XLMRobertaXLOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states, input_tensor):
        # 将输入hidden_states传入全连接层self.dense，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果与输入input_tensor相加
        hidden_states = hidden_states + input_tensor
        return hidden_states


class XLMRobertaXLLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设定用于分块的前馈传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度为1
        self.seq_len_dim = 1
        # 创建XLMRobertaXLAttention对象并赋值给self.attention
        self.attention = XLMRobertaXLAttention(config)
        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 判断是否添加交叉注意力机制
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # 如果添加交叉注意力机制但不是解码器，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建带有绝对位置嵌入类型的XLMRobertaXLAttention对象并赋值给self.crossattention
            self.crossattention = XLMRobertaXLAttention(config, position_embedding_type="absolute")
        # 创建XLMRobertaXLIntermediate对象并赋值给self.intermediate
        self.intermediate = XLMRobertaXLIntermediate(config)
        # 创建XLMRobertaXLOutput对象并赋值给self.output
        self.output = XLMRobertaXLOutput(config)
        # 创建具有指定参数的LayerNorm对象并赋值给self.LayerNorm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 如果过去的键/值对存在，则只保留自注意力部分的前两个位置
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力层进行计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 如果当前层是解码器层，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 提取除了自注意力缓存以外的所有输出
            outputs = self_attention_outputs[1:-1]
            # 获取当前自注意力的键/值对
            present_key_value = self_attention_outputs[-1]
        else:
            # 否则，包括自注意力权重输出
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，还要包括自注意力
          
        # 初始化交叉注意力的键/值对为 None
        cross_attn_present_key_value = None
        # 如果是解码器并且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果模型没有交叉注意力层，则抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值对存在，则提取交叉注意力部分的位置 3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 调用交叉注意力层进行计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力层的输出
            attention_output = cross_attention_outputs[0]
            # 添加交叉注意力的输出到总输出中，排除注意力权重以外的部分
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力缓存添加到当前键/值对的位置 3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用块分片处理函数到前向输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将处理后的层输出添加到总输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，则将注意力的键/值对作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 定义前馈块处理函数
    def feed_forward_chunk(self, attention_output):
        # 对注意力输出进行层归一化
        intermediate_output = self.LayerNorm(attention_output)
        # 应用中间层
        intermediate_output = self.intermediate(intermediate_output)
        # 应用输出层
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义一个用于编码的 XLM-Roberta 模型的编码器类
class XLMRobertaXLEncoder(nn.Module):
    # 初始化方法，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象保存到实例变量中
        self.config = config
        # 创建一个由多个 XLM-Roberta 层组成的模块列表
        self.layer = nn.ModuleList([XLMRobertaXLLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建一个 LayerNorm 层，用于对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 是否启用梯度检查点技术，默认为关闭状态
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 如果启用了梯度检查点且处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 如果 use_cache 设置为 True，则与梯度检查点不兼容，发出警告并强制设置为 False
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False
            # 如果不需要输出隐藏状态，则初始化空的隐藏状态元组
            all_hidden_states = () if output_hidden_states else None
            # 如果不需要输出注意力权重，则初始化空的自注意力权重元组
            all_self_attentions = () if output_attentions else None
            # 如果不需要输出注意力权重或者不含交叉注意力层，则初始化空的交叉注意力权重元组
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

            # 如果 use_cache 设置为 False，则初始化空的下一个解码器缓存元组
            next_decoder_cache = () if use_cache else None
            # 遍历每个 Transformer 层
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 获取当前层的头部掩码，如果未提供则为 None
                layer_head_mask = head_mask[i] if head_mask is not None else None
                # 获取过去的键值对，如果未提供则为 None
                past_key_value = past_key_values[i] if past_key_values is not None else None

                # 如果启用了梯度检查点且处于训练模式下
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点函数调用当前层模块，传入相关参数
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
                    # 否则直接调用当前层模块，传入相关参数
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                # 更新隐藏状态为当前层模块的输出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果 use_cache 设置为 True，则将当前层模块的输出的最后一个元素添加到 next_decoder_cache 中
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                # 如果需要输出注意力权重，则将当前层模块的输出的第二个元素添加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    # 如果模型配置中包含交叉注意力层，则将当前层模块的输出的第三个元素添加到 all_cross_attentions 中
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # 对最终的隐藏状态应用 LayerNorm 归一化
            hidden_states = self.LayerNorm(hidden_states)

            # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不需要返回字典形式的输出
            if not return_dict:
                # 返回包含非 None 值的元组，包括隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力权重、所有交叉注意力权重
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
            # 否则返回包含所有输出的 BaseModelOutputWithPastAndCrossAttentions 对象
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
# Copied from transformers.models.bert.modeling_bert.BertPooler
class XLMRobertaXLPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        # Pass the first token's hidden state through a linear layer
        pooled_output = self.dense(first_token_tensor)
        # Apply activation function (Tanh) to the pooled output
        pooled_output = self.activation(pooled_output)
        return pooled_output


class XLMRobertaXLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = XLMRobertaXLConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Initialize weights of a linear layer with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # Initialize biases to zeros
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # If padding index is specified, initialize those weights to zeros
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm biases to zeros and weights to ones
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


XLM_ROBERTA_XL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.) This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`XLMRobertaXLConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XLM_ROBERTA_XL_INPUTS_DOCSTRING = r"""
    """
    Placeholder for documenting model input parameters, to be filled with specific details.
    """
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中词汇表中的标记索引。可以使用 [`AutoTokenizer`] 获得索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取详细信息。[什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力操作的掩码。掩码取值范围 `[0, 1]`：

            # - 1 表示**未被掩盖**的标记，
            # - 0 表示**被掩盖**的标记。
            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，指示输入的第一部分和第二部分。索引取值 `[0, 1]`：

            # - 0 对应*句子 A* 的标记，
            # - 1 对应*句子 B* 的标记。
            [什么是分段标记 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列中每个标记的位置索引，在位置嵌入中选择范围 `[0, config.max_position_embeddings - 1]`。[什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于空置自注意力模块中选择头部的掩码。掩码取值范围 `[0, 1]`：

            # - 1 表示**未被掩盖**的头部，
            # - 0 表示**被掩盖**的头部。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，可以直接传递嵌入表示，而不是传递 `input_ids`。如果要比模型内部的嵌入查找矩阵更精确地控制如何将 `input_ids` 索引转换为相关向量，则此选项很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回张量下的 `attentions` 获取更多详细信息。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回张量下的 `hidden_states` 获取更多详细信息。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
    """
    # 定义 XLM-RoBERTa-XL 模型类，继承自 XLMRobertaXLPreTrainedModel
    @add_start_docstrings(
        "The bare XLM-RoBERTa-XL Model transformer outputting raw hidden-states without any specific head on top.",
        XLM_ROBERTA_XL_START_DOCSTRING,
    )
    class XLMRobertaXLModel(XLMRobertaXLPreTrainedModel):
        """
        The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
        cross-attention is added between the self-attention layers, following the architecture described in *Attention is
        all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
        Kaiser and Illia Polosukhin. To behave as an decoder the model needs to be initialized with the `is_decoder`
        argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with
        both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as
        an input to the forward pass. .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
        """

        # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制并修改为 XLM-RoBERTa-XL
        def __init__(self, config, add_pooling_layer=True):
            # 调用父类构造函数
            super().__init__(config)
            # 初始化模型配置
            self.config = config

            # 初始化词嵌入层
            self.embeddings = XLMRobertaXLEmbeddings(config)
            # 初始化编码器
            self.encoder = XLMRobertaXLEncoder(config)

            # 添加池化层，如果指定要添加
            self.pooler = XLMRobertaXLPooler(config) if add_pooling_layer else None

            # 执行初始化权重和最终处理
            self.post_init()

        # 获取输入词嵌入
        def get_input_embeddings(self):
            return self.embeddings.word_embeddings

        # 设置输入词嵌入
        def set_input_embeddings(self, value):
            self.embeddings.word_embeddings = value

        # 剪枝模型中的注意力头部
        def _prune_heads(self, heads_to_prune):
            """
            Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
            class PreTrainedModel
            """
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)

        @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=BaseModelOutputWithPoolingAndCrossAttentions,
            config_class=_CONFIG_FOR_DOC,
        )
        # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        # 输入的 token IDs 张量，可选
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码张量，指示输入中哪些是 padding 的，可选
        attention_mask: Optional[torch.Tensor] = None,
        # 分段 ID 张量，用于区分不同句子或片段，可选
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID 张量，标识输入中每个 token 的位置信息，可选
        position_ids: Optional[torch.Tensor] = None,
        # 头部掩码张量，用于屏蔽特定的注意力头部，可选
        head_mask: Optional[torch.Tensor] = None,
        # 嵌入的输入张量，用于直接提供输入的嵌入表示，可选
        inputs_embeds: Optional[torch.Tensor] = None,
        # 编码器的隐藏状态张量，可选
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器的注意力掩码张量，用于屏蔽编码器注意力，可选
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 过去的键值列表，用于生成缓存，可选
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 是否使用缓存，可选
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回结果，可选
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """XLM-RoBERTa-XL Model with a `language modeling` head on top for CLM fine-tuning.""",
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLForCausalLM(XLMRobertaXLPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 初始化父类构造函数，配置模型为是否解码器
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # 创建语言模型的头部
        self.lm_head = XLMRobertaXLLMHead(config)

        # 初始化模型权重
        self.init_weights()

    def get_output_embeddings(self):
        # 返回语言模型头部的解码器部分
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的解码器部分为新的嵌入层
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ):
        """
        模型的前向传播函数，支持条件语言建模（CLM）。
        """
        # 准备生成的输入，包括输入的 ID、注意力掩码和过去的键值对
        input_shape = input_ids.shape
        # 如果注意力掩码为空，则创建全为1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果传入了过去的键值对，则修剪输入 ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 有些生成方法可能只传入最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回输入的字典，包含修剪后的输入 ID、注意力掩码和过去的键值对
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 定义一个方法 `_reorder_cache`，用于重新排序缓存中的过去键值
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化一个空元组 `reordered_past` 用于存储重新排序后的过去键值
        reordered_past = ()
        # 遍历每个层级的过去键值
        for layer_past in past_key_values:
            # 对每个层级的过去状态进行重新排序，并添加到 `reordered_past` 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值 `reordered_past`
        return reordered_past
# 为 XLM-RoBERTa-XL 模型添加一个在顶部的语言建模头部
@add_start_docstrings(
    """XLM-RoBERTa-XL Model with a `language modeling` head on top.""", XLM_ROBERTA_XL_START_DOCSTRING
)
class XLMRobertaXLForMaskedLM(XLMRobertaXLPreTrainedModel):
    # 定义了共享权重的关键键列表，用于语言建模头部
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置指定为解码器，发出警告，建议设置为双向自注意力模型
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 XLM-RoBERTa-XL 模型，不添加池化层
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # 初始化语言建模头部
        self.lm_head = XLMRobertaXLLMHead(config)

        # 初始化模型权重
        self.init_weights()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 定义前向传播方法，接受一系列输入参数和返回值，并用注释来描述每个参数和返回值的含义
    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
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
        # 根据 return_dict 参数确定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 RoBERTa 模型进行前向传播，获取输出
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
        # 获取 RoBERTa 模型的序列输出
        sequence_output = outputs[0]
        # 使用语言模型头部对序列输出进行预测得分计算
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果提供了 labels，计算掩码语言模型的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则按照非字典类型的方式返回输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则按照 MaskedLMOutput 类型返回输出
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class XLMRobertaXLLMHead(nn.Module):
    """XLM-RoBERTa-XL Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 线性层，用于将输入特征从隐藏大小转换为隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 归一化层，对输入进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 解码层，将隐藏特征映射到词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 偏置参数，与解码层相关联
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 线性转换
        x = self.dense(features)
        # GELU激活函数
        x = gelu(x)
        # 归一化
        x = self.layer_norm(x)

        # 使用解码层映射到词汇表大小
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果偏置被重新设置（如在TPU上或偏置大小变化时），将解码层的偏置与模型的偏置参数关联
        self.bias = self.decoder.bias


@add_start_docstrings(
    """
    XLM-RoBERTa-XL Model transformer with a sequence classification/regression head on top (a linear layer on top
    of the pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLForSequenceClassification(XLMRobertaXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 类别数目
        self.num_labels = config.num_labels
        self.config = config

        # XLM-RoBERTa-XL 模型，不添加池化层
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # 分类器，用于在顶部进行序列分类
        self.classifier = XLMRobertaXLClassificationHead(config)

        # 初始化模型权重
        self.init_weights()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        ):
        # 模型前向传播函数，详细参数如下所示
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保 return_dict 不为 None，则使用 self.config.use_return_dict，否则设为 None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 模型处理输入数据，并获取输出
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
        # 从 RoBERTa 输出中获取序列输出
        sequence_output = outputs[0]
        # 使用分类器对序列输出进行分类得到 logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果未指定问题类型，则根据条件自动确定
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数进行计算
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()  # 使用均方误差损失函数
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()  # 使用带 logits 的二元交叉熵损失函数
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典，则返回输出和损失
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、logits、隐藏状态和注意力权重的序列分类器输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
XLM-RoBERTa-XL Model with a multiple choice classification head on top (a linear layer on top of the pooled
output and a softmax) e.g. for RocStories/SWAG tasks.
"""
# 基于 XLM-RoBERTa-XL 模型，添加一个多选分类头部（池化输出之上的线性层和 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    XLM_ROBERTA_XL_START_DOCSTRING,
)
# XLM-RoBERTa-XLForMultipleChoice 类定义，继承自 XLMRobertaXLPreTrainedModel
class XLMRobertaXLForMultipleChoice(XLMRobertaXLPreTrainedModel):
    
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 初始化 XLM-RoBERTa-XL 模型
        self.roberta = XLMRobertaXLModel(config)
        
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 分类器，线性层，输入尺寸为隐藏层大小，输出尺寸为 1（二元分类）
        self.classifier = nn.Linear(config.hidden_size, 1)
        
        # 初始化权重
        self.init_weights()

    # forward 方法
    @add_start_docstrings_to_model_forward(
        XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
        """
        XLM-RoBERTa-XL 模型的前向传播方法。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入的 token IDs. Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): token 类型 IDs. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): 注意力遮罩. Defaults to None.
            labels (Optional[torch.LongTensor], optional): 标签. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): 位置 IDs. Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): 头部遮罩. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): 输入嵌入. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出注意力. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典格式的输出. Defaults to None.

        Returns:
            MultipleChoiceModelOutput: 多选分类模型的输出.
        """
        # 省略具体的前向传播逻辑
        pass
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保返回字典不为空，根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选项的数量，如果输入的input_ids不为空，则为其第二维的大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的input_ids展平成二维数组，如果input_ids不为空
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将position_ids展平成二维数组，如果position_ids不为空
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将token_type_ids展平成二维数组，如果token_type_ids不为空
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将attention_mask展平成二维数组，如果attention_mask不为空
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将inputs_embeds展平成三维数组，如果inputs_embeds不为空
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用RoBERTa模型处理展平后的输入
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
        # 获取RoBERTa模型的汇聚输出
        pooled_output = outputs[1]

        # 对汇聚输出进行dropout操作
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类，得到logits
        logits = self.classifier(pooled_output)
        # 将logits重新形状为(batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了labels，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不使用返回字典，则返回一个元组，包含reshaped_logits和额外的输出信息
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则返回一个MultipleChoiceModelOutput对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    XLM-RoBERTa-XL Model with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLForTokenClassification(XLMRobertaXLPreTrainedModel):
    """
    XLM-RoBERTa-XL模型，顶部带有一个标记分类头部（在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 XLM-RoBERTa-XL 模型，不添加池化层
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)

        # 如果配置中指定了分类器的dropout，则使用其值；否则使用隐藏层dropout的值
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # 线性分类器，将隐藏状态大小映射到标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型权重
        self.init_weights()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        """
        XLM-RoBERTa-XL 模型的前向传播方法，接受多种输入参数，并返回标记分类的输出。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入的 token IDs，shape 为 [batch_size, sequence_length]。
            attention_mask (Optional[torch.FloatTensor], optional): 注意力遮罩，shape 为 [batch_size, sequence_length]。
            token_type_ids (Optional[torch.LongTensor], optional): token 类型 IDs，shape 为 [batch_size, sequence_length]。
            position_ids (Optional[torch.LongTensor], optional): 位置 IDs，shape 为 [batch_size, sequence_length]。
            head_mask (Optional[torch.FloatTensor], optional): 头部遮罩，shape 为 [num_heads] 或者 [num_layers, num_heads]。
            inputs_embeds (Optional[torch.FloatTensor], optional): 嵌入的输入，shape 为 [batch_size, sequence_length, hidden_size]。
            labels (Optional[torch.LongTensor], optional): 标签，shape 为 [batch_size, sequence_length]。
            output_attentions (Optional[bool], optional): 是否输出注意力权重。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态。
            return_dict (Optional[bool], optional): 是否返回输出的字典格式。

        Returns:
            TokenClassifierOutput: 标记分类器的输出，包括 logits、损失和可能的额外内容。
        """
        # 略
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 参数为 None，则使用 self.config.use_return_dict 决定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 模型进行前向传播
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

        # 获取模型输出的序列输出（通常是最后一层的隐藏状态）
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        
        # 对处理后的序列输出进行分类预测
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只保留 loss 中与 attention_mask 激活部分对应的部分
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                # 计算所有 logits 与 labels 之间的 loss
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # 如果不要求返回字典类型的输出，则按元组形式返回
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典类型的输出，则构造 TokenClassifierOutput 对象返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class XLMRobertaXLClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择分类器的 dropout rate，若未指定则使用隐藏层 dropout rate
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层，全连接层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 从 features 中取出第一个 token 的隐藏状态，相当于取出了 [CLS] token
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)  # 使用 dropout 处理隐藏状态
        x = self.dense(x)  # 全连接层处理隐藏状态
        x = torch.tanh(x)  # 使用 tanh 激活函数
        x = self.dropout(x)  # 再次使用 dropout
        x = self.out_proj(x)  # 输出层处理结果
        return x


@add_start_docstrings(
    """
    XLM-RoBERTa-XL Model with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_XL_START_DOCSTRING,
)
class XLMRobertaXLForQuestionAnswering(XLMRobertaXLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 XLM-RoBERTa 模型，不包括 pooling 层
        self.roberta = XLMRobertaXLModel(config, add_pooling_layer=False)
        # QA 输出层，全连接层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()  # 初始化模型权重

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_XL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # Initialize return_dict to either the provided value or the default from model configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the Roberta model with specified inputs and optional arguments
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

        # Extract the sequence output from the model outputs
        sequence_output = outputs[0]

        # Pass sequence output through QA output layer to get logits
        logits = self.qa_outputs(sequence_output)
        
        # Split logits into start and end logits along the last dimension
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # Squeeze out unnecessary dimensions and ensure contiguous memory layout
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Adjust start_positions and end_positions if they have extra dimensions
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # Clamp positions to ignore indices outside of model input length
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Compute CrossEntropyLoss for start and end positions
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            # Calculate total loss as the average of start and end losses
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # Prepare output tuple if return_dict is False
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # Return structured output using QuestionAnsweringModelOutput class if return_dict is True
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的 input_ids 中创建位置 ID，用于模型的位置编码
# 非填充符号被替换为它们的位置编号，位置编号从 padding_idx+1 开始计数，填充符号被忽略。
# 这个函数是根据 fairseq 的 `utils.make_positions` 修改而来。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: torch.Tensor，输入的 token IDs
        padding_idx: int，填充符的索引
        past_key_values_length: int，过去的键值长度，用于增量索引

    Returns:
        torch.Tensor，位置 ID 的张量
    """
    # 创建一个掩码，标记非填充符号位置为1，填充符号位置为0
    mask = input_ids.ne(padding_idx).int()
    # 计算递增的位置索引，这里的类型转换和累加被精心设计以便与 ONNX 导出和 XLA 兼容
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 最终的位置 ID 是递增索引加上 padding_idx，转换为长整型
    return incremental_indices.long() + padding_idx
```