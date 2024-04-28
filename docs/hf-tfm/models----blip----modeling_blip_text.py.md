# `.\transformers\models\blip\modeling_blip_text.py`

```py
# 导入必要的模块
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, device, nn
from torch.nn import CrossEntropyLoss

# 从BLIP模型的相关模块导入配置和输出类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)


# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#L52 改编而来的 BLIP 文本嵌入层
class BlipTextEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        # 定义词嵌入层，将词索引映射为词向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，将位置索引映射为位置向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 使用 LayerNorm 进行归一化处理
        # 这里的 self.LayerNorm 名称未采用蛇形命名法，以与 TensorFlow 模型变量名保持一致，以便能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 将位置 id (1, 最大位置嵌入长度) 在内存中连续，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入类型，默认为"absolute"，即绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # 存储配置
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果输入的 token IDs 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取输入嵌入张量的形状，去除最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置 IDs 为空，则使用预训练模型中的位置 IDs
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果输入嵌入张量为空，则将 token IDs 转换为预训练模型支持的设备，并获取其嵌入表示
        if inputs_embeds is None:
            input_ids = input_ids.to(self.word_embeddings.weight.device)
            inputs_embeds = self.word_embeddings(input_ids)

        # 将嵌入张量初始化为输入嵌入张量
        embeddings = inputs_embeds

        # 如果位置嵌入类型是“绝对”的话
        if self.position_embedding_type == "absolute":
            # 获取位置嵌入，并将其加到嵌入张量中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 将嵌入张量进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入张量进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入张量
        return embeddings
# BLIP 文本自注意力机制模块的定义
class BlipTextSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        # 如果隐藏大小不能整除注意力头数，则抛出数值错误异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型为相对位置嵌入之一，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    # 保存注意力梯度
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    # 获取保存的注意力梯度
    def get_attn_gradients(self):
        return self.attn_gradients

    # 保存注意力映射
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    # 获取保存的注意力映射
    def get_attention_map(self):
        return self.attention_map

    # 将输入张量调整为适合计算注意力得分的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # BLIP 文本自注意力模块的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    # 初始化方法，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNormalization 层，输入大小为 config.hidden_size，设置 epsilon 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，设置 dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，用于定义模型的前向计算逻辑
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过全连接层 dense，得到新的 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对新的 hidden_states 进行 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的 hidden_states 和输入的 input_tensor 相加，然后经过 LayerNormalization 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过处理后的 hidden_states
        return hidden_states
# BlipTextAttention 类定义，用于实现 BLIP 模型中的文本注意力机制
class BlipTextAttention(nn.Module):
    # 初始化方法
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 创建 BlipTextSelfAttention 对象
        self.self = BlipTextSelfAttention(config, is_cross_attention)
        # 创建 BlipTextSelfOutput 对象
        self.output = BlipTextSelfOutput(config)
        # 初始化被剪枝的注意力头集合
        self.pruned_heads = set()

    # 剪枝注意力头
    def prune_heads(self, heads):
        # 如果没有要剪枝的头，则直接返回
        if len(heads) == 0:
            return
        # 寻找可剪枝的头以及它们的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头
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
        # 调用 BlipTextSelfAttention 的前向传播方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 对自注意力输出进行处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回输出
        return outputs


# BlipTextIntermediate 类定义，用于实现 BLIP 模型中的文本中间层
class BlipTextIntermediate(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 创建线性层，将隐藏状态映射到中间层维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果激活函数是字符串形式，则使用对应的激活函数，否则使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层映射到中间层维度
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回结果
        return hidden_states


# BlipTextOutput 类定义，用于实现 BLIP 模型中的文本输出层
class BlipTextOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个张量参数，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states传入全连接层，得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的输出与input_tensor相加，然后传入LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回LayerNorm层的输出
        return hidden_states
class BlipTextLayer(nn.Module):
    # 定义 BLIP 文本层模块
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BlipTextAttention(config)
        self.layer_num = layer_num
        # 如果是解码器，创建交叉注意力对象
        if self.config.is_decoder:
            self.crossattention = BlipTextAttention(config, is_cross_attention=self.config.is_decoder)
        self.intermediate = BlipTextIntermediate(config)
        self.output = BlipTextOutput(config)

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
        # 获取解码器单向自注意力的缓存键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if encoder_hidden_states is not None:
            # 进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力
        # 应用分块处理到前向传播
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 进行中间层和输出层的前向传播
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#L386 改编
class BlipTextEncoder(nn.Module):
    # 定义 BLIP 文本编码器模块
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建多个 BLIP 文本层模块
        self.layer = nn.ModuleList([BlipTextLayer(config, i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    # 定义了模型的前向传播方法，用于生成模型的输出
    def forward(
        self,
        # 输入的隐藏状态张量，即模型的输入
        hidden_states: torch.Tensor,
        # 注意力掩码，用于指示哪些位置需要被忽略
        attention_mask: Optional[torch.FloatTensor] = None,
        # 头部掩码，用于指示哪些注意力头应该被忽略
        head_mask: Optional[torch.FloatTensor] = None,
        # 编码器的隐藏状态，用于注意力计算的上下文
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 编码器的注意力掩码，用于指示编码器哪些位置需要被忽略
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 过去的键值对，用于存储前一个时间步的注意力信息
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 是否使用缓存，用于加速多次调用
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = False,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = False,
        # 是否返回字典格式的输出
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果启用了梯度检查点并且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果使用缓存，则警告梯度检查点不兼容，设置use_cache为False
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        # 初始化存储隐藏状态、注意力权重和交叉注意力权重的元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None

        # 如果使用缓存，则初始化下一个解码器缓存
        next_decoder_cache = () if use_cache else None

        # 遍历每个隐藏层
        for i in range(self.config.num_hidden_layers):
            # 获取当前隐藏层模块
            layer_module = self.layer[i]
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码和过去的键值对
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数计算当前层的输出
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
                # 否则直接调用当前层模块计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # ��新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的注意力权重和交叉注意力权重添加到对应的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回包含隐藏状态、下一个解码器缓存、所有隐藏状态、自注意力权重和交叉注意力权重的元组
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
        # 否则返回包含最终隐藏状态、下一个解码器缓存、所有隐藏状态、自注意力权重和交叉注意力权重的BaseModelOutputWithPastAndCrossAttentions对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPooler复制，将Bert->BlipText
class BlipTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将隐藏状态维度转换为与配置相同的隐藏状态维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数，使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取对应于第一个标记的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制，将Bert->BlipText
class BlipTextPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将隐藏状态维度转换为与配置相同的隐藏状态维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制，将Bert->BlipText
class BlipTextLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用BlipTextPredictionHeadTransform转换隐藏状态
        self.transform = BlipTextPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来确保偏置与“resize_token_embeddings”一起正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制，将Bert->BlipText
class BlipTextOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用BlipTextLMPredictionHead构建预测头
        self.predictions = BlipTextLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从https://github.com/salesforce/BLIP/blob/main/models/med.py#L548适应
class BlipTextPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化的抽象类，用于下载和加载预训练的简单接口
    models.
    """

# 定义一个类，继承自BertPreTrainedModel类，并为文本分类任务配置BlipTextConfig
class BlipTextForPreTraining(BertPreTrainedModel):
    # 使用BlipTextConfig类作为配置类
    config_class = BlipTextConfig
    # 设置基础模型前缀为"bert"
    base_model_prefix = "bert"

    # 初始化权重函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果module是nn.Linear或者nn.Embedding类型
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正态分布初始化权重，均值为0，标准差为config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果module是nn.LayerNorm类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
        # 如果module是nn.Linear类型并且有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置项初始化为零
            module.bias.data.zero_()
# 根据给定的配置创建一个 BLIP 文本模型类
# 可以作为编码器（仅自注意力）或解码器（此时在自注意力层之间添加一层交叉注意力），遵循 [Attention is all you need] 中描述的架构
# 作者为 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin
class BlipTextModel(BlipTextPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置信息
        self.config = config

        # 创建 BLIP 文本模型的嵌入层
        self.embeddings = BlipTextEmbeddings(config)
        # 创建 BLIP 文本模型的编码器层
        self.encoder = BlipTextEncoder(config)
        # 如果需要添加池化层，则创建池化层，否则设为 None
        self.pooler = BlipTextPooler(config) if add_pooling_layer else None

        # 完成初始化后的操作
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 从 BERT 模型中复制并修剪头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 获取扩展的注意力掩码
    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_decoder: Optional[bool] = False,
# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#L811 改编
class BlipTextLMHeadModel(BlipTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 创建 BLIP 文本语言模型的 BERT 部分
        self.bert = BlipTextModel(config, add_pooling_layer=False)
        # 创建 BLIP 文本语言模型的分类层
        self.cls = BlipTextOnlyMLMHead(config)

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接收多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码
        labels: Optional[torch.Tensor] = None,  # 标签
        past_key_values: Optional[List[torch.Tensor]] = None,  # 过去的键值
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 输出注意力
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
        return_logits: Optional[bool] = False,  # 返回对数
        is_decoder: Optional[bool] = True,  # 是否为解码器
        reduction: Optional[str] = "mean",  # 减少方式
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全为1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去的键值，截取输入的 token ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    # 重新排序缓存中的过去键值
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```