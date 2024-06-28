# `.\models\blip\modeling_blip_text.py`

```
# coding=utf-8
# 版权 2022 年 Salesforce 团队作者和 HuggingFace 团队。保留所有权利。
#
# 根据 BSD-3-clause 许可证授权（“许可证”）;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# 除非适用法律要求或书面同意，否则依照“原样”分发的软件
# 不附带任何形式的明示或暗示担保或条件。
# 有关特定语言的详细信息，请参阅许可证。
#

import math  # 导入数学模块
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 工具
from torch import Tensor, device, nn  # 从 PyTorch 导入 Tensor、device、nn 等
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出相关的类
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import (  # 导入模型工具函数
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging  # 导入日志工具
from .configuration_blip import BlipTextConfig  # 导入 BLIP 文本配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#L52 进行了适配
class BlipTextEmbeddings(nn.Module):
    """根据单词和位置嵌入构建嵌入层。"""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm 未使用蛇形命名，以保持与 TensorFlow 模型变量名一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids（1，长度位置嵌入）在序列化时是连续的内存并可导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 定义函数的输入和输出类型，此处返回一个 torch.Tensor 对象
    ) -> torch.Tensor:
        # 如果传入了 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状，排除最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，这里假设 input_shape 是一个元组，其第二个维度表示序列长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从预设的 position_ids 中选择对应序列长度的部分
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果 inputs_embeds 为空，则使用 input_ids 加载对应设备的 word_embeddings 来获取 inputs_embeds
        if inputs_embeds is None:
            input_ids = input_ids.to(self.word_embeddings.weight.device)
            inputs_embeds = self.word_embeddings(input_ids)

        # 将 embeddings 初始化为 inputs_embeds
        embeddings = inputs_embeds

        # 如果使用绝对位置编码（absolute），则添加位置编码到 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对 embeddings 进行 LayerNorm（归一化处理）
        embeddings = self.LayerNorm(embeddings)

        # 对 embeddings 应用 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的 embeddings
        return embeddings
# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L97
# 定义一个自注意力机制模块，继承自 nn.Module
class BlipTextSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        # 检查隐藏大小是否可以被注意力头数整除，若不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义注意力概率的 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 根据配置确定位置嵌入的类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 如果是相对位置编码，则初始化距离嵌入的 Embedding 层
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

    # 将输入 tensor 转置为 scores 的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
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
    ):
        # 此处省略了具体的前向传播逻辑，根据输入参数计算注意力输出
        pass

# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并修改为 BlipTextSelfOutput
# 定义一个自注意力输出模块，继承自 nn.Module
class BlipTextSelfOutput(nn.Module):
    # 初始化函数，用于初始化一个新的实例
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化函数
        super().__init__()
        # 创建一个全连接层，输入和输出的维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对输入进行归一化处理，eps是用于数值稳定性的小值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，以config.hidden_dropout_prob的概率随机将输入置零，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了模型的计算流程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理输入的隐藏状态，生成新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对新的隐藏状态进行 Dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的隐藏状态和输入张量相加，然后经过 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#242 改编而来的代码

# 定义了一个用于 BlipText 模型中注意力机制的自定义 PyTorch 模块
class BlipTextAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        # 初始化 self 层，使用 BlipTextSelfAttention 类处理自注意力或交叉注意力
        self.self = BlipTextSelfAttention(config, is_cross_attention)
        # 初始化 output 层，用于处理自注意力的输出
        self.output = BlipTextSelfOutput(config)
        # 初始化一个用于记录被修剪掉的注意力头的集合
        self.pruned_heads = set()

    # 修剪模型中的注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可以修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 在 self 层中修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录修剪掉的注意力头
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
        # 调用 self 层的前向传播，获取自注意力的输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力的输出传递给 output 层进行处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，将它们添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，添加注意力
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 BlipTextIntermediate
# 定义了 BlipText 模型中间层的自定义 PyTorch 模块
class BlipTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态转换为中间层状态
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置中的激活函数设置中间层的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 中间层的前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层计算中间层的输出
        hidden_states = self.dense(hidden_states)
        # 使用配置中指定的激活函数对中间层进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 BlipTextOutput
# 定义了 BlipText 模型输出层的自定义 PyTorch 模块
class BlipTextOutput(nn.Module):
    # 初始化函数，用于设置模型的各个组件和参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度为config中的中间大小，输出维度为config中的隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个Layer Normalization层，输入维度为config中的隐藏大小，设置epsilon为config中的layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于随机置零输入张量的一些元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了模型的计算流程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入张量经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的张量应用Dropout层
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的张量与输入张量相加，并应用Layer Normalization
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过处理后的张量作为输出
        return hidden_states
# BLIP 文本层的定义，继承自 nn.Module 类
class BlipTextLayer(nn.Module):
    # 初始化函数，接受配置对象 config 和层编号 layer_num 作为参数
    def __init__(self, config, layer_num):
        super().__init__()
        # 将配置对象保存在实例中
        self.config = config
        # 设置前馈过程的块大小为配置中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设定为 1
        self.seq_len_dim = 1
        # 创建 BLIP 文本注意力层对象
        self.attention = BlipTextAttention(config)
        # 保存层编号
        self.layer_num = layer_num
        # 如果配置中包含解码器，则创建 BLIP 文本交叉注意力层对象
        if self.config.is_decoder:
            self.crossattention = BlipTextAttention(config, is_cross_attention=self.config.is_decoder)
        # 创建 BLIP 文本中间层对象
        self.intermediate = BlipTextIntermediate(config)
        # 创建 BLIP 文本输出层对象
        self.output = BlipTextOutput(config)

    # 前向传播函数，接受多个输入参数并返回一个元组
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
        # 如果过去键值不为空，提取解码器单向自注意力的缓存键/值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力层的前向传播函数，传入隐藏状态等参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 提取除了第一个和最后一个元素之外的所有输出，用于后续处理
        outputs = self_attention_outputs[1:-1]
        # 获取当前键值
        present_key_value = self_attention_outputs[-1]

        # 如果存在编码器隐藏状态，调用交叉注意力层的前向传播函数
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力层的输出
            attention_output = cross_attention_outputs[0]
            # 如果输出注意力权重，则添加到输出中
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights
        
        # 将注意力输出应用到前馈块的处理函数中
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将层输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 添加当前键值到输出元组中
        outputs = outputs + (present_key_value,)

        # 返回所有输出的元组
        return outputs

    # 前馈块处理函数，接受注意力输出并返回处理后的层输出
    def feed_forward_chunk(self, attention_output):
        # 调用中间层对象处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 调用输出层对象生成最终的层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的层输出
        return layer_output


# 从 https://github.com/salesforce/BLIP/blob/main/models/med.py#L386 调整的代码
# BLIP 文本编码器的定义，继承自 nn.Module 类
class BlipTextEncoder(nn.Module):
    # 初始化函数，接受配置对象 config 作为参数
    def __init__(self, config):
        super().__init__()
        # 保存配置对象到实例中
        self.config = config
        # 创建包含多个 BLIP 文本层对象的模块列表，根据配置中的隐藏层数量进行创建
        self.layer = nn.ModuleList([BlipTextLayer(config, i) for i in range(config.num_hidden_layers)])
        # 关闭梯度检查点
        self.gradient_checkpointing = False
    # 定义一个方法用于执行前向传播，用于处理Transformer网络中的一个步骤

    self,
        # 指向当前实例的引用，允许方法访问实例的属性和方法

        hidden_states: torch.Tensor,
        # 输入参数：表示模型中的隐藏状态张量，通常是模型输入的表示

        attention_mask: Optional[torch.FloatTensor] = None,
        # 可选参数：用于屏蔽注意力机制中不需要考虑的位置，可以是一个张量或None

        head_mask: Optional[torch.FloatTensor] = None,
        # 可选参数：头部掩码，用于控制哪些注意力头参与计算

        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 可选参数：编码器的隐藏状态张量，用于某些模型需要对编码器输出进行注意

        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 可选参数：编码器的注意力掩码，用于编码器输出的屏蔽机制

        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 可选参数：用于存储过去的键值对，用于处理长序列时的记忆优化

        use_cache: Optional[bool] = None,
        # 可选参数：指示是否使用缓存，以便重复计算可以复用中间结果

        output_attentions: Optional[bool] = False,
        # 可选参数：指示是否输出注意力权重

        output_hidden_states: Optional[bool] = False,
        # 可选参数：指示是否输出所有隐藏状态，而不仅仅是最后一层的隐藏状态

        return_dict: Optional[bool] = True,
        # 可选参数：指示是否以字典形式返回结果，如果为False，则返回元组形式的结果
    # 返回值类型声明，函数返回一个元组，包含 torch.Tensor 类型的元素或者 BaseModelOutputWithPastAndCrossAttentions 类型的对象
    -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果启用了梯度检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 被设置为 True，则警告 use_cache=True 与梯度检查点不兼容，并强制设置 use_cache=False
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        # 如果不输出隐藏状态，则初始化 all_hidden_states 为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化 all_self_attentions 为空元组，否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果不输出注意力权重或者当前模型不是解码器，则初始化 all_cross_attentions 为空元组，否则为 None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None

        # 如果 use_cache 为 True，则初始化 next_decoder_cache 为空元组，否则为 None
        next_decoder_cache = () if use_cache else None

        # 遍历每个隐藏层
        for i in range(self.config.num_hidden_layers):
            # 获取第 i 层的模块
            layer_module = self.layer[i]
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取第 i 层的头部掩码，如果 head_mask 不为 None，则为其赋值，否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取第 i 层的过去键值对，如果 past_key_values 不为 None，则为其赋值，否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来计算当前层的输出
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
                # 直接调用当前层的模块来计算输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果 use_cache 为 True，则将当前层的输出的最后一个元素添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层输出的第二个元素添加到 all_self_attentions 中，
            # 并将当前层输出的第三个元素（如果存在）添加到 all_cross_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回包含非 None 元素的元组
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
        # 如果 return_dict 为 True，则返回一个 BaseModelOutputWithPastAndCrossAttentions 对象，
        # 包含指定的隐藏状态、缓存、隐藏状态历史、自注意力权重和交叉注意力权重
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->BlipText
class BlipTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义一个全连接层，输入输出维度都是 config.hidden_size
        self.activation = nn.Tanh()  # 定义激活函数为双曲正切函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]  # 获取每个样本中第一个 token 对应的隐藏状态
        pooled_output = self.dense(first_token_tensor)  # 将第一个 token 的隐藏状态通过全连接层
        pooled_output = self.activation(pooled_output)  # 应用激活函数
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->BlipText
class BlipTextPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义一个全连接层，输入输出维度都是 config.hidden_size
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]  # 根据配置选择激活函数
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 应用 Layer Normalization

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 通过全连接层变换隐藏状态
        hidden_states = self.transform_act_fn(hidden_states)  # 应用激活函数
        hidden_states = self.LayerNorm(hidden_states)  # 应用 Layer Normalization
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->BlipText
class BlipTextLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BlipTextPredictionHeadTransform(config)  # 使用上面定义的头部变换层

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 定义一个线性层，将隐藏状态映射到词汇表大小的空间
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 定义一个偏置参数，大小为词汇表大小

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias  # 将偏置参数与解码器层关联，以便与 `resize_token_embeddings` 正确调整大小

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # 应用头部变换层
        hidden_states = self.decoder(hidden_states)  # 将变换后的隐藏状态映射到词汇表空间
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->BlipText
class BlipTextOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BlipTextLMPredictionHead(config)  # 使用上面定义的语言模型预测头部

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)  # 生成预测分数
        return prediction_scores


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L548
class BlipTextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    """
    pass  # 空类，用于处理权重初始化和预训练模型的简单下载和加载接口
    models.
    """

    # 定义配置类为BlipTextConfig
    config_class = BlipTextConfig
    # 设置基础模型前缀为"bert"
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层或嵌入层
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正态分布初始化权重，均值为0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1.0
            module.weight.data.fill_(1.0)
        # 如果模块是线性层并且具有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置项初始化为零
            module.bias.data.zero_()
# Adapted from https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/med.py#L571
class BlipTextModel(BlipTextPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Initialize embeddings layer for text
        self.embeddings = BlipTextEmbeddings(config)
        # Initialize encoder layer for processing text
        self.encoder = BlipTextEncoder(config)
        # Optionally initialize pooling layer if specified
        self.pooler = BlipTextPooler(config) if add_pooling_layer else None

        # Perform any post-initialization steps
        self.post_init()

    def get_input_embeddings(self):
        # Return the word embeddings from the embeddings layer
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # Set new word embeddings in the embeddings layer
        self.embeddings.word_embeddings = value

    # Copied from transformers.models.bert.modeling_bert.BertModel._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # Prune specified heads in the attention mechanism of each layer
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool
    ):
        # Create an extended attention mask to handle different attention scenarios
        # Not fully implemented in the provided snippet
        pass

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
    ):
        # Forward pass through the model, not fully implemented here
        pass

# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L811
class BlipTextLMHeadModel(BlipTextPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # Initialize the base BlipTextModel for language modeling
        self.bert = BlipTextModel(config, add_pooling_layer=False)
        # Initialize the MLM (Masked Language Modeling) head
        self.cls = BlipTextOnlyMLMHead(config)
        # Define label smoothing factor
        self.label_smoothing = config.label_smoothing

    def get_output_embeddings(self):
        # Return the decoder part of the MLM head's predictions
        return self.cls.predictions.decoder
    # 设置新的输出嵌入到模型预测的解码器中
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 模型的前向传播函数，接受多个输入参数并返回模型输出或损失
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token ID序列，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示哪些token需要注意力，默认为None
        position_ids: Optional[torch.Tensor] = None,  # 位置ID，用于指示每个token的位置信息，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，用于控制不同头部的注意力，默认为None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，默认为None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，默认为None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码，默认为None
        labels: Optional[torch.Tensor] = None,  # 真实标签，默认为None
        past_key_values: Optional[List[torch.Tensor]] = None,  # 过去的键值对，用于生成，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回，默认为None
        return_logits: Optional[bool] = False,  # 是否返回logits，默认为False
        is_decoder: Optional[bool] = True,  # 是否作为解码器，默认为True
        reduction: Optional[str] = "mean",  # 损失函数的减少方式，默认为"mean"
    ):
        # 准备生成输入的函数，为生成模型准备输入数据
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            input_shape = input_ids.shape
            # 如果注意力掩码为None，则创建一个全1的注意力掩码，形状与输入ID相同
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果过去的键值对不为None，则根据过去的长度截取输入ID
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 一些生成方法已经只传递了最后一个输入ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认保留最后一个ID
                    remove_prefix_length = input_ids.shape[1] - 1

                input_ids = input_ids[:, remove_prefix_length:]

            # 返回准备好的输入数据作为字典形式
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
                "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
                "is_decoder": True,
            }

    # 重新排序缓存中的过去键值对，根据beam索引重排
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```