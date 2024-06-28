# `.\models\mpnet\modeling_mpnet.py`

```
# coding=utf-8
# 版权 2018 年 HuggingFace Inc. 团队，Microsoft 公司。
# 版权所有 (c) 2018 年 NVIDIA 公司。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据许可证分发的软件
# 是基于“按原样”提供，没有任何明示或暗示的担保或条件。
# 有关许可证的详细信息，请参阅许可证。
"""PyTorch MPNet 模型。"""


import math  # 导入数学函数库
from typing import Optional, Tuple, Union  # 引入类型提示相关的库

import torch  # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN, gelu  # 导入激活函数
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
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 导入模型工具函数
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入工具函数和日志模块
from .configuration_mpnet import MPNetConfig  # 导入 MPNet 模型的配置文件


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "microsoft/mpnet-base"  # 预训练模型的检查点名称
_CONFIG_FOR_DOC = "MPNetConfig"  # MPNet 模型的配置类名


MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/mpnet-base",  # MPNet 预训练模型存档列表
]


class MPNetPreTrainedModel(PreTrainedModel):
    config_class = MPNetConfig  # 使用的配置类
    pretrained_model_archive_map = MPNET_PRETRAINED_MODEL_ARCHIVE_LIST  # 预训练模型的映射表
    base_model_prefix = "mpnet"  # 基础模型前缀名称

    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在 padding 索引，将其对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的权重和偏置
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MPNetEmbeddings(nn.Module):
    # 初始化函数，接收配置参数并进行初始化
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置填充索引为1
        self.padding_idx = 1
        # 创建词嵌入层，使用配置中的词汇表大小和隐藏层大小，设置填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # 创建位置嵌入层，使用配置中的最大位置数和隐藏层大小，设置填充索引
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # 创建 LayerNorm 层，使用隐藏层大小和配置中的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，使用配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 注册位置索引的缓冲区，生成一个从0到最大位置数的索引张量
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播函数，接收输入张量或嵌入向量，并返回处理后的嵌入向量
    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
        # 如果位置索引为 None
        if position_ids is None:
            # 如果输入的是 input_ids
            if input_ids is not None:
                # 根据 input_ids 创建位置索引，使用填充索引
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                # 否则，根据 inputs_embeds 创建位置索引
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入的是 input_ids
        if input_ids is not None:
            # 获取输入的形状
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状，去掉最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置索引仍然为 None，则使用预先创建的位置索引的子集
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果输入的嵌入向量为 None，则使用输入的 input_ids 获得词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 使用位置嵌入层获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)

        # 将词嵌入和位置嵌入相加作为最终的嵌入向量
        embeddings = inputs_embeds + position_embeddings
        # 对嵌入向量进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 Dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入向量
        return embeddings

    # 根据输入的嵌入向量创建位置索引
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入的形状，去掉最后一个维度
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成从填充索引+1到序列长度+填充索引+1的位置索引张量，使用输入的设备
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置索引张量扩展为与输入形状相匹配
        return position_ids.unsqueeze(0).expand(input_shape)
class MPNetSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的倍数，若不是且没有嵌入尺寸配置则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值和输出线性层
        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        self.o = nn.Linear(config.hidden_size, config.hidden_size)

        # 初始化注意力概率的丢弃层
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
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        # 计算查询、键、值的线性变换
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # 将查询、键、值调整为注意力头形式
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 计算注意力分数，通过查询和键的点积得到原始注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果提供了位置偏置，则添加相对位置嵌入
        if position_bias is not None:
            attention_scores += position_bias

        # 如果提供了注意力遮罩，则将其应用于注意力分数
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对注意力分数进行归一化，转换为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 应用注意力概率的丢弃层
        attention_probs = self.dropout(attention_probs)

        # 如果提供了头部遮罩，则将其应用于注意力概率
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权后的值向量
        c = torch.matmul(attention_probs, v)

        # 调整输出形状，将注意力头合并回原来的形状
        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.all_head_size,)
        c = c.view(*new_c_shape)

        # 经过输出线性层得到最终的注意力输出
        o = self.o(c)

        # 返回注意力输出及可能的注意力概率，根据需要决定是否输出注意力分布
        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs


class MPNetAttention(nn.Module):
    # 初始化函数，用于初始化一个新的实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化自注意力层
        self.attn = MPNetSelfAttention(config)
        # 初始化层归一化，设置隐藏层大小和层归一化的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 层，设置隐藏层的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 存储被修剪的注意力头的索引集合
        self.pruned_heads = set()

    # 方法用于修剪自注意力层中的头
    def prune_heads(self, heads):
        # 如果传入的头列表为空，则直接返回
        if len(heads) == 0:
            return
        
        # 调用函数找到可修剪的头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attn.num_attention_heads, self.attn.attention_head_size, self.pruned_heads
        )

        # 对自注意力层中的 q、k、v、o 进行线性层修剪
        self.attn.q = prune_linear_layer(self.attn.q, index)
        self.attn.k = prune_linear_layer(self.attn.k, index)
        self.attn.v = prune_linear_layer(self.attn.v, index)
        self.attn.o = prune_linear_layer(self.attn.o, index, dim=1)

        # 更新注意力头的数量和总的头大小
        self.attn.num_attention_heads = self.attn.num_attention_heads - len(heads)
        self.attn.all_head_size = self.attn.attention_head_size * self.attn.num_attention_heads
        # 将修剪的头添加到已修剪头的集合中
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，用于执行模型的前向计算
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        # 调用自注意力层进行前向传播计算
        self_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias,
            output_attentions=output_attentions,
        )
        # 计算注意力输出并进行层归一化、dropout
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + hidden_states)
        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力权重，则添加它们
        return outputs
# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class MPNetIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # 如果 config.hidden_act 是字符串，则从预定义的 ACT2FN 字典中选择对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行前向传播，得到中间层的输出
        hidden_states = self.dense(hidden_states)
        # 应用中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class MPNetOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入维度为 config.intermediate_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化 LayerNorm 层，输入维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行前向传播，得到输出的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 对隐藏状态应用 LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MPNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层、中间层和输出层
        self.attention = MPNetAttention(config)
        self.intermediate = MPNetIntermediate(config)
        self.output = MPNetOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        # 使用自注意力层处理隐藏状态，可能包括注意力掩码、头部掩码和位置偏置等参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，将其添加到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 通过中间层处理自注意力层的输出
        intermediate_output = self.intermediate(attention_output)
        # 通过输出层处理中间层的输出和自注意力层的输出
        layer_output = self.output(intermediate_output, attention_output)
        # 将层输出与可能存在的注意力权重输出合并到 outputs 中
        outputs = (layer_output,) + outputs
        return outputs


class MPNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        # 使用 MPNetLayer 构建编码器的多层堆叠，层数由 config.num_hidden_layers 决定
        self.layer = nn.ModuleList([MPNetLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化相对注意力偏置 Embedding，用于编码器中的每个注意头
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.n_heads)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        # 计算位置偏置，用于注意力机制中的相对位置编码
        position_bias = self.compute_position_bias(hidden_states)
        
        # 如果需要输出隐藏状态，则初始化一个空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        
        # 如果需要输出注意力权重，则初始化一个空元组用于存储所有注意力权重
        all_attentions = () if output_attentions else None
        
        # 遍历每个层次的 Transformer 层进行前向传播
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                # 如果需要输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 调用当前层的前向传播函数，获取当前层的输出
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                position_bias,
                output_attentions=output_attentions,
                **kwargs,
            )
            
            # 更新当前隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            if output_attentions:
                # 如果需要输出注意力权重，则将当前层的注意力权重加入到 all_attentions 中
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 的设置决定返回结果的形式
        if not return_dict:
            # 如果不需要返回一个字典，则返回包含非空元素的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        else:
            # 如果需要返回一个字典形式的结果，则使用 BaseModelOutput 构造结果并返回
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        # 获取输入张量 x 的尺寸信息
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        
        # 如果提供了位置 ids，则使用这些 ids；否则，创建默认的位置 ids
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        # 计算相对位置矩阵
        relative_position = memory_position - context_position

        # 将相对位置映射到固定数量的桶中
        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)
        rp_bucket = rp_bucket.to(x.device)
        
        # 获取相对注意力偏置值
        values = self.relative_attention_bias(rp_bucket)
        
        # 调整值的维度顺序以匹配注意力矩阵的形状要求
        values = values.permute([2, 0, 1]).unsqueeze(0)
        
        # 扩展值的维度以匹配输入张量 x 的尺寸
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        
        # 返回计算得到的位置偏置值
        return values

    @staticmethod
    # 定义函数 `relative_position_bucket`，计算相对位置的桶索引
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        # 初始化返回值为0
        ret = 0
        # 将相对位置取反
        n = -relative_position

        # 将桶的数量除以2，这是为了后面计算的方便
        num_buckets //= 2
        # 根据条件将ret加上一个整数值（0或num_buckets），使用torch的long类型
        ret += (n < 0).to(torch.long) * num_buckets
        # 取n的绝对值
        n = torch.abs(n)

        # 定义最大精确值为桶数的一半
        max_exact = num_buckets // 2
        # 判断n是否小于最大精确值
        is_small = n < max_exact

        # 如果n较大，计算大值的情况
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)

        # 将大值的情况限制在桶数减1以内
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        # 根据is_small的条件选择填入的值，返回最终的ret
        ret += torch.where(is_small, n, val_if_large)
        # 返回计算结果
        return ret
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Tensor containing the hidden states of the input sequences. Typically comes from the output of the
                last layer of the model.
        
        Returns:
            :obj:`torch.Tensor`: The pooled output tensor of shape :obj:`(batch_size, hidden_size)`.
                This tensor represents the "pooled" output, i.e., the output corresponding to the first token of each
                input sequence in the batch.
        """
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。使用 AutoTokenizer 可获取这些索引。
            # 参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 获取更多细节。
            # 什么是输入 ID？请参阅 ../glossary#input-ids

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于避免在填充标记索引上执行注意力计算。
            # 遮罩值在 [0, 1] 范围内选择：
            # - 1 表示不遮罩的标记，
            # - 0 表示遮罩的标记。
            # 什么是注意力遮罩？请参阅 ../glossary#attention-mask

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。选择范围为 [0, config.max_position_embeddings - 1]。
            # 什么是位置 ID？请参阅 ../glossary#position-ids

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置零的遮罩。
            # 遮罩值在 [0, 1] 范围内选择：
            # - 1 表示不遮罩的头部，
            # - 0 表示遮罩的头部。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，可以直接传递嵌入表示而不是传递 input_ids。如果需要对如何将 input_ids 索引转换为关联向量
            # 拥有更多控制权，这将非常有用，胜过于模型的内部嵌入查找矩阵。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 以获取更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 以获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回 `utils.ModelOutput` 而不是普通元组。
# 使用自定义的装饰器为模型类添加文档字符串，描述模型输出原始隐藏状态的特性
@add_start_docstrings(
    "The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.",
    MPNET_START_DOCSTRING,
)
# 定义 MPNetModel 类，继承自 MPNetPreTrainedModel 类
class MPNetModel(MPNetPreTrainedModel):
    # 初始化方法，接受配置参数和是否添加池化层的标志
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 存储配置信息
        self.config = config

        # 初始化嵌入层
        self.embeddings = MPNetEmbeddings(config)
        # 初始化编码器
        self.encoder = MPNetEncoder(config)
        # 如果设置了添加池化层的标志，则初始化池化层；否则设为 None
        self.pooler = MPNetPooler(config) if add_pooling_layer else None

        # 执行初始化后的处理
        self.post_init()

    # 返回输入嵌入的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头进行剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 使用装饰器添加前向传播方法的文档字符串，描述其输入参数
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例的文档字符串，描述前向传播方法的返回类型、检查点、输出配置
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法定义，接受多个输入参数和关键字参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        # 确定是否输出注意力权重，默认为配置文件中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认为配置文件中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的输出，默认为配置文件中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出 ValueError 异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 检查是否存在 padding 但未提供 attention_mask 的情况，如果有则发出警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 获取 inputs_embeds 的形状，排除最后一个维度（通常是 embedding 维度）
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出 ValueError 异常
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 确定使用的设备，如果 input_ids 存在则使用其设备，否则使用 inputs_embeds 的设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供 attention_mask，则创建一个全为 1 的张量作为默认 attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 获取扩展后的 attention_mask，确保其形状与输入数据匹配
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # 获取头部遮罩，用于控制每层的注意力头部是否生效
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 使用 embeddings 层生成嵌入向量
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        # 将嵌入向量输入到编码器（encoder）中，得到编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从编码器的输出中获取序列输出（sequence_output）
        sequence_output = encoder_outputs[0]
        # 如果存在池化器（pooler），则对序列输出进行池化操作，得到池化输出（pooled_output）
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不要求返回字典形式的输出，则返回元组形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典形式的输出，则构建 BaseModelOutputWithPooling 对象并返回
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class MPNetForMaskedLM(MPNetPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder"]

    def __init__(self, config):
        super().__init__(config)

        # 初始化 MPNet 模型，不添加池化层
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # 初始化 MPNetLMHead，即语言模型头部
        self.lm_head = MPNetLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回 lm_head 的 decoder，用于输出嵌入
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置 lm_head 的 decoder，用于更新输出嵌入
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
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
        """
        # 确定是否使用返回字典，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 MPNet 模型
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出并通过 lm_head 进行预测
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 如果提供了标签，则计算掩码语言建模损失
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不使用返回字典，则输出包含预测分数和其他输出的元组
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果使用返回字典，则返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    """MPNet Head for masked and permuted language modeling."""
    
    # 定义一个名为 MPNet 的类，用于处理掩码和置换语言建模的头部任务
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征的大小映射到隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个层归一化层，对隐藏大小的特征进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
        # 创建一个线性层，将隐藏大小映射到词汇表大小，没有偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 创建一个偏置参数，大小为词汇表大小
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
        # 需要一个链接来保证偏置能够在 `resize_token_embeddings` 中正确调整大小
        self.decoder.bias = self.bias
    
    # 定义前向传播方法，处理输入特征和其他关键字参数
    def forward(self, features, **kwargs):
        # 将特征输入到全连接层
        x = self.dense(features)
        # 应用 GELU 激活函数
        x = gelu(x)
        # 应用层归一化
        x = self.layer_norm(x)
    
        # 使用线性层将特征映射回词汇表大小，并加上偏置
        x = self.decoder(x)
    
        # 返回结果张量 x
        return x
# 使用装饰器添加模型的文档字符串，描述该模型是基于 MPNet 的序列分类/回归模型，例如用于 GLUE 任务
@add_start_docstrings(
    """
    MPNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MPNET_START_DOCSTRING,  # 添加 MPNet 的起始文档字符串
)
# 定义 MPNetForSequenceClassification 类，继承自 MPNetPreTrainedModel
class MPNetForSequenceClassification(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 从配置中获取类别数目并赋值给对象属性
        self.num_labels = config.num_labels
        # 使用 MPNetModel 初始化 MPNet 模型，不添加池化层
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # 使用 MPNetClassificationHead 初始化分类器
        self.classifier = MPNetClassificationHead(config)

        # 执行初始化权重和最终处理
        self.post_init()

    # 使用装饰器添加 forward 方法的文档字符串，描述输入参数及其作用
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例的文档字符串，指定相关的检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 forward 方法，接收多个输入参数并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数描述完毕，没有注释的代码
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        
        # 初始化返回字典，若未提供则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MPNet 模型处理输入数据
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        
        # 使用分类器层生成 logits（分类预测）
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        
        # 如果提供了标签
        if labels is not None:
            # 根据模型配置决定问题类型（回归、单标签分类、多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类，计算交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类，计算带 logits 的二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不需要返回字典形式的结果，则只返回 logits 和其它附加输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # 返回 SequenceClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MPNET_START_DOCSTRING,
)
class MPNetForMultipleChoice(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 MPNet 模型
        self.mpnet = MPNetModel(config)
        # 使用指定的隐藏单元丢弃概率初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化分类器线性层，输入大小为隐藏状态的大小，输出大小为1（用于二分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        
        # Determine whether to use the provided return_dict or the default from configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Determine the number of choices (second dimension size) based on input_ids or inputs_embeds
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    
        # Flatten input tensors if not None to prepare for model input
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
    
        # Pass the flattened inputs to the model, along with other optional arguments
        outputs = self.mpnet(
            flat_input_ids,
            position_ids=flat_position_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Extract the pooled output (typically pooled for classification tasks)
        pooled_output = outputs[1]
    
        # Apply dropout to the pooled output
        pooled_output = self.dropout(pooled_output)
        
        # Pass the pooled output through the classifier to get logits
        logits = self.classifier(pooled_output)
        
        # Reshape logits to match the shape required for multiple choice tasks
        reshaped_logits = logits.view(-1, num_choices)
    
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
    
        # Prepare output based on return_dict flag
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]  # Include additional model outputs if not using return_dict
            return ((loss,) + output) if loss is not None else output
        else:
            # Return as a MultipleChoiceModelOutput object if return_dict is True
            return MultipleChoiceModelOutput(
                loss=loss,
                logits=reshaped_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
@add_start_docstrings(
    """
    MPNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    MPNET_START_DOCSTRING,
)
class MPNetForTokenClassification(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize MPNet model with no pooling layer
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """

        # Determine whether to return as dictionary based on configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through MPNet model
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Apply dropout to the output of the MPNet model
        sequence_output = self.dropout(sequence_output)
        # Classify the sequence output using a linear layer
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Calculate cross-entropy loss if labels are provided
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # Return output as a tuple if return_dict is False
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return TokenClassifierOutput named tuple if return_dict is True
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def __init__(self, config):
        super().__init__()
        # 使用 config 中的 hidden_size 参数定义一个全连接层，输入和输出维度都是 hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用 config 中的 hidden_dropout_prob 参数定义一个 dropout 层，用于在训练时随机置零输入张量的元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 使用 config 中的 hidden_size 和 num_labels 参数定义一个全连接层，将 hidden_size 维度的输入映射到 num_labels 维度的输出
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 从 features 张量中取出所有样本的第一个位置的向量，通常用于获取整体序列的表示（例如 BERT 的 [CLS] token 表示整个序列）
        x = features[:, 0, :]
        # 对取出的向量应用 dropout 操作，以防止过拟合
        x = self.dropout(x)
        # 将经过 dropout 后的向量输入到全连接层 dense 中进行线性变换
        x = self.dense(x)
        # 对 dense 层的输出应用 tanh 激活函数，增加模型的非线性特性
        x = torch.tanh(x)
        # 再次对经过 tanh 激活后的向量应用 dropout 操作
        x = self.dropout(x)
        # 最后将经过两次 dropout 和一次全连接后得到的向量输入到全连接层 out_proj 中，得到最终的输出
        x = self.out_proj(x)
        # 返回神经网络的前向传播结果
        return x
# 定义一个 MPNet 问题回答模型，用于像 SQuAD 这样的抽取式问答任务，在隐藏状态输出的基础上增加了一个用于计算“起始位置 logits”和“结束位置 logits”的线性层。
@add_start_docstrings(
    """
    MPNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MPNET_START_DOCSTRING,
)
class MPNetForQuestionAnswering(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设置标签数等于配置文件中的标签数
        self.num_labels = config.num_labels
        # 初始化 MPNet 模型，不添加池化层
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # QA 输出层，使用线性变换将隐藏状态大小转换为标签数
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
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

        # 根据返回值的需求确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法，获取输出
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入问答输出层得到 logits
        logits = self.qa_outputs(sequence_output)
        
        # 将 logits 按最后一个维度分割成起始位置和结束位置 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # 去除多余的维度并保持连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # 如果提供了起始位置和结束位置，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果是多 GPU 训练，需要添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 忽略超出模型输入范围的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始位置和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典形式的输出，则返回原始的输出元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果需要返回字典形式的输出，则构造 QuestionAnsweringModelOutput 对象并返回
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的输入 ID 中创建位置 ID。非填充符号被替换为它们的位置号码。位置号码从 padding_idx+1 开始计算，填充符号被忽略。
# 这是从 fairseq 的 `utils.make_positions` 修改而来。
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`. :param torch.Tensor x: :return torch.Tensor:
    """
    # 创建一个掩码，标记出非填充符号的位置为1，填充符号位置为0
    mask = input_ids.ne(padding_idx).int()
    # 计算每个位置上非填充符号的累积数量，并将结果转换为与 mask 张量相同的数据类型
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    # 将累积的位置索引转换为 long 类型，并加上 padding_idx，以得到最终的位置 ID
    return incremental_indices.long() + padding_idx
```