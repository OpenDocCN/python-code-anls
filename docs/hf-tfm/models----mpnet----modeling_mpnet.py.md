# `.\transformers\models\mpnet\modeling_mpnet.py`

```py
# 设定编码格式为 UTF-8
# 版权声明：HuggingFace Inc.团队，Microsoft Corporation，NVIDIA CORPORATION。保留所有权利。
# 根据 Apache 许可证 2.0 版本使用本文件
# 你可以在遵守许可证的前提下使用本文件
# 可以在以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“原样”分发的，没有任何明示或暗示的保证或条件。
# 请参阅许可证以了解特定语言的权限和限制。
"""PyTorch MPNet 模型。"""


# 导入需要的库
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数和模型输出
from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型工具函数和配置
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_mpnet import MPNetConfig


# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "microsoft/mpnet-base"
_CONFIG_FOR_DOC = "MPNetConfig"

# MPNet 模型的预训练模型存档列表
MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/mpnet-base",
]


# MPNet 预训练模型的基类
class MPNetPreTrainedModel(PreTrainedModel):
    # MPNet 模型的配置类
    config_class = MPNetConfig
    # MPNet 预训练模型的存档映射
    pretrained_model_archive_map = MPNET_PRETRAINED_MODEL_ARCHIVE_LIST
    # MPNet 模型的基类前缀
    base_model_prefix = "mpnet"

    # 初始化权重
    def _init_weights(self, module):
        """初始化权重"""
        # 如果是线性层，使用标准正态分布初始化权重和偏置
        if isinstance(module, nn.Linear):
            # 和 TF 版本略有不同，TF 版本使用截断的正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层，使用标准正态分布初始化权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层，初始化偏置为零，权重为一
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# MPNet 模型的嵌入层
class MPNetEmbeddings(nn.Module):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 设置填充索引
        self.padding_idx = 1
        # 创建词嵌入层对象
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # 创建位置嵌入层对象
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # 创建 LayerNorm 层对象
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建丢弃层对象
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 注册位置 ID 缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播方法
    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
        # 如果未指定位置 ID
        if position_ids is None:
            # 如果输入 ID 不为空
            if input_ids is not None:
                # 根据输入 ID 创建位置 ID
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                # 根据输入的嵌入数据创建位置 ID
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入 ID 不为空
        if input_ids is not None:
            # 获取输入形状
            input_shape = input_ids.size()
        else:
            # 获取嵌入数据的形状
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置 ID 为空
        if position_ids is None:
            # 使用预设的位置 ID
            position_ids = self.position_ids[:, :seq_length]

        # 如果嵌入数据为空
        if inputs_embeds is None:
            # 使用词嵌入层获取嵌入数据
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入数据
        position_embeddings = self.position_embeddings(position_ids)

        # 合并词嵌入数据和位置嵌入数据
        embeddings = inputs_embeds + position_embeddings
        # 运行 LayerNorm 层
        embeddings = self.LayerNorm(embeddings)
        # 运行丢弃层
        embeddings = self.dropout(embeddings)
        # 返回嵌入数据
        return embeddings

    # 从输入嵌入数据创建位置 ID 的方法
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入数据形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成位置 ID 序列
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
class MPNetSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建线性层 q，用于将输入的 hidden_states 映射到所有 attention heads 的维度
        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 创建线性层 k，用于将输入的 hidden_states 映射到所有 attention heads 的维度
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 创建线性层 v，用于将输入的 hidden_states 映射到所有 attention heads 的维度
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 创建线性层 o，用于将计算得到的 attention 结果映射回原始的 hidden_states 维度
        self.o = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 创建 dropout 层，用于在计算 attention probabilities 时进行随机失活
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量 x 转换为 attention scores 对应的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        # 计算 q、k、v 的线性映射结果
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # 将线性映射结果转换为对应的 attention scores 形状
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 计算 "query" 和 "key" 之间的点积，得到原始的 attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果提供了 position_bias，将其加到 attention_scores 中
        if position_bias is not None:
            attention_scores += position_bias

        # 如果提供了 attention_mask，则将其加到 attention_scores 中
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 将 attention_scores 归一化为概率形式
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 在计算 attention_probs 时使用 dropout 进行随机失活
        attention_probs = self.dropout(attention_probs)

        # 如果提供了 head_mask，则将其应用到 attention_probs 上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算最终的 context 向量 c
        c = torch.matmul(attention_probs, v)

        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.all_head_size,)
        c = c.view(*new_c_shape)

        # 将 context 向量 c 映射回原始 hidden_states 的维度
        o = self.o(c)

        # 如果需要输出 attention 的话，则将 attention_probs 一同返回
        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs


class MPNetAttention(nn.Module):
    def __init__(self, config):
        # 继承父类的初始化方法
        super().__init__()
        # 创建 MPNetSelfAttention 对象
        self.attn = MPNetSelfAttention(config)
        # 创建 LayerNorm 对象
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 对象
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 保存被剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 如果要剪枝的注意力头为空，则直接返回
        if len(heads) == 0:
            return
        
        # 根据要剪枝的注意力头和当前模型的头数量和头尺寸，寻找可剪枝的头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attn.num_attention_heads, self.attn.attention_head_size, self.pruned_heads
        )

        # 剪枝注意力层中 query、key、value、output 的线性层
        self.attn.q = prune_linear_layer(self.attn.q, index)
        self.attn.k = prune_linear_layer(self.attn.k, index)
        self.attn.v = prune_linear_layer(self.attn.v, index)
        self.attn.o = prune_linear_layer(self.attn.o, index, dim=1)

        # 更新剩余的注意力头数量和头总尺寸
        self.attn.num_attention_heads = self.attn.num_attention_heads - len(heads)
        self.attn.all_head_size = self.attn.attention_head_size * self.attn.num_attention_heads
        # 更新剪枝后的注意力头集合
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
        **kwargs,
    ):
        # 调用 MPNetSelfAttention 的 forward 方法，计算自注意力
        self_outputs = self.attn(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias,
            output_attentions=output_attentions,
        )
        # 应用 LayerNorm、Dropout 并将注意力输出与原始输入相加
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + hidden_states)
        # 将注意力输出和其它输出（如果有）作为结果返回
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate中复制而来的MPNetIntermediate类
class MPNetIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度是config.hidden_size，输出维度是config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则使用ACT2FN字典中对应的激活函数；否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层操作，将hidden_states作为输入，输出经过全连接后的结果
        hidden_states = self.dense(hidden_states)
        # 经过激活函数处理后的结果
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput中复制而来的MPNetOutput类
class MPNetOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度是config.intermediate_size，输出维度是config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入维度是config.hidden_size，epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，丢弃率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层操作，将hidden_states作为输入，输出经过全连接后的结果
        hidden_states = self.dense(hidden_states)
        # dropout操作
        hidden_states = self.dropout(hidden_states)
        # LayerNorm操作，对全连接结果和输入进行残差连接，并进行LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MPNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个MPNetAttention层
        self.attention = MPNetAttention(config)
        # 创建一个MPNetIntermediate层
        self.intermediate = MPNetIntermediate(config)
        # 创建一个MPNetOutput层
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
        # 调用self.attention进行注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加到outputs中

        # 经过中间层处理
        intermediate_output = self.intermediate(attention_output)
        # 经过输出层处理
        layer_output = self.output(intermediate_output, attention_output)
        # 将本层的输出加入outputs中
        outputs = (layer_output,) + outputs
        return outputs


class MPNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 获取配置信息
        self.config = config
        # 获取注意力头数
        self.n_heads = config.num_attention_heads
        # 创建一个由config.num_hidden_layers个MPNetLayer组成的ModuleList
        self.layer = nn.ModuleList([MPNetLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建一个相对位置编码的Embedding层，维度为config.relative_attention_num_buckets * self.n_heads
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.n_heads)
    # 前向传播函数，接受隐藏状态、注意力掩码、头部掩码等参数，并返回模型输出
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
        # 计算位置偏置
        position_bias = self.compute_position_bias(hidden_states)
        # 初始化存储隐藏状态和注意力的变量
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，保存当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的前向传播函数
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                position_bias,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力，保存当前层的注意力
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，按顺序返回对应的值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 以模型输出对象形式返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    # 计算位置偏置，用于生成相对位置编码
    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        # 获取输入张量的尺寸信息
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        # 根据位置 ID 计算相对位置
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position

        # 将相对位置映射到预定义的桶中
        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)
        rp_bucket = rp_bucket.to(x.device)
        # 获取相对位置偏置值，重塑形状并扩展张量
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

    # 静态方法
    @staticmethod
    # 定义一个函数，用于将相对位置映射到指定数量和距离范围内的桶中
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        # 初始化结果
        ret = 0
        # 将相对位置取反，转换为正值
        n = -relative_position
    
        # 指定桶的数量为一半
        num_buckets //= 2
        # 如果相对位置为负数，将结果加上桶的数量一半
        ret += (n < 0).to(torch.long) * num_buckets
        # 取相对位置的绝对值
        n = torch.abs(n)
    
        # 计算最大精确值
        max_exact = num_buckets // 2
        # 判断相对位置是否小于最大精确值
        is_small = n < max_exact
    
        # 如果相对位置较大，进行非线性映射
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
    
        # 确保映射值不超过桶的数量减一
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        # 根据相对位置大小选择映射结果或非线性映射结果
        ret += torch.where(is_small, n, val_if_large)
        # 返回最终结果
        return ret
# 从transformers.models.bert.modeling_bert.BertPooler中复制了代码，定义了一个MPNetPooler类
class MPNetPooler(nn.Module):
    # 初始化函数，接受一个配置参数config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 定义一个全连接层，输入和输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个激活函数为Tanh
        self.activation = nn.Tanh()

    # 前向传播函数，接受一个torch.Tensor类型的hidden_states作为输入，返回一个torch.Tensor类型的输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记对应的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数处理全连接层的输出
        pooled_output = self.activation(pooled_output)
        # 返回处理后的池化输出
        return pooled_output


# MPNet模型的文档字符串的起始部分
MPNET_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MPNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# MPNet模型输入的文档字符串
MPNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。有关详情，请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充标记索引执行注意力的掩码。掩码值选在 `[0, 1]` 之间：
            # - 1 表示**未被掩蔽**的标记，
            # - 0 表示**被掩蔽**的标记。

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 置空自注意力模块中选择的头部的掩码。掩码值选在 `[0, 1]` 之间：
            # - 1 表示**未被掩蔽**的头部，
            # - 0 表示**被掩蔽**的头部。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示，而不是传递 `input_ids`，这在想要更好地控制如何将 *input_ids* 索引转换为关联向量时很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详情，请参见返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详情，请参见返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是纯元组。
# 定义一个 MPNet 模型类，继承自 MPNetPreTrainedModel
@add_start_docstrings(
    "The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.",  # 添加模型说明
    MPNET_START_DOCSTRING,  # 添加 MPNet 文档字符串
)
class MPNetModel(MPNetPreTrainedModel):  # 声明 MPNetModel 类，继承自 MPNetPreTrainedModel
    def __init__(self, config, add_pooling_layer=True):  # 初始化方法，接受配置和是否添加池化层参数
        super().__init__(config)  # 调用父类初始化方法
        self.config = config  # 设置配置属性

        self.embeddings = MPNetEmbeddings(config)  # 创建 MPNetEmbeddings 对象
        self.encoder = MPNetEncoder(config)  # 创建 MPNetEncoder 对象
        self.pooler = MPNetPooler(config) if add_pooling_layer else None  # 创建 MPNetPooler 对象，若添加池化层，则创建，否则为 None

        # Initialize weights and apply final processing
        self.post_init()  # 调用初始化权重和应用最终处理方法

    def get_input_embeddings(self):  # 获取输入嵌入方法
        return self.embeddings.word_embeddings  # 返回嵌入词嵌入属性

    def set_input_embeddings(self, value):  # 设置输入嵌入方法
        self.embeddings.word_embeddings = value  # 设置嵌入词嵌入属性为指定值

    def _prune_heads(self, heads_to_prune):  # 剪枝头部方法
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():  # 遍历需要剪枝的头部
            self.encoder.layer[layer].attention.prune_heads(heads)  # 对指定层的注意力头部进行剪枝

    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))  # 添加输入文档说明
    @add_code_sample_docstrings(  # 添加代码示例文档说明
        checkpoint=_CHECKPOINT_FOR_DOC,  # 添加检查点文档说明
        output_type=BaseModelOutputWithPooling,  # 添加输出类型文档说明
        config_class=_CONFIG_FOR_DOC,  # 添加配置文档说明
    )
    def forward(  # 前向传播方法
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的索引序列
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩
        position_ids: Optional[torch.LongTensor] = None,  # 位置索引
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入
        output_attentions: Optional[bool] = None,  # 输出注意力
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
        **kwargs,  # 关键字参数
    # 定义一个方法，接受输入参数并返回一个torch张量或者带有池化的基础模型输出
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        # 如果没有指定output_attentions，则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有指定output_hidden_states，则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有指定return_dict，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果input_ids和inputs_embeds都不为None，则引发异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果input_ids不为None
        elif input_ids is not None:
            # 如果存在填充和没有注意力掩码则发出警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取input_ids的形状
            input_shape = input_ids.size()
        # 如果inputs_embeds不为None
        elif inputs_embeds is not None:
            # 获取inputs_embeds的形状
            input_shape = inputs_embeds.size()[:-1]
        # 否则
        else:
            # 抛出数值错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取input_ids的设备（如果存在）或者inputs_embeds的设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果attention_mask为None，则创建一个全1的注意力掩码张量
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 得到扩展后的注意力掩码张量
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # 获取头部掩码
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 获取嵌入输出
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        # 获取编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = encoder_outputs[0]
        # 如果存在池化层，则获取池化输出
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不返回字典
        if not return_dict:
            # 返回元组
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 返回带有池化的基础模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个 MPNetForMaskedLM 类，继承自 MPNetPreTrainedModel 类
class MPNetForMaskedLM(MPNetPreTrainedModel):
    # 定义一个私有变量，包含 lm_head.decoder，用于表示权重共享的键值
    _tied_weights_keys = ["lm_head.decoder"]

    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 MPNetModel 对象，并设置 add_pooling_layer 参数为 False
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # 创建一个 MPNetLMHead 对象
        self.lm_head = MPNetLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数，返回一个元组或 MaskedLMOutput 对象
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
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 mpnet 的前向传播函数，获取输出
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

        # 获取序列输出和预测分数
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # 初始化 masked_lm_loss 为 None
        masked_lm_loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算 masked_lm_loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回额外的输出信息
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 定义 MPNetLMHead 类
class MPNetLMHead(nn.Module):
    """MPNet Head for masked and permuted language modeling."""

    # 初始化 MPNet Head，用于掩码和置换语言建模
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度和输出维度都是隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于对隐藏层进行正则化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建一个线性层，将隐藏层映射回词汇表大小，不含偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 创建一个 bias 参数，用于偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要对这两个变量进行关联，以便在调整 token embeddings 时正确调整偏置
        self.decoder.bias = self.bias

    # 正向传播函数
    def forward(self, features, **kwargs):
        # 将输入 features 输入至 dense 线性层
        x = self.dense(features)
        # 经过激活函数 gelu 处理
        x = gelu(x)
        # 对 x 进行 LayerNorm
        x = self.layer_norm(x)

        # 使用 decoder 线性层将 x 投影回词汇表大小，并添加偏置
        x = self.decoder(x)

        # 返回结果
        return x
# 在 MPNet 模型的基础上增加一个顶部的序列分类/回归头部（在池化输出的顶部增加了一个线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    MPNet 模型的变压器，顶部带有一个序列分类/回归头部（在池化输出的顶部是一个线性层），例如用于 GLUE 任务。
    """,
    MPNET_START_DOCSTRING,
)
class MPNetForSequenceClassification(MPNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 获取标签数目
        self.num_labels = config.num_labels
        # 创建 MPNet 模型
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # 创建分类器
        self.classifier = MPNetClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将前向传播的注释添加到模型的前向传播中
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例注释
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否返回字典格式的输出，如果未指定则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过 MPNet 进行模型的前向传播
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 通过分类器获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果问题类型未指定
            if self.config.problem_type is None:
                # 根据标签数量确定问题类型
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
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
        
        # 如果不返回字典格式的输出
        if not return_dict:
            # 构造输出元组，包括 logits 和其他输出（如果有），以及损失（如果有）
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典格式的输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 在 MPNetForMultipleChoice 类上添加 start_docstrings
@add_start_docstrings(
    """
    MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MPNET_START_DOCSTRING,
)
class MPNetForMultipleChoice(MPNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)

        # 创建 MPNetModel 实例
        self.mpnet = MPNetModel(config)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建用于分类的线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的 forward 方法添加 start_docstrings 和 code_sample_docstrings
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # 输入 IDs
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码
        attention_mask: Optional[torch.FloatTensor] = None,
        # 位置 IDs
        position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签
        labels: Optional[torch.LongTensor] = None,
        # 是否输出注意力
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """

        # 检查是否需要返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入 tensors 的第二维度大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的 input_ids 展开成二维张量
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将输入的 position_ids 展开成二维张量
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将输入的 attention_mask 展开成二维张量
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将输入的 inputs_embeds 展开成三维张量
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用模型处理展开后的输入数据
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
        # 获取汇聚后的输出
        pooled_output = outputs[1]

        # 对汇聚后的输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器得出 logits
        logits = self.classifier(pooled_output)
        # 重新调整 logits 的形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果有标签，则计算损失值
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回字典，并且损失值不为空，则返回 loss 和 output
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则返回 MultipleChoiceModelOutput 格式的结果
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
# 定义 MPNet 用于标记分类任务的模型类
class MPNetForTokenClassification(MPNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)
        # 初始化标签数量
        self.num_labels = config.num_labels

        # 初始化 MPNet 模型，不添加池化层
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # 初始化丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
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

        # 如果未指定返回字典，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 MPNet 模型的前向传播
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

        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出应用丢弃层
        sequence_output = self.dropout(sequence_output)
        # 通过分类器线性层获取分类 logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果存在标签，则计算分类损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典，则组装输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    # 类的初始化方法，传入配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入和输出的维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个丢弃层，以config.hidden_dropout_prob作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，输入维度是config.hidden_size，输出维度是config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    # 前向传播方法，接受features作为输入，kwargs是其他参数
    def forward(self, features, **kwargs):
        # 取出features的第一个元素的全部元素，相当于取出<BOS>标记对应的张量（类似BERT的[CLS]标记）
        x = features[:, 0, :]  # take <s> token (equiv. to BERT's [CLS] token)
        # 对x进行丢弃操作
        x = self.dropout(x)
        # 将x传入全连接层
        x = self.dense(x)
        # 对x进行tanh激活
        x = torch.tanh(x)
        # 对x进行丢弃操作
        x = self.dropout(x)
        # 将x传入输出全连接层
        x = self.out_proj(x)
        # 返回结果
        return x
# 为 MPNetForQuestionAnswering 类添加详细文档注释
@add_start_docstrings(
    """
    MPNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MPNET_START_DOCSTRING,
)
class MPNetForQuestionAnswering(MPNetPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建 MPNetModel 对象
        self.mpnet = MPNetModel(config, add_pooling_layer=False)
        # 创建线性层，用于计算 span 开始和结束的 logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为前向传播函数添加详细文档注释
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        # 输入 token 的序列
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # 输入注意力掩码
        attention_mask: Optional[torch.FloatTensor] = None,
        # 输入位置 id
        position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入向量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 起始位置
        start_positions: Optional[torch.LongTensor] = None,
        # 结束位置
        end_positions: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用mpnet模型进行预测
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

        # 获取问题回答的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果处于多GPU环境，则添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时候起始/结束位置在模型输入之外，这些位置会被忽略
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回answer模型的输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的id中创建位置id，非填充符号替换为它们的位置数字。位置数字从padding_idx+1开始。忽略填充符号。这是从fairseq的`utils.make_positions`修改而来。
def create_position_ids_from_input_ids(input_ids, padding_idx):
    # 这里的一系列类型转换和转换非常谨慎地平衡，既可以与ONNX导出一起工作，也可以与XLA一起工作。
    # 创建一个mask来表示哪些位置不是填充符号，转换成整型
    mask = input_ids.ne(padding_idx).int()
    # 逐层累加每一行的mask的非填充位置，并转换为和mask相同的类型，乘以mask
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    # 返回加上padding_idx的long类型的递增索引
    return incremental_indices.long() + padding_idx
```