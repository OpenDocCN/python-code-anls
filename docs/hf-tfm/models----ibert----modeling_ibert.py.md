# `.\models\ibert\modeling_ibert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者信息和许可证信息
# 本代码基于 Apache License, Version 2.0 许可证发布
# 详细许可证信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
# 本代码根据适用法律或书面同意分发，基于"AS IS"基础分发，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

"""PyTorch I-BERT model."""

# 导入所需的库和模块
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关的模型输出类和工具类
from ...activations import gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "kssteven/ibert-roberta-base"
_CONFIG_FOR_DOC = "IBertConfig"

# 预训练模型存档列表
IBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kssteven/ibert-roberta-base",
    "kssteven/ibert-roberta-large",
    "kssteven/ibert-roberta-large-mnli",
]

# 定义 IBertEmbeddings 类，与 BertEmbeddings 类相同，但对位置嵌入索引进行微小调整
class IBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 设置量化模式
        self.quant_mode = config.quant_mode
        # 设置嵌入位数
        self.embedding_bit = 8
        # 设置嵌入激活位数
        self.embedding_act_bit = 16
        # 设置激活位数
        self.act_bit = 8
        # 设置输入 LayerNorm 位数
        self.ln_input_bit = 22
        # 设置输出 LayerNorm 位数
        self.ln_output_bit = 32

        # 初始化词嵌入层
        self.word_embeddings = QuantEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )
        # 初始化 token_type 嵌入层
        self.token_type_embeddings = QuantEmbedding(
            config.type_vocab_size, config.hidden_size, weight_bit=self.embedding_bit, quant_mode=self.quant_mode
        )

        # 创建 position_ids 缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入类型
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # 初始化位置嵌入层
        self.padding_idx = config.pad_token_id
        self.position_embeddings = QuantEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
            weight_bit=self.embedding_bit,
            quant_mode=self.quant_mode,
        )

        # 初始化嵌入层激活函数1
        self.embeddings_act1 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        # 初始化嵌入层激活函数2
        self.embeddings_act2 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)

        # 设置 LayerNorm，保持与 TensorFlow 模型变量名一致
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        # 初始化输出激活函数
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
        # 如果位置 id 为 None
        if position_ids is None:
            # 如果输入 id 不为 None
            if input_ids is not None:
                # 从输入 token id 创建位置 id。任何填充的 token 保持填充状态。
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                # 从输入嵌入创建位置 id
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入 id 不为 None
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 如果 token 类型 id 为 None
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为 None
        if inputs_embeds is None:
            # 使用 word_embeddings 方法获取输入嵌入和缩放因子
            inputs_embeds, inputs_embeds_scaling_factor = self.word_embeddings(input_ids)
        else:
            inputs_embeds_scaling_factor = None
        # 获取 token 类型嵌入和缩放因子
        token_type_embeddings, token_type_embeddings_scaling_factor = self.token_type_embeddings(token_type_ids)

        # 获取嵌入和缩放因子
        embeddings, embeddings_scaling_factor = self.embeddings_act1(
            inputs_embeds,
            inputs_embeds_scaling_factor,
            identity=token_type_embeddings,
            identity_scaling_factor=token_type_embeddings_scaling_factor,
        )

        # 如果位置嵌入类型为 "absolute"
        if self.position_embedding_type == "absolute":
            # 获取位置嵌入和缩放因子
            position_embeddings, position_embeddings_scaling_factor = self.position_embeddings(position_ids)
            # 获取嵌入和缩放因子
            embeddings, embeddings_scaling_factor = self.embeddings_act1(
                embeddings,
                embeddings_scaling_factor,
                identity=position_embeddings,
                identity_scaling_factor=position_embeddings_scaling_factor,
            )

        # 获取 LayerNorm 后的嵌入和缩放因子
        embeddings, embeddings_scaling_factor = self.LayerNorm(embeddings, embeddings_scaling_factor)
        # 对嵌入进行 dropout
        embeddings = self.dropout(embeddings)
        # 获取输出激活后的嵌入和缩放因子
        embeddings, embeddings_scaling_factor = self.output_activation(embeddings, embeddings_scaling_factor)
        # 返回嵌入和缩放因子
        return embeddings, embeddings_scaling_factor

    # 从输入嵌入创建位置 id
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入形状
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 生成顺序位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 定义 IBertSelfAttention 类，继承自 nn.Module
class IBertSelfAttention(nn.Module):
    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，并且配置中没有嵌入大小属性，则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置量化模式、权重位数、偏置位数和激活位数
        self.quant_mode = config.quant_mode
        self.weight_bit = 8
        self.bias_bit = 32
        self.act_bit = 8

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义 Q, K, V 的线性层
        self.query = QuantLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.key = QuantLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        self.value = QuantLinear(
            config.hidden_size,
            self.all_head_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )

        # 定义 Q, K, V 激活的重新量化层
        self.query_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.key_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.value_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)

        # 定义 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型不是绝对位置嵌入，则抛出数值错误
        if self.position_embedding_type != "absolute":
            raise ValueError("I-BERT only supports 'absolute' for `config.position_embedding_type`")

        # 定义 Softmax 层
        self.softmax = IntSoftmax(self.act_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)

    # 将输入张量转换为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
class IBertSelfOutput(nn.Module):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 设置量化模式
        self.quant_mode = config.quant_mode
        # 设置激活位数
        self.act_bit = 8
        # 设置权重位数
        self.weight_bit = 8
        # 设置偏置位数
        self.bias_bit = 32
        # 设置输入层归一化位数
        self.ln_input_bit = 22
        # 设置输出层归一化位数
        self.ln_output_bit = 32

        # 创建量化线性层对象
        self.dense = QuantLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        # 创建输入层激活量化对象
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        # 创建整数层归一化对象
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        # 创建输出层激活量化对象
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 创建丢弃层对象
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        # 线性层前向传播
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        # 执行丢弃操作
        hidden_states = self.dropout(hidden_states)
        # 输入层激活量化
        hidden_states, hidden_states_scaling_factor = self.ln_input_act(
            hidden_states,
            hidden_states_scaling_factor,
            identity=input_tensor,
            identity_scaling_factor=input_tensor_scaling_factor,
        )
        # 整数层归一化
        hidden_states, hidden_states_scaling_factor = self.LayerNorm(hidden_states, hidden_states_scaling_factor)

        # 输出层激活量化
        hidden_states, hidden_states_scaling_factor = self.output_activation(
            hidden_states, hidden_states_scaling_factor
        )
        # 返回隐藏状态和缩放因子
        return hidden_states, hidden_states_scaling_factor
class IBertAttention(nn.Module):
    # 定义 IBertAttention 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化函数
        self.quant_mode = config.quant_mode
        # 设置 quant_mode 属性为传入的 config 的 quant_mode 属性
        self.self = IBertSelfAttention(config)
        # 创建 IBertSelfAttention 对象并赋值给 self.self
        self.output = IBertSelfOutput(config)
        # 创建 IBertSelfOutput 对象并赋值给 self.output
        self.pruned_heads = set()
        # 创建一个空集合用于存储被剪枝的头部

    def prune_heads(self, heads):
        # 定义 prune_heads 方法，接受 heads 参数
        if len(heads) == 0:
            return
        # 如果 heads 长度为 0，则直接返回
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # 调用 find_pruneable_heads_and_indices 函数，获取可剪枝的头部和索引

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        # 剪枝 self.query 线性层
        self.self.key = prune_linear_layer(self.self.key, index)
        # 剪枝 self.key 线性层
        self.self.value = prune_linear_layer(self.self.value, index)
        # 剪枝 self.value 线性层
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # 剪枝 self.output.dense 线性层

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        # 更新 num_attention_heads 属性
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        # 更新 all_head_size 属性
        self.pruned_heads = self.pruned_heads.union(heads)
        # 将剪枝的头部添加到 pruned_heads 集合中

    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 定义前向传播方法，接受多个参数
        self_outputs, self_outputs_scaling_factor = self.self(
            hidden_states,
            hidden_states_scaling_factor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 调用 self 对象的前向传播方法
        attention_output, attention_output_scaling_factor = self.output(
            self_outputs[0], self_outputs_scaling_factor[0], hidden_states, hidden_states_scaling_factor
        )
        # 调用 output 对象的前向传播方法
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 将 attention_output 和 self_outputs 的其余部分组成元组
        outputs_scaling_factor = (attention_output_scaling_factor,) + self_outputs_scaling_factor[1:]
        # 将 attention_output_scaling_factor 和 self_outputs_scaling_factor 的其余部分组成元组
        return outputs, outputs_scaling_factor
        # 返回结果元组


class IBertIntermediate(nn.Module):
    # 定义 IBertIntermediate 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化函数
        self.quant_mode = config.quant_mode
        # 设置 quant_mode 属性为传入的 config 的 quant_mode 属性
        self.act_bit = 8
        # 设置 act_bit 属性为 8
        self.weight_bit = 8
        # 设置 weight_bit 属性为 8
        self.bias_bit = 32
        # 设置 bias_bit 属性为 32
        self.dense = QuantLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        # 创建 QuantLinear 对象并赋值给 self.dense
        if config.hidden_act != "gelu":
            raise ValueError("I-BERT only supports 'gelu' for `config.hidden_act`")
        # 如果 config 的 hidden_act 不是 "gelu"，则抛出异常
        self.intermediate_act_fn = IntGELU(quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        # 创建 IntGELU 对象并赋值给 self.intermediate_act_fn
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 创建 QuantAct 对象并赋值给 self.output_activation
    # 前向传播函数，接受隐藏状态和隐藏状态缩放因子作为输入
    def forward(self, hidden_states, hidden_states_scaling_factor):
        # 使用全连接层对隐藏状态进行处理，并更新隐藏状态缩放因子
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        # 使用激活函数对处理后的隐藏状态进行处理，并更新隐藏状态缩放因子
        hidden_states, hidden_states_scaling_factor = self.intermediate_act_fn(
            hidden_states, hidden_states_scaling_factor
        )

        # 重新量化：32位 -> 8位
        # 使用输出激活函数对处理后的隐藏状态进行处理，并更新隐藏状态缩放因子
        hidden_states, hidden_states_scaling_factor = self.output_activation(
            hidden_states, hidden_states_scaling_factor
        )
        # 返回处理后的隐藏状态和更新后的隐藏状态缩放因子
        return hidden_states, hidden_states_scaling_factor
class IBertOutput(nn.Module):
    # 定义 IBertOutput 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化函数
        self.quant_mode = config.quant_mode
        # 设置量化模式为 config 中的量化模式
        self.act_bit = 8
        # 设置激活位数为 8
        self.weight_bit = 8
        # 设置权重位数为 8
        self.bias_bit = 32
        # 设置偏置位数为 32
        self.ln_input_bit = 22
        # 设置输入层归一化位数为 22
        self.ln_output_bit = 32
        # 设置输出层归一化位数为 32

        self.dense = QuantLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            weight_bit=self.weight_bit,
            bias_bit=self.bias_bit,
            quant_mode=self.quant_mode,
            per_channel=True,
        )
        # 创建一个量化线性层，设置权重和偏置位数，以及量化模式和通道量化为真
        self.ln_input_act = QuantAct(self.ln_input_bit, quant_mode=self.quant_mode)
        # 创建一个输入层激活量化层，设置位数和量化模式
        self.LayerNorm = IntLayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            output_bit=self.ln_output_bit,
            quant_mode=self.quant_mode,
            force_dequant=config.force_dequant,
        )
        # 创建一个整数层归一化层，设置隐藏层大小、epsilon、输出位数、量化模式和是否强制去量化
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 创建一个输出激活量化层，设置激活位数和量化模式
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个 dropout 层，设置 dropout 概率

    def forward(self, hidden_states, hidden_states_scaling_factor, input_tensor, input_tensor_scaling_factor):
        # 前向传播函数，接受隐藏状态、隐藏状态缩放因子、输入张量和输入张量缩放因子
        hidden_states, hidden_states_scaling_factor = self.dense(hidden_states, hidden_states_scaling_factor)
        # 使用量化线性层处理隐藏状态和缩放因子
        hidden_states = self.dropout(hidden_states)
        # 使用 dropout 处理隐藏状态
        hidden_states, hidden_states_scaling_factor = self.ln_input_act(
            hidden_states,
            hidden_states_scaling_factor,
            identity=input_tensor,
            identity_scaling_factor=input_tensor_scaling_factor,
        )
        # 使用输入层激活量化层处理隐藏状态和缩放因子，传��输入张量和输入张量缩放因子
        hidden_states, hidden_states_scaling_factor = self.LayerNorm(hidden_states, hidden_states_scaling_factor)
        # 使用整数层归一化层处理隐藏状态和缩放因子

        hidden_states, hidden_states_scaling_factor = self.output_activation(
            hidden_states, hidden_states_scaling_factor
        )
        # 使用输出激活量化层处理隐藏状态和缩放因子
        return hidden_states, hidden_states_scaling_factor
        # 返回处理后的隐藏状态和缩放因子


class IBertLayer(nn.Module):
    # 定义 IBertLayer 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化函数，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化函数
        self.quant_mode = config.quant_mode
        # 设置量化模式为 config 中的量化模式
        self.act_bit = 8
        # 设置激活位数为 8

        self.seq_len_dim = 1
        # 设置序列长度维度为 1
        self.attention = IBertAttention(config)
        # 创建一个 IBertAttention 对象
        self.intermediate = IBertIntermediate(config)
        # 创建一个 IBertIntermediate 对象
        self.output = IBertOutput(config)
        # 创建一个 IBertOutput 对象

        self.pre_intermediate_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 创建一个预中间激活量化层，设置激活位数和量化模式
        self.pre_output_act = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        # 创建一个预输出激活量化层，设置激活位数和量化模式

    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    # 定义一个方法，用于处理一个 Transformer 层的前向传播
    def forward(self, hidden_states, hidden_states_scaling_factor, attention_mask=None, head_mask=None, output_attentions=False
        ):
        # 使用自注意力机制处理隐藏状态，得到自注意力输出和缩放因子
        self_attention_outputs, self_attention_outputs_scaling_factor = self.attention(
            hidden_states,
            hidden_states_scaling_factor,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力输出的第一个元素作为注意力输出
        attention_output = self_attention_outputs[0]
        attention_output_scaling_factor = self_attention_outputs_scaling_factor[0]

        # 如果需要输出注意力权重，则将自注意力输出添加到输出中
        outputs = self_attention_outputs[1:]

        # 使用前馈网络处理注意力输出，得到层输出和缩放因子
        layer_output, layer_output_scaling_factor = self.feed_forward_chunk(
            attention_output, attention_output_scaling_factor
        )
        # 将层输出添加到输出中
        outputs = (layer_output,) + outputs

        # 返回输出
        return outputs

    # 定义一个方法，用于处理前馈网络的一部分
    def feed_forward_chunk(self, attention_output, attention_output_scaling_factor):
        # 使用预激活函数处理注意力输出，得到中间输出和缩放因子
        attention_output, attention_output_scaling_factor = self.pre_intermediate_act(
            attention_output, attention_output_scaling_factor
        )
        # 使用中间层处理中间输出，得到中间层输出和缩放因子
        intermediate_output, intermediate_output_scaling_factor = self.intermediate(
            attention_output, attention_output_scaling_factor
        )

        # 使用预输出激活函数处理中间层输出，得到最终层输出和缩放因子
        intermediate_output, intermediate_output_scaling_factor = self.pre_output_act(
            intermediate_output, intermediate_output_scaling_factor
        )
        # 使用输出层处理中间层输出，得到最终层输出和缩放因子
        layer_output, layer_output_scaling_factor = self.output(
            intermediate_output, intermediate_output_scaling_factor, attention_output, attention_output_scaling_factor
        )
        # 返回最终层输出和缩放因子
        return layer_output, layer_output_scaling_factor
class IBertEncoder(nn.Module):
    # 定义 IBertEncoder 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化方法
        self.config = config
        # 将传入的 config 参数保存到实例变量中
        self.quant_mode = config.quant_mode
        # 从 config 中获取 quant_mode，并保存到实例变量中
        self.layer = nn.ModuleList([IBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建一个包含多个 IBertLayer 实例的 ModuleList，数量为 config.num_hidden_layers

    def forward(
        self,
        hidden_states,
        hidden_states_scaling_factor,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 定义前向传播方法，接受多个参数
        all_hidden_states = () if output_hidden_states else None
        # 如果 output_hidden_states 为 True，则初始化 all_hidden_states 为空元组，否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果 output_attentions 为 True，则初始化 all_self_attentions 为空元组，否则为 None
        all_cross_attentions = None  # `config.add_cross_attention` is not supported
        # 初始化 all_cross_attentions 为 None，不支持 config.add_cross_attention
        next_decoder_cache = None  # `config.use_cache` is not supported
        # 初始化 next_decoder_cache 为 None，不支持 config.use_cache

        for i, layer_module in enumerate(self.layer):
            # 遍历 self.layer 中的每个 IBertLayer 实例
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果 output_hidden_states 为 True，则将当前 hidden_states 添加到 all_hidden_states 中

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的头部掩码

            layer_outputs = layer_module(
                hidden_states,
                hidden_states_scaling_factor,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )
            # 调用当前层的前向传播方法

            hidden_states = layer_outputs[0]
            # 更新 hidden_states 为当前层的输出的第一个元素
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果 output_attentions 为 True，则将当前层的注意力输出添加到 all_self_attentions 中

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # 如果 output_hidden_states 为 True，则将最后一个 hidden_states 添加到 all_hidden_states 中

        if not return_dict:
            # 如果 return_dict 为 False
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
            # 返回包含非 None 值的元组
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        # 返回包含各种输出的 BaseModelOutputWithPastAndCrossAttentions 实例


class IBertPooler(nn.Module):
    # 定义 IBertPooler 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法，接受一个 config 参数
        super().__init__()
        # 调用父类的初始化方法
        self.quant_mode = config.quant_mode
        # 从 config 中获取 quant_mode，并保存到实例变量中
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个线性层，输入和输出维度都为 config.hidden_size
        self.activation = nn.Tanh()
        # 创建一个 Tanh 激活函数实例

    def forward(self, hidden_states):
        # 定义前向传播方法，接受 hidden_states 参数
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 我们通过简单地取第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 获取第一个标记对应的隐藏状态
        pooled_output = self.dense(first_token_tensor)
        # 将第一个标记对应的隐藏状态通过线性层 dense 处理
        pooled_output = self.activation(pooled_output)
        # 将处理后的结果通过 Tanh 激活函数处理
        return pooled_output
        # 返回池化后的输出


class IBertPreTrainedModel(PreTrainedModel):
    # 定义 IBertPreTrainedModel 类，继承自 PreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    # 定义配置类为 IBertConfig
    config_class = IBertConfig
    # 定义基础模型前缀为 "ibert"
    base_model_prefix = "ibert"

    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是 QuantLinear 或 nn.Linear 类型
        if isinstance(module, (QuantLinear, nn.Linear)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 QuantEmbedding 或 nn.Embedding 类型
        elif isinstance(module, (QuantEmbedding, nn.Embedding)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 IntLayerNorm 或 nn.LayerNorm 类型
        elif isinstance(module, (IntLayerNorm, nn.LayerNorm)):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)

    # 调整 token embeddings 的大小
    def resize_token_embeddings(self, new_num_tokens=None):
        # 抛出未实现错误，因为 I-BERT 不支持调整 token embeddings 的大小
        raise NotImplementedError("`resize_token_embeddings` is not supported for I-BERT.")
# IBERT_START_DOCSTRING 是一个原始字符串，包含了关于 IBertModel 类的文档字符串
# 这个模型继承自 PreTrainedModel 类，可以查看超类文档以了解库实现的通用方法（如下载或保存、调整输入嵌入、修剪头等）
# 这个模型也是 PyTorch 的 torch.nn.Module 子类，可以像普通的 PyTorch 模块一样使用，并参考 PyTorch 文档了解一般用法和行为
# 参数:
#     config ([IBertConfig]): 包含模型所有参数的模型配置类。使用配置文件初始化不会加载与模型关联的权重，只加载配置。查看 from_pretrained 方法以加载模型权重



# IBERT_INPUTS_DOCSTRING 是一个原始字符串，包含了关于 IBertModel 类输入的文档字符串
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引。索引选择在 `[0, 1]` 之间：
            # - 0 对应于*句子 A* 标记，
            # - 1 对应于*句子 B* 标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]` 之间。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的选定头部置零的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示头部**未被掩码**，
            # - 0 表示头部**被掩码**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多控制如何将 `input_ids` 索引转换为关联向量，这将很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
"""

@add_start_docstrings(
    "The bare I-BERT Model transformer outputting raw hidden-states without any specific head on top.",
    IBERT_START_DOCSTRING,
)
class IBertModel(IBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的构造函数
        super().__init__(config)
        # 设置配置
        self.config = config
        # 获取量化模式
        self.quant_mode = config.quant_mode

        # 初始化嵌入层
        self.embeddings = IBertEmbeddings(config)
        # 初始化编码器
        self.encoder = IBertEncoder(config)

        # 如果需要添加池化层，则初始化池化层
        self.pooler = IBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
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



@add_start_docstrings("""I-BERT Model with a `language modeling` head on top.""", IBERT_START_DOCSTRING)
class IBertForMaskedLM(IBertPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.bias", "lm_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 初始化 I-BERT 模型
        self.ibert = IBertModel(config, add_pooling_layer=False)
        # 初始化语言模型头
        self.lm_head = IBertLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.decoder
    # 设置输出嵌入层的新嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数并返回预测结果
    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ibert 模型进行前向传播
        outputs = self.ibert(
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
        sequence_output = outputs[0]
        # 使用 lm_head 模型对序列输出进行预测
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果存在标签，则计算 masked language modeling 损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回输出结果
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 MaskedLMOutput 对象，包含损失、预测结果、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class IBertLMHead(nn.Module):
    """I-BERT Head for masked language modeling."""

    def __init__(self, config):
        # 初始化 I-BERT 语言模型头部
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入维度是 config.hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建一个全连接层，输入维度是 config.hidden_size，输出维度是 config.vocab_size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 创建一个偏置参数，维度是 config.vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将偏置参数赋值给 decoder 层的偏置
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 将 features 输入到全连接层
        x = self.dense(features)
        # 使用 gelu 激活函数
        x = gelu(x)
        # 对结果进行 LayerNorm
        x = self.layer_norm(x)

        # 通过 decoder 层将结果映射回词汇表大小
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果两个权重断开连接（在 TPU 上或者偏置参数被调整大小），则将这两个权重绑定在一起
        self.bias = self.decoder.bias


@add_start_docstrings(
    """
    I-BERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    IBERT_START_DOCSTRING,
)
class IBertForSequenceClassification(IBertPreTrainedModel):
    def __init__(self, config):
        # 初始化 I-BERT 序列分类/回归模型
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建一个 IBertModel 模型，不添加池化层
        self.ibert = IBertModel(config, add_pooling_layer=False)
        # 创建一个 IBertClassificationHead 分类头部
        self.classifier = IBertClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未指定则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用ibert模型，传入各种参数
        outputs = self.ibert(
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
        # 获取序列输出
        sequence_output = outputs[0]
        # 通过分类器获取logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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
        # 如果不需要返回字典，则返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有多选分类头部的 I-BERT 模型（在汇总输出之上有一个线性层和一个 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    I-BERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    IBERT_START_DOCSTRING,
)
class IBertForMultipleChoice(IBertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 I-BERT 模型
        self.ibert = IBertModel(config)
        # 创建一个 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        ) -> Union[MultipleChoiceModelOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的 input_ids 展平
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将输入的 position_ids 展平
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将输入的 token_type_ids 展平
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将输入的 attention_mask 展平
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将输入的 inputs_embeds 展平
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用 ibert 模型进行前向传播
        outputs = self.ibert(
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
        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类
        logits = self.classifier(pooled_output)
        # 重塑 logits
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果存在 labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个基于IBERT模型的标记分类头部的模型，用于命名实体识别（NER）任务
class IBertForTokenClassification(IBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 初始化IBERT模型，不添加池化层
        self.ibert = IBertModel(config, add_pooling_layer=False)
        # 添加dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
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
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果return_dict为None，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取IBERT模型的输出
        outputs = self.ibert(
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

        # 获取序列输出
        sequence_output = outputs[0]

        # 应用dropout
        sequence_output = self.dropout(sequence_output)
        # 获取分类器输出
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class IBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        # 初始化函数，定义了一个用于句子级分类任务的头部
        super().__init__()
        # 调用父类的初始化函数
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个全连接层，输入和输出维度都是隐藏层大小
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个dropout层，用于防止过拟合
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        # 定义一个全连接层，用于输出分类结果

    def forward(self, features, **kwargs):
        # 前向传播函数，接收特征和其他参数
        hidden_states = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 从特征中取出第一个位置的隐藏状态，相当于取出了特殊的 [CLS] 标记
        hidden_states = self.dropout(hidden_states)
        # 对隐藏状态进行dropout操作
        hidden_states = self.dense(hidden_states)
        # 将隐藏状态通过全连接层
        hidden_states = torch.tanh(hidden_states)
        # 使用tanh激活函数
        hidden_states = self.dropout(hidden_states)
        # 再次进行dropout操作
        hidden_states = self.out_proj(hidden_states)
        # 将结果通过输出全连接层
        return hidden_states
        # 返回隐藏状态

@add_start_docstrings(
    """
    I-BERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    IBERT_START_DOCSTRING,
)
class IBertForQuestionAnswering(IBertPreTrainedModel):
    # 定义一个用于问题回答任务的I-BERT模型
    def __init__(self, config):
        # 初始化函数
        super().__init__(config)
        # 调用父类的初始化函数
        self.num_labels = config.num_labels
        # 获取标签数量

        self.ibert = IBertModel(config, add_pooling_layer=False)
        # 创建一个I-BERT模型实例
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # 定义一个全连接层用于输出问题回答结果

        # Initialize weights and apply final processing
        self.post_init()
        # 初始化权重并进行最终处理

    @add_start_docstrings_to_model_forward(IBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[QuestionAnsweringModelOutput, Tuple[torch.FloatTensor]]:
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
        # 设置返回字典，如果未指定则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用ibert模型进行推理
        outputs = self.ibert(
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

        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 将序列输出传入qa_outputs模型获取logits
        logits = self.qa_outputs(sequence_output)
        # 将logits拆分为起始和结束logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除多余维度并保持连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出模型输入范围，忽略这些项
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不返回字典，则返回元组
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回问题回答模型输出
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从输入的 input_ids 中创建位置 id，非填充符号替换为它们的位置数字。位置数字从 padding_idx+1 开始。填充符号将被忽略。这是从 fairseq 的 *utils.make_positions* 修改而来。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's *utils.make_positions*.

    Args:
    input_ids (`torch.LongTensor`):
           Indices of input sequence tokens in the vocabulary.

    Returns: torch.Tensor
    """
    # 将非填充符号替换为它们的位置数字
    mask = input_ids.ne(padding_idx).int()
    # 计算累积位置索引，同时考虑过去的键值长度
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回位置 id，加上填充索引
    return incremental_indices.long() + padding_idx
```