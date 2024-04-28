# `.\transformers\models\nezha\modeling_nezha.py`

```
# 设置编码格式为 UTF-8
# 版权声明及许可证信息
# 该模块实现了 PyTorch 下的 Nezha 模型
# 导入所需的库和模块
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数映射字典
from ...activations import ACT2FN
# 导入模型输出类
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型基类
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入工具函数和类
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 Nezha 配置类
from .configuration_nezha import NezhaConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中用到的检查点和配置
_CHECKPOINT_FOR_DOC = "sijunhe/nezha-cn-base"
_CONFIG_FOR_DOC = "NezhaConfig"

# 预训练模型存档列表
NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sijunhe/nezha-cn-base",
    "sijunhe/nezha-cn-large",
    "sijunhe/nezha-base-wwm",
    "sijunhe/nezha-large-wwm",
    # 查看所有 Nezha 模型 https://huggingface.co/models?filter=nezha
]

# 加载 TensorFlow 检查点的函数
def load_tf_weights_in_nezha(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        # 导入所需的库
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 若导入失败，输出错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 检查点路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 打印信息，显示正在转换的 TensorFlow 检查点路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 初始化名称和数组列表
    names = []
    arrays = []
    # 遍历 TF 模型中的变量
    for name, shape in init_vars:
        # 打印信息，显示正在加载的 TF 权重名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载 TF 变量
        array = tf.train.load_variable(tf_path, name)
        # 将名称和数组添加到对应的列表中
        names.append(name)
        arrays.append(array)
``` 
    # 遍历给定的名称和数组列表
    for name, array in zip(names, arrays):
        # 将名称按照'/'分割，处理嵌套结构的变量名
        name = name.split("/")
        
        # 检查名称中是否包含特定的变量，如果包含则跳过
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        # 初始化指针指向模型
        pointer = model
        
        # 遍历处理分割后的名称部分
        for m_name in name:
            # 使用正则表达式判断是否满足指定格式
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据不同的scope名称对指针进行设置
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 处理名称中包含序号的情况
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 如果名称以'_embeddings'结尾，则将指针指向权重
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        # 如果名称为'kernel'，则对数组进行转置
        elif m_name == "kernel":
            array = np.transpose(array)
        
        # 检查指针和数组的形状是否匹配
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        # 打印初始化PyTorch权重信息
        logger.info(f"Initialize PyTorch weight {name}")
        # 将数组转换为PyTorch张量并赋值给指针
        pointer.data = torch.from_numpy(array)
    
    # 返回模型
    return model
class NezhaRelativePositionsEncoding(nn.Module):
    """实现函数式相对位置编码"""

    def __init__(self, length, depth, max_relative_position=127):
        super().__init__()
        vocab_size = max_relative_position * 2 + 1  # 计算词汇表大小
        range_vec = torch.arange(length)  # 生成长度为length的序列
        range_mat = range_vec.repeat(length).view(length, length)  # 将序列重复length次并reshape成length*length的矩阵
        distance_mat = range_mat - torch.t(range_mat)  # 生成距离矩阵
        distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)  # 对距离矩阵进行截断
        final_mat = distance_mat_clipped + max_relative_position  # 对距离矩阵进行平移处理

        embeddings_table = torch.zeros(vocab_size, depth)  # 初始化一个大小为vocab_size*depth的全零张量
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)  # 生成位置向量
        div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))  # 计算除数项
        embeddings_table[:, 0::2] = torch.sin(position * div_term)  # 计算sin位置编码
        embeddings_table[:, 1::2] = torch.cos(position * div_term)  # 计算cos位置编码

        flat_relative_positions_matrix = final_mat.view(-1)  # 将处理后的距离矩阵展平
        one_hot_relative_positions_matrix = torch.nn.functional.one_hot(
            flat_relative_positions_matrix, num_classes=vocab_size
        ).float()  # 生成独热编码的相对位置矩阵
        positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)  # 计算相对位置编码
        my_shape = list(final_mat.size())  # 获取final_mat的形状
        my_shape.append(depth)  # 将深度信息加入形状中
        positions_encoding = positions_encoding.view(my_shape)  # 将位置编码reshape成指定形状
        self.register_buffer("positions_encoding", positions_encoding, persistent=False)  # 注册位置编码张量缓存，不持久化

    def forward(self, length):
        return self.positions_encoding[:length, :length, :]  # 返回指定长度的位置编码


class NezhaEmbeddings(nn.Module):
    """从词和标记类型嵌入构造嵌入"""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  # 初始化词嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # 初始化标记类型嵌入层

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化LayerNorm层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 初始化Dropout层
        self.register_buffer(
            "token_type_ids", torch.zeros((1, config.max_position_embeddings), dtype=torch.long), persistent=False
        )  # 注册标记类型张量缓存，不持久化

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 如果传入了 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 否则获取 inputs_embeds 的形状，不包括最后一个维度（通常是序列长度）
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即 input_shape 的第二个维度
        seq_length = input_shape[1]

        # 如果 inputs_embeds 为 None，则使用 word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果 token_type_ids 为 None，则设置为全零张量，形状与 inputs_embeds 相同
        if token_type_ids is None:
            # 如果模型具有注册的 token_type_ids 缓冲区，则使用其值，否则设置为全零
            if hasattr(self, "token_type_ids"):
                # 获取注册的 token_type_ids 缓冲区的值，并截取到序列长度
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                # 将缓冲区的 token_type_ids 扩展到与 input_shape 相同的形状
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                # 将扩展后的 token_type_ids 赋值给 token_type_ids
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果没有注册的 token_type_ids 缓冲区，则创建全零张量，与 inputs_embeds 相同的形状
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)

        # 使用 token_type_embeddings 对 token_type_ids 进行嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入向量与 token type 嵌入向量相加
        embeddings = inputs_embeds + token_type_embeddings
        # 对嵌入向量进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings
# 定义类 NezhaSelfAttention，继承自 nn.Module
class NezhaSelfAttention(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 如果隐藏层大小不能整除注意力头数，抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 使用线性变换生成查询、键和值的方式
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 使用丢弃层进行概率丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 初始化相对位置编码
        self.relative_positions_encoding = NezhaRelativePositionsEncoding(
            length=config.max_position_embeddings,
            depth=self.attention_head_size,
            max_relative_position=config.max_relative_position,
        )
        self.is_decoder = config.is_decoder

    # 定义一个转置函数，用于调整张量的尺寸和维度用于计算注意力分数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播方法，接受各种输入参数
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
# 定义类 NezhaSelfOutput，继承自 nn.Module
class NezhaSelfOutput(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 使用线性变换将隐藏层的特征转换为相同大小的特征
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用 LayerNorm 进行层标准化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用丢弃层进行概率丢弃
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态和输入张量作为输入
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行特征转换
        hidden_states = self.dense(hidden_states)
        # 使用丢弃层进行概率丢弃
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 进行残差连接和层标准化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states

# 定义类 NezhaAttention，继承自 nn.Module
class NezhaAttention(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 创建自注意力和自注意力输出对象
        self.self = NezhaSelfAttention(config)
        self.output = NezhaSelfOutput(config)
        # 初始化剪枝头部集合
        self.pruned_heads = set()
    def prune_heads(self, heads):
        # 如果头列表为空，则直接返回
        if len(heads) == 0:
            return
        # 找到可以被修剪的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        # 对自身进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 通过output函数对attention_output进行后续处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将attention_output与self_outputs的其它部分组合成outputs
        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出，添加到outputs中
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate中复制代码，并将Bert替换为Nezha
class NezhaIntermediate(nn.Module):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 使用全连接层将隐藏状态映射到中间层大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置参数中的隐藏激活函数是字符串形式，则从ACT2FN字典中查找对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受隐藏状态输入hidden_states，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOutput中复制代码，并将Bert替换为Nezha
class NezhaOutput(nn.Module):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 使用全连接层将中间层大小映射回隐藏状态大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化LayerNorm层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout层，用于随机丢弃部分隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态输入hidden_states和原始输入input_tensor，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 随机丢弃部分隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 归一化处理后的隐藏状态和原始输入的隐藏状态
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义Nezha层的类
class NezhaLayer(nn.Module):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义分块前馈大小和序列长度维度
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化NezhaAttention、NezhaIntermediate、NezhaOutput模块
        self.attention = NezhaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果需要添加交叉注意力，则验证是否为解码器模型，否则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = NezhaAttention(config)
        self.intermediate = NezhaIntermediate(config)
        self.output = NezhaOutput(config)

    # 前向传播方法，接受隐藏状态输入和多种注意力相关参数，返回处理后的隐藏状态
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
        # 使用先前缓存的键/值对执行自注意力机制
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用先前缓存的键/值对执行跨注意力机制
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
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对注意力输出进行前向传播并将结果分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 处理前向传播的分块函数
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制而来，将Bert更换为Nezha
class NezhaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 为编码器层创建一个nn.ModuleList，其中包含多个NezhaLayer对象，数量由config中的num_hidden_layers确定
        self.layer = nn.ModuleList([NezhaLayer(config) for _ in range(config.num_hidden_layers)])
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
    # 如果输出隐藏状态，则初始化一个空元组用于存储所有隐藏状态
    all_hidden_states = () if output_hidden_states else None
    # 如果输出注意力值，则初始化一个空元组用于存储所有自注意力值
    all_self_attentions = () if output_attentions else None
    # 如果输出注意力值并且配置中包含交叉注意力，则初始化一个空元组用于存储所有交叉注意力值
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    # 梯度检查点并且处于训练状态时，如果使用缓存，则给出警告并将use_cache设为False
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # 如果使用缓存，则初始化一个空元组用于存储下一个解码器缓存
    next_decoder_cache = () if use_cache else None
    # 遍历self.layer中的层模块
    for i, layer_module in enumerate(self.layer):
        # 如果输出隐藏状态，将当前隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果存在头部掩码，则将对应索引的掩码提取出来，否则为None
        layer_head_mask = head_mask[i] if head_mask is not None else None
        # 如果存在过去键值对，则将对应索引的过去键值对提取出来，否则为None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        # 如果梯度检查点并且处于训练状态，则使用梯度检查点方法对当前层进行计算
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
        else:
            # 否则正常调用当前层模块进行计算
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        # 更新隐藏状态为当前层计算结果的第一个元素
        hidden_states = layer_outputs[0]
        # 如果使用缓存，则将当前层输出的最后一个元素添加到下一个解码器缓存中
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        # 如果输出注意力值，则将当前层输出结果的第二个元素添加到所有自注意力值中
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # 如果配置中包含交叉注意力，则将当前层输出结果的第三个元素添加到所有交叉注意力值中
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    # 如果输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态中
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # 如果不返回字典，则按特定顺序返回一组结果元素
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
    # 返回包含隐藏状态、过去键值对、所有隐藏状态、所有自注意力值和所有交叉注意力值的模型输出
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )
# 从transformers.models.bert.modeling_bert.BertPooler复制代码，并将Bert->Nezha
class NezhaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态的大小从config.hidden_size转换为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用双曲正切函数作为激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地取对应于第一个标记的隐藏状态来"池化"模型。
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态传入线性层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数处理线性层的输出
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制代码，并将Bert->Nezha
class NezhaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态的大小从config.hidden_size转换为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            # 如果config.hidden_act是字符串，则使用ACT2FN字典中对应的激活函数
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用LayerNorm对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理变换后的隐藏状态
        hidden_states = self.transform_act_fn(hidden_states)
        # 对变换后的隐藏状态进行LayerNorm归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制代码，并将Bert->Nezha
class NezhaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建NezhaPredictionHeadTransform类型的变换层
        self.transform = NezhaPredictionHeadTransform(config)

        # 输出权重和输入嵌入层相同，但每个标记都有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要两个变量之间的链接，以便偏置能够正确调整大小以与“resize_token_embeddings”���配
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 通过变换处理隐藏状态
        hidden_states = self.transform(hidden_states)
        # 通过解码器获取最终的隐藏状态
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制代码，并将Bert->Nezha
class NezhaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 预测层是NezhaLMPredictionHead类型的
        self.predictions = NezhaLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 获取预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从transformers.models.bert.modeling_bert.BertOnlyNSPHead复制代码，并将Bert->Nezha
class NezhaOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层处理隐藏状态，输出大小为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 定义一个方法，参数为pooled_output，用于向前传播神经网络
    def forward(self, pooled_output):
        # 计算序列关系得分，传入pooled_output作为输入
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回计算得到的序列关系得分
        return seq_relationship_score
# 从 transformers.models.bert.modeling_bert.BertPreTrainingHeads 复制类，并替换 Bert 为 Nezha
class NezhaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 NezhaLMPredictionHead 实例，用于预测
        self.predictions = NezhaLMPredictionHead(config)
        # 初始化全连接层，用于序列关系预测
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 使用输入的序列输出进行预测，得到预测得分
        prediction_scores = self.predictions(sequence_output)
        # 使用池化输出进行序列关系预测，得到关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测得分和关系得分
        return prediction_scores, seq_relationship_score


class NezhaPreTrainedModel(PreTrainedModel):
    """
    抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 配置类为 NezhaConfig
    config_class = NezhaConfig
    # 加载 TensorFlow 权重的方法为 load_tf_weights_in_nezha
    load_tf_weights = load_tf_weights_in_nezha
    # 基础模型的前缀为 "nezha"
    base_model_prefix = "nezha"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0.0，标准差为配置的初始范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0.0，标准差为配置的初始范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将填充位置权重设为 0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为 0
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)


@dataclass
class NezhaForPreTrainingOutput(ModelOutput):
    """
    [`NezhaForPreTraining`] 的输出类型。
    """
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
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
    # 损失：如果 `labels` 被提供，则返回，形状为 `(1,)` 的 `torch.FloatTensor`
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头的预测分数，形状为 `(batch_size, sequence_length, config.vocab_size)` 的 `torch.FloatTensor`
    prediction_logits: torch.FloatTensor = None
    # 下一序列预测（分类）头的预测分数，形状为 `(batch_size, 2)` 的 `torch.FloatTensor`
    seq_relationship_logits: torch.FloatTensor = None
    # 隐藏状态：返回 `output_hidden_states=True` 时或 `config.output_hidden_states=True` 时，包含 `(batch_size, sequence_length, hidden_size)` 形状的元组
    # 第一个元素是嵌入的输出，其余元素是每个层的输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重：返回 `output_attentions=True` 时或 `config.output_attentions=True` 时，包含 `(batch_size, num_heads, sequence_length, sequence_length)` 形状的元组
    # 用于计算自注意力头中的加权平均值的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None
NEZHA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`NezhaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

NEZHA_INPUTS_DOCSTRING = r"""

# NEZHA_INPUTS_DOCSTRING变量用于存储模型的输入文档字符串
# 在此处可添加模型输入的文档字符串，以便为模型的输入参数提供说明
# 注意：在此变量中添加的文档字符串应该与模型的输入参数相关
# 你可以使用reStructuredText或其他格式来编写文档字符串
# 如果想要了解更多关于如何编写好的文档字符串的信息，可以查阅相关文档和示例
# 请确保文档字符串的格式清晰易读，方便用户理解模型的输入参数及其含义
# 文档字符串应该包含有关输入参数的信息，例如其类型、形状、范围等
# 在编写文档字符串时，建议参考其他模型的输入文档字符串示例，以保持一致性和规范性
# 如果有必要，还可以在文档字符串中提供示例用法，以帮助用户更好地理解模型的使用方法

"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            输入序列标记在词汇表中的索引。
            
            索引可以通过 [`AutoTokenizer`] 获取。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
    
            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            遮蔽填充标记索引进行注意力计算。遮蔽值选择在 `[0, 1]` 之间：
    
            - 1 代表**未遮蔽**的标记，
            - 0 代表**已遮蔽**的标记。
    
            [什么是注意力遮蔽？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]` 之间：
    
            - 0 对应于*句子 A* 标记，
            - 1 对应于*句子 B* 标记。
    
            [什么是标记类型 ID？](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于屏蔽自注意力模块中特定头部的掩码。掩码值选择在 `[0, 1]` 之间：
    
            - 1 表示头部**未被遮蔽**，
            - 0 表示头部**已被遮蔽**。
    
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            可选择直接传递嵌入表示，而不是传递 `input_ids`。如果需要更多控制将 `input_ids` 索引转换为关联向量的方式，比模型的内部嵌入查找矩阵更有用。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请查看返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请查看返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
# 这是一个 Nezha 模型的实现,该模型是一种预训练语言模型,基于 Transformer 架构。

# 此类用于输出原始隐藏状态,不带任何特定的头部。
@add_start_docstrings(
    "The bare Nezha Model transformer outputting raw hidden-states without any specific head on top.",
    NEZHA_START_DOCSTRING,
)
class NezhaModel(NezhaPreTrainedModel):
    """
    这个模型可以作为编码器(仅使用自注意力)或解码器使用。作为解码器时,在自注意力层之间添加一层交叉注意力,遵循"Attention is all you need"论文中描述的架构。

    要把这个模型作为解码器使用,需要在配置中将 `is_decoder` 参数设置为 `True`。
    要在一个 Seq2Seq 模型中使用,需要将 `is_decoder` 和 `add_cross_attention` 都设置为 `True`,并且需要将 `encoder_hidden_states` 作为输入传入。
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 创建 Nezha 嵌入层
        self.embeddings = NezhaEmbeddings(config)
        # 创建 Nezha 编码器层
        self.encoder = NezhaEncoder(config)

        # 如果需要,创建 Nezha 池化层
        self.pooler = NezhaPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型头
    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型头。heads_to_prune: 一个字典,键为层号,值为该层需要剪枝的头的列表。
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 模型前向传播
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        ...
    # sentence prediction (classification) 头部的文档字符串
    """
    sentence prediction (classification) 头部的文档字符串
    """,
    # 导入 NEZHA_START_DOCSTRING 常量
    NEZHA_START_DOCSTRING,
# Nezha预训练模型。定义了Nezha预训练模型的结构和前向传递方法
class NezhaForPreTraining(NezhaPreTrainedModel):
    # 预定义在模型内被绑定权重的模块名称
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化函数，根据配置参数构建预训练模型的结构
    def __init__(self, config):
        super().__init__(config)

        # 创建Nezha模型对象
        self.nezha = NezhaModel(config)
        # 创建Nezha预训练模型头对象
        self.cls = NezhaPreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传递方法
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NezhaForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    
# Nezha预训练模型。将`language modeling`头添加到Nezha模型顶部。
class NezhaForMaskedLM(NezhaPreTrainedModel):
    # 预定义在模型内被绑定权重的模块名称
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化函数，根据配置参数构建预训练模型的结构
    def __init__(self, config):
        super().__init__(config)

        # 如果配置表示该模型是decoder，发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `NezhaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建Nezha模型对象，不添加池化层
        self.nezha = NezhaModel(config, add_pooling_layer=False)
        # 创建只有MLM头的Nezha模型对象
        self.cls = NezhaOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传递方法
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个forward方法，用于执行前向传播计算
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的tokens的ids序列
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，用来指示哪些token需要被注重
        token_type_ids: Optional[torch.Tensor] = None,  # 标记token的类型，如segment A和segment B
        head_mask: Optional[torch.Tensor] = None,  # 头部mask，用于控制哪些attention头需要被使用
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力遮罩
        labels: Optional[torch.Tensor] = None,  # 用于计算MLM损失的标签序列
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        
        # 检查是否指定了返回字典格式的输出，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给nezha模型进行前向计算
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取输出中的序列输出和预测分数
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果指定了标签，则计算MLM损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典格式的输出，则返回预测分数和可能的额外输出（如隐藏状态）
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回带有损失、预测分数、隐藏状态和注意力权重的MaskedLMOutput对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # 获取输入张量的形状
        input_shape = input_ids.shape
        # 获取有效批次大小
        effective_batch_size = input_shape[0]

        # 添加一个虚拟标记
        # 检查是否定义了 PAD 标记，如果没有则抛出异常
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 将注意力掩码张量在最后一个维度上连接一个全为0的张量
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全为 PAD 标记的虚拟标记张量
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 在输入张量的最后一个维度上连接虚拟标记张量
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回包含修改后的输入张量和注意力掩码的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用给定的文档字符串初始化 Nezha 模型，并在顶部添加一个“下一句预测（分类）”头部
@add_start_docstrings(
    """Nezha Model with a `next sentence prediction (classification)` head on top.""",
    NEZHA_START_DOCSTRING,
)
class NezhaForNextSentencePrediction(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Nezha 模型
        self.nezha = NezhaModel(config)
        # 初始化只包含下一句预测头部的模型
        self.cls = NezhaOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 方法，并为其添加文档字符串和返回值类型注释
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
):
    ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, NezhaForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("sijunhe/nezha-cn-base")
        >>> model = NezhaForNextSentencePrediction.from_pretrained("sijunhe/nezha-cn-base")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        # 如果kwargs中有"next_sentence_label"参数，则发出警告，该参数已被弃用，并且将在将来的版本中移除，建议使用"labels"
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给Nezha模型，获取模型输出
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取汇总的输出
        pooled_output = outputs[1]

        # 使用分类器对汇总的输出进行序列关系分数计算
        seq_relationship_scores = self.cls(pooled_output)

        # 初始化下一个句子的损失为None
        next_sentence_loss = None
        # 如果有标签，则计算下一个句子的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 如果不返回字典，则返回模型输出和下一个句子的损失
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 如果返回字典，则构造NextSentencePredictorOutput并返回
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```  
# 使用预定义的文档字符串和配置信息添加序列分类/回归头顶部的Nezha模型变换器（在汇总输出的顶部上方的线性层），例如用于GLUE任务
@add_start_docstrings(
    """
    Nezha Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    NEZHA_START_DOCSTRING,
)
class NezhaForSequenceClassification(NezhaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置类别标签数量
        self.num_labels = config.num_labels
        # 存储配置信息
        self.config = config

        # 创建Nezha模型对象
        self.nezha = NezhaModel(config)
        # 设置分类器的dropout参数
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 初始化丢失层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性层，输入尺寸为配置中的隐藏大小，输出尺寸为配置中的类别标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用指定的文档字符串和模型前向功能配置信息添加模型前向功能说明
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 这是一个Transformer模型的分类任务的前向传播过程
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # 判断是否使用返回词典的方式，如果未设置则使用模型配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用Transformer模型的前向传播
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 从输出中获取池化后的输出
        pooled_output = outputs[1]
    
        # 对池化后的输出进行dropout
        pooled_output = self.dropout(pooled_output)
    
        # 将dropout后的输出送入分类器得到logits
        logits = self.classifier(pooled_output)
    
        # 初始化loss为None
        loss = None
    
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 如果问题类型未设置，根据标签和类别数自动判断
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
    
            # 根据问题类型计算损失
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
    
        # 如果不使用返回词典，返回logits和其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 否则返回SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加模型文档字符串，描述模型是Nezha模型，具有多项选择分类头部
class NezhaForMultipleChoice(NezhaPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 创建Nezha模型
        self.nezha = NezhaModel(config)
        # 根据配置参数设置分类器的dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向方法的文档字符串，描述输入参数和输出类型
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义这个函数的返回值类型为 Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        # 根据输入参数 return_dict 设置是否使用 return_dict
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据 input_ids 或 inputs_embeds 的 shape 计算 num_choices 的值
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        # 如果 input_ids 不为 None，则展平为 2D 张量
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果 attention_mask 不为 None，则展平为 2D 张量
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果 token_type_ids 不为 None，则展平为 2D 张量
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果 inputs_embeds 不为 None，则展平为 3D 张量
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
    
        # 调用 self.nezha 函数进行前向计算，得到输出
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取池化输出
        pooled_output = outputs[1]
        print(pooled_output.shape)
        # 对池化输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 将池化输出传入分类器得到逻辑值
        logits = self.classifier(pooled_output)
        print(logits.shape)
        print(num_choices)
        # 根据 num_choices 将逻辑值重塑为 2D 张量
        reshaped_logits = logits.view(-1, num_choices)
    
        # 如果存在标签 labels，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
    
        # 根据 return_dict 返回合适的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用指定的文档字符串作为 NezhaForTokenClassification 类的起始文档字符串，并添加 Nezha 模型头部的标记分类头（在隐藏状态输出之上的线性层），
# 例如用于命名实体识别（NER）任务。
class NezhaForTokenClassification(NezhaPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 初始化 Nezha 模型，不添加池化层
        self.nezha = NezhaModel(config, add_pooling_layer=False)
        # 获取分类器的丢弃率，如果未指定，则使用隐藏状态丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用丢弃率初始化丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用指定的文档字符串添加到模型前向传播的文档字符串，并添加代码示例的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果未指定返回字典，则使用配置中的返回字典参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对输入进行前向传播，获取输出
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 应用丢弃层到序列输出
        sequence_output = self.dropout(sequence_output)
        # 应用线性分类器到序列输出，得到分类结果
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典，则返回元组
        if not return_dict:
            # 构造输出元组
            output = (logits,) + outputs[2:]
            # 如果损失不为空，则添加到输出元组中
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则构造 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 导入必要的库
@add_start_docstrings(
    """
    Nezha Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    NEZHA_START_DOCSTRING,
)
# 定义 NezhaForQuestionAnswering 类，用于问答任务，继承自 NezhaPreTrainedModel
class NezhaForQuestionAnswering(NezhaPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置类别数
        self.num_labels = config.num_labels

        # 实例化 NezhaModel，不添加池化层
        self.nezha = NezhaModel(config, add_pooling_layer=False)
        # 线性层，用于计算答案起始位置和结束位置的 logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(NEZHA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入 token 的 ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 ID
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入式输入
        start_positions: Optional[torch.Tensor] = None,  # 答案起始位置
        end_positions: Optional[torch.Tensor] = None,  # 答案结束位置
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
    # 定义一个函数，该函数接受输入参数并返回一个元组或QuestionAnsweringModelOutput类型的对象
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        # 如果return_dict不为None，就使用return_dict，否则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用模型的nezha方法，并传入相应参数
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取模型的序列输出
        sequence_output = outputs[0]
    
        # 对输出进行分类，获取开始和结束的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    
        total_loss = None
        # 如果开始和结束位置存在，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果处于多GPU状态，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时开始/结束位置超出了模型输入范围，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
    
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
    
        # 如果不需要返回dict，返回相应输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        # 返回QuestionAnsweringModelOutput对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```