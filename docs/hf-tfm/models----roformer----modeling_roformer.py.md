# `.\transformers\models\roformer\modeling_roformer.py`

```py
# 设置编码格式为 utf-8
# 版权声明和许可证信息，这段代码标明了代码的版权归属和使用许可
# 根据 Apache 许可证，只有遵守许可证规定才能使用该文件中的代码
# 详细的版权和许可信息可在 http://www.apache.org/licenses/LICENSE-2.0 找到
# 根据适用法律或书面协议，软件以“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证，了解具体语言版本和限制条件
# PyTorch RoFormer 模型

# 导入所需的库
import math
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义激活函数和模型输出类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型父类和其他模块
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 RoFormer 配置
from .configuration_roformer import RoFormerConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# 用于文档中引用的预训练模型和配置
_CHECKPOINT_FOR_DOC = "junnyu/roformer_chinese_base"
_CONFIG_FOR_DOC = "RoFormerConfig"

# RoFormer 的预训练模型
ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "junnyu/roformer_chinese_small",
    "junnyu/roformer_chinese_base",
    "junnyu/roformer_chinese_char_small",
    "junnyu/roformer_chinese_char_base",
    "junnyu/roformer_small_discriminator",
    "junnyu/roformer_small_generator",
    # 查看所有 RoFormer 模型：https://huggingface.co/models?filter=roformer
]

# 这个类用于生成任意长度的正弦位置嵌入
# 与 "transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding" 相似，只是将 Marian 改为 RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__(num_positions, embedding_dim)
        # 初始化权重
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter) -> nn.Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        # 初始化位置编码权重。将位置值映射成正弦和余弦函数的值。维度为[out的行数, out的列数]
        # n_pos表示位置编码的数量，dim表示位置编码的维度
        n_pos, dim = out.shape
        # 创建位置编码的二维数组，元素值为位置除以10000的2 * (j // 2) / dim次方
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        # 设置out不需要梯度计算
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        # 如果维度是偶数，则取dim的一半作为sentinel，否则取dim的一半加1作为sentinel
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        # 将位置编码的奇数列的sin函数值赋给out的前sentinel列
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        # 将位置编码的偶数列的cos函数值赋给out的后(sentinel - dim)列
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        # 将out从计算图中分离
        out.detach_()
        # 返回更新后的out
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 获取batch size和sequence length
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置编码，从past_key_values_length到past_key_values_length + seq_len
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的forward方法生成位置嵌入
        return super().forward(positions)
def load_tf_weights_in_roformer(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    # 尝试导入需要的库
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，输出错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TF 检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name.replace("bert", "roformer"))
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # 排除不需要加载的变量
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据变量名的前缀来确定权重矩阵或偏置向量的位置
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
            # 处理带有数字后缀的变量名
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            # 检查权重矩阵的形状是否匹配
            if not pointer.shape == array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        # 将 NumPy 数组转换为 PyTorch 张量，并初始化模型的权重
        pointer.data = torch.from_numpy(array)
    # 返回加载了 TF 权重的 PyTorch 模型
    return model
class RoFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""
    
    # 初始化函数，设置embedding层，LayerNorm和Dropout
    def __init__(self, config):
        super().__init__()
        # 创建word embeddings层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 创建token_type embeddings层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        
        # 对Transformer里的LayerNorm进行设置，保持和TensorFlow模型变量名一致
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接收input_ids, token_type_ids和inputs_embeds
    def forward(self, input_ids=None, token_type_ids=None, inputs_embeds=None):
        # 根据input_ids和inputs_embeds的情况设置input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        # 如果inputs_embeds不存在，则根据input_ids生成
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 如果token_type_ids不存在，则初始化为全0矩阵
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class RoFormerSelfAttention(nn.Module):
    # 初始化函数，设置注意力头数目和相关变量
    def __init__(self, config):
        super().__init__()
        # 校验hidden_size是否为attention头的整数倍
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 根据hidden_size创建query, key, value线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.is_decoder = config.is_decoder
        self.rotary_value = config.rotary_value
    
    # 将输入reshape以便进行attention计算
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    # 前向传播函数，接收多种输入参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    **kwargs):
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # 将输入的位置编码拆分为正弦值和余弦值
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # 将正弦值重复两次并重新塑形，得到与输入相同形状的正弦位置编码
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # 将余弦值重复两次并重新塑形，得到与输入相同形状的余弦位置编码
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # 将查询层的一半旋转，并将旋转后结果与原查询层相加，生成新的查询层
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # 将键层的一半旋转，并将旋转后结果与原键层相加，生成新的键层
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # 若存在值层，则将值层的一半旋转，并将旋转后结果与原值层相加，生成新的值层
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        # 若不存在值层，则返回新的查询层和键层
        return query_layer, key_layer
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并将 Bert 替换为 RoFormer 的自注意力输出层
class RoFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态进行全连接变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 将变换后的隐藏状态与输入张量相加，并进行 LayerNorm 归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertAttention 复制并将 Bert 替换为 RoFormer 的注意力层
class RoFormerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # RoFormer 自注意力层
        self.self = RoFormerSelfAttention(config)
        # RoFormer 自注意力输出层
        self.output = RoFormerSelfOutput(config)
        # 用于存储已剪枝的注意力头
        self.pruned_heads = set()

    # 从 transformers.models.bert.modeling_bert.BertAttention.prune_heads 复制
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到待剪枝的注意力头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # End Copy
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # RoFormer 自注意力层前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # RoFormer 自注意力输出层前向传播
        attention_output = self.output(self_outputs[0], hidden_states)
        # 返回注意力输出和可能的其他输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，添加注意力
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并将 Bert 替换为 RoFormer 的中间层
class RoFormerIntermediate(nn.Module):
    # 初始化函数，传入配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则使用ACT2FN字典中对应的激活函数，否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    
    # 前向传播函数，输入为hidden_states张量，输出为经过神经网络层处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性层处理
        hidden_states = self.dense(hidden_states)
        # 经过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput复制代码，并将Bert->RoFormer
class RoFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，对输入的隐藏状态进行处理，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用dropout进行正则化处理
        hidden_states = self.dropout(hidden_states)
        # 使用LayerNorm进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


class RoFormerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的一些参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RoFormerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # 如果要添加交叉注意力，且不是解码器模型，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RoFormerAttention(config)
        # RoFormer中间层
        self.intermediate = RoFormerIntermediate(config)
        # RoFormer输出层
        self.output = RoFormerOutput(config)

    # RoFormer层的前向传播函数，对输入的隐藏状态进行处理，并返回处理后的隐藏状态
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果是解码器，单向自注意力缓存的键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self_attn_past_key_value作为过去的键/值，计算自注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            sinusoidal_pos,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # 如果是解码器，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 获取除了最后一个元素以外的所有输出
            outputs = self_attention_outputs[1:-1]
            # 获取自注意力缓存的键/值
            present_key_value = self_attention_outputs[-1]
        else:
            # 获取除了第一个元素以外的所有输出，如果需要输出注意力权重则添加自注意力
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        # 如果是解码器且有编码器隐状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未定义交叉注意力层，则报错
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention "
                    "layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # 交叉注意力的缓存的键/值元组在过去键值元组的第3,4位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用cross_attn_past_key_value作为过去的键/值，计算交叉注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                sinusoidal_pos,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力输出添加到输出中，如果需要输出注意力权重则添加交叉注意力
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            # 将交叉注意力缓存添加到目前的键/值元组的第3,4位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将处理后的输出添加到结果中
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        # 如果是解码器，则返回注意力的键/值作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 对注意力输出进行前向分块处理
    def feed_forward_chunk(self, attention_output):
        # 计算注意力输出的中间输出
        intermediate_output = self.intermediate(attention_output)
        # 计算最终输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
class RoFormerEncoder(nn.Module):
    # RoFormer 编码器类
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 保存配置信息
        self.config = config
        # 初始化位置嵌入层
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads
        )
        # 初始化多层 RoFormer 层
        self.layer = nn.ModuleList([RoFormerLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点
        self.gradient_checkpointing = False

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
class RoFormerPredictionHeadTransform(nn.Module):
    # RoFormer 预测头变换类
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 全连接层
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        # 激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm 归一化
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 前向传播函数
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RoFormerLMPredictionHead(nn.Module):
    # RoFormer 语言模型预测头类
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 初始化预测头变换
        self.transform = RoFormerPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个变量之间的链接，以便偏置在 `resize_token_embeddings` 中正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 前向传播函数
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制而来，将 Bert 改为 RoFormer
class RoFormerOnlyMLMHead(nn.Module):
    # 仅包含 RoFormer 语言模型头的类
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 初始化预测头
        self.predictions = RoFormerLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 前向传播函数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RoFormerPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和一个简单的接口，用于下载和加载预训练模型。
    """

    config_class = RoFormerConfig
    load_tf_weights = load_tf_weights_in_roformer
    base_model_prefix = "roformer"
    supports_gradient_checkpointing = True
    # 设定是否支持梯度检查点的标志为真

    def _init_weights(self, module):
        # 将权重初始化为0均值、标准差为配置中的初始化范围的正态分布随机数
        if isinstance(module, nn.Linear):
            # 略微不同于TF版本，TF版本使用截断正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，则将偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是RoFormerSinusoidalPositionalEmbedding的实例，则不进行操作
        elif isinstance(module, RoFormerSinusoidalPositionalEmbedding):
            pass
        # 如果是nn.Embedding的实例，则将权重初始化为0均值、标准差为配置中的初始化范围的正态分布随机数
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果设置了填充索引，则将对应位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是nn.LayerNorm的实例，则将偏置初始化为0, 将权重初始化为全1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
# RoFormer 模型的文档字符串开头，提供一般用法和行为的相关信息
ROFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RoFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# RoFormer 模型输入的文档字符串，描述输入参数和其用途
ROFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 添加文档字符串起始标记
@add_start_docstrings(
    # 创建一个字符串，描述 RoFormer 模型的基本描述
    "The bare RoFormer Model transformer outputting raw hidden-states without any specific head on top.",
    # RoFormer 模型的文档字符串的起始标记
    ROFORMER_START_DOCSTRING,
    )
class RoFormerModel(RoFormerPreTrainedModel):
    """

    # 该模型可以作为编码器（只有自注意力）或解码器，此时在自注意力层之间添加了一层交叉注意力，遵循 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中描述的体系结构，作者是Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin。
    # 要将模型用作解码器，模型需要使用配置的 `is_decoder` 参数初始化为 `True`。要在 Seq2Seq 模型中使用，模型需要用两个参数初始化，即 `is_decoder` 和 `add_cross_attention` 设置为 `True`；然后预期在前向传播中输入 `encoder_hidden_states`。

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = RoFormerEmbeddings(config)

        # 如果配置中的嵌入维度不等于隐藏状态维度，则进行嵌入投影
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        self.encoder = RoFormerEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剔除模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        剔除模型的头部。heads_to_prune: 表示{层编号: 要在此层中剔除的头部列表}的字典。参见 PreTrainedModel 基类。
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings("""RoFormer Model with a `language modeling` head on top.""", ROFORMER_START_DOCSTRING)
class RoFormerForMaskedLM(RoFormerPreTrainedModel):
    # 定义了一个列表，存储需要共享权重的关键词
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]
    
    # 继承了 nn.Module 的 RoFormerForMaskedLM 类
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 如果 config 中 is_decoder 属性为 True，打印警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `RoFormerForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        
        # 创建 RoFormerModel 实例
        self.roformer = RoFormerModel(config)
        # 创建 RoFormerOnlyMLMHead 实例
        self.cls = RoFormerOnlyMLMHead(config)
        
        # 初始化权重并执行最终处理
        self.post_init()
    
    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder
    
    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播逻辑在此实现
        pass
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 RoFormer 模型进行前向传播
        outputs = self.roformer(
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

        # 获取序列输出
        sequence_output = outputs[0]
        # 通过分类器得到预测分数
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 计算掩码语言建模损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 若不返回字典格式的输出，则按顺序返回结果
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回字典格式的 MaskedLMOutput
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        # 获取输入形状和有效批次大小
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 添加一个虚拟 token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回准备好的生成输入
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 添加开始文档字符串，描述这个 RoFormer 模型被用于 CLM 微调
@add_start_docstrings(
    """RoFormer Model with a `language modeling` head on top for CLM fine-tuning.""", ROFORMER_START_DOCSTRING
)
# 定义一个 RoFormerForCausalLM 类，继承自 RoFormerPreTrainedModel
class RoFormerForCausalLM(RoFormerPreTrainedModel):
    # 定义需要打包在一起的权重键
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置没有设置 is_decoder 为 True，则输出警告信息
        if not config.is_decoder:
            logger.warning("If you want to use `RoFormerForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建 RoFormerModel 和 RoFormerOnlyMLMHead 对象
        self.roformer = RoFormerModel(config)
        self.cls = RoFormerOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 添加开始文档字符串到模型前向传播
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回文档字符串
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义模型前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入的形状
        input_shape = input_ids.shape

        # 如果没有提供 attention_mask，则创建一个全 1 的 attention_mask
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了 past_key_values，则剪掉输入的前缀长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回准备好的输入
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排序缓存中的过去键值对
    def _reorder_cache(self, past_key_values, beam_idx):
        # 创建空的重新排序后的过去键值对
        reordered_past = ()
        # 遍历每个层的过去键值对
        for layer_past in past_key_values:
            # 将每层的过去键值对中的过去状态按照beam_idx重新排序，并组成新的元组
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        # 返回重新排序后的过去键值对
        return reordered_past
# 用于句级分类任务的分类头部
class RoFormerClassificationHead(nn.Module):
    
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个全连接层，输入大小为配置的隐藏层大小，输出大小也为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个dropout层，以指定的概率丢弃输入元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个全连接层，输入大小为隐藏层大小，输出大小为标签类别数
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        
        # 保存配置信息
        self.config = config

    # 前向传播函数
    def forward(self, features, **kwargs):
        # 取出特征的第一个元素，即[CLS]token的特征向量
        x = features[:, 0, :]
        # 对特征向量应用dropout
        x = self.dropout(x)
        # 通过第一个全连接层
        x = self.dense(x)
        # 应用激活函数
        x = ACT2FN[self.config.hidden_act](x)
        # 再次应用dropout
        x = self.dropout(x)
        # 通过最后一个全连接层得到分类输出
        x = self.out_proj(x)
        return x


# 带有序列分类/回归头的 RoFormer 模型
@add_start_docstrings(
    """
    RoFormer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ROFORMER_START_DOCSTRING,
)
class RoFormerForSequenceClassification(RoFormerPreTrainedModel):
    
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存标签数量
        self.num_labels = config.num_labels
        # 创建 RoFormer 模型
        self.roformer = RoFormerModel(config)
        # 创建分类头部
        self.classifier = RoFormerClassificationHead(config)
        
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 在此处添加注释
        pass
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        Forward pass of the RoFormerForSequenceClassification model.
    
        Args:
            input_ids (torch.LongTensor of shape `(batch_size, sequence_length)`):
                Indices of the input sequence tokens in the vocabulary.
            attention_mask (torch.LongTensor of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding tokens.
            token_type_ids (torch.LongTensor of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. Only necessary for models with
                a sequence classification head.
            position_ids (torch.LongTensor of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings. Selected in the range
                [0, config.max_position_embeddings - 1].
                - if `config.pad_token_id` is defined, pad token positions are replaced by `config.pad_token_id`.
                - if ``config.pad_token_id`` is not defined, input positions are not replaced.
            head_mask (torch.FloatTensor of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (torch.FloatTensor of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (torch.LongTensor of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            output_attentions (bool, *optional*):
                Whether to also ouput attentions weights computed by the model.
            output_hidden_states (bool, *optional*):
                Whether to also output all hidden-states of the model.
            return_dict (bool, *optional*):
                Whether to return a `SequenceClassifierOutput` instead of a tuple of torch tensors. If `return_dict=True`,
                the `SequenceClassifierOutput` object will be returned.
    
        Returns:
            - If `return_dict=False` (default): Returns a tuple of `logits` (torch.FloatTensor) and `hidden_states`
              (torch.FloatTensor) or optional `attentions` (tuple(torch.FloatTensor)).
    
            - If `return_dict=True`: Returns a `SequenceClassifierOutput` instead of a plain tuple.
    
    
        """
        # If return_dict is not set, use the value from the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Perform the forward pass of the RoFormer model
        outputs = self.roformer(
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
    
        sequence_output = outputs[0]  # Retrieve the last hidden state output
        logits = self.classifier(sequence_output)  # Pass the last hidden state through the classifier layer
    
        loss = None
        
        # Compute the loss if labels are provided
        if labels is not None:
            # Infer the problem type based on the number of labels and label data type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"  # Regression problem
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"  # Single-label classification problem
                else:
                    self.config.problem_type = "multi_label_classification"  # Multi-label classification problem
    
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()  # Mean-Square loss
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # Compute loss for single-label regression
                else:
                    loss = loss_fct(logits, labels)  # Compute loss for multi-label regression
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()  # Cross-Entropy loss
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # Compute loss for single-label classification
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()  # Binary Cross-Entropy loss
                loss = loss_fct(logits, labels)  # Compute loss for multi-label classification
        
        # Return the loss and outputs based on return_dict
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 给RoFormer模型添加一个多选分类头部（线性层位于池化输出之上并带有softmax），例如用于RocStories/SWAG任务
class RoFormerForMultipleChoice(RoFormerPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化RoFormer模型
        self.roformer = RoFormerModel(config)
        # 序列总结
        self.sequence_summary = SequenceSummary(config)
        # 分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
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
    ) -> Union[MultipleChoiceModelOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            用于计算多项选择分类损失的标签。索引应该在 `[0, ..., num_choices-1]` 范围内，其中 `num_choices` 是输入张量的第二个维度的大小。 (参见上面的 `input_ids`)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果传入的 return_dict 参数不为空，使用传入的值；否则使用预训练模型配置中的值
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑 input_ids，并对应重塑 attention_mask 和 token_type_ids
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        # 重塑 inputs_embeds
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 输入模型并返回结果
        outputs = self.roformer(
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

        # 对序列输出进行池化
        pooled_output = self.sequence_summary(sequence_output)
        # 使用分类器对池化输出进行分类
        logits = self.classifier(pooled_output)
        # 重塑 logits
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回字典，则返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、logits、隐藏状态和注意力权重的字典
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 RoFormer 模型在顶部增加一个标记分类头部，例如用于命名实体识别（NER）任务
# 此处的标记分类头部指的是在隐藏状态输出之上的线性层
# config：RoFormer 模型的配置对象
class RoFormerForTokenClassification(RoFormerPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用 RoFormerPreTrainedModel 类的初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels

        # 实例化 RoFormerModel，并传入配置对象
        self.roformer = RoFormerModel(config)
        # 实例化 Dropout 层，使用隐藏层丢失概率进行初始化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 实例化线性层，输入大小为隐藏大小，输出大小为类别数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用配置中的 return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # RoFormerModel 的前向传播
        outputs = self.roformer(
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

        # 应用 Dropout
        sequence_output = self.dropout(sequence_output)
        # 线性层，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典，则组合输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            # 如果损失不为 None，则将损失添加到输出中
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # RoFormer 模型，顶部带有用于提取式问答任务（如 SQuAD）的跨度分类头（在隐藏状态输出之上的线性层，用于计算“跨度起始标记”和“跨度结束标记”）。
    # ROFORMER_START_DOCSTRING
# 定义 RoFormerForQuestionAnswering 类，继承自 RoFormerPreTrainedModel 类
class RoFormerForQuestionAnswering(RoFormerPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置 config.num_labels 为 2
        config.num_labels = 2
        # 将 config.num_labels 赋值给 self.num_labels
        self.num_labels = config.num_labels

        # 创建 RoFormerModel 对象
        self.roformer = RoFormerModel(config)
        # 创建一个线性层，作为问答输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 调用 post_init 方法进行权重初始化和最终处理
        # 这里假设 post_init 方法在父类中
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(ROFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[QuestionAnsweringModelOutput, Tuple[torch.Tensor]]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            用于计算标记位置（索引）的开始位置的标签，用于计算令牌分类损失。
            位置被限制在序列的长度(`sequence_length`)内。超出序列范围的位置不会计入损失计算。
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            用于计算标记位置（索引）的结束位置的标签，用于计算令牌分类损失。
            位置被限制在序列的长度(`sequence_length`)内。超出序列范围的位置不会计入损失计算。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU运行，则添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时开始/结束位置超出我们的模型输入，我们忽略这些项
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```