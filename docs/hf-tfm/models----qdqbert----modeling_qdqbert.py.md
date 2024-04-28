# `.\transformers\models\qdqbert\modeling_qdqbert.py`

```
# 设置代码文件的编码格式为 UTF-8
# 版权声明信息，版权属于 NVIDIA Corporation 和 The HuggingFace Team
# 版权声明信息，版权所有 (c) 2018-2021, NVIDIA CORPORATION
# 基于 Apache License, Version 2.0 许可证
# 除非符合许可证的要求，否则禁止使用该文件
# 可以在以下链接获取许可证的拷贝： http://www.apache.org/licenses/LICENSE-2.0
# 未经适用法律要求或书面同意，软件将根据“实际情况”分发，无论是明示的还是隐含的，均无任何担保或条件
# 详细了解许可证规定，包括责任限制和权限限制请参考许可证文档

""" PyTorch QDQBERT model."""

# 导入模块
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_pytorch_quantization_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_qdqbert import QDQBertConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 软依赖
# 如果 PyTorch 量化可用
if is_pytorch_quantization_available():
    try:
        # 导入 PyTorch 量化模块
        from pytorch_quantization import nn as quant_nn
        # 导入张量量化器
        from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
    except OSError:
        logger.error(
            "QDQBERT model are not usable since `pytorch_quantization` can't be loaded. Please try to reinstall it"
            " following the instructions here:"
            " https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
        )

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
# 用于文档的配置
_CONFIG_FOR_DOC = "QDQBertConfig"

# QDQBERT 预训练模型的存档列表
QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    # 查看所有 BERT 模型：https://huggingface.co/models?filter=bert
]

def load_tf_weights_in_qdqbert(model, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 模型路径的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 打印日志，提示正在转换 TensorFlow 检查点
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []  # 保存变量名称
    arrays = []  # 保存变量值
    # 循环处理每个变量
    for name, shape in init_vars:
        # 打印日志，提示正在加载 TF 权重
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 从 TF 模型中获取变量值
        array = tf.train.load_variable(tf_path, name)
        names.append(name)  # 将变量名称添加到列表中
        arrays.append(array)  # 将变量值添加到列表中

    # 遍历每个变量的名称和值
    for name, array in zip(names, arrays):
        name = name.split("/")  # 按斜杠分割变量名
        # 判断变量名是否为不需要加载的特殊变量
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 打印日志，跳过不需要加载的特殊变量
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model  # 指针指向模型
        # 遍历变量名的每个部分
        for m_name in name:
            # 判断变量名是否带有数字后缀
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据变量名的不同部分更新指针位置
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
                    # 打印日志，跳过无法识别的变量名
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # 如果变量名有多个部分，更新指针位置
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果变量名以 "_embeddings" 结尾，指向权重
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        # 如果变量名为 "kernel"，将数组转置
        elif m_name == "kernel":
            array = np.transpose(array)
        # 检查指针和数组形状是否匹配
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # 打印日志，提示初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)  # 将数组转换为 PyTorch 张量
    return model  # 返回加载了权重的模型
# 从transformers.models.bert.modeling_bert.BertEmbeddings复制而来，并将类名修改为QDQBertEmbeddings
class QDQBertEmbeddings(nn.Module):
    """从单词、位置和token类型的嵌入构建嵌入。"""

    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 创建单词嵌入对象，词汇大小为config.vocab_size，隐藏层大小为config.hidden_size，填充id为config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入对象，最大位置为config.max_position_embeddings，隐藏层大小为config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建token类型嵌入对象，类型词汇大小为config.type_vocab_size，隐藏层大小为config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm的命名不采用蛇形命名方式，以保留与TensorFlow模型变量名称的一致性，并能够加载任何TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout对象，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids（1，长度位置嵌入）在内存中是连续的，并且在序列化时被导出
        # 根据config.position_embedding_type的值来确定位置嵌入类型是绝对还是相对
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册名为position_ids的缓冲区，值为0到config.max_position_embeddings-1
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册名为token_type_ids的缓冲区，值为全零，数据类型为长整型
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 函数签名指定了输入参数及其类型，并指定了返回值类型为 torch.Tensor
    ) -> torch.Tensor:
        # 如果输入参数 input_ids 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取输入参数 inputs_embeds 的形状，除了最后一个维度（通常是 batch 维度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置编码参数 position_ids 为空，则使用已注册的位置编码（position_ids）中的一部分，以匹配当前序列长度
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置 token_type_ids 为构造函数中注册的缓冲区，其中所有元素为零，通常在自动生成时发生，
        # 注册的缓冲区有助于用户在跟踪模型时不传递 token_type_ids，解决问题 #5664
        if token_type_ids is None:
            # 如果模型有注册 token_type_ids，则使用其值
            if hasattr(self, "token_type_ids"):
                # 获取已注册的 token_type_ids，并且扩展以匹配当前序列长度
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则创建一个零填充的 token_type_ids 张量，与输入形状相同
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入参数 inputs_embeds 为空，则通过 word_embeddings 层获取输入的嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token_type_embeddings 相加得到总嵌入
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型为 "absolute"，则加上位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对嵌入进行 LayerNormalization
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings
class QDQBertSelfAttention(nn.Module):
    # 定义 QDQBertSelfAttention 类
    def __init__(self, config):
        # 初始化方法，接受一个config对象
        super().__init__()
        # 调用父类构造函数

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果隐藏大小不能整除注意力头数且config没有嵌入大小
            raise ValueError(
                # 抛出值错误异常
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        # 设置注意力头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 计算所有头的大小

        self.query = quant_nn.QuantLinear(config.hidden_size, self.all_head_size)
        # 创建查询线性层
        self.key = quant_nn.QuantLinear(config.hidden_size, self.all_head_size)
        # 创建键线性层
        self.value = quant_nn.QuantLinear(config.hidden_size, self.all_head_size)
        # 创建值线性层

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 使用指定的dropout概率
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 获取位置嵌入类型，如果不存在则使用绝对位置

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 如果是相对键或相对键查询
            self.max_position_embeddings = config.max_position_embeddings
            # 获取最大位置嵌入
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
            # 创建距离嵌入

        self.is_decoder = config.is_decoder
        # 判断是否为解码器

        self.matmul_q_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_k_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_v_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        self.matmul_a_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        # 初始化输入的量化器

    def transpose_for_scores(self, x):
        # 定义转置函数，用于调整形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 计算新的形状
        x = x.view(*new_x_shape)
        # 调整形状
        return x.permute(0, 2, 1, 3)
        # 对特定维度进行转置

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
class QDQBertSelfOutput(nn.Module):
    # 定义 QDQBertSelfOutput 类
    def __init__(self, config):
        # 初始化方法，接受一个config对象
        super().__init__()
        # 调用父类构造函数

        # Quantize Linear layer
        # 量化线性层
        self.dense = quant_nn.QuantLinear(config.hidden_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用LayerNorm层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 使用dropout层

        # Quantize the inputs to the residual add
        # 量化残差加法的输入
        self.add_local_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        # 输入本地量化器
        self.add_residual_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        # 输入残差量化器
```    
    # 前向传播函数，接收隐藏状态和输入张量作为输入
    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态进行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对全连接结果进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 对残差输入进行量化处理
        add_local = self.add_local_input_quantizer(hidden_states)
        # 对输入张量进行量化处理
        add_residual = self.add_residual_input_quantizer(input_tensor)
        # 将量化后的残差输入和量化后的输入张量相加，并通过LayerNorm进行归一化处理
        hidden_states = self.LayerNorm(add_local + add_residual)
        # 返回处理后的隐藏状态
        return hidden_states
# 根据 transformers.models.bert.modeling_bert.BertAttention 改写而来，将类名由 BertAttention 改为 QDQBertAttention
class QDQBertAttention(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建 QDQBertSelfAttention 对象
        self.self = QDQBertSelfAttention(config)
        # 创建 QDQBertSelfOutput 对象
        self.output = QDQBertSelfOutput(config)
        # 创建 pruned_heads 集合，用于存储被剪枝的 attention heads
        self.pruned_heads = set()

    # 剪枝函数，用于剪枝 attention heads
    def prune_heads(self, heads):
        # 如果要剪枝的 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 方法，根据要剪枝的 heads 列表，self.self 中的参数，以及之前被剪枝的 heads，获取要被剪枝的 heads 列表以及对应的 indices
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # 剪枝 linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # 更新超参数并存储剪枝的 heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
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
        # 调用 self.self 的 forward 方法进行 self attention 计算，得到 self_outputs
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 self.output 的 forward 方法，将 self_outputs[0] 和 hidden_states 作为输入，得到 attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构建 outputs，将 attention_output 放在第一个位置，然后是 self_outputs 的其它部分（如果 output_attentions 为 True，则还会包含 attention weights）
        outputs = (attention_output,) + self_outputs[1:]
        # 返回 outputs
        return outputs


class QDQBertIntermediate(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建 quantized Linear layer
        self.dense = quant_nn.QuantLinear(config.hidden_size, config.intermediate_size)
        # 根据 config.hidden_act 的类型选择激活函数，如果 hidden_act 是字符串类型，则在 ACT2FN 字典中找到对应的激活函数，否则直接使用 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states):
        # 将 hidden_states 输入 dense 层得到新的 hidden_states
        hidden_states = self.dense(hidden_states)
        # 将新的 hidden_states 输入激活函数，得到 intermediate_hidden_states
        intermediate_hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回 intermediate_hidden_states
        return intermediate_hidden_states


class QDQBertOutput(nn.Module):
    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 创建一个量化线性层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = quant_nn.QuantLinear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为 config.hidden_size，使用 config.layer_norm_eps 作为eps值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，使用 config.hidden_dropout_prob 作为dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建一个输入量化器，用于量化到residual add的局部输入
        self.add_local_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)
        # 创建一个输入量化器，用于量化到residual add的残差输入
        self.add_residual_input_quantizer = TensorQuantizer(quant_nn.QuantLinear.default_quant_desc_input)

    # 前向传播函数，接受 hidden_states 和 input_tensor 作为输入
    def forward(self, hidden_states, input_tensor):
        # 通过量化线性层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用dropout处理 hidden_states
        hidden_states = self.dropout(hidden_states)
        # 量化到 residual add 的局部输入
        add_local = self.add_local_input_quantizer(hidden_states)
        # 量化到 residual add 的残差输入
        add_residual = self.add_residual_input_quantizer(input_tensor)
        # LayerNorm 处理 residual add 的局部输入和残差输入的和，并将结果赋给 hidden_states
        hidden_states = self.LayerNorm(add_local + add_residual)
        # 返回处理后的 hidden_states
        return hidden_states
# 基于 transformers.models.bert.modeling_bert.BertLayer 实现的 QDQBertLayer 模块
class QDQBertLayer(nn.Module):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 序列长度维度设置为 1
        self.seq_len_dim = 1
        # 创建 QDQBertAttention 模块
        self.attention = QDQBertAttention(config)
        # 判断是否为解码器模型
        self.is_decoder = config.is_decoder
        # 判断是否需要添加交叉注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 如果需要添加交叉注意力，且不是解码器模型，则抛出错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建 QDQBertAttention 模块用于交叉注意力
            self.crossattention = QDQBertAttention(config)
        # 创建 QDQBertIntermediate 模块
        self.intermediate = QDQBertIntermediate(config)
        # 创建 QDQBertOutput 模块
        self.output = QDQBertOutput(config)

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
        # 如果过去的键/值缓存不为空，则decoder的uni-directional自注意力的缓存键/值元组位于位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用自注意力机制处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 输出不包括最后一个元素，即自注意力缓存
            outputs = self_attention_outputs[1:-1]
            # 获取当前时刻的键/值
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果输出注意权重，添加自注意力
            outputs = self_attention_outputs[1:]  # 如果我们输出注意权重，添加自注意力
        

        cross_attn_present_key_value = None
        # 如果是decoder且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未设置crossattention，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 跨注意力的缓存键/值元组位于过去键/值元组的第3,4位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用crossattention处理自注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取crossattention输出
            attention_output = cross_attention_outputs[0]
            # 输出不包括crossattention缓存
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果我们输出注意权重，添加crossattention

            # 将cross-attn缓存添加到当前键/值元组的第3,4位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用前馈网络处理注意力输出
        layer_output = self.feed_forward_chunk(attention_output)
        # 将前馈网络输出作为输出的第一个元素
        outputs = (layer_output,) + outputs

        # 如果是decoder，将注意力键/值作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 前馈网络的部分处理
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 基于 transformers.models.bert.modeling_bert.BertEncoder 实现的 QDQBertEncoder 类
class QDQBertEncoder(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建 QDQBertLayer 模块列表，数量为配置中指定的隐藏层数
        self.layer = nn.ModuleList([QDQBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点标志为 False
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
        ):
            # 初始化存储所有隐藏状态的元组
            all_hidden_states = () if output_hidden_states else None
            # 初始化存储所有自注意力权重的元组
            all_self_attentions = () if output_attentions else None
            # 若输出注意力权重且存在交叉注意力，则初始化存储所有交叉注意力权重的元组
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

            # 若使用缓存，则初始化下一个解码器缓存的元组
            next_decoder_cache = () if use_cache else None
            # 遍历每个编码器层
            for i, layer_module in enumerate(self.layer):
                # 若输出隐藏状态，则将当前隐藏状态存入所有隐藏状态元组中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 获取当前层的头部掩码
                layer_head_mask = head_mask[i] if head_mask is not None else None
                # 获取过去的键值对
                past_key_value = past_key_values[i] if past_key_values is not None else None

                # 如果启用渐变检查点并处于训练模式，则执行渐变检查点函数
                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        # 如果使用缓存与渐变检查点不兼容，则警告并关闭使用缓存
                        logger.warning_once(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False
                    # 执行梯度检查点函数
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
                    # 否则直接通过当前层模块计算输出
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        past_key_value,
                        output_attentions,
                    )

                # 更新隐藏状态为当前层的输出
                hidden_states = layer_outputs[0]
                # 若使用缓存，则将当前层的输出加入下一个解码器缓存中
                if use_cache:
                    next_decoder_cache += (layer_outputs[-1],)
                # 若输出注意力权重，则将当前层的注意力权重加入所有自注意力的元组中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                    # 若模型具有交叉注意力，将当前层的交叉注意力加入所有交叉注意力的元组中
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # 若输出隐藏状态，则将最终隐藏状态加入所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 若不返回字典形式的结果，则以非字典形式返回结果元组
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
            # 否则以字典形式返回模型输出，包括过去的键值对和交叉注意力
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_decoder_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )
# 从 transformers.models.bert.modeling_bert.BertPooler 复制并修改为 QDQBertPooler 类
class QDQBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个激活函数对象，使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个 token 对应的隐藏状态来对模型进行“池化”
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 对应的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 将全连接层的输出通过激活函数进行激活
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 从 transformers.models.bert.modeling_bert.BertPredictionHeadTransform 复制并修改为 QDQBertPredictionHeadTransform 类
class QDQBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果 config.hidden_act 是字符串类型，则根据 ACT2FN 字典映射激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果通过激活函数进行激活
        hidden_states = self.transform_act_fn(hidden_states)
        # 对激活后的隐藏状态进行归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 返回变换后的隐藏状态
        return hidden_states


# 基于 transformers.models.bert.modeling_bert.BertLMPredictionHead 复制并修改为 QDQBertLMPredictionHead 类
class QDQBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 QDQBertPredictionHeadTransform 实例
        self.transform = QDQBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个 token 都有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个参数 tensor 用于偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来确保偏置能够与 `resize_token_embeddings` 正确地调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 通过 QDQBertPredictionHeadTransform 对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        # 通过 decoder 进行线性变换
        hidden_states = self.decoder(hidden_states)
        # 返回变换后的隐藏状态
        return hidden_states


# 基于 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制并修改为 QDQBertOnlyMLMHead 类
class QDQBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 QDQBertLMPredictionHead 实例
        self.predictions = QDQBertLMPredictionHead(config)

    def forward(self, sequence_output):
        # 通过 predictions 对序列输出进行预测
        prediction_scores = self.predictions(sequence_output)
        # 返回预测分数
        return prediction_scores


# 从 transformers.models.bert.modeling_bert.BertOnlyNSPHead 复制并修改为 QDQBertOnlyNSPHead 类
class QDQBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度是 config.hidden_size，输出维度是 2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 定义一个方法，用于对输入的池化输出进行向前传播
    def forward(self, pooled_output):
        # 通过调用 seq_relationship 方法计算序列关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回计算得到的序列关系得分
        return seq_relationship_score
# 基于PreTrainedModel类的初始化权重和下载和加载预训练模型的简单接口的抽象基类
class QDQBertPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化，并提供一个简单的接口来下载和加载预训练模型。

    config类：QDQBertConfig
    load_tf_weights方法：load_tf_weights_in_qdqbert
    base_model_prefix：bert
    supports_gradient_checkpointing：True

    _init_weights方法：初始化权重
"""
    # 配置类
    config_class = QDQBertConfig
    # 加载TF格式的权重
    load_tf_weights = load_tf_weights_in_qdqbert
    # 模型前缀为bert
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """
        初始化权重
        """
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 从标准正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将偏置项初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 从标准正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将填充索引的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)


# 基于BertPreTrainingHeads类的初始化和前向传播方法的子类，将Bert替换为QDQBert
class QDQBertPreTrainingHeads(nn.Module):
    """
    基于BertPreTrainingHeads类的初始化和前向传播方法的子类，将Bert替换为QDQBert
    """
    def __init__(self, config):
        super().__init__()
        # 创建QDQBertLMPredictionHead对象
        self.predictions = QDQBertLMPredictionHead(config)
        # 用线性层将config.hidden_size维度的向量映射为2维向量
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 调用QDQBertLMPredictionHead的forward方法进行预测，并获得预测分数
        prediction_scores = self.predictions(sequence_output)
        # 调用线性层，将pooled_output映射为2维向量，用于判断句子之间的关系
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测分数和句子关系分数
        return prediction_scores, seq_relationship_score


QDQBERT_START_DOCSTRING = r"""

    本模型继承自[`PreTrainedModel`]。请查看超类文档以了解库实现的常见方法，例如下载或保存模型，调整输入嵌入大小，修剪头等。

    该模型还是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。
    将其用作常规的PyTorch模块，并参考PyTorch文档以获取有关一般用法和行为的所有事项。

    参数:
        config ([`QDQBertConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件进行初始化不会加载与模型关联的权重，只会加载配置信息。
            可以查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

QDQBERT_INPUTS_DOCSTRING = r"""
    # 定义函数输入参数
    Args:
        # input_ids 是输入序列标记在词汇表中的索引
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            # 通过 AutoTokenizer 可以获得索引。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 获取更多细节。
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
        
        # attention_mask 是用来避免在填充标记索引上执行注意力的掩码
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 代表不被遮蔽的标记
            - 0 代表被遮蔽的标记
        # token_type_ids 是用来指示输入的第一部分和第二部分的片段标记索引
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 0 对应一个句子 A 的标记
            - 1 对应一个句子 B 的标记
        # position_ids 是输入序列标记在位置嵌入中的位置索引
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
        # head_mask 是用来清除自注意力模块的选择头部的掩码
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 表示头部没有被遮蔽
            - 0 表示头部被遮蔽
        # inputs_embeds 是直接传递嵌入表示
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. 
            # 如果希望对如何将 input_ids 索引转换为关联向量有更多控制权，这将非常有用
        # output_attentions 标志着是否返回所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # output_hidden_states 标志着是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # return_dict 标志着是否返回一个 ModelOutput 对象而不是一个普通元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# QDQBERT 模型类，输出原始隐藏状态而不带任何特定的顶部头信息
# QDQBERT_START_DOCSTRING 模型的文档字符串
class QDQBertModel(QDQBertPreTrainedModel):
    """

    模型可以作为编码器（仅具有自注意力）以及解码器使用，在后一种情况下，在自注意力层之间添加了一个交叉注意力层，遵循 [注意力就是一切](https://arxiv.org/abs/1706.03762) 的架构，由 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser 和 Illia Polosukhin 描述。

    要作为解码器工作，模型需要使用配置集中的 `is_decoder` 参数初始化为 `True`。要在 Seq2Seq 模型中使用，模型需要使用 `is_decoder` 参数和 `add_cross_attention` 参数均初始化为 `True`；然后预期将 `encoder_hidden_states` 作为前向传递的输入。
    """

    def __init__(self, config, add_pooling_layer: bool = True):
        requires_backends(self, "pytorch_quantization")
        调用基类的初始化函数，传入配置参数
        super().__init__(config)
        self.config = config

        初始化嵌入层模块
        self.embeddings = QDQBertEmbeddings(config)
        初始化编码器模块
        self.encoder = QDQBertEncoder(config)

        如果 add_pooling_layer 为真，则初始化池化层模块, 否则为 None
        self.pooler = QDQBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        调用后续初始化函数
        self.post_init()

    返回输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 精简模型的头部
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        遍历需要精简的层和头部列表
        for layer, heads in heads_to_prune.items():
            调用基类 PreTrainedModel 的 attention.prune_heads 函数对头部进行裁剪

    在模型的前向传递上添加注释
    上面关于 QDQBERT 输入参数的文档字符串
    # 添加代码示例的文档字符串
    # 正向传播函数，用于模型的前向推断过程
    def forward(
        self,
        # 输入的 token ID 序列，可选参数，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力遮罩，可选参数，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,
        # 分段 ID 序列，可选参数，默认为 None
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 ID 序列，可选参数，默认为 None
        position_ids: Optional[torch.LongTensor] = None,
        # 头部遮罩，可选参数，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入的嵌入张量，可选参数，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 编码器隐藏状态，可选参数，默认为 None
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 编码器注意力遮罩，可选参数，默认为 None
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 过去的键值对，可选参数，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 是否使用缓存，可选参数，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，可选参数，默认为 None
        return_dict: Optional[bool] = None,
# 导入所需要的模块
@add_start_docstrings(
    """QDQBERT Model with a `language modeling` head on top for CLM fine-tuning.""", QDQBERT_START_DOCSTRING
)
class QDQBertLMHeadModel(QDQBertPreTrainedModel):
    # 合并部分权重信息
    _tied_weights_keys = ["predictions.decoder.weight", "predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 如果不是解码器的话，给出警告
        if not config.is_decoder:
            logger.warning("If you want to use `QDQBertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建BERT模型
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        # 创建MLM头部模型
        self.cls = QDQBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播的函数
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.LongTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 生成预测输入
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor],
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs,
    ):
        # 获取输入的形状
        input_shape = input_ids.shape
        # 如果模型用作编码器-解码器模型中的解码器，那么将在需要时创建解码器注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用过去的键值对，则切断解码器输入ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧的行为：仅保留最后的ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回输入ID、注意力掩码和过去的键值对
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存内容
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        # 遍历过去的键值对，重新排序后添加到reordered_past中
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 给 QDQBERT 一个带有顶层 `语言建模` 头部的模型
@add_start_docstrings("""QDQBERT Model with a `language modeling` head on top.""", QDQBERT_START_DOCSTRING)
class QDQBertForMaskedLM(QDQBertPreTrainedModel):
    # 指定需要绑定权重的键值对
    _tied_weights_keys = ["predictions.decoder.weight", "predictions.decoder.bias"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置中设置为decoder，则警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `QDQBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建bert模型
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        # 创建仅包含MLM头部的cls
        self.cls = QDQBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出embedding
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出embedding
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个方法，用于生成给定输入的掩码语言建模损失输出
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
    
        # 检查是否使用返回字典，如果为 None 则使用配置文件中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 使用 BERT 模型处理输入
        outputs = self.bert(
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
    
        # 获取序列输出
        sequence_output = outputs[0]
        # 使用分类线性层得到预测分数
        prediction_scores = self.cls(sequence_output)
    
        # 初始化掩码语言建模损失为空
        masked_lm_loss = None
        # 如果存在标签，则计算掩码语言建模损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    
        # 如果不返回字典，则按特定顺序返回输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
        # 返回掩码语言建模的输出对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 为生成准备输入，添加一个虚拟 token
    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, attention_mask: Optional[torch.FloatTensor] = None, **model_kwargs
    ):
        # 获取输入的形状和有效批次大小
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]
    
        # 如果 PAD token 未定义，抛出 ValueError
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
    
        # 为每个输入添加一个虚拟 token，并扩展对应的注意力掩码
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
    
        # 返回包含输入和注意力掩码的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 以一个带有“下一个句子预测（分类）”头部的Bert模型
@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    QDQBERT_START_DOCSTRING,
)
class QDQBertForNextSentencePrediction(QDQBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化Bert模型
        self.bert = QDQBertModel(config)
        # 初始化仅包含NSP头部的模块
        self.cls = QDQBertOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受输入，并返回输出
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
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
        **kwargs,
    ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, QDQBertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = QDQBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```"""

        # 检查是否传入了旧版参数"next_sentence_label"，如果是，发出警告，并使用"labels"替代
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        # 检查是否需要返回字典形式的输出，如果未指定，则根据模型配置决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给BERT模型，获取输出
        outputs = self.bert(
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

        # 从BERT模型的输出中提取池化后的表示
        pooled_output = outputs[1]

        # 使用分类器层对池化后的表示进行分类，得到下一句关系的分数
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        # 如果传入了标签，则计算下一句预测的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 如果不需要以字典形式返回输出，则将输出组装成元组形式返回
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 如果需要以字典形式返回输出，则构造NextSentencePredictorOutput对象返回
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 BERT 模型进行序列分类/回归任务的变换器，顶部有一个线性层 (在汇总输出之上)，例如用于 GLUE 任务。
class QDQBertForSequenceClassification(QDQBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        self.config = config

        # 创建 QDQBertModel 实例
        self.bert = QDQBertModel(config)
        # 用于随机关闭单元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 线性层，处理隐藏层数据
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 初始化权重并应用最终处理
        self.post_init()

    # 向前传递方法
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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0,...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 BERT 模型进行处理
        outputs = self.bert(
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

        # 获取汇聚输出
        pooled_output = outputs[1]

        # 使用 Dropout 进行汇聚输出的处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器预测
        logits = self.classifier(pooled_output)

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
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 基于 Bert 模型构建一个多选题分类模型，用于例如 RocStories/SWAG 等任务
class QDQBertForMultipleChoice(QDQBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型参数
        super().__init__(config)

        # 初始化 Bert 模型
        self.bert = QDQBertModel(config)
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个线性层用于分类，输出一个值（用于二分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接收多个输入参数并输出结果
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
    # 定义一个多选分类任务的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        # 如果 return_dict 为 None，则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算输入的选项数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    
        # 将输入重塑为 (batch_size * num_choices, seq_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
    
        # 通过 BERT 模型前向传播
        outputs = self.bert(
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
    
        # 获取池化后的输出
        pooled_output = outputs[1]
    
        # 对池化后的输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 通过分类器得到逻辑分数
        logits = self.classifier(pooled_output)
        # 将逻辑分数重塑为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)
    
        # 计算损失函数
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
    
        # 根据 return_dict 返回不同的输出格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 在 QDQBERT 模型上增加一个标记分类头部（位于隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务
class QDQBertForTokenClassification(QDQBertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels

        # QDQBERT 模型
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        标签（`torch.LongTensor` 格式为 `(batch_size, sequence_length)`，*可选*）：
            用于计算标记分类损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 内。
        """
        # 若 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输出
        outputs = self.bert(
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

        # 序列输出
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        # 计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要返回字典
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 对 QDQBERT 模型添加起始文档字符串
"""
    # 这是一个 QDQBERT 模型的描述文档字符串。
    # QDQBERT 模型是一个用于抽取式问答任务（如 SQuAD）的模型
    # 它在隐藏状态的输出上添加了一个跨度分类头
    # 用于计算 span start logits 和 span end logits
    QDQBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    QDQBERT_START_DOCSTRING,
# 定义 QDQBertForQuestionAnswering 类，继承自 QDQBertPreTrainedModel
class QDQBertForQuestionAnswering(QDQBertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取配置中的标签数量
        self.num_labels = config.num_labels

        # 初始化 BERT 模型，不添加池化层
        self.bert = QDQBertModel(config, add_pooling_layer=False)
        # 初始化输出层，将隐藏层的输出映射到标签的数量上
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(QDQBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用提供的输入参数调用BERT模型
        outputs = self.bert(
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

        # 从BERT输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入QA输出层获得logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上操作，增加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入范围的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # 计算起始和结束位置的损失
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```