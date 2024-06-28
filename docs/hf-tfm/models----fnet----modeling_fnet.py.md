# `.\models\fnet\modeling_fnet.py`

```py
# 设置源代码文件的编码格式为UTF-8
# 版权声明，2021年由Google Research和HuggingFace Inc.团队保留所有权利
# 根据Apache许可证2.0版（“许可证”）许可，除非符合许可证的使用，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“许可证”分发的软件是基于“原样”提供的，不提供任何形式的明示或暗示担保或条件。
# 有关特定语言的权限，请参阅许可证。

""" PyTorch FNet model."""

# 导入警告模块
import warnings
# 导入dataclass用于数据类
from dataclasses import dataclass
# 导入partial函数用于创建偏函数
from functools import partial
# 导入类型提示相关模块
from typing import Optional, Tuple, Union

# 导入PyTorch相关模块
import torch
# 导入PyTorch中的checkpoint功能
import torch.utils.checkpoint
# 导入PyTorch中的神经网络模块
from torch import nn
# 导入PyTorch中的损失函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 判断是否存在SciPy库，用于后续可能的特定操作
from ...utils import is_scipy_available

# 如果SciPy库可用，则导入linalg模块
if is_scipy_available():
    from scipy import linalg

# 导入激活函数映射表
from ...activations import ACT2FN
# 导入模型输出相关类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型工具函数和预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入PyTorch工具函数，用于前向计算时的分块应用
from ...pytorch_utils import apply_chunking_to_forward
# 导入通用工具函数，包括日志记录等
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入FNet模型的配置类
from .configuration_fnet import FNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "google/fnet-base"
_CONFIG_FOR_DOC = "FNetConfig"

# 预训练模型的存档列表
FNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/fnet-base",
    "google/fnet-large",
    # 更多FNet模型详见https://huggingface.co/models?filter=fnet
]

# 从https://github.com/google-research/google-research/blob/master/f_net/fourier.py适配而来
def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """Applies 2D matrix multiplication to 3D input arrays."""
    # 获取序列长度
    seq_length = x.shape[1]
    # 裁剪矩阵以匹配序列长度
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    # 将输入张量转换为复数类型
    x = x.type(torch.complex64)
    # 执行张量乘法操作
    return torch.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)


# 从https://github.com/google-research/google-research/blob/master/f_net/fourier.py适配而来
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """Applies 2D matrix multiplication to 3D input arrays."""
    # 调用内部函数_two_dim_matmul执行操作
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)


# 从https://github.com/google-research/google-research/blob/master/f_net/fourier.py适配而来
def fftn(x):
    """
    Applies n-dimensional Fast Fourier Transform (FFT) to input array.

    Args:
        x: Input n-dimensional array.

    Returns:
        n-dimensional Fourier transform of input n-dimensional array.
    """
    # 将输入直接返回，实际实现可能在此基础上增加FFT操作
    out = x
    # 对输入张量 x 进行逆序遍历其除了最后一个轴以外的所有轴
    for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
        # 对张量 out 在指定的轴上应用 FFT 变换
        out = torch.fft.fft(out, axis=axis)
    # 返回应用完 FFT 后的张量 out
    return out
class FNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，用于将词汇索引映射为隐藏状态向量，支持填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，用于将位置索引映射为隐藏状态向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化标记类型嵌入层，用于将标记类型索引映射为隐藏状态向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 初始化层归一化层，用于归一化隐藏状态向量，保持与 TensorFlow 模型变量名的一致性以便加载 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        # 初始化投影层，用于将隐藏状态向量投影到另一个隐藏状态空间
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化丢弃层，用于在训练过程中随机丢弃部分隐藏状态向量，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 注册缓冲区 position_ids，用于存储位置索引，作为持久化数据不会随模型参数保存
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # 注册缓冲区 token_type_ids，用于存储标记类型索引，初始化为全零
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    # 定义模型的前向传播函数，接收输入的标识符、标记类型ID、位置ID和嵌入输入
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果传入了input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取inputs_embeds的形状，排除最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 如果未提供position_ids，则使用模型中注册的缓冲区，截取到seq_length长度的部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 设置token_type_ids为模型构造函数中注册的缓冲区，通常是全零，用于在不传递token_type_ids时帮助用户追踪模型，解决问题#5664
        if token_type_ids is None:
            # 如果模型具有token_type_ids属性，则使用它的缓冲区值
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则创建一个全零的tensor作为token_type_ids，类型为长整型，位于与self.position_ids设备相同的设备上
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供inputs_embeds，则使用word_embeddings对input_ids进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 根据token_type_ids获取token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的嵌入向量，包括输入的嵌入和token type embeddings
        embeddings = inputs_embeds + token_type_embeddings

        # 根据position_ids获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)
        
        # 将位置嵌入加到当前的嵌入向量中
        embeddings += position_embeddings
        
        # 对嵌入向量进行LayerNormalization处理
        embeddings = self.LayerNorm(embeddings)
        
        # 将嵌入向量投影到最终的输出空间
        embeddings = self.projection(embeddings)
        
        # 对投影后的向量进行dropout操作，用于防止过拟合
        embeddings = self.dropout(embeddings)
        
        # 返回最终的嵌入向量作为前向传播的结果
        return embeddings
# 定义 FNetBasicFourierTransform 类，继承自 nn.Module
class FNetBasicFourierTransform(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 调用 _init_fourier_transform 方法进行初始化
        self._init_fourier_transform(config)

    # 初始化傅里叶变换方法
    def _init_fourier_transform(self, config):
        # 如果配置指示不使用 TPU 傅里叶优化
        if not config.use_tpu_fourier_optimizations:
            # 使用 torch.fft.fftn 作为傅里叶变换的部分函数，指定变换维度为 (1, 2)
            self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))
        # 如果配置中最大位置嵌入小于等于 4096
        elif config.max_position_embeddings <= 4096:
            # 检查是否有 SciPy 库可用
            if is_scipy_available():
                # 注册隐藏大小的 DFT（离散傅里叶变换）矩阵为缓冲区
                self.register_buffer(
                    "dft_mat_hidden", torch.tensor(linalg.dft(config.hidden_size), dtype=torch.complex64)
                )
                # 注册序列长度的 DFT 矩阵为缓冲区
                self.register_buffer(
                    "dft_mat_seq", torch.tensor(linalg.dft(config.tpu_short_seq_length), dtype=torch.complex64)
                )
                # 使用自定义的两个维度矩阵乘法作为傅里叶变换的部分函数
                self.fourier_transform = partial(
                    two_dim_matmul, matrix_dim_one=self.dft_mat_seq, matrix_dim_two=self.dft_mat_hidden
                )
            else:
                # 如果没有找到 SciPy 库，则记录警告并使用 fftn 作为傅里叶变换
                logging.warning(
                    "SciPy is needed for DFT matrix calculation and is not found. Using TPU optimized fast fourier"
                    " transform instead."
                )
                self.fourier_transform = fftn
        else:
            # 如果不满足上述条件，则使用 fftn 作为傅里叶变换
            self.fourier_transform = fftn

    # 前向传播方法
    def forward(self, hidden_states):
        # 输出通过傅里叶变换后的实部
        outputs = self.fourier_transform(hidden_states).real
        return (outputs,)


# 定义 FNetBasicOutput 类，继承自 nn.Module
class FNetBasicOutput(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法
    def forward(self, hidden_states, input_tensor):
        # 对输入张量和隐藏状态进行 LayerNorm 处理
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        return hidden_states


# 定义 FNetFourierTransform 类，继承自 nn.Module
class FNetFourierTransform(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 创建 FNetBasicFourierTransform 实例作为 self 属性
        self.self = FNetBasicFourierTransform(config)
        # 创建 FNetBasicOutput 实例作为 output 属性
        self.output = FNetBasicOutput(config)

    # 前向传播方法
    def forward(self, hidden_states):
        # 调用 self 实例的前向传播方法
        self_outputs = self.self(hidden_states)
        # 将 self 的输出与隐藏状态作为输入，调用 output 实例的前向传播方法
        fourier_output = self.output(self_outputs[0], hidden_states)
        # 返回输出元组
        outputs = (fourier_output,)
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 FNetIntermediate 类
class FNetIntermediate(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏大小转换为中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串，则从 ACT2FN 字典获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则使用配置中的激活函数
            self.intermediate_act_fn = config.hidden_act
    # 定义一个方法 `forward`，接收一个名为 `hidden_states` 的张量参数，并返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层 `self.dense` 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的张量应用激活函数 `self.intermediate_act_fn`
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过线性变换和激活函数处理后的张量结果
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertOutput 复制代码，并将 Bert 替换为 FNet
class FNetOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将中间大小的特征向量映射到隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 层，对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃隐藏状态中的一部分特征，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 和原始输入的残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 用于分块前馈的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度所在的维度
        self.seq_len_dim = 1  # The dimension which has the sequence length
        # Fourier 变换层
        self.fourier = FNetFourierTransform(config)
        # 中间层
        self.intermediate = FNetIntermediate(config)
        # 输出层
        self.output = FNetOutput(config)

    def forward(self, hidden_states):
        # Fourier 变换的输出
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]

        # 将前馈应用到每个块
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, fourier_output
        )

        outputs = (layer_output,)

        return outputs

    def feed_forward_chunk(self, fourier_output):
        # 中间层的输出
        intermediate_output = self.intermediate(fourier_output)
        # 输出层的输出
        layer_output = self.output(intermediate_output, fourier_output)
        return layer_output


class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 多层 FNetLayer 堆叠
        self.layer = nn.ModuleList([FNetLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        # 是否输出所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 如果启用梯度检查点，使用梯度检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states)
            else:
                # 否则直接调用层的前向传播
                layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回模型输出，包括最后一个隐藏状态和所有隐藏状态（如果输出所有隐藏状态）
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
    # 初始化方法，接受一个config对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入和输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数选择为双曲正切函数
        self.activation = nn.Tanh()

    # 前向传播方法，接受一个形状为[batch_size, sequence_length, hidden_size]的张量作为输入，
    # 返回一个形状为[batch_size, hidden_size]的张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从隐藏状态张量中取出每个样本的第一个token的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个token的隐藏状态传入全连接层进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 对线性变换后的结果应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回激活后的结果作为最终的输出张量
        return pooled_output
# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制代码，将Bert->FNet
class FNetPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出大小都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果config.hidden_act是字符串，则使用ACT2FN字典中对应的激活函数；否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 初始化LayerNorm层，输入大小为config.hidden_size，设置eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用LayerNorm进行归一化
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 复制自transformers.models.bert.modeling_bert.BertLMPredictionHead，将Bert->FNet
class FNetLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测头部的变换层
        self.transform = FNetPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个token有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化偏置参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将decoder的偏置设置为初始化的偏置参数
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 输入经过变换层
        hidden_states = self.transform(hidden_states)
        # 经过线性层得到预测分数
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self):
        # 如果权重断开连接（在TPU上或者调整偏置大小时），重新绑定偏置
        self.bias = self.decoder.bias


# 复制自transformers.models.bert.modeling_bert.BertOnlyMLMHead，将Bert->FNet
class FNetOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化MLM头部的预测
        self.predictions = FNetLMPredictionHead(config)

    def forward(self, sequence_output):
        # 使用预测层进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 复制自transformers.models.bert.modeling_bert.BertOnlyNSPHead，将Bert->FNet
class FNetOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化NSP头部的序列关系预测层
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 使用线性层计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# 复制自transformers.models.bert.modeling_bert.BertPreTrainingHeads，将Bert->FNet
class FNetPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测头部和序列关系头部
        self.predictions = FNetLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 分别计算预测分数和序列关系分数
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# 继承自PreTrainedModel的FNet预训练模型
class FNetPreTrainedModel(PreTrainedModel):
    """
    FNet预训练模型基类
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FNetConfig
    base_model_prefix = "fnet"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 注意：原始代码中偏置的初始化和权重相同
            if module.bias is not None:
                # 将偏置数据初始化为零
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有 padding_idx，则将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置数据初始化为零
            module.bias.data.zero_()
            # 将权重数据初始化为 1.0
            module.weight.data.fill_(1.0)
    """
    FNetForPreTrainingOutput 类定义了预训练模型的输出类型，继承自 ModelOutput。

    Args:
        loss (torch.FloatTensor, 可选): 当提供 `labels` 时返回，表示总损失，包括掩码语言建模损失和下一个序列预测（分类）损失，形状为 `(1,)`。
        prediction_logits (torch.FloatTensor): 语言建模头部的预测分数，即 SoftMax 之前的每个词汇标记的分数，形状为 `(batch_size, sequence_length, config.vocab_size)`。
        seq_relationship_logits (torch.FloatTensor): 下一个序列预测（分类）头部的预测分数，即 SoftMax 之前的 True/False 连续性的分数，形状为 `(batch_size, 2)`。
        hidden_states (tuple(torch.FloatTensor), 可选): 当 `output_hidden_states=True` 被传递或 `config.output_hidden_states=True` 时返回，包含模型每一层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`。

    """

FNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`FNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FNET_INPUTS_DOCSTRING = r"""
    输入参数说明：
    # 接受输入的索引序列，表示输入序列中的单词在词汇表中的索引
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        # 表示段落标记索引，用于指示输入的第一部分和第二部分。索引值为 0 或 1：
        # - 0 对应于*句子 A* 的标记，
        # - 1 对应于*句子 B* 的标记。
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        # 表示输入序列中每个单词的位置索引，用于位置嵌入。索引值选择在范围 `[0, config.max_position_embeddings - 1]` 内。
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        # 选项参数，可以选择直接传递嵌入表示而不是传递 `input_ids`。如果您想要比模型内部嵌入查找矩阵更多控制权，则这是有用的。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        # 是否返回所有层的隐藏状态。有关更多细节，请查看返回张量中的 `hidden_states`。
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare FNet Model transformer outputting raw hidden-states without any specific head on top.",
    FNET_START_DOCSTRING,
)
class FNetModel(FNetPreTrainedModel):
    """

    The model can behave as an encoder, following the architecture described in [FNet: Mixing Tokens with Fourier
    Transforms](https://arxiv.org/abs/2105.03824) by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 FNetEmbeddings 对象，用于处理模型的嵌入层
        self.embeddings = FNetEmbeddings(config)
        
        # 初始化 FNetEncoder 对象，用于处理模型的编码器层
        self.encoder = FNetEncoder(config)

        # 如果需要添加池化层，则初始化 FNetPooler 对象，否则为 None
        self.pooler = FNetPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层的单词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置嵌入层的单词嵌入
        self.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids
        elif input_ids is not None:
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        # 如果指定了 inputs_embeds
        elif inputs_embeds is not None:
            # 获取 inputs_embeds 的形状，排除最后一维
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出数值错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果配置中启用了 TPU Fourier 优化，并且序列长度小于等于 4096，并且配置的 tpu_short_seq_length 不等于当前序列长度
        if (
            self.config.use_tpu_fourier_optimizations
            and seq_length <= 4096
            and self.config.tpu_short_seq_length != seq_length
        ):
            # 抛出数值错误，提示需要设置正确的 tpu_short_seq_length
            raise ValueError(
                "The `tpu_short_seq_length` in FNetConfig should be set equal to the sequence length being passed to"
                " the model when using TPU optimizations."
            )

        # 根据是否存在 input_ids 来确定设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果 token_type_ids 未指定
        if token_type_ids is None:
            # 如果 embeddings 拥有 token_type_ids 属性
            if hasattr(self.embeddings, "token_type_ids"):
                # 从 embeddings 中获取 token_type_ids，并截取到序列长度的部分，然后扩展到整个 batch
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果 embeddings 没有 token_type_ids 属性，则创建一个全零的 tensor
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 使用 embeddings 进行前向传播
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # 使用 encoder 进行编码器的前向传播
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]

        # 如果存在 pooler，则使用 pooler 对序列输出进行池化
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不要求返回字典形式的输出，则返回元组形式的结果
        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]

        # 否则，返回一个带池化的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
        )
"""
FNet Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
sentence prediction (classification)` head.
"""
# 导入所需的函数和类，用于添加文档字符串
@add_start_docstrings(
    """
    FNet Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    FNET_START_DOCSTRING,
)
# 定义 FNetForPreTraining 类，继承自 FNetPreTrainedModel
class FNetForPreTraining(FNetPreTrainedModel):
    # 定义用于权重共享的关键键列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 初始化 FNetModel 模型
        self.fnet = FNetModel(config)
        # 初始化 FNetPreTrainingHeads 模型
        self.cls = FNetPreTrainingHeads(config)

        # 调用后处理函数，初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层的函数
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层的函数
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=FNetForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数：输入的张量数据、标签、下一个句子标签、是否返回隐藏状态、是否返回字典类型的结果

        return self.fnet(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            next_sentence_label=next_sentence_label,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


注释：

        # 调用 FNetModel 的前向传播方法，将参数传递给 fnet 对象
        return self.fnet(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            next_sentence_label=next_sentence_label,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


这段代码定义了一个 FNetForPreTraining 类，它包含了 FNet 模型的预训练结构，包括一个掩码语言建模头和一个下一个句子预测头。它实现了前向传播方法，调用了内部的 FNetModel 的前向传播函数。
    ) -> Union[Tuple, FNetForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:
            Returns an instance of `FNetForPreTrainingOutput` containing various outputs from the model.

        Example:

        ```
        >>> from transformers import AutoTokenizer, FNetForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        >>> model = FNetForPreTraining.from_pretrained("google/fnet-base")
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```"""
        # Determine whether to use the return_dict mode based on the input argument or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the FNet model with specified inputs and configurations
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the relevant outputs from the FNet model's output
        sequence_output, pooled_output = outputs[:2]

        # Perform classification on the extracted outputs
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # Initialize total_loss to None
        total_loss = None

        # Compute total loss if both labels and next_sentence_label are provided
        if labels is not None and next_sentence_label is not None:
            # Define the CrossEntropyLoss criterion
            loss_fct = CrossEntropyLoss()
            
            # Calculate the masked language modeling (MLM) loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            
            # Calculate the next sentence prediction (NSP) loss
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            
            # Aggregate total loss from MLM and NSP losses
            total_loss = masked_lm_loss + next_sentence_loss

        # Return the appropriate outputs based on the return_dict flag
        if not return_dict:
            # If return_dict is False, return a tuple of outputs
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        else:
            # If return_dict is True, return FNetForPreTrainingOutput object
            return FNetForPreTrainingOutput(
                loss=total_loss,
                prediction_logits=prediction_scores,
                seq_relationship_logits=seq_relationship_score,
                hidden_states=outputs.hidden_states,
            )
# 使用装饰器为类添加文档字符串，描述了其带有语言建模头部的 FNet 模型
@add_start_docstrings("""FNet Model with a `language modeling` head on top.""", FNET_START_DOCSTRING)
class FNetForMaskedLM(FNetPreTrainedModel):
    # 指定了共享权重的键列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 初始化 FNet 模型和仅包含 MLM 头部的组件
        self.fnet = FNetModel(config)
        self.cls = FNetOnlyMLMHead(config)

        # 执行后期初始化，包括权重初始化和最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回 MLM 头部的预测解码器
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置 MLM 头部的预测解码器的新嵌入
        self.cls.predictions.decoder = new_embeddings

    # 为 forward 方法添加文档字符串，描述了其输入和输出以及一些示例代码
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 如果没有指定 return_dict，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 FNet 模型进行前向传播
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取序列输出和预测分数
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # 计算掩蔽语言建模损失
            loss_fct = CrossEntropyLoss()  # -100 索引表示填充令牌
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不需要返回字典，则输出结果元组
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果需要返回字典，则返回 MaskedLMOutput 对象
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states)


# 使用装饰器为类添加文档字符串，描述了其带有下一句预测分类头部的 FNet 模型
@add_start_docstrings(
    """FNet Model with a `next sentence prediction (classification)` head on top.""",
    FNET_START_DOCSTRING,
)
class FNetForNextSentencePrediction(FNetPreTrainedModel):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数 config
        super().__init__(config)

        # 创建 FNetModel 对象，使用给定的配置参数 config
        self.fnet = FNetModel(config)
        # 创建 FNetOnlyNSPHead 对象，使用给定的配置参数 config
        self.cls = FNetOnlyNSPHead(config)

        # 调用本类中的 post_init 方法，用于初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，添加了一些文档字符串用于描述函数的输入和输出
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
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

        ```
        >>> from transformers import AutoTokenizer, FNetForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        >>> model = FNetForNextSentencePrediction.from_pretrained("google/fnet-base")
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```"""

        # 如果 kwargs 中包含 `next_sentence_label`，则发出警告并使用其值作为 labels
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        # 决定是否返回字典格式的输出，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 FNet 模型进行下一句预测任务的计算
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 FNet 输出的第二个元素中提取池化后的输出，用于后续分类任务
        pooled_output = outputs[1]

        # 将池化后的输出传递给分类器，得到下一句关系的分数
        seq_relationship_scores = self.cls(pooled_output)

        # 初始化下一句预测的损失为 None
        next_sentence_loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 根据 return_dict 决定返回的结果格式
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 使用 NextSentencePredictorOutput 类封装并返回结果
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
        )
@add_start_docstrings(
    """
    FNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    FNET_START_DOCSTRING,
)
class FNetForSequenceClassification(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化时从配置中获取标签数量
        self.num_labels = config.num_labels
        # 初始化一个FNet模型实例
        self.fnet = FNetModel(config)

        # Dropout层，使用配置中的隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器，线性层，输入大小为配置中的隐藏层大小，输出大小为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Perform a forward pass of the FNetForSequenceClassification model.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs of shape (batch_size, sequence_length).
            token_type_ids (torch.Tensor, optional): Input token type IDs of shape (batch_size, sequence_length).
            position_ids (torch.Tensor, optional): Input token position IDs of shape (batch_size, sequence_length).
            inputs_embeds (torch.Tensor, optional): Embedded representations of inputs.
            labels (torch.Tensor, optional): Labels for classification task.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary as output.

        Returns:
            SequenceClassifierOutput: Classification output consisting of logits, hidden states, etc.
        """
        # 实现FNetForSequenceClassification模型的前向传播

        # 省略具体的前向传播逻辑，由于代码中未提供实现细节
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化 return_dict 变量，如果 return_dict 不为 None，则使用给定的值，否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.fnet 进行前向传播，获取模型输出
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取 pooled_output（通常是 CLS token 的输出）
        pooled_output = outputs[1]
        # 对 pooled_output 应用 dropout 操作，用于防止过拟合
        pooled_output = self.dropout(pooled_output)
        # 使用分类器 self.classifier 对 pooled_output 进行分类预测，得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果 self.config.problem_type 未定义，则根据 num_labels 自动定义问题类型
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
                    # 对于单标签回归任务，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则按顺序返回 logits 和额外的 hidden states
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 SequenceClassifierOutput 类型的对象，包括损失、logits 和 hidden states
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
@add_start_docstrings(
    """
    FNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    FNET_START_DOCSTRING,
)
class FNetForMultipleChoice(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 FNet 模型
        self.fnet = FNetModel(config)
        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器线性层，将 FNet 输出映射到单一数值（用于二元分类）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行后续处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        FNet 模型的前向传播方法。

        Args:
            input_ids (torch.Tensor, optional): 输入的 token IDs 张量.
            token_type_ids (torch.Tensor, optional): token 类型 IDs 张量.
            position_ids (torch.Tensor, optional): 位置 IDs 张量.
            inputs_embeds (torch.Tensor, optional): 嵌入输入张量.
            labels (torch.Tensor, optional): 标签张量 (用于训练时).
            output_hidden_states (bool, optional): 是否输出隐藏状态.
            return_dict (bool, optional): 是否返回字典格式输出.

        Returns:
            返回一个包含多个选择的模型输出.

        """
        # 实现 FNet 模型的具体前向传播逻辑
        # （具体实现代码在这里，不包含在注释内）
        pass
    # 返回一个字典，如果没有指定则使用配置中的默认设置
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # 获取输入中第二维的大小作为选择项的数量
    num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    
    # 如果输入的input_ids不为None，则将其视图重新调整为二维张量
    input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
    
    # 如果token_type_ids不为None，则将其视图重新调整为二维张量
    token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
    
    # 如果position_ids不为None，则将其视图重新调整为二维张量
    position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
    
    # 如果inputs_embeds不为None，则将其视图重新调整为三维张量
    inputs_embeds = (
        inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
        if inputs_embeds is not None
        else None
    )
    
    # 使用给定的输入调用模型的前向传播函数fnet，返回输出结果
    outputs = self.fnet(
        input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    # 从模型输出中获取池化后的输出，一般在位置1
    pooled_output = outputs[1]
    
    # 对池化后的输出应用dropout操作
    pooled_output = self.dropout(pooled_output)
    
    # 将dropout后的输出通过分类器得到logits（对数概率）
    logits = self.classifier(pooled_output)
    
    # 将logits重新调整为二维张量，以匹配选择项的数量
    reshaped_logits = logits.view(-1, num_choices)
    
    # 初始化损失值为None
    loss = None
    
    # 如果提供了标签labels，则计算交叉熵损失
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)
    
    # 如果不需要返回字典形式的输出，则按指定格式返回结果
    if not return_dict:
        output = (reshaped_logits,) + outputs[2:]  # 将预测值和可能的其他输出组合成元组
        return ((loss,) + output) if loss is not None else output  # 如果有损失值则将其包含在返回结果中
    
    # 如果需要返回字典形式的输出，则构建MultipleChoiceModelOutput对象返回
    return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states)
"""
FNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 导入必要的库函数
@add_start_docstrings(
    """
    FNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    FNET_START_DOCSTRING,
)
# 定义 FNetForTokenClassification 类，继承自 FNetPreTrainedModel
class FNetForTokenClassification(FNetPreTrainedModel):
    
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量为配置中的 num_labels
        self.num_labels = config.num_labels

        # 初始化 FNetModel 对象，并保存在 self.fnet 中
        self.fnet = FNetModel(config)

        # 使用配置中的隐藏层 dropout 概率创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建线性分类器层，将隐藏层输出映射到 num_labels 维度
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 定义 forward 方法，处理输入并返回结果
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # forward 方法的详细文档字符串
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.fnet 对象的 forward 方法，处理输入数据
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取序列输出
        sequence_output = outputs[0]

        # 应用 dropout 层
        sequence_output = self.dropout(sequence_output)
        # 将序列输出传入分类器层，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算分类损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只保留损失的有效部分
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则创建 TokenClassifierOutput 对象并返回
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 从配置对象中获取标签数目并存储在实例变量中
        self.num_labels = config.num_labels

        # 创建一个FNetModel的实例并存储在实例变量self.fnet中
        self.fnet = FNetModel(config)

        # 创建一个线性层用于输出，输入尺寸为配置对象中的隐藏层大小，输出尺寸为标签数目
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 调用自定义的初始化方法，用于初始化权重并进行最终处理
        # 在此方法中可能包含权重初始化和其他必要的处理步骤
        self.post_init()

    # 前向传播函数，用于模型的前向计算
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
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
        # Determine if we should use the return_dict from the config or from the function argument
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the transformer network with given inputs
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the final sequence output from the transformer network
        sequence_output = outputs[0]

        # Generate logits from the sequence output using the question answering head
        logits = self.qa_outputs(sequence_output)
        
        # Split logits into start and end logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # Calculate total loss if start_positions and end_positions are provided
        if start_positions is not None and end_positions is not None:
            # If the batch size is greater than 1, squeeze extra dimensions
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # Clamp positions to be within the valid range of sequence length
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # Define loss function and compute start and end position losses
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # If return_dict is False, prepare outputs as tuple
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # If return_dict is True, return QuestionAnsweringModelOutput
        return QuestionAnsweringModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states
        )
```