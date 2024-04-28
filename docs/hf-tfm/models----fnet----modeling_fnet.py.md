# `.\models\fnet\modeling_fnet.py`

```
# 设置编码格式为 utf-8
# 版权声明和许可声明
# 导入必要的包和模块
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...utils import is_scipy_available

# 如果 scipy 可用，则导入 linalg 模块
if is_scipy_available():
    from scipy import linalg

# 导入各种模型输出类和辅助函数
from ...activations import ACT2FN
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
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 从配置文件中导入 FNetConfig 类
from .configuration_fnet import FNetConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 文档中的检查点和配置
_CHECKPOINT_FOR_DOC = "google/fnet-base"
_CONFIG_FOR_DOC = "FNetConfig"

# FNet 的预训练模型列表
FNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/fnet-base",
    "google/fnet-large",
    # 更多的 FNet 模型可以在 https://huggingface.co/models?filter=fnet 上查看
]

# 从 https://github.com/google-research/google-research/blob/master/f_net/fourier.py 调整而来
def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """对三维输入数组应用二维矩阵乘法。"""
    seq_length = x.shape[1]
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    x = x.type(torch.complex64)
    return torch.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)

# 对外的接口，对外暴露矩阵乘法函数
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)

# 计算 n 维的快速傅里叶变换（FFT）
def fftn(x):
    """
    对输入的 n 维数组应用 n 维快速傅里叶变换（FFT）。

    参数:
        x: 输入的 n 维数组。

    返回:
        输入数组的 n 维傅里叶变换。
    """
    out = x
    # 对于输入张量的除最后一个轴以外的所有轴进行反向遍历
    for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
        # 在指定的轴上对输入张量执行快速傅立叶变换
        out = torch.fft.fft(out, axis=axis)
    # 返回处理后的输出张量
    return out
class FNetEmbeddings(nn.Module):
    """构建来自单词、位置和标记类型嵌入的嵌入层。"""

    def __init__(self, config):
        super().__init__()
        # 定义单词嵌入层，将单词映射到隐藏状态空间
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义位置嵌入层，将位置索引映射到隐藏状态空间
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 定义标记类型嵌入层，将标记类型映射到隐藏状态空间
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 不使用蛇形命名法以保持与 TensorFlow 模型变量名一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 注意：这是投影层，将被使用。原始代码允许不同的嵌入和不同的模型维度。
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义 dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在内存中是连续的，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # token_type_ids (与 position_ids 大小相同) 在内存中是连续的，并在序列化时导出
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
    # 定义前向传播函数，用于模型的正向计算
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入的 input_ids 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状，排除最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则使用模型中存储的 position_ids 的部分内容，保留与序列长度相对应的部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 设置 token_type_ids 为在构造函数中注册的缓冲区，其中所有值为零，通常在自动生成时出现，
        # 注册的缓冲区可以在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            # 如果模型具有 token_type_ids 属性，则获取其部分内容，并扩展以匹配输入形状
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则创建与输入形状相同的零张量作为 token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则使用 word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 使用 token_type_embeddings 对 token_type_ids 进行嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入与 token_type 嵌入相加
        embeddings = inputs_embeds + token_type_embeddings

        # 使用 position_embeddings 对 position_ids 进行嵌入，并将结果与之前相加得到的 embeddings 相加
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        # 将 embeddings 输入 LayerNorm 进行归一化
        embeddings = self.LayerNorm(embeddings)

        # 将归一化后的 embeddings 输入 projection 进行投影
        embeddings = self.projection(embeddings)

        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的 embeddings
        return embeddings
# 定义一个类 FNetBasicFourierTransform，继承自 nn.Module 类
class FNetBasicFourierTransform(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 调用 _init_fourier_transform 方法进行初始化
        self._init_fourier_transform(config)

    # 定义一个内部方法 _init_fourier_transform，接收一个 config 参数
    def _init_fourier_transform(self, config):
        # 如果不使用 tpu_fourier_optimizations
        if not config.use_tpu_fourier_optimizations:
            # 使用 torch.fft.fftn 函数进行傅里叶变换，计算维度为 (1, 2) 的傅里叶变换
            self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))
        # 如果 max_position_embeddings 小于等于 4096
        elif config.max_position_embeddings <= 4096:
            # 如果安装了 scipy 库
            if is_scipy_available():
                # 注册隐藏层 DFT（离散傅里叶变换）矩阵作为缓冲区
                self.register_buffer(
                    "dft_mat_hidden", torch.tensor(linalg.dft(config.hidden_size), dtype=torch.complex64)
                )
                # 注册序列 DFT 矩阵作为缓冲区
                self.register_buffer(
                    "dft_mat_seq", torch.tensor(linalg.dft(config.tpu_short_seq_length), dtype=torch.complex64)
                )
                # 使用自定义的矩阵乘法函数进行傅里叶变换，参数为两个 DFT 矩阵
                self.fourier_transform = partial(
                    two_dim_matmul, matrix_dim_one=self.dft_mat_seq, matrix_dim_two=self.dft_mat_hidden
                )
            else:
                # 如果没有安装 scipy 库，则输出警告信息
                logging.warning(
                    "SciPy is needed for DFT matrix calculation and is not found. Using TPU optimized fast fourier"
                    " transform instead."
                )
                # 使用 fftn 函数进行傅里叶变换
                self.fourier_transform = fftn
        else:
            # 如果不满足上述条件，则使用 fftn 函数进行傅里叶变换
            self.fourier_transform = fftn

    # 前向传播方法，接收 hidden_states 参数
    def forward(self, hidden_states):
        # 执行傅里叶变换，并取实部作为输出
        outputs = self.fourier_transform(hidden_states).real
        # 返回输出结果
        return (outputs,)


# 定义一个类 FNetBasicOutput，继承自 nn.Module 类
class FNetBasicOutput(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接收 hidden_states 和 input_tensor 两个参数
    def forward(self, hidden_states, input_tensor):
        # 对输入进行 LayerNorm 处理，并与 hidden_states 相加得到输出
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        # 返回输出结果
        return hidden_states


# 定义一个类 FNetFourierTransform，继承自 nn.Module 类
class FNetFourierTransform(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 self 层为 FNetBasicFourierTransform 类的实例
        self.self = FNetBasicFourierTransform(config)
        # 初始化 output 层为 FNetBasicOutput 类的实例
        self.output = FNetBasicOutput(config)

    # 前向传播方法，接收 hidden_states 参数
    def forward(self, hidden_states):
        # 对输入进行 FNetBasicFourierTransform 和 FNetBasicOutput 处理，得到最终输出
        self_outputs = self.self(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        outputs = (fourier_output,)
        # 返回输出结果
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制成 FNetIntermediate
class FNetIntermediate(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 dense 层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 hidden_act 是字符串类型，则使用对应的激活函数；否则使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义了前向传播函数，接收隐藏状态张量作为输入，并返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量输入到全连接层中，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的隐藏状态张量输入到激活函数中，进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过线性变换和激活函数处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput复制代码，并将Bert->FNet
class FNetOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将隐藏层维度转换为中间层维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 对隐藏层进行层归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout 对隐藏层进行 dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 首先对隐藏层进行全连接层变换
        hidden_states = self.dense(hidden_states)
        # 然后对结果进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 最后对结果进行层归一化，并加上输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# FNetLayer 模块
class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设定前馈网络的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 定义序列长度所在的维度
        self.seq_len_dim = 1  # The dimension which has the sequence length
        # 创建 Fourier 变换
        self.fourier = FNetFourierTransform(config)
        # 创建中间层
        self.intermediate = FNetIntermediate(config)
        # 创建输出层
        self.output = FNetOutput(config)

    def forward(self, hidden_states):
        # 对隐藏层进行 Fourier 变换
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]

        # 将前馈网络应用到 Fourier 变换输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, fourier_output
        )

        # 返回层输出
        outputs = (layer_output,)

        return outputs

    # 前馈网络的块函数
    def feed_forward_chunk(self, fourier_output):
        # 对 Fourier 变换输出进行中间层计算
        intermediate_output = self.intermediate(fourier_output)
        # 对中间层输出进行输出层计算
        layer_output = self.output(intermediate_output, fourier_output)
        return layer_output


# FNetEncoder
class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建 FNetLayer 模块列表
        self.layer = nn.ModuleList([FNetLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states)
            else:
                layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

# 从transformers.models.bert.modeling_bert.BertPooler复制代码，并将Bert->FNet
class FNetPooler(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入和输出大小都是隐藏层的大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    # 前向传播函数，接受一个张量作为输入，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取第一个标记对应的隐藏状态来“池化”模型。
        # 获取第一个标记对应的张量
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态通过全连接层
        pooled_output = self.dense(first_token_tensor)
        # 对全连接层的输出进行激活
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output
# 用于将 BERT 模型中的预测头部转换为 FNet 模型中的预测头部
class FNetPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集连接层，将隐藏状态映射到相同维度的空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 隐藏激活函数，根据配置选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 层归一化，对密集连接层输出进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 密集连接层前向传播
        hidden_states = self.dense(hidden_states)
        # 激活函数前向传播
        hidden_states = self.transform_act_fn(hidden_states)
        # 层归一化前向传播
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 用于 FNet 模型中的语言模型预测头部
class FNetLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 预测头部转换层
        self.transform = FNetPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个令牌有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

        # 为每个词汇设置偏置参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 预测头部转换层前向传播
        hidden_states = self.transform(hidden_states)
        # 预测头部前向传播
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self):
        # 如果它们断开连接（在 TPU 上或当偏置被重新调整时），将这两个权重联系起来
        self.bias = self.decoder.bias


# 用于 FNet 模型中的仅 MLM 预测头部
class FNetOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MLM 预测头部
        self.predictions = FNetLMPredictionHead(config)

    def forward(self, sequence_output):
        # 进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 用于 FNet 模型中的仅 NSP 预测头部
class FNetOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 下一句预测线性层
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 预测下一句
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# 用于 FNet 模型中的预训练头部
class FNetPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MLM 预测头部
        self.predictions = FNetLMPredictionHead(config)
        # NSP 预测头部
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 预测 MLM
        prediction_scores = self.predictions(sequence_output)
        # 预测 NSP
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# FNet 预训练模型的基类
class FNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义配置类为FNetConfig
    config_class = FNetConfig
    # 基础模型前缀为"fnet"
    base_model_prefix = "fnet"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重，使用正态分布，平均值为0.0，标准差为配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 注意：原始代码对偏置使用与权重相同的初始化
            if module.bias is not None:
                # 将偏置数据置零
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重，使用正态分布，平均值为0.0，标准差为配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引
            if module.padding_idx is not None:
                # 将填充索引的权重置零
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置数据置零
            module.bias.data.zero_()
            # 将权重数据填充为1.0
            module.weight.data.fill_(1.0)
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            - 1 for tokens that are NOT MASKED,
            - 0 for MASKED tokens.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size - 1]``.
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels in ``[0, ..., config.vocab_size - 1]``.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to return the hidden states of all layers. See `hidden_states`.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to return the attentions tensors of all layers. See `attentions`.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
    # 定义函数参数和其类型说明
    Args:
        # input_ids 是一个形状为 ({0}) 的 torch.LongTensor，表示词汇表中输入序列标记的索引
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 用 AutoTokenizer 获取索引，参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__ 获取详情
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            # 查看输入 ID 的含义
            [What are input IDs?](../glossary#input-ids)

        # token_type_ids 是一个形状为 ({0}) 的 torch.LongTensor，表示输入的段标记索引，指示输入的第一部分和第二部分。索引在 [0,1] 之间选择：
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 0 对应一个“句子 A”标记，1 对应一个“句子 B”标记
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            # 查看令牌类型 ID 的含义
            [What are token type IDs?](../glossary#token-type-ids)

        # position_ids 是一个形状为 ({0}) 的 torch.LongTensor，表示位置嵌入中每个输入序列标记的位置索引。在范围 [0, config.max_position_embeddings - 1] 中选取。
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 查看位置 ID 的含义
            [What are position IDs?](../glossary#position-ids)

        # inputs_embeds 是一个形状为 ({0}, hidden_size) 的 torch.FloatTensor，可选参数，代表直接传递嵌入表示而不是传递 input_ids。如果想要更好地控制如何将 input_ids 索引转换为相关联的向量，则这是有用的，而不是使用模型的内部嵌入查找矩阵。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选传递嵌入表示的情况说明
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.

        # output_hidden_states 是一个布尔值，可选参数，表示是否返回所有层的隐藏状态。关于返回的张量下的 hidden_states 有更多细节可以查看。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态的情况说明
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        # return_dict 是一个布尔值，可选参数，表示是否返回一个 `utils.ModelOutput` 而不是一个普通的元组。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `utils.ModelOutput` 的情况说明
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
    
    # 初始化FNetModel类
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法，传入配置
        super().__init__(config)
        # 保存配置信息
        self.config = config

        # 创建FNetEmbeddings实例
        self.embeddings = FNetEmbeddings(config)
        # 创建FNetEncoder实例
        self.encoder = FNetEncoder(config)

        # 创建FNetPooler实例，如果add_pooling_layer为True
        self.pooler = FNetPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 重写forward方法，输入参数和输出注释在装饰器中定义
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
    # 定义函数，指定输入和输出类型
    ) -> Union[tuple, BaseModelOutput]:
        # 如果未指定隐藏状态输出，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        # 如果指定了 inputs_embeds
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        # 如果都未指定，则抛出数值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果开启了 TPU 傅里叶优化，并且序列长度符合要求，则抛出数值错误
        if (
            self.config.use_tpu_fourier_optimizations
            and seq_length <= 4096
            and self.config.tpu_short_seq_length != seq_length
        ):
            raise ValueError(
                "The `tpu_short_seq_length` in FNetConfig should be set equal to the sequence length being passed to"
                " the model when using TPU optimizations."
            )

        # 获取输入的设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未指定 token_type_ids，则根据情况处理
        if token_type_ids is None:
            # 如果 embeddings 中包含 token_type_ids，则使用该信息，否则创建全零的 token_type_ids
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 生成嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        # 通过编码器处理嵌入输出，得到编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        # 如果有池化层，则对序列输出进行池化操作
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]

        # 否则返回带池化的 BaseModelOutput
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 添加模型前置文档字符串，描述 FNet 模型的两个头部：掩码语言建模头部和下一句预测（分类）头部
@add_start_docstrings(
    """
    FNet Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    FNET_START_DOCSTRING,
)
# 定义 FNet 用于预训练的模型类，继承自 FNetPreTrainedModel
class FNetForPreTraining(FNetPreTrainedModel):
    # 定义共享权重的关键键列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建 FNet 模型实例
        self.fnet = FNetModel(config)
        # 创建 FNet 预训练头部实例
        self.cls = FNetPreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 重写前向传播函数，并添加模型输入的文档字符串和输出的替换文档字符串
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
    ) -> Union[Tuple, FNetForPreTrainingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            计算掩码语言建模损失的标签。索引应在`[-100, 0, ..., config.vocab_size]`范围内（参见`input_ids`文档字符串）。索引设置为`-100`的标记将被忽略（掩码），损失仅计算具有标签在`[0, ..., config.vocab_size]`范围内的标记。
        next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算下一个序列预测（分类）损失的标签。输入应该是一个序列对（参见`input_ids`文档字符串）。索引应在`[0, 1]`范围内：

            - 0 表示序列 B 是序列 A 的延续，
            - 1 表示序列 B 是随机序列。
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            用于隐藏已被弃用的旧参数的字典。

        Returns:
            返回值：

        Example:
            示例：

        ```python
        >>> from transformers import AutoTokenizer, FNetForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
        >>> model = FNetForPreTraining.from_pretrained("google/fnet-base")
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return FNetForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
        )
# 根据给定的文档字符串和 FNet 的起始文档字符串，添加文档字符串注释
@add_start_docstrings("""FNet Model with a `language modeling` head on top.""", FNET_START_DOCSTRING)
class FNetForMaskedLM(FNetPreTrainedModel):
    # 设置共享的权重键值对
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 初始化 FNetModel 和 FNetOnlyMLMHead
        self.fnet = FNetModel(config)
        self.cls = FNetOnlyMLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输出的嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 给 forward 函数添加文档字符串
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
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 运行 FNet 模型
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出和预测分数
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 计算掩码语言建模损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回输出元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 Masked LM 输出对象
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states)


# 添加起始文档字符串注释
@add_start_docstrings(
    """FNet Model with a `next sentence prediction (classification)` head on top.""",
    FNET_START_DOCSTRING,
)
class FNetForNextSentencePrediction(FNetPreTrainedModel):
    # 初始化函数，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 创建一个 FNetModel 对象
        self.fnet = FNetModel(config)
        # 创建一个 FNetOnlyNSPHead 对象
        self.cls = FNetOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    # 模型前向传播函数
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
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

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

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
    # 初始化函数，接收一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 保存标签数量
        self.num_labels = config.num_labels
        # 创建 FNet 模型对象
        self.fnet = FNetModel(config)

        # 创建一个 dropout 层，用于模型训练过程中的随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，用于分类器的输出
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 此函数用于模型前向传播
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
        # 参数说明，接收输入序列的各种张量以及模型的配置参数
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典的默认值，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用预训练模型的前向传播方法
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取池化后的表示
        pooled_output = outputs[1]
        # 对池化后的表示进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 将池化后的表示输入分类器，得到预测的 logits
        logits = self.classifier(pooled_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果问题类型未指定，则根据标签的数据类型和类别数量来确定问题类型
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
        # 如果不需要返回字典，则将输出打包成元组返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则创建一个 SequenceClassifierOutput 对象返回
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
    # 添加模型文档字符串，描述模型的多选分类头部分
    # 包括线性层和 softmax 的处理，例如 RocStories/SWAG 任务
    # 对应 FNetForMultipleChoice 类的文档字符串
    @add_start_docstrings(
        """
        FNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
        softmax) e.g. for RocStories/SWAG tasks.
        """,
        FNET_START_DOCSTRING,
    )
    class FNetForMultipleChoice(FNetPreTrainedModel):
        # 初始化方法
        def __init__(self, config):
            # 调用父类的初始化方法
            super().__init__(config)
    
            # 创建 FNetModel 对象
            self.fnet = FNetModel(config)
            # 创建 Dropout 层
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            # 创建线性分类器
            self.classifier = nn.Linear(config.hidden_size, 1)
    
            # 初始化权重并应用最终处理
            self.post_init()
    
        # 将文档字符串添加到模型前向传播方法
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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否需要返回字典格式的输出，若未指定，则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选项的数量，若存在 input_ids 则取其第二个维度大小，否则取 inputs_embeds 的第二个维度大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 若存在 input_ids，则将其形状重塑为 (-1, input_ids.size(-1))，否则为 None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 若存在 token_type_ids，则将其形状重塑为 (-1, token_type_ids.size(-1))，否则为 None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 若存在 position_ids，则将其形状重塑为 (-1, position_ids.size(-1))，否则为 None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 若存在 inputs_embeds，则将其形状重塑为 (-1, inputs_embeds.size(-2), inputs_embeds.size(-1))，否则为 None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将输入参数传递给模型前向传播函数，获取模型输出
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取汇聚输出
        pooled_output = outputs[1]

        # 对汇聚输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对汇聚输出进行分类，得到 logits
        logits = self.classifier(pooled_output)
        # 将 logits 重塑为 (-1, num_choices) 形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 若 labels 不为空，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 若不需要返回字典格式的输出，则将输出重塑为对应格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型输出，包括损失、logits 和隐藏状态
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states)
# 为FNetForTokenClassification类添加文档字符串，描述该类是一个带有标记分类头部的FNet模型，例如用于命名实体识别（NER）任务
class FNetForTokenClassification(FNetPreTrainedModel):
    # 初始化方法，接收一个config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置num_labels属性为config.num_labels
        self.num_labels = config.num_labels

        # 实例化一个FNetModel对象
        self.fnet = FNetModel(config)

        # 实例化一个Dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 实例化一个全连接层，输入特征数为config.hidden_size，输出特征数为config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 定义forward方法
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对输入进行FNet模型的前向传播
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 对序列输出进行dropout操作
        sequence_output = self.dropout(sequence_output)
        # 将序列输出传入分类器得到预测值logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只保留损失的活跃部分
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

# 为FNetForQuestionAnswering类添加文档字符串，描述该类是一个带有用于提取式问答任务的跨度分类头部的FNet模型
class FNetForQuestionAnswering(FNetPreTrainedModel):
    # 初始化函数，初始化模型参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
    
        # 设置标签数量
        self.num_labels = config.num_labels
    
        # 创建 FNetModel 模型
        self.fnet = FNetModel(config)
        # 创建一个线性层，用于问题回答输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
    
        # 初始化权重并应用最终处理
        self.post_init()
    
    # 前向传播函数，接受输入和参数，返回模型输出
    @add_start_docstrings_to_model_forward(FNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入文本的 token 序列
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型的序列
        position_ids: Optional[torch.Tensor] = None,  # token 的位置序列
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        start_positions: Optional[torch.Tensor] = None,  # 答案起始位置
        end_positions: Optional[torch.Tensor] = None,  # 答案结束位置
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏层状态
        return_dict: Optional[bool] = None,  # 是否返回字典类型的输出
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
        # 确保返回值字典的正确性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用预训练模型的forward方法，获取输出结果
        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 获取答案起始位置和结束位置的预测结果
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 初始化总损失值
        total_loss = None
        # 如果提供了答案起始位置和结束位置的标签
        if start_positions is not None and end_positions is not None:
            # 如果是在多GPU环境下，扩展维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时候答案的起始位置和结束位置超出了模型输入的范围，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # 计算起始位置和结束位置的损失
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失值
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典类型的结果
        if not return_dict:
            # 组装输出结果
            output = (start_logits, end_logits) + outputs[2:]
            # 返回结果元组
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回QuestionAnsweringModelOutput类型的结果
        return QuestionAnsweringModelOutput(
            loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states
        )
```