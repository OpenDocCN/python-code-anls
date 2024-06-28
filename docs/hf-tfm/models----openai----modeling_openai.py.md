# `.\models\openai\modeling_openai.py`

```
# 导入必要的库和模块
import json  # 导入处理 JSON 格式数据的模块
import math  # 导入数学运算模块
import os    # 导入操作系统相关功能的模块
from dataclasses import dataclass  # 导入用于定义数据类的装饰器
from typing import Any, Dict, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 深度学习库
from torch import nn  # 导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数模块

# 导入自定义的激活函数
from ...activations import gelu_new, silu  
# 导入模型输出相关的类
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput  
# 导入模型相关的基类和函数
from ...modeling_utils import PreTrainedModel, SequenceSummary  
# 导入与 PyTorch 相关的实用函数和类
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer  
# 导入一般实用函数
from ...utils import (
    ModelOutput,  # 模型输出相关的类
    add_code_sample_docstrings,  # 添加代码示例的文档字符串
    add_start_docstrings,  # 添加起始文档字符串
    add_start_docstrings_to_model_forward,  # 添加模型前向方法的起始文档字符串
    logging,  # 日志记录模块
    replace_return_docstrings,  # 替换返回结果的文档字符串
)

# 导入 OpenAI GPT 的配置类
from .configuration_openai import OpenAIGPTConfig  


# 获取全局日志记录器对象
logger = logging.get_logger(__name__)

# 定义用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "openai-community/openai-gpt"
_CONFIG_FOR_DOC = "OpenAIGPTConfig"

# OpenAI GPT 预训练模型的存档列表
OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-community/openai-gpt",
    # 查看所有 OpenAI GPT 模型的存档列表 https://huggingface.co/models?filter=openai-community/openai-gpt
]


def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path):
    """Load tf pre-trained weights in a pytorch model (from NumPy arrays here)"""
    import re  # 导入正则表达式模块
    import numpy as np  # 导入 NumPy 数据处理模块

    # 如果文件路径包含 ".ckpt"，则截取其所在文件夹路径
    if ".ckpt" in openai_checkpoint_folder_path:
        openai_checkpoint_folder_path = os.path.dirname(openai_checkpoint_folder_path)

    # 记录加载权重的日志信息
    logger.info(f"Loading weights from {openai_checkpoint_folder_path}")

    # 从 JSON 文件中读取参数名称
    with open(openai_checkpoint_folder_path + "/parameters_names.json", "r", encoding="utf-8") as names_handle:
        names = json.load(names_handle)
    # 从 JSON 文件中读取参数形状
    with open(openai_checkpoint_folder_path + "/params_shapes.json", "r", encoding="utf-8") as shapes_handle:
        shapes = json.load(shapes_handle)
    # 计算参数的偏移量
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    # 加载所有分片的参数数据并组装成初始参数列表
    init_params = [np.load(openai_checkpoint_folder_path + f"/params_{n}.npy") for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]

    # 曾经用于将位置和标记的单个嵌入矩阵合并
    # init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
    # 删除初始化参数列表中的第二个元素（索引为1）
    init_params = [arr.squeeze() for arr in init_params]

    # 检查模型的token和position embeddings的权重维度是否与初始化参数匹配
    if model.tokens_embed.weight.shape != init_params[1].shape:
        raise ValueError(
            f"tokens_embed.weight.shape: {model.tokens_embed.weight.shape} does not match init_param[1].shape:"
            f" {init_params[1].shape}"
        )

    # 检查模型的positions embeddings的权重维度是否与初始化参数匹配
    if model.positions_embed.weight.shape != init_params[0].shape:
        raise ValueError(
            f"positions_embed.weight.shape: {model.positions_embed.weight.shape} does not match init_param[0].shape:"
            f" {init_params[0].shape}"
        )

    # 将numpy数组转换为PyTorch张量，并赋值给模型的token embeddings权重
    model.tokens_embed.weight.data = torch.from_numpy(init_params[1])
    
    # 将numpy数组转换为PyTorch张量，并赋值给模型的positions embeddings权重
    model.positions_embed.weight.data = torch.from_numpy(init_params[0])
    
    # 移除列表中的第一个元素
    names.pop(0)
    
    # 弹出位置和token embedding数组
    init_params.pop(0)
    init_params.pop(0)

    # 遍历names和init_params，进行模型参数初始化
    for name, array in zip(names, init_params):
        # 跳过字符串"model/"，截取名称的一部分
        name = name[6:]

        # 检查名称是否以":0"结尾
        if name[-2:] != ":0":
            raise ValueError(f"Layer {name} does not end with :0")

        # 去除名称末尾的":0"
        name = name[:-2]

        # 按"/"分割名称
        name = name.split("/")
        
        # 指针初始化为模型本身
        pointer = model

        # 遍历名称中的每一部分
        for m_name in name:
            # 如果名称符合"[A-Za-z]+\d+"的格式，则拆分为作用域名称和数字
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]

            # 根据作用域名称更新指针位置
            if scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "w":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])

            # 如果作用域名称长度大于等于2，则进一步根据数字索引更新指针位置
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        # 确保指针和数组具有兼容的形状
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        # 记录初始化的PyTorch权重信息
        logger.info(f"Initialize PyTorch weight {name}")
        
        # 将numpy数组转换为PyTorch张量，并赋值给指针的数据属性
        pointer.data = torch.from_numpy(array)

    # 返回更新后的模型
    return model
ACT_FNS = {"relu": nn.ReLU(), "silu": silu, "gelu": gelu_new, "swish": silu}

# 定义 Attention 类，继承自 nn.Module
class Attention(nn.Module):
    def __init__(self, nx, n_positions, config, scale=False):
        super().__init__()
        n_state = nx  # 在 Attention 中，n_state=768 (nx=n_embd)
        # [将 nx => n_state 从 Block 转到 Attention 以保持与 TF 实现一致]
        if n_state % config.n_head != 0:
            # 如果 n_state 不能被 config.n_head 整除，则抛出异常
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")
        # 注册缓冲区 bias，用于存储下三角矩阵的张量
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions),
            persistent=False,
        )
        self.n_head = config.n_head  # 头数
        self.split_size = n_state  # 分割大小
        self.scale = scale  # 是否进行缩放

        # 线性卷积层，用于计算注意力权重和投影
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 注意力的 dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 残差的 dropout
        self.pruned_heads = set()  # 被修剪的注意力头集合

    # 修剪多余的注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的注意力头并索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # 对 conv1d 层进行修剪
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # 更新超参数
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    # 计算注意力权重
    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)  # 计算注意力分数
        if self.scale:
            w = w / math.sqrt(v.size(-1))  # 如果需要，进行缩放

        # 截取与注意力矩阵大小相匹配的下三角部分，用于屏蔽无效的注意力
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = w * b + -1e4 * (1 - b)  # 应用下三角 mask

        if attention_mask is not None:
            # 应用额外的注意力 mask
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)  # softmax 归一化
        w = self.attn_dropout(w)  # 对注意力权重应用 dropout

        # 如果需要，对头部进行 mask
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]  # 计算加权后的值
        if output_attentions:
            outputs.append(w)  # 如果需要输出注意力权重，则添加到输出中
        return outputs

    # 合并注意力头
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()  # 调整维度顺序以便合并
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # 合并多头注意力的状态
    # 将输入张量 x 按照指定的维度拆分为多个头部，以便进行多头注意力计算
    def split_heads(self, x, k=False):
        # 计算新的张量形状，保持前面所有维度，最后两个维度分别为 self.n_head 和 x.size(-1) // self.n_head
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # 重新调整张量 x 的形状，将其拆分为多个头部
        x = x.view(*new_x_shape)  # 在 Tensorflow 实现中称为 split_states
        if k:
            # 如果 k 为 True，交换特定维度顺序，便于后续计算
            return x.permute(0, 2, 3, 1)
        else:
            # 否则，按照默认的维度顺序返回
            return x.permute(0, 2, 1, 3)

    # 前向传播函数，用于处理输入 x，可选的注意力掩码和头部掩码，以及是否输出注意力信息
    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        # 应用自注意力机制层，处理输入 x
        x = self.c_attn(x)
        # 将处理后的 x 按照特定维度分割成查询、键、值
        query, key, value = x.split(self.split_size, dim=2)
        # 将查询部分按照多头进行拆分
        query = self.split_heads(query)
        # 将键部分按照多头进行拆分，并进行特定的维度交换（如果 k=True）
        key = self.split_heads(key, k=True)
        # 将值部分按照多头进行拆分
        value = self.split_heads(value)

        # 执行多头注意力计算，并返回注意力输出及可能的注意力信息
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        # 合并多头计算得到的注意力输出
        a = self.merge_heads(a)
        # 应用额外的投影层
        a = self.c_proj(a)
        # 应用残差连接和丢弃层
        a = self.resid_dropout(a)

        # 返回最终输出结果，包括 a 和可能的注意力信息
        outputs = [a] + attn_outputs[1:]
        return outputs  # 返回 a 和 (attentions)
# 定义多层感知机（MLP）神经网络模型
class MLP(nn.Module):
    # 初始化函数，设置网络结构
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        # 第一个卷积层，输入通道数为 n_embd，输出通道数为 n_state
        self.c_fc = Conv1D(n_state, nx)
        # 第二个卷积层，输入通道数为 n_state，输出通道数为 n_embd
        self.c_proj = Conv1D(nx, n_state)
        # 激活函数，根据配置选择激活函数类型
        self.act = ACT_FNS[config.afn]
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.resid_pdrop)

    # 前向传播函数
    def forward(self, x):
        # 第一层卷积层后接激活函数
        h = self.act(self.c_fc(x))
        # 第二层卷积层
        h2 = self.c_proj(h)
        # 应用 Dropout
        return self.dropout(h2)


# 定义一个 Transformer 模型的基本模块 Block
class Block(nn.Module):
    # 初始化函数，设置模块的结构
    def __init__(self, n_positions, config, scale=False):
        super().__init__()
        nx = config.n_embd
        # 注意力机制层
        self.attn = Attention(nx, n_positions, config, scale)
        # Layer Normalization 层
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        # 多层感知机（MLP）模型
        self.mlp = MLP(4 * nx, config)
        # 再次应用 Layer Normalization 层
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    # 前向传播函数
    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        # 使用注意力机制层处理输入
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        # 获取注意力机制层的输出
        a = attn_outputs[0]

        # 应用 Layer Normalization 和残差连接
        n = self.ln_1(x + a)
        # 应用多层感知机模型
        m = self.mlp(n)
        # 再次应用 Layer Normalization 和残差连接
        h = self.ln_2(n + m)

        # 返回模块的输出
        outputs = [h] + attn_outputs[1:]
        return outputs


# 定义一个抽象类，用于处理权重初始化、预训练模型的下载和加载接口
class OpenAIGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 类属性，指定配置类
    config_class = OpenAIGPTConfig
    # 类方法，用于加载 TensorFlow 格式的权重
    load_tf_weights = load_tf_weights_in_openai_gpt
    # 模型名称前缀
    base_model_prefix = "transformer"

    # 初始化权重函数
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # 对线性层和卷积层进行权重初始化
            # 与 TensorFlow 版本稍有不同，这里使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层进行权重初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 Layer Normalization 层进行权重初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 定义一个用于 OpenAIGPT 模型输出的基类
@dataclass
class OpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """

    # 这里没有具体的实现代码，只是一个数据类的定义
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
            语言建模的损失值（当提供`labels`时返回）。
        mc_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `mc_labels` is provided):
            Multiple choice classification loss.
            多项选择分类的损失值（当提供`mc_labels`时返回）。
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言建模头部的预测分数（SoftMax之前每个词汇标记的分数）。
        mc_logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
            多项选择分类头部的预测分数（SoftMax之前每个选项的分数）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每层输出的隐藏状态，加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，经过注意力SoftMax后的结果，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
OPENAI_GPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OpenAIGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

OPENAI_GPT_INPUTS_DOCSTRING = r"""
    Model input descriptions for OpenAI GPT models, detailing expected inputs and their formats.

    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of input sequence tokens in the vocabulary.

        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding tokens.

        token_type_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to differentiate two sequences in the same input (e.g. question/answer).

        position_ids (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.

        head_mask (:obj:`torch.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.

        inputs_embeds (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Instead of passing :obj:`input_ids`, you can directly pass an embedded representation.

    Returns:
        :obj:`tuple(torch.FloatTensor)`: A tuple of :obj:`torch.FloatTensor` (or :obj:`tuple` of :obj:`torch.FloatTensor`
        if :obj:`return_dict` is True) containing various elements depending on the configuration (:class:`~transformers.GPT2Config`)
        and inputs.

    Examples::

        >>> from transformers import GPT2Tokenizer, GPT2Model
        >>> import torch

        >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        >>> model = GPT2Model.from_pretrained('gpt2')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
"""
    # 接收模型输入的输入序列的token索引，形状为(batch_size, sequence_length)，每个值是词汇表中对应token的索引
    input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
    
        # 通过AutoTokenizer可以获取这些索引。参见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__以获取更多细节
        Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
        [`PreTrainedTokenizer.__call__`] for details.
    
        [What are input IDs?](../glossary#input-ids)
    
    # 注意力遮罩，形状为(batch_size, sequence_length)，用于避免对填充token的索引进行注意力操作。遮罩值选在[0, 1]之间：
    
        - 1表示**未遮罩**的token，
        - 0表示**已遮罩**的token。
    attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
    
        [What are attention masks?](../glossary#attention-mask)
    
    # token类型ID，形状为(batch_size, sequence_length)，用于指示输入的第一和第二部分。索引选在[0, 1]之间：
    
        - 0对应一个*sentence A*的token，
        - 1对应一个*sentence B*的token。
    token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    
        [What are token type IDs?](../glossary#token-type-ids)
    
    # 位置ID，形状为(batch_size, sequence_length)，用于指示每个输入序列token在位置嵌入中的位置索引。选在范围[0, config.max_position_embeddings - 1]内。
    
        [What are position IDs?](../glossary#position-ids)
    position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    
        # 自注意力模块中选择性地将某些头部置零的遮罩。遮罩值选在[0, 1]之间：
    
            - 1表示头部**未遮罩**，
            - 0表示头部**已遮罩**。
    head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
    
        # 可选项，可以直接传递嵌入表示而不是传递input_ids。如果需要更精确地控制如何将input_ids索引转换为相关向量，这很有用，而不是使用模型的内部嵌入查找矩阵。
    inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
    
        # 是否返回所有注意力层的注意力张量。有关更多细节，请参见返回的张量下的attentions。
    output_attentions (`bool`, *optional*):
    
        # 是否返回所有层的隐藏状态。有关更多细节，请参见返回的张量下的hidden_states。
    output_hidden_states (`bool`, *optional*):
    
        # 是否返回[`~utils.ModelOutput`]而不是普通元组。
    return_dict (`bool`, *optional*):
"""
OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
class OpenAIGPTLMHeadModel(OpenAIGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 OpenAIGPTModel，即底层的 Transformer 模型
        self.transformer = OpenAIGPTModel(config)
        
        # 定义语言模型头部，线性层的输出维度与词汇表大小相同，且没有偏置
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
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
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否应该返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用transformer模型进行前向传播
        transformer_outputs = self.transformer(
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
        # 获取transformer模型的隐藏状态
        hidden_states = transformer_outputs[0]
        # 使用语言模型头部生成logits
        lm_logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果存在labels，则计算损失
        if labels is not None:
            # 调整logits和labels的形状以便计算损失
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 如果不需要返回字典格式的结果，则输出元组格式的输出
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的结果，则输出CausalLMOutput对象
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        # 准备生成的输入，将input_ids放入字典中并返回
        return {"input_ids": input_ids}
@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 1
        # 初始化 OpenAIGPTModel 模型作为 Transformer 模型的基础
        self.transformer = OpenAIGPTModel(config)
        # 初始化 lm_head 线性层，用于语言建模任务
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 初始化 multiple_choice_head 用于多项选择任务的序列摘要层
        self.multiple_choice_head = SequenceSummary(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回 lm_head 作为输出的嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置 lm_head 的新嵌入层
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OpenAIGPTDoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mc_token_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        mc_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此方法定义了 OpenAIGPTDoubleHeadsModel 的前向传播逻辑
        # （具体逻辑在下文的具体实现中，这里主要是声明参数和返回类型）

@add_start_docstrings(
    """
    The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).
    [`OpenAIGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the
    last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding
    token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
    it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
    the last value in each row of the batch).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTForSequenceClassification(OpenAIGPTPreTrainedModel):
    # 省略此处的类定义和初始化方法注释，因为在本示例中并未包含
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置对象中的标签数量赋给实例变量 num_labels
        self.num_labels = config.num_labels
        # 使用 OpenAIGPTModel 根据配置对象创建一个转换器实例，并赋给实例变量 transformer
        self.transformer = OpenAIGPTModel(config)
        # 创建一个线性层实例 score，用于输出预测结果，输入维度为 config.n_embd，输出维度为 num_labels，无偏置项
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # 调用 post_init 方法完成权重初始化和最终处理

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多个输入参数，可以选择是否返回一个字典作为输出
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
```