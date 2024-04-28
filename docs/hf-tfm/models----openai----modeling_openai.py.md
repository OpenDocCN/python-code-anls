# `.\transformers\models\openai\modeling_openai.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# Apache 2.0 许可证
# 导入所需的库
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

# 导入 torch 库
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义库和模型配置
from ...activations import gelu_new, silu
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_openai import OpenAIGPTConfig

# 设置日志记录
logger = logging.get_logger(__name__)

# 文档中的模型和配置
_CHECKPOINT_FOR_DOC = "openai-gpt"
_CONFIG_FOR_DOC = "OpenAIGPTConfig"

# 预训练模型归档列表
OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai-gpt",
    # 具体模型列表网址: https://huggingface.co/models?filter=openai-gpt
]

# 导入 TensorFlow 预训练权重到 PyTorch 模型中
def load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path):
    """Load tf pre-trained weights in a pytorch model (from NumPy arrays here)"""
    import re
    import numpy as np

    if ".ckpt" in openai_checkpoint_folder_path:
        openai_checkpoint_folder_path = os.path.dirname(openai_checkpoint_folder_path)

    logger.info(f"Loading weights from {openai_checkpoint_folder_path}")

    # 从 JSON 文件中加载参数名称
    with open(openai_checkpoint_folder_path + "/parameters_names.json", "r", encoding="utf-8") as names_handle:
        names = json.load(names_handle)
    # 从 JSON 文件中加载参数形状
    with open(openai_checkpoint_folder_path + "/params_shapes.json", "r", encoding="utf-8") as shapes_handle:
        shapes = json.load(shapes_handle)
    # 计算参数数据在文件中的偏移量
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    # 从文件中加载初始化参数
    init_params = [np.load(openai_checkpoint_folder_path + f"/params_{n}.npy") for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]

    # This was used when we had a single embedding matrix for positions and tokens
    # init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
    # del init_params[1]
    # 初始化参数去掉多余的维度，将所有参数数组进行压缩
    init_params = [arr.squeeze() for arr in init_params]

    # 检查 token 和 position embeddings 的权重维度是否与 init 参数的维度匹配
    if model.tokens_embed.weight.shape != init_params[1].shape:
        raise ValueError(
            f"tokens_embed.weight.shape: {model.tokens_embed.weight.shape} does not match init_param[1].shape:"
            f" {init_params[1].shape}"
        )

    if model.positions_embed.weight.shape != init_params[0].shape:
        raise ValueError(
            f"positions_embed.weight.shape: {model.positions_embed.weight.shape} does not match init_param[0].shape:"
            f" {init_params[0].shape}"
        )

    # 将 token 和 position embeddings 的权重数据替换为从 numpy 数组初始化的 PyTorch 张量
    model.tokens_embed.weight.data = torch.from_numpy(init_params[1])
    model.positions_embed.weight.data = torch.from_numpy(init_params[0])

    # 移除 names 列表中的第一个元素
    names.pop(0)
    # 弹出位置和标记嵌入数组
    init_params.pop(0)
    init_params.pop(0)

    for name, array in zip(names, init_params):  # 遍历 names 和 init_params 中的元素
        name = name[6:]  # 跳过 "model/" 前缀
        if name[-2:] != ":0":
            raise ValueError(f"Layer {name} does not end with :0")
        name = name[:-2]
        name = name.split("/")
        pointer = model
        # 遍历 names 中的各级层级名
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "w":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        # 确保 pointer 和 array 具有兼容的形状
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")

        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    # 返回模型
    return model
# 定义一个字典，包含了不同激活函数的名称和对应的激活函数对象
ACT_FNS = {"relu": nn.ReLU(), "silu": silu, "gelu": gelu_new, "swish": silu}

# 定义了一个名为Attention的类，继承自nn.Module
class Attention(nn.Module):
    def __init__(self, nx, n_positions, config, scale=False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        if n_state % config.n_head != 0:
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")
        # 注册一个缓冲区，包含对角线以下的矩阵，用于注意力机制中的遮罩操作
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_positions, n_positions)).view(1, 1, n_positions, n_positions),
            persistent=False,
        )
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        # 1D卷积层，用于计算注意力矩阵
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    # 按heads参数裁剪注意力头部
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # 裁剪卷积层
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # 更新超参数和裁剪头部集合
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    # 计算注意力矩阵
    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs

    # 合并注意力头部
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implementation: fct merge_states
    # 将输入 x 沿最后一个维度分割成多个头
    def split_heads(self, x, k=False):
        # 计算新的 x 的 shape
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # 将 x 重塑为新的 shape
        x = x.view(*new_x_shape)
        # 如果 k 为 True，则将 x 转置为 (batch_size, n_head, x_size//n_head, seq_len)
        if k:
            return x.permute(0, 2, 3, 1)
        # 否则将 x 转置为 (batch_size, n_head, seq_len, x_size//n_head)
        else:
            return x.permute(0, 2, 1, 3)
    
    # 前向传播
    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        # 将 x 传入 self.c_attn 进行线性变换
        x = self.c_attn(x)
        # 将变换后的 x 分割为 query、key 和 value
        query, key, value = x.split(self.split_size, dim=2)
        # 将 query、key 和 value 分割成多个头
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
    
        # 计算注意力输出
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]
    
        # 合并注意力输出的多个头
        a = self.merge_heads(a)
        # 将合并后的结果传入 self.c_proj 进行线性变换
        a = self.c_proj(a)
        # 对变换后的结果应用 dropout
        a = self.resid_dropout(a)
    
        # 返回注意力输出以及可能的其他输出
        outputs = [a] + attn_outputs[1:]
        return outputs
# MLP (Multilayer Perceptron) 模块定义
class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 获取 config 中的 n_embd 参数
        nx = config.n_embd
        # 创建两个线性变换层
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        # 根据 config 中的 afn 参数获取激活函数
        self.act = ACT_FNS[config.afn]
        # 创建随机失活层
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        # 通过第一个线性变换层，激活函数，第二个线性变换层，随机失活层
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


# Block 模块定义
class Block(nn.Module):
    def __init__(self, n_positions, config, scale=False):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 获取 config 中的 n_embd 参数
        nx = config.n_embd
        # 创建注意力机制层
        self.attn = Attention(nx, n_positions, config, scale)
        # 创建两个层归一化层
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        # 计算注意力输出
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        a = attn_outputs[0]

        # 进行层归一化、MLP 计算、层归一化
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        # 返回计算结果和可选的注意力输出
        outputs = [h] + attn_outputs[1:]
        return outputs


# 预训练模型的基类定义
class OpenAIGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类、加载权重函数、基础模型前缀
    config_class = OpenAIGPTConfig
    load_tf_weights = load_tf_weights_in_openai_gpt
    base_model_prefix = "transformer"

    def _init_weights(self, module):
        """Initialize the weights."""
        # 根据模块类型初始化权重
        if isinstance(module, (nn.Linear, Conv1D)):
            # 线性层使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Embedding 层使用正态分布初始化权重，并将 padding_idx 对应的权重置零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 层归一化层将偏差置零，权重置为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 双头模型输出定义
@dataclass
class OpenAIGPTDoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.
    """
    pass
    # 这是一个 Transformer 模型的输出类型注解
    # 包含以下几个属性:
    # loss: 语言建模损失函数值(可选)
    # mc_loss: 多选分类损失函数值(可选)
    # logits: 语言建模预测得分(批大小, 选项数, 序列长度, 词汇表大小)
    # mc_logits: 多选分类预测得分(批大小, 选项数)
    # hidden_states: 模型每一层的隐藏状态(批大小, 序列长度, 隐藏层大小)
    # attentions: 模型每一层的注意力权重(批大小, 注意力头数, 序列长度, 序列长度)
        loss: Optional[torch.FloatTensor] = None
        mc_loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        mc_logits: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
# OpenAI GPT 模型的文档字符串，用于说明该模型的继承关系和一般用法
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

# OpenAI GPT 模型的输入文档字符串，用于说明模型输入的格式和含义
OPENAI_GPT_INPUTS_DOCSTRING = r"""
    # 参数说明：
    # input_ids: 输入序列令牌在词汇表中的索引，形状为(batch_size, sequence_length)
    # attention_mask: 避免在填充令牌索引上执行注意力操作的掩码。掩码值在[0, 1]范围内：
    #                - 1表示**未掩码**的令牌，
    #                - 0表示**掩码**的令牌。
    # token_type_ids: 指示输入的第一部分和第二部分的段标记索引。索引在[0, 1]范围内：
    #                - 0对应于*句子A*的令牌，
    #                - 1对应于*句子B*的令牌。
    # position_ids: 输入序列令牌在位置嵌入中的位置索引。在范围[0, config.max_position_embeddings - 1]内选择。
    # head_mask: 空值化自注意力模块的选定头部的掩码。掩码值在[0, 1]范围内：
    #                - 1表示**未掩码**的头部，
    #                - 0表示**掩码**的头部。
    # inputs_embeds: 可选地，可以直接传递嵌入表示而不是传递`input_ids`。如果您想要更多控制如何将`input_ids`索引转换为相关向量，这将非常有用。
    # output_attentions: 是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的`attentions`。
    # output_hidden_states: 是否返回所有层的隐藏状态。有关更多细节，请参见返回张量中的`hidden_states`。
    # return_dict: 是否返回`~utils.ModelOutput`而不是普通元组。
"""
定义一个 OpenAI GPT 模型，输出原始隐藏状态而不是顶部特定头部的裸变压器模型

"""
@add_start_docstrings(
    "The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.",
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 为 token 构建嵌入 (embedding) 层
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        # 构建位置嵌入层
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        # 初始化丢弃层
        self.drop = nn.Dropout(config.embd_pdrop)
        # 创建列表，其中每个元素都是 Block 类的实例，数量为 config.n_layer
        self.h = nn.ModuleList([Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)])

        # 注册位置 id 缓冲
        self.register_buffer("position_ids", torch.arange(config.n_positions), persistent=False)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

# 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
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
# 添加开始文档字符串
@add_start_docstrings(
    """
    OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
    # 定义 forward 方法，接收各种输入，并返回模型输出或损失
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None, # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None, # 分段 token ID
        position_ids: Optional[torch.LongTensor] = None, # 位置 ID
        head_mask: Optional[torch.FloatTensor] = None, # 头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None, # 输入嵌入向量
        labels: Optional[torch.LongTensor] = None, # 用于语言建模的标签
        output_attentions: Optional[bool] = None, # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典格式结果
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        # 确定是否返回字典格式结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Transformer 层进行前向传播
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
        hidden_states = transformer_outputs[0] # 获取隐藏状态
        lm_logits = self.lm_head(hidden_states) # 使用 lm_head 生成语言模型的输出

        loss = None
        if labels is not None:
            # 将 logits 和 labels 进行偏移以进行计算
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 将 tokens 展平以计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            # 如果不需要返回字典格式结果，则返回 loss 和输出
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含 loss、logits、hidden_states 和 attentions 的 CausalLMOutput 对象
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # 准备生成时的输入格式
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
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
    # 定义需要共享权重的关键字
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 将分类数目设定为1
        config.num_labels = 1
        # 初始化transformer模型
        self.transformer = OpenAIGPTModel(config)
        # 初始化语言模型头部
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 初始化多项选择头部
        self.multiple_choice_head = SequenceSummary(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
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
        """
        The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).
        [`OpenAIGPTForSequenceClassification`] uses the last token in order to do the classification, as other causal
        models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the
        last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding
        token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
        it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
        the last value in each row of the batch).
        """
        # ...
    # 初始化函数，设置配置参数和模型组件
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 从配置参数中获取标签数量
        self.num_labels = config.num_labels
        # 创建 OpenAIGPTModel 模型组件
        self.transformer = OpenAIGPTModel(config)
        # 创建一个线性层，将模型输出映射到标签数量的维度
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 模型前向传播函数，输入参数包括不同类型的张量和操作标志
    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        # 添加代码示例注释
        checkpoint=_CHECKPOINT_FOR_DOC,
        # 输出类型为 SequenceClassifierOutput
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数定义
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