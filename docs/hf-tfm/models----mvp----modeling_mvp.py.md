# `.\transformers\models\mvp\modeling_mvp.py`

```
# 设置文件编码为utf-8
# 版权声明
# 基于Apache许可证2.0授权，可以在不违反许可证的情况下使用此文件
# 可以通过以下链接获得许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则在"AS IS"的基础上分发软件
# 没有任何形式的保证或条件，无论明示或暗示
# 请参阅许可证以获取特定语言的权限和限制

""" PyTorch MVP model."""
# 上面的文字是程序的版权声明和许可证内容
import copy  # 导入拷贝模块
import math  # 导入数学模块
from typing import List, Optional, Tuple, Union  # 导入类型提示模块

import torch  # 导入torch模块
import torch.utils.checkpoint  # 导入torch.utils.checkpoint模块
from torch import nn  # 从torch中导入nn模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从torch.nn中导入BCEWithLogitsLoss、CrossEntropyLoss、MSELoss类

from ...activations import ACT2FN  # 从指定路径中导入ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 从指定路径中导入_prepare_4d_attention_mask、_prepare_4d_causal_attention_mask函数
from ...modeling_outputs import (  # 从指定路径中导入各种输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 从指定路径中导入PreTrainedModel类
from ...utils import (  # 从指定路径中导入各种辅助函数
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mvp import MvpConfig  # 从指定路径中导入MvpConfig类

logger = logging.get_logger(__name__)  # 通过logging模块获取日志对象

_CHECKPOINT_FOR_DOC = "RUCAIBox/mvp"  # 定义_CHECKPOINT_FOR_DOC变量
_CONFIG_FOR_DOC = "MvpConfig"  # 定义_CONFIG_FOR_DOC变量

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]  # 定义_EXPECTED_OUTPUT_SHAPE变量

MVP_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 定义MVP_PRETRAINED_MODEL_ARCHIVE_LIST变量
    "RUCAIBox/mvp",
    "RUCAIBox/mvp-data-to-text",
    "RUCAIBox/mvp-open-dialog",
    "RUCAIBox/mvp-question-answering",
    "RUCAIBox/mvp-question-generation",
    "RUCAIBox/mvp-story",
    "RUCAIBox/mvp-summarization",
    "RUCAIBox/mvp-task-dialog",
    "RUCAIBox/mtl-data-to-text",
    "RUCAIBox/mtl-multi-task",
    "RUCAIBox/mtl-open-dialog",
    "RUCAIBox/mtl-question-answering",
    "RUCAIBox/mtl-question-generation",
    "RUCAIBox/mtl-story",
    "RUCAIBox/mtl-summarization",
    # See all MVP models at https://huggingface.co/models?filter=mvp
]

# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 将输入id向右移动一个标记位
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        # 抛出异常
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 用pad_token_id替换标签中可能的-100值
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    # 返回移位后的输入 ID 序列
        return shifted_input_ids
# 这个类实现了可学习的位置编码，用于模型中的位置信息编码
class MvpLearnedPositionalEmbedding(nn.Embedding):
    """
    这个模块学习到固定最大尺寸的位置编码。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # MVP 模型中如果指定了 padding_idx，则位置编码 id 会偏移 2，并相应调整 num_embeddings
        # 其他模型没有这种处理方式
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """输入 input_ids 的形状应该是 [bsz x seqlen]。"""

        bsz, seq_len = input_ids.shape[:2]
        # 根据当前序列长度和之前的 key/value 计算位置编码索引
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        # 返回对应的位置编码
        return super().forward(positions + self.offset)


# 这个类实现了多头注意力机制
class MvpAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 检查嵌入维度是否能被头数整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 定义注意力机制所需的线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状以适应多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        attn_prompt: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 多头注意力机制的前向传播

class MvpEncoderLayer(nn.Module):
    # MVP 编码器层的实现
    # 初始化函数，接受一个MvpConfig类型的参数config
    def __init__(self, config: MvpConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为config中的模型维度
        self.embed_dim = config.d_model
        # 初始化self attention层，使用MvpAttention类
        self.self_attn = MvpAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 初始化self attention层后的LayerNorm层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置dropout概率为config中的dropout概率
        self.dropout = config.dropout
        # 设置激活函数为config中指定的激活函数类型对应的函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的dropout概率为config中的激活函数dropout概率
        self.activation_dropout = config.activation_dropout
        # 初始化全连接层fc1，输入维度为嵌入维度，输出维度为config中的FFN维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 初始化全连接层fc2，输入维度为config中的FFN维度，输出维度为嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 初始化最终的LayerNorm层，输入维度为嵌入维度
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受多个torch.FloatTensor类型的参数
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        self_attn_prompt: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            self_attn_prompt (`torch.FloatTensor`): prompt of self attention of shape
                `(2, encoder_attention_heads, pro_len, head_dim)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存输入的隐藏状态，用于残差连接
        residual = hidden_states
        # 进行自注意力计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            attn_prompt=self_attn_prompt,
            output_attentions=output_attentions,
        )
        # 对隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 对得到的结果进行Layer Norm
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存上一次的隐藏状态，用于残差连接
        residual = hidden_states
        # 使用激活函数计算隐藏状态
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 通过全连接层计算隐藏状态
        hidden_states = self.fc2(hidden_states)
        # 对隐藏状态进行dropout操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 残差连接
        hidden_states = residual + hidden_states
        # 对得到的结果进行Layer Norm
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果隐藏状态的数据类型为torch.float16且包含无穷大或NaN值
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 根据数据类型设置clamp_value的值
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            # 对隐藏状态进��值裁剪
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 设置输出为隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重
        if output_attentions:
            # 将注意力权重也加入到输出中
            outputs += (attn_weights,)

        # 返回输出值
        return outputs
# 定义MvpDecoderLayer类，继承自nn.Module
class MvpDecoderLayer(nn.Module):
    # 初始化方法接受MvpConfig类型的config参数
    def __init__(self, config: MvpConfig):
        super().__init__()
        # 将d_model值赋给embed_dim属性
        self.embed_dim = config.d_model

        # 初始化self_attn属性为MvpAttention实例，传入相关参数
        self.self_attn = MvpAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 将dropout的值赋给dropout属性
        self.dropout = config.dropout
        # 将指定的activation_function转换为对应激活函数的函数，并赋给activation_fn属性
        self.activation_fn = ACT2FN[config.activation_function]
        # 将activation_dropout的值赋给activation_dropout属性

        self.activation_dropout = config.activation_dropout

        # 初始化self_attn_layer_norm属性为nn.LayerNorm的实例，传入embed_dim参数
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化encoder_attn属性为MvpAttention实例，传入相关参数
        self.encoder_attn = MvpAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        # 初始化encoder_attn_layer_norm属性为nn.LayerNorm的实例，传入embed_dim参数
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化fc1属性为nn.Linear的实例，传入embed_dim和decoder_ffn_dim参数
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化fc2属性为nn.Linear的实例，传入decoder_ffn_dim和embed_dim参数
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化final_layer_norm属性为nn.LayerNorm的实例，传入embed_dim参数

    # 前向传播方法，接受多个参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        self_attn_prompt: Optional[torch.Tensor] = None,
        cross_attn_prompt: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
# 定义MvpClassificationHead类，继承自nn.Module，用于句子级分类任务
class MvpClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # 初始化方法接受input_dim、inner_dim、num_classes、pooler_dropout参数
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        # 初始化dense属性为nn.Linear的实例，传入input_dim和inner_dim参数
        self.dense = nn.Linear(input_dim, inner_dim)
        # 初始化dropout属性为nn.Dropout的实例，传入pooler_dropout参数
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 初始化out_proj属性为nn.Linear的实例，传入inner_dim和num_classes参数

    # 前向传播方法，接受hidden_states参数，返回torch.Tensor类型的值
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将经过dense层的结果赋给hidden_states
        hidden_states = self.dense(hidden_states)
        # 对hidden_states进行tanh激活函数操作
        hidden_states = torch.tanh(hidden_states)
        # 再次对hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将经过out_proj层的结果赋给hidden_states
        hidden_states = self.out_proj(hidden_states)
        # 返回hidden_states
        return hidden_states


# 定义MvpPrompt类，层次提示器，用于编码器或解码器
class MvpPrompt(nn.Module):
    """Layer-wise prompt for encoder or decoder."""
    # 构建自定义的 Transformer 编码器模型
    def __init__(self, config, num_layers, num_heads):
        # 调用父类的初始化方法
        super().__init__()
        # 设置提示长度
        self.prompt_length = config.prompt_length
        # 设置层数
        self.num_layers = num_layers
        # 设置注意力头数
        self.num_heads = num_heads
        # 计算每个注意力头的维度
        self.head_dim = config.d_model // num_heads
        # 创建一个 Dropout 层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建一个可学习的提示词嵌入层
        self.prompt_embedding = nn.Embedding(config.prompt_length, config.d_model)
        # 创建一个包含三个线性层和一个 GELU 激活函数的序列模型
        self.prompt_trans = nn.Sequential(
            nn.Linear(config.d_model, config.prompt_mid_dim),
            nn.GELU(),
            nn.Linear(config.prompt_mid_dim, num_layers * 2 * config.d_model),
        )
    
    # 前向传播
    def forward(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        # 根据提示词 ID 获取提示词的嵌入向量
        prompt = self.prompt_trans(self.prompt_embedding(prompt_ids))
        # 调整提示词向量的形状为 (prompt_length, num_layers * 2, num_heads, head_dim)
        prompt = prompt.view(self.prompt_length, self.num_layers * 2, self.num_heads, self.head_dim)
        # 对提示词向量应用 Dropout
        prompt = self.dropout(prompt)
        # 将提示词向量的维度顺序调整为 (num_layers * 2, num_heads, prompt_length, head_dim)
        prompt = prompt.permute([1, 2, 0, 3]).split(2)
        # 返回分裂后的提示词向量
        return prompt
class MvpPreTrainedModel(PreTrainedModel):
    # 设置配置类为 MvpConfig
    config_class = MvpConfig
    # 设置基础模型前缀为 "model"
    base_model_prefix = "model"
    # 开启梯度检查点支持
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # 从配置中获取初始化标准差
        std = self.config.init_std
        # 如果是线性层模块
        if isinstance(module, nn.Linear):
            # 权重初始化为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # 偏置初始化为零
                module.bias.data.zero_()
        # 如果是嵌入层模块
        elif isinstance(module, nn.Embedding):
            # 权重初始化为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # 对应填充索引的权重初始化为零
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        # 获取填充标记
        pad_token = self.config.pad_token_id
        # 创建输入张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs



MVP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MvpConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MVP_INPUTS_DOCSTRING = r"""
"""



MVP_CONDITIONAL_GENERATION_EXAMPLE = r"""
    Example of summarization:

    Fine-tuning a model
    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, MvpForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
    >>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")

    >>> inputs = tokenizer(
    ...     "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
    ...     return_tensors="pt",
    ... )
    >>> labels = tokenizer("Bad Reasons To Quit Your Job", return_tensors="pt")["input_ids"]

    >>> loss = model(**inputs, labels=labels).loss
    >>> loss.backward()
    ```

    Inference after the model fine-tuned
    ```python
    >>> with torch.no_grad():
    ...     generated_ids = model.generate(**inputs)

    >>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ```
"""



MVP_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example of single-label classification:

    Fine-tuning a model on `num_labels` classes
    ```python
    # 导入 PyTorch 库
    import torch
    # 从 transformers 库中导入 AutoTokenizer 和 MvpForSequenceClassification 类
    from transformers import AutoTokenizer, MvpForSequenceClassification
    
    # 设定类别数目，例如，这是一个二分类任务
    num_labels = 2  
    # 使用预训练模型的标记器进行初始化
    tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
    # 使用预训练的 MVP 模型进行序列分类任务的初始化
    model = MvpForSequenceClassification.from_pretrained("RUCAIBox/mvp", num_labels=num_labels)
    
    # 对输入文本进行标记化并转换为 PyTorch 张量
    inputs = tokenizer("Classify: Hello, my dog is cute", return_tensors="pt")
    # 为输入文本设置真实标签
    labels = torch.tensor(1)  
    
    # 使用模型对输入进行前向传播并计算损失
    loss = model(**inputs, labels=labels).loss
    # 对损失进行反向传播
    loss.backward()
    
    # 模型微调后的推理
    with torch.no_grad():
        # 使用模型进行推理得到 logits
        logits = model(**inputs).logits
    
    # 根据 logits 得到预测的类别 ID
    predicted_class_id = logits.argmax()
# MVP问答样例代码
MVP_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    # 微调问答模型，支持抽取式和生成式问答
    利用 `BartForConditionalGeneration` 模型进行微调
    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, MvpForQuestionAnswering

    # 加载预训练的 tokenizer 和问答模型
    >>> tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
    >>> model = MvpForQuestionAnswering.from_pretrained("RUCAIBox/mvp")

    # 输入问题和上下文，转换为模型输入格式
    >>> inputs = tokenizer(
    ...     "Answer the following question: Who was Jim Henson? [SEP] Jim Henson was a nice puppet",
    ...     return_tensors="pt",
    ... )
    # 设置正确答案的起始和结束位置
    >>> target_start_index = torch.tensor([18])
    >>> target_end_index = torch.tensor([19])

    # 计算损失并反向传播
    >>> loss = model(**inputs, start_positions=target_start_index, end_positions=target_end_index).loss
    >>> loss.backward()
    ```

    # 微调后的推理过程
    ```python
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    # 预测答案的起始和结束位置
    >>> answer_start_index = outputs.start_logits.argmax()
    >>> answer_end_index = outputs.end_logits.argmax()

    # 根据预测位置获取答案
    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> predict_answer = tokenizer.decode(predict_answer_tokens)
    ```
"""


# MVP编码器模块
class MvpEncoder(MvpPreTrainedModel):
    """
    Transformer 编码器，包含 *config.encoder_layers* 个自注意力层。每个层都是 [`MvpEncoderLayer`]。

    参数:
        config: MvpConfig 配置
        embed_tokens (nn.Embedding): 输出嵌入层
        use_prompt (bool): 是否使用提示
    """

    def __init__(
        self, config: MvpConfig, embed_tokens: Optional[nn.Embedding] = None, use_prompt: Optional[bool] = False
    ):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 如果提供了嵌入层，使用它；否则创建一个新的嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 位置嵌入层
        self.embed_positions = MvpLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 编码器层列表
        self.layers = nn.ModuleList([MvpEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.use_prompt = use_prompt
        if use_prompt:
            self.prompt_length = config.prompt_length
            self.self_attn_prompt = MvpPrompt(
                config,
                config.encoder_layers,
                config.encoder_attention_heads,
            )

        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()
    # 获取输入嵌入
    def get_input_embeddings(self):
        # 返回嵌入令牌
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        # 将输入嵌入设置为给定值
        self.embed_tokens = value

    # 前向传播函数，接收一系列输入，并返回模型的输出
    def forward(
        self,
        input_ids: torch.LongTensor = None,                 # 输入的token IDs
        attention_mask: Optional[torch.Tensor] = None,      # 表示哪些token要被attention，哪些不用
        head_mask: Optional[torch.Tensor] = None,           # 头部遮罩，用于屏蔽不需要的attention头
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        output_attentions: Optional[bool] = None,           # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,        # 是否输出隐藏状态
        return_dict: Optional[bool] = None,                 # 是否以字典形式返回输出
# 定义 MvpDecoder 类，继承自 MvpPreTrainedModel
class MvpDecoder(MvpPreTrainedModel):
    """
    Transformer 解码器，由 config.decoder_layers 指定的层数组成。每个层是一个 MvpDecoderLayer 类的实例。
    
    参数:
        config: MvpConfig 配置对象
        embed_tokens (nn.Embedding): 输出嵌入
        use_prompt (bool): 是否使用提示
    """

    def __init__(
        # 初始化函数，接收 MvpConfig 配置对象、嵌入张量、以及是否使用提示的标志
        self, config: MvpConfig, embed_tokens: Optional[nn.Embedding] = None, use_prompt: Optional[bool] = False
    ):
        # 调用基类的初始化函数
        super().__init__(config)
        # 设置解码器的 dropout 和 layerdrop
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引和目标最大位置
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        # 根据配置决定嵌入的缩放因子
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果提供了嵌入张量，使用该张量；否则，创建一个新的嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 创建位置嵌入层
        self.embed_positions = MvpLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # 创建多个解码器层，并添加到模块列表中
        self.layers = nn.ModuleList([MvpDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建 LayerNorm 层，用于嵌入层
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 设置是否使用提示，并根据配置创建提示模块
        self.use_prompt = use_prompt
        if use_prompt:
            # 设置提示长度
            self.prompt_length = config.prompt_length
            # 创建自注意力提示模块
            self.self_attn_prompt = MvpPrompt(
                config,
                config.decoder_layers,
                config.decoder_attention_heads,
            )
            # 创建交叉注意力提示模块
            self.cross_attn_prompt = MvpPrompt(
                config,
                config.decoder_layers,
                config.decoder_attention_heads,
            )

        # 设置梯度检查点标志，默认值为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传递方法，接收输入和可选参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 添加起始文档字符串
@add_start_docstrings(
    "输出原始隐藏状态的 MVP 模型，没有任何特定的头部。",
    MVP_START_DOCSTRING,
)
# 定义 MvpModel 类，继承自 MvpPreTrainedModel
class MvpModel(MvpPreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = ["final_logits_bias"]
    # 模型预定义的绑定权重键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    
    # 初始化模型
    def __init__(self, config: MvpConfig):
        # 调用父类的初始化方法
        super().__init__(config)
    
        # 从配置中获取 padding_idx 和 vocab_size
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 获取是否使用提示的配置
        self.use_prompt = config.use_prompt
        # 创建共享的词嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
    
        # 创建编码器和解码器模块
        self.encoder = MvpEncoder(config, self.shared, config.use_prompt)
        self.decoder = MvpDecoder(config, self.shared, config.use_prompt)
    
        # 初始化权重并应用最终处理
        self.post_init()
    
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared
    
    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
    
    # 获取编码器模块
    def get_encoder(self):
        return self.encoder
    
    # 获取解码器模块
    def get_decoder(self):
        return self.decoder
    
    # 设置轻量级微调
    def set_lightweight_tuning(self):
        # 确保使用了提示
        assert self.use_prompt, "If you want to use lightweight tuning, make sure that `use_prompt=True`."
    
        # 冻结所有参数
        self.requires_grad_(False)
        # 只微调提示参数
        self.encoder.self_attn_prompt.requires_grad_(True)
        self.decoder.self_attn_prompt.requires_grad_(True)
        self.decoder.cross_attn_prompt.requires_grad_(True)
    
    # 前向传播
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 在此处编写注释
# 使用装饰器添加模型文档字符串的起始部分，描述了带语言建模头部的 MVP 模型，可用于各种文本生成任务
@add_start_docstrings(
    "The MVP Model with a language modeling head. Can be used for various text generation tasks.", MVP_START_DOCSTRING
)
# 定义了一个 MvpForConditionalGeneration 类，继承自 MvpPreTrainedModel 类
class MvpForConditionalGeneration(MvpPreTrainedModel):
    # 定义了一个列表，包含需要共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接收一个 MvpConfig 类型的参数 config
    def __init__(self, config: MvpConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 MvpModel 对象并赋值给 self.model
        self.model = MvpModel(config)
        # 使用零张量初始化 final_logits_bias 属性，其形状为 (1, self.model.shared.num_embeddings)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层 lm_head，输入维度为 config.d_model，输出维度为 self.model.shared.num_embeddings，没有偏置项
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 调用后处理方法
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整 final_logits_bias 的大小以匹配新的 token 数量
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # 调整 final_logits_bias 的大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于旧的 token 数量
        if new_num_tokens <= old_num_tokens:
            # 则截取现有的 final_logits_bias
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            # 否则创建额外的偏置项，然后拼接到 final_logits_bias 后面
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册调整大小后的 final_logits_bias
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出的 embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置轻量级调参
    def set_lightweight_tuning(self):
        # 设置模型为轻量级调参模式
        self.model.set_lightweight_tuning()
        # 冻结 lm_head 的梯度
        self.lm_head.requires_grad_(False)

    # 添加模型前向传播方法的文档字符串的起始部分
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    # 替换模型前向传播方法的返回值文档字符串
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向传播方法的文档字符串的结尾部分
    @add_end_docstrings(MVP_CONDITIONAL_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的序列的id表示，数据类型为torch.LongTensor，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 输入的序列的注意力掩码，数据类型为torch.Tensor，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入序列的id表示，数据类型为torch.LongTensor，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的输入序列的注意力掩码，数据类型为torch.LongTensor，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码，数据类型为torch.Tensor，默认为None
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器多头注意力机制的掩码，数据类型为torch.Tensor，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力机制的多头掩码，数据类型为torch.Tensor，默认为None
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器的输出，数据类型为List[torch.FloatTensor]，默认为None
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 编码器过去的键值对，数据类型为List[torch.FloatTensor]，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入后的输入序列，数据类型为torch.FloatTensor，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入后的解码器输入序列，数据类型为torch.FloatTensor，默认为None
        labels: Optional[torch.LongTensor] = None,  # 标签序列的id表示，数据类型为torch.LongTensor，默认为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，数据类型为bool，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，数据类型为bool，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，数据类型为bool，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，数据类型为bool，默认为None
    # 此函数用于计算序列到序列的语言模型的输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # 判断是否使用返回字典的形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 如果提供了标签(labels)
        if labels is not None:
            # 如果使用了缓存(use_cache)，则警告并禁用缓存
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # 如果没有decoder_input_ids和decoder_inputs_embeds，则从labels中移位生成decoder_input_ids
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
    
        # 调用self.model方法获取模型输出
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 计算语言模型预测输出logits
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    
        # 如果提供了标签(labels)，计算masked language modeling损失
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    
        # 根据return_dict标志返回不同的输出格式
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    # 为生成器准备输入，根据参数准备输入数据和配置信息
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,  # 过去的键值，用于生成
        attention_mask=None,  # 注意力掩码
        head_mask=None,  # 头部掩码
        decoder_head_mask=None,  # 解码器头部掩码
        cross_attn_head_mask=None,  # 交叉注意力头部掩码
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器输出
        **kwargs,
    ):
        # 如果使用过去的键值
        if past_key_values is not None:
            # 获取过去的键值的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果解码器输入的长度大于过去的长度
            if decoder_input_ids.shape[1] > past_length:
                # 移除前缀部分的长度为过去的长度
                remove_prefix_length = past_length
            else:
                # 默认的旧行为：只保留最后一个输入 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 重新赋值解码器输入，只保留后缀
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # 从标签准备解码器输入 ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 重新排列缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排列 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 使用特定的文档字符串初始化一个包含序列分类/头部的 Mvp 模型（在汇总输出之上的线性层），例如 GLUE 任务
@add_start_docstrings(
    """
    Mvp model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MVP_START_DOCSTRING,
)
class MvpForSequenceClassification(MvpPreTrainedModel):
    # 用于指定共享权重的键值对
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MvpConfig, **kwargs):
        super().__init__(config, **kwargs)
        # 初始化 MvpModel
        self.model = MvpModel(config)
        # 初始化序列分类头
        self.classification_head = MvpClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 设置轻量级调整
    def set_lightweight_tuning(self):
        self.model.set_lightweight_tuning()
        self.classification_head.requires_grad_(False)

    # 前向传播方法，添加了文档字符串注释
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    @add_end_docstrings(MVP_SEQUENCE_CLASSIFICATION_SAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 使用特定的文档字符串初始化一个包含用于抽取式问答任务的跨度分类头部的 Mvp 模型（用于计算 `跨度起始对数` 和 `跨度结束对数` 的隐藏状态输出之上的线性层）
@add_start_docstrings(
    """
    MVP Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MVP_START_DOCSTRING,
)
class MvpForQuestionAnswering(MvpPreTrainedModel):
    # 用于指定共享权重的键值对
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 设定类别数为 2（开始和结束的跨度）
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 初始化 MvpModel
        self.model = MvpModel(config)
        # 初始化 QA 输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 设置轻量级调整
    def set_lightweight_tuning(self):
        self.model.set_lightweight_tuning()
        self.qa_outputs.requires_grad_(False)

    # 前向传播方法，添加了文档字符串注释
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    @add_end_docstrings(MVP_QUESTION_ANSWERING_SAMPLE)
    # 定义一个方法，用于进行前向传播计算
    def forward(
        self,
        # 输入的标识符张量，默认为None
        input_ids: torch.Tensor = None,
        # 注意力掩码，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入的标识符张量，默认为None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，默认为None
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头掩码，默认为None
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头掩码，默认为None
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头掩码，默认为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，默认为None
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 开始位置，默认为None
        start_positions: Optional[torch.LongTensor] = None,
        # 结束位置，默认为None
        end_positions: Optional[torch.LongTensor] = None,
        # 输入的嵌入，默认为None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入的嵌入，默认为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 是否使用缓存，默认为None
        use_cache: Optional[bool] = None,
        # 输出注意力，默认为None
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态，默认为None
        output_hidden_states: Optional[bool] = None,
        # 返回字典，默认为None
        return_dict: Optional[bool] = None,
# 从transformers.models.bart.modeling_bart.BartDecoderWrapper复制而来，将Bart替换为Mvp
class MvpDecoderWrapper(MvpPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化MvpDecoder对象
        self.decoder = MvpDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class MvpForCausalLM(MvpPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深拷贝config对象
        config = copy.deepcopy(config)
        # 设置config中的is_decoder和is_encoder_decoder属性
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 初始化MvpDecoderWrapper对象
        self.model = MvpDecoderWrapper(config)

        # 初始化一个线性层用于LM头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入的嵌入对象
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入的嵌入对象
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出的嵌入对象
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出的嵌入对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 设置轻量级调整
    def set_lightweight_tuning(self):
        self.model.set_lightweight_tuning()
        self.lm_head.requires_grad_(False)

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    # 该函数用于创建用于解码器模型的注意力掩码
    def prepare_decoder_input_ids_from_labels(
        self,
        labels,
        decoder_input_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
    ):
        # 如果没有传入注意力掩码，则新建一个全1的注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
    
        # 如果有过去的关键值（用于生成）
        if past_key_values:
            # 获取过去序列的长度
            past_length = past_key_values[0][0].shape[2]
    
            # 如果输入序列长度大于过去序列长度，则从输入序列中除去过去序列长度部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            # 否则，保留输入序列的最后一个ID
            else:
                remove_prefix_length = input_ids.shape[1] - 1
    
            # 从输入序列中除去过去序列部分
            input_ids = input_ids[:, remove_prefix_length:]
    
        # 返回包含输入ID、注意力掩码、过去关键值和是否使用缓存的字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
    
    # 该函数用于根据给定的 beam_idx 重新排序过去的关键值
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化重新排序后的过去关键值
        reordered_past = ()
        # 遍历每一层的过去关键值
        for layer_past in past_key_values:
            # 对每一层的过去关键值进行重新排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去关键值
        return reordered_past
```