# `.\transformers\models\biogpt\modeling_biogpt.py`

```
# 设置编码为 UTF-8
# 版权声明
# 版权所有 2022 年 HuggingFace 团队和微软研究 AI4Science。保留所有权利。
# 
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则软件
# 根据“按原样”分发的基础分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch BioGPT 模型。"""

# 导入所需库
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义库和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_biogpt import BioGptConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的模型和配置信息
_CHECKPOINT_FOR_DOC = "microsoft/biogpt"
_CONFIG_FOR_DOC = "BioGptConfig"

# 预训练模型存档列表
BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/biogpt",
    "microsoft/BioGPT-Large",
    # 查看所有 BioGPT 模型 https://huggingface.co/models?filter=biogpt
]

# 从 transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding 复制的代码，用于 BioGpt
class BioGptLearnedPositionalEmbedding(nn.Embedding):
    """
    此模块学习固定最大大小的位置嵌入。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 对于 BioGpt，如果指定了 padding_idx，则通过偏移嵌入 id 2 并相应地调整 num_embeddings。
        # 其他模型没有此偏移的设置。
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` 期望为 [bsz x seqlen]。"""
        attention_mask = attention_mask.long()

        # 根据 attention_mask 创建位置
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # 如果 past_key_values_length > 0，则截断位置
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)

# 从 transformers.models.bart.modeling_bart.BartAttention 复制的代码，用于 BioGpt
class BioGptAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 定义多头注意力机制的初始化方法
    def __init__(
        self,
        embed_dim: int,  # 输入嵌入维度
        num_heads: int,  # 头数
        dropout: float = 0.0,  # dropout 概率，默认为 0
        is_decoder: bool = False,  # 是否为解码器，默认为 False
        bias: bool = True,  # 是否包含偏置，默认为 True
        is_causal: bool = False,  # 是否是因果注意力，默认为 False
        config: Optional[BioGptConfig] = None,  # 配置对象，默认为 None
    ):
        super().__init__()
        # 初始化参数
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.dropout = dropout  # dropout 概率
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        self.config = config  # 配置对象

        # 检查嵌入维度是否可以被头数整除
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否是因果注意力

        # 初始化线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # K 投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # V 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # Q 投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影层

    # 将张量变换为期望形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入张量
        key_value_states: Optional[torch.Tensor] = None,  # K/V 张量
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 上一步的 K/V
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        layer_head_mask: Optional[torch.Tensor] = None,  # 层级头掩码
        output_attentions: bool = False,  # 是否输出注意力权重
```  
class BioGptDecoderLayer(nn.Module):
    # 定义一个类，表示 BioGpt 解码器的一个层
    def __init__(self, config: BioGptConfig):
        # 初始化函数
        super().__init__()
        # 隐藏层大小即嵌入维度
        self.embed_dim = config.hidden_size

        # 自注意力机制层
        self.self_attn = BioGptAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )
        # 随机失活率
        self.dropout = config.hidden_dropout_prob
        # 激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 激活函数的失活率
        self.activation_dropout = config.activation_dropout

        # 自注意力机制层的 LayerNorm 归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        # 全连接层2
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        # 最终的 LayerNorm 归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # 前向传播函数，用于计算模型的输出
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        # 保存输入 hidden_states 作为残差连接的基准
        residual = hidden_states

        # 对 hidden_states 进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # 获取过去的 key 和 value 用于缓存
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对 hidden_states 进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到当前 hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        # 保存当前 hidden_states 作为残差连接的基准
        residual = hidden_states
        # 对 hidden_states 进行 Layer Normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 对 hidden_states 进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 进行 Dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到当前 hidden_states
        hidden_states = residual + hidden_states

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，则添加到输出元组中
        if use_cache:
            outputs += (present_key_value,)

        return outputs
class BioGptPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和一个简单的接口，用于下载和加载预训练模型。
    """

    # 指定模型配置类
    config_class = BioGptConfig
    # 模型名称前缀
    base_model_prefix = "biogpt"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 对于线性层，使用正态分布初始化权重
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在 padding_idx，则将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，初始化偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


BIOGPT_START_DOCSTRING = r"""
    此模型是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其用作
    常规 PyTorch 模块，并参考 PyTorch 文档以了解与一般用法和行为相关的所有内容。

    参数:
        config ([`~BioGptConfig`]): 具有模型所有参数的模型配置类。
            使用配置文件进行初始化不会加载与模型关联的权重，仅加载配置。查看 [`~PreTrainedModel.from_pretrained`] 
            方法以加载模型权重。
"""

BIOGPT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "输出原始隐藏状态而没有特定头部的裸BioGPT模型变换器。",
    BIOGPT_START_DOCSTRING,
)
class BioGptModel(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig):
        super().__init__(config)
        self.config = config
        self.layerdrop = config.layerdrop
        self.dropout = config.hidden_dropout_prob
        self.embed_dim = config.hidden_size
        self.padding_idx = config.pad_token_id
        # 如果配置要求，将嵌入层缩放
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 初始化嵌入层和位置嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, self.embed_dim, self.padding_idx)
        self.embed_positions = BioGptLearnedPositionalEmbedding(config.max_position_embeddings, self.embed_dim)

        # 初始化 Transformer 的各层
        self.layers = nn.ModuleList([BioGptDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化 LayerNorm 层
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        # 梯度检查点标志，默认关闭
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # 将模型输入的描述添加到模型前向传播方法的文档字符串中
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 将代码示例的描述添加到模型前向传播方法的文档字符串中
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，可选
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选
# 使用装饰器为类添加文档字符串，描述了该模型的作用以及用途
@add_start_docstrings(
    """BioGPT Model with a `language modeling` head on top for CLM fine-tuning.""", BIOGPT_START_DOCSTRING
)
# 定义了一个新的类 BioGptForCausalLM，继承自 BioGptPreTrainedModel 类
class BioGptForCausalLM(BioGptPreTrainedModel):
    # 定义了一个列表，包含了需要绑定权重的关键字
    _tied_weights_keys = ["output_projection.weight"]

    # 初始化方法，接受一个参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 BioGptModel 类的实例，并将其赋值给 self.biogpt 属性
        self.biogpt = BioGptModel(config)
        # 创建一个线性层，将输入的特征映射到输出词汇表大小的空间，并将其赋值给 self.output_projection 属性
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用自定义的后初始化方法
        self.post_init()

    # 返回输出嵌入层的方法
    def get_output_embeddings(self):
        return self.output_projection

    # 设置输出嵌入层的方法
    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 确定是否返回字典形式的输出结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用预训练模型进行前向传播
        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 将序列输出投影到词汇表大小的空间
        prediction_scores = self.output_projection(sequence_output)

        # 初始化语言模型损失为 None
        lm_loss = None
        # 如果提供了标签
        if labels is not None:
            # 我们进行下一个标记的预测；将预测分数和输入 id 向后移动一个位置
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            # 使用交叉熵损失函数计算语言模型损失
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回字典形式的输出结果
        if not return_dict:
            # 组装输出元组
            output = (prediction_scores,) + outputs[1:]
            # 返回输出元组，如果语言模型损失不为 None 则包含在其中
            return ((lm_loss,) + output) if lm_loss is not None else output

        # 返回带有交叉注意力的因果语言模型输出
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, attention_mask, inputs_embeds=None, past_key_values=None, **kwargs
        # 如果 past_key_values 不为 None，则只保留输入 ID 的最后一个 token
        if past_key_values is not None:
            # 获取 past_key_values 中第一个元素的 shape 的第三个维度，即过去的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入 ID 的长度大于过去的长度，则移除前缀长度为过去的长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 如果 inputs_embeds 不为 None 且 past_key_values 为 None，则使用 inputs_embeds 作为模型输入
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 否则，使用 input_ids 作为模型输入
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，包括 attention_mask、past_key_values 和 use_cache
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        # 返回更新后的 model_inputs
        return model_inputs

    # 重新排序缓存 past_key_values，根据 beam_idx
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 遍历 past_key_values 中的每一层
        for layer_past in past_key_values:
            # 对每个 past_state 根据 beam_idx 进行重新排序，并添加到 reordered_past 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的 past_key_values
        return reordered_past
# 定义一个带有标记分类头部的 BioGPT 模型，用于命名实体识别等任务
class BioGptForTokenClassification(BioGptPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建一个 BioGptModel 模型
        self.biogpt = BioGptModel(config)
        # 检查是否有分类器丢弃率，如果没有则使用隐藏层丢弃率
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        else:
            classifier_dropout = config.hidden_dropout_prob
        # 创建一个丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 执行初始化后的操作
        self.post_init()

    # 重写 forward 函数
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用传入的值；否则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型进行前向传播
        transformer_outputs = self.biogpt(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 BERT 输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态进行 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 使用分类器对隐藏状态进行分类，得到预测的 logits
        logits = self.classifier(hidden_states)

        loss = None
        # 如果 labels 不为空，则计算损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 仅保留损失的有效部分
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                # 仅保留激活的标签
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                # 计算损失
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典，则返回模型输出
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 定义一个带有顺序分类头的 BioGpt 模型转换器（线性层）
# [`BioGptForSequenceClassification`] 使用最后一个标记进行分类，与其他因果模型（例如 GPT-2）一样。
# 由于它在最后一个标记上进行分类，因此需要知道最后一个标记的位置。如果在配置中定义了 `pad_token_id`，则在每一行中找到不是填充标记的最后一个标记。如果没有定义 `pad_token_id`，则简单地取每一行批次中的最后一个值。由于无法猜测当传递 `inputs_embeds` 而不是 `input_ids` 时的填充标记，它执行相同的操作（取每一行批次中的最后一个值）。
class BioGptForSequenceClassification(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 初始化 BioGpt 模型
        self.biogpt = BioGptModel(config)
        # 初始化线性层
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def get_input_embeddings(self):
        # 返回嵌入标记
        return self.biogpt.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入
        self.biogpt.embed_tokens = value
```