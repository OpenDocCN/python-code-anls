# `.\models\deprecated\mmbt\modeling_mmbt.py`

```
# 设置编码为utf-8，以便支持各种字符集
# 版权声明，版权所有Facebook，Inc.和其关联公司，版权所有HuggingFace Inc.团队
#
# 根据Apache License，Version 2.0许可，仅在遵守许可的情况下才可使用该文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则根据许可证分发的软件都是“基于现状”分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言。
"""PyTorch MMBT model."""


# 导入所需的库
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ....modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ....modeling_utils import ModuleUtilsMixin
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "MMBTConfig"

# 定义ModalEmbeddings类，用于处理modal embedding
class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding."""

    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        # 使用线性层进行modal_hidden_size到hidden_size的投影
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
    # 定义一个方法用于生成输入模态的前向传播结果
    def forward(self, input_modal, start_token=None, end_token=None, position_ids=None, token_type_ids=None):
        # 使用编码器对输入模态进行编码，并经过投影层得到 token_embeddings
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)  # 计算 token_embeddings 的序列长度

        # 如果存在起始标记，则将起始标记的词嵌入与 token_embeddings 进行拼接
        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1  # 序列长度加一
            token_embeddings = torch.cat([start_token_embeds.unsqueeze(1), token_embeddings], dim=1)

        # 如果存在结束标记，则将结束标记的词嵌入与 token_embeddings 进行拼接
        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1  # 序列长度加一
            token_embeddings = torch.cat([token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)

        # 如果 position_ids 为空，则根据序列长度创建 position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(input_modal.size(0), seq_length)

        # 如果 token_type_ids 为空，则创建全零的 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_modal.size(0), seq_length), dtype=torch.long, device=input_modal.device
            )

        # 将 position_ids 和 token_type_ids 转化为对应的词嵌入
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将 token_embeddings、position_embeddings、token_type_embeddings 相加得到最终的嵌入结果
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)  # 对嵌入结果进行层标准化
        embeddings = self.dropout(embeddings)  # 对嵌入结果进行 dropout 处理
        return embeddings  # 返回处理后的嵌入结果
MMBT_START_DOCSTRING = r"""
    MMBT model was proposed in [Supervised Multimodal Bitransformers for Classifying Images and
    Text](https://github.com/facebookresearch/mmbt) by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
    It's a supervised multimodal bitransformer model that fuses information from text and other image encoders, and
    obtain state-of-the-art performance on various multimodal classification benchmark tasks.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MMBTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration.
        transformer (`nn.Module`): A text transformer that is used by MMBT.
            It should have embeddings, encoder, and pooler attributes.
        encoder (`nn.Module`): Encoder for the second modality.
            It should take in a batch of modal inputs and return k, n dimension embeddings.
"""

MMBT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare MMBT Model outputting raw hidden-states without any specific head on top.",
    MMBT_START_DOCSTRING,
)
class MMBTModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.modal_encoder = ModalEmbeddings(config, encoder, transformer.embeddings)

    @add_start_docstrings_to_model_forward(MMBT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings(
    """
    MMBT Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    """,
    MMBT_START_DOCSTRING,
    # 提供了MMBT模型的输入文档字符串
    MMBT_INPUTS_DOCSTRING,
# 定义 MMTCForClassification 类，继承自 nn.Module
class MMBTForClassification(nn.Module):
    r"""
    **labels**: (*optional*) `torch.LongTensor` of shape `(batch_size,)`:
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
        config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
        `config.num_labels > 1` a classification loss is computed (Cross-Entropy). 

    Returns: *Tuple* comprising various elements depending on the configuration (config) and inputs: **loss**:
    (*optional*, returned when `labels` is provided) `torch.FloatTensor` of shape `(1,)`: Classification (or
    regression if config.num_labels==1) loss. **logits**: 
        `torch.FloatTensor` of shape `(batch_size, config.num_labels)` Classification (or regression if
        config.num_labels==1) scores (before SoftMax). 
    **hidden_states**: (*optional*, returned when `output_hidden_states=True`) list of `torch.FloatTensor` (one for 
    the output of each layer + the output of the embeddings) of shape `(batch_size, sequence_length, hidden_size)`: 
    Hidden-states of the model at the output of each layer plus the initial embedding outputs. **attentions**: 
    (*optional*, returned when `output_attentions=True`) list of `torch.FloatTensor` (one for each layer) of shape 
    `(batch_size, num_heads, sequence_length, sequence_length)`: Attentions weights after the attention softmax, used 
    to compute the weighted average in the self-attention heads. 

    Examples:

    ```python
    # For example purposes. Not runnable.
    transformer = BertModel.from_pretrained("bert-base-uncased")
    encoder = ImageEncoder(args)
    model = MMBTForClassification(config, transformer, encoder)
    outputs = model(input_modal, input_ids, labels=labels)
    loss, logits = outputs[:2]
    """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.num_labels = config.num_labels

        # 初始化 MMBTForClassification 类的成员变量
        self.mmbt = MMBTModel(config, transformer, encoder)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):  # 定义函数结束的括号，函数参数的起始
        # 如果 return_dict 参数未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 mmbt 模型进行推理
        outputs = self.mmbt(
            input_modal=input_modal,  # 模态输入数据
            input_ids=input_ids,  # 输入的文本数据的token IDs
            modal_start_tokens=modal_start_tokens,  # 模态输入数据的起始token
            modal_end_tokens=modal_end_tokens,  # 模态输入数据的结束token
            attention_mask=attention_mask,  # 注意力掩码
            token_type_ids=token_type_ids,  # token类型IDs
            modal_token_type_ids=modal_token_type_ids,  # 模态token类型IDs
            position_ids=position_ids,  # 位置IDs
            modal_position_ids=modal_position_ids,  # 模态位置IDs
            head_mask=head_mask,  # 头部掩码
            inputs_embeds=inputs_embeds,  # 输入的嵌入向量
            return_dict=return_dict,  # 是否返回字典形式的输出
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出进行dropout
        pooled_output = self.dropout(pooled_output)
        # 将dropout后的输出传入分类器进行分类
        logits = self.classifier(pooled_output)

        # 初始化损失值为None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果标签的数量为1，则执行回归任务
            if self.num_labels == 1:
                # 使用均方误差作为损失函数
                loss_fct = MSELoss()
                # 计算均方误差损失值
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 如果标签数量大于1，则执行分类任务
                loss_fct = CrossEntropyLoss()
                # 计算交叉熵损失值
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典形式的输出
        if not return_dict:
            # 将输出组装为元组
            output = (logits,) + outputs[2:]
            # 如果存在损失值，则将损失值与输出一起返回，否则只返回输出
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器的输出
        return SequenceClassifierOutput(
            loss=loss,  # 损失值
            logits=logits,  # 输出的logits
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )
```