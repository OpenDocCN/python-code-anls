# `.\models\deprecated\mmbt\modeling_mmbt.py`

```
# coding=utf-8
# 文件编码声明为UTF-8，确保可以处理各种字符
# Copyright (c) Facebook, Inc. and its affiliates.
# 版权声明，版权归Facebook及其关联公司所有
# Copyright (c) HuggingFace Inc. team.
# 版权声明，版权归HuggingFace Inc.团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache License 2.0许可证授权，详情参见许可证文档
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用本文件
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 可以在上述链接获取许可证副本
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则按"原样"分发软件，无论是明示的还是暗示的
# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证，了解特定语言下的权限和限制
"""PyTorch MMBT model."""
# PyTorch MMBT模型的定义

import torch
# 引入PyTorch库
from torch import nn
# 从PyTorch中引入神经网络模块
from torch.nn import CrossEntropyLoss, MSELoss
# 从PyTorch的神经网络模块中引入交叉熵损失和均方误差损失

from ....modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
# 从上层目录中的modeling_outputs模块引入BaseModelOutputWithPooling和SequenceClassifierOutput
from ....modeling_utils import ModuleUtilsMixin
# 从上层目录中的modeling_utils模块引入ModuleUtilsMixin
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 从上层目录中的utils模块引入add_start_docstrings、add_start_docstrings_to_model_forward、logging和replace_return_docstrings函数

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

_CONFIG_FOR_DOC = "MMBTConfig"
# 用于文档的配置名称为MMBTConfig

class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding."""
    # 通用的模态嵌入，接受一个编码器和一个Transformer嵌入

    def __init__(self, config, encoder, embeddings):
        super().__init__()
        # 调用父类的初始化方法
        self.config = config
        # 设置配置属性
        self.encoder = encoder
        # 设置编码器属性
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.hidden_size)
        # 使用线性层映射模态隐藏大小到隐藏大小
        self.position_embeddings = embeddings.position_embeddings
        # 设置位置嵌入属性
        self.token_type_embeddings = embeddings.token_type_embeddings
        # 设置token类型嵌入属性
        self.word_embeddings = embeddings.word_embeddings
        # 设置单词嵌入属性
        self.LayerNorm = embeddings.LayerNorm
        # 设置LayerNorm属性
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # 使用指定的丢弃概率创建Dropout层
    # 定义前向传播函数，接受输入模态数据，可选的起始和结束标记，位置编码和标记类型编码
    def forward(self, input_modal, start_token=None, end_token=None, position_ids=None, token_type_ids=None):
        # 获取输入模态数据的token embeddings
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        # 获取当前token embeddings的序列长度
        seq_length = token_embeddings.size(1)

        # 如果存在起始标记，则添加起始标记的embedding到token embeddings序列中
        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat([start_token_embeds.unsqueeze(1), token_embeddings], dim=1)

        # 如果存在结束标记，则添加结束标记的embedding到token embeddings序列末尾
        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat([token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)

        # 如果未提供位置编码，则根据序列长度创建默认位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(input_modal.size(0), seq_length)

        # 如果未提供标记类型编码，则创建全零的标记类型编码
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_modal.size(0), seq_length), dtype=torch.long, device=input_modal.device
            )

        # 根据位置编码获取位置embedding
        position_embeddings = self.position_embeddings(position_ids)
        # 根据标记类型编码获取标记类型embedding
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # 将token embeddings、位置embedding和标记类型embedding相加得到最终的嵌入表示
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        # 对嵌入表示进行Layer Normalization
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入表示进行dropout处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入表示作为前向传播的输出
        return embeddings
"""
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



"""
The bare MMBT Model outputting raw hidden-states without any specific head on top.
Inherits documentation from MMBT_START_DOCSTRING.

This class represents the core MMBT model without any additional task-specific head, providing raw hidden states.

Attributes:
    config: Model configuration instance containing all model parameters.
    transformer: Text transformer module used by MMBT for processing textual inputs.
    modal_encoder: Module for handling the encoding of the second modality (e.g., images).

Methods:
    forward: Defines the forward pass of the model, detailing how inputs propagate through the network.
    get_input_embeddings: Retrieves the word embedding layer of the model.
    set_input_embeddings: Sets the word embedding layer of the model.

"""
    MMBT_INPUTS_DOCSTRING,


注释：


    # 使用 MMBT_INPUTS_DOCSTRING 常量
# 定义一个名为MMBTForClassification的神经网络模型类，继承自nn.Module类，用于分类任务
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
    transformer = BertModel.from_pretrained("google-bert/bert-base-uncased")
    encoder = ImageEncoder(args)
    model = MMBTForClassification(config, transformer, encoder)
    outputs = model(input_modal, input_ids, labels=labels)
    loss, logits = outputs[:2]
    ```"""

    # 初始化方法，定义了模型的结构
    def __init__(self, config, transformer, encoder):
        super().__init__()
        # 设置类别数量
        self.num_labels = config.num_labels

        # 初始化MMBT模型，传入配置、transformer和encoder
        self.mmbt = MMBTModel(config, transformer, encoder)
        # 添加一个dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器层，线性变换，将隐藏状态的大小映射到类别数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    # 前向传播方法，定义了数据从输入到输出的流程
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
        ):
            # 如果 return_dict 不是 None，则使用传入的 return_dict，否则使用 self.config.use_return_dict
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # 使用 MMBT 模型进行推理，传入各种输入参数
            outputs = self.mmbt(
                input_modal=input_modal,
                input_ids=input_ids,
                modal_start_tokens=modal_start_tokens,
                modal_end_tokens=modal_end_tokens,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                modal_token_type_ids=modal_token_type_ids,
                position_ids=position_ids,
                modal_position_ids=modal_position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                return_dict=return_dict,
            )

            # 获取 MMBT 模型输出的池化后的表示
            pooled_output = outputs[1]

            # 对池化后的表示进行 dropout 操作
            pooled_output = self.dropout(pooled_output)
            # 将 dropout 后的表示输入分类器获取 logits
            logits = self.classifier(pooled_output)

            loss = None
            # 如果有标签，则计算损失
            if labels is not None:
                if self.num_labels == 1:
                    # 如果只有一个标签，说明是回归任务
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    # 多标签分类任务，使用交叉熵损失函数
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # 如果 return_dict 为 False，则返回一个元组，包括 logits 和额外的输出
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            # 如果 return_dict 为 True，则返回一个 SequenceClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
```