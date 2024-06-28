# `.\models\deprecated\retribert\modeling_retribert.py`

```py
# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
RetriBERT model
"""


import math  # 导入数学模块
from typing import Optional  # 导入类型提示模块

import torch  # 导入PyTorch
import torch.utils.checkpoint as checkpoint  # 导入PyTorch的checkpoint模块
from torch import nn  # 导入PyTorch的神经网络模块

from ....modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ....utils import add_start_docstrings, logging  # 导入文档字符串添加和日志记录工具
from ...bert.modeling_bert import BertModel  # 导入BERT模型
from .configuration_retribert import RetriBertConfig  # 导入RetriBERT的配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 定义RetriBERT预训练模型存档列表
    "yjernite/retribert-base-uncased",
    # See all RetriBert models at https://huggingface.co/models?filter=retribert
]


# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #
class RetriBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RetriBertConfig  # 设置配置类为RetriBertConfig
    load_tf_weights = None  # 不使用TensorFlow权重加载
    base_model_prefix = "retribert"  # 设置基础模型前缀为retribert

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


RETRIBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


"""
    Parameters:
        config ([`RetriBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
@add_start_docstrings(
    """Bert Based model to embed queries or document for document retrieval.""",
    RETRIBERT_START_DOCSTRING,
)
"""
# 使用 `add_start_docstrings` 装饰器添加文档字符串，描述这是一个基于BERT的模型，用于嵌入查询或文档以进行文档检索
class RetriBertModel(RetriBertPreTrainedModel):
    def __init__(self, config: RetriBertConfig) -> None:
        super().__init__(config)
        # 初始化模型，设定投影维度
        self.projection_dim = config.projection_dim

        # 建立查询BERT模型
        self.bert_query = BertModel(config)
        # 如果配置中不共享编码器，建立文档BERT模型；否则设为None
        self.bert_doc = None if config.share_encoders else BertModel(config)
        # 设定Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 设定查询投影层
        self.project_query = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
        # 设定文档投影层
        self.project_doc = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 设定交叉熵损失函数
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")

        # 初始化权重并进行最终处理
        self.post_init()

    def embed_sentences_checkpointed(
        self,
        input_ids,
        attention_mask,
        sent_encoder,
        checkpoint_batch_size=-1,
    ):
        # 使用检查点技术重现BERT前向传播
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            # 如果不使用检查点或者输入量小于检查点批大小，直接进行前向传播
            return sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            # 准备隐式变量
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = sent_encoder.get_extended_attention_mask(
                attention_mask, input_shape
            )

            # 定义用于检查点的函数
            def partial_encode(*inputs):
                encoder_outputs = sent_encoder.encoder(
                    inputs[0],
                    attention_mask=inputs[1],
                    head_mask=head_mask,
                )
                sequence_output = encoder_outputs[0]
                pooled_output = sent_encoder.pooler(sequence_output)
                return pooled_output

            # 将所有输入一次性运行嵌入层
            embedding_output = sent_encoder.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            # 每次处理一个小批次的编码和汇总
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)
    # Embedding questions by processing input_ids using the specified BERT model (self.bert_query).
    # If attention_mask is provided, it's used to mask certain tokens during embedding.
    # Utilizes checkpointing if checkpoint_batch_size is specified.
    def embed_questions(
        self,
        input_ids,
        attention_mask=None,
        checkpoint_batch_size=-1,
    ):
        # Embedding sentences using the checkpointed embedding method with BERT for queries.
        q_reps = self.embed_sentences_checkpointed(
            input_ids,
            attention_mask,
            self.bert_query,
            checkpoint_batch_size,
        )
        # Projecting the embedded query representations to a different space if needed.
        return self.project_query(q_reps)

    # Embedding answers by processing input_ids using either self.bert_query or self.bert_doc BERT models.
    # Choice depends on the availability of self.bert_doc; defaults to self.bert_query if not available.
    # Utilizes checkpointing if checkpoint_batch_size is specified.
    def embed_answers(
        self,
        input_ids,
        attention_mask=None,
        checkpoint_batch_size=-1,
    ):
        # Embedding sentences using the checkpointed embedding method with BERT for answers.
        a_reps = self.embed_sentences_checkpointed(
            input_ids,
            attention_mask,
            self.bert_query if self.bert_doc is None else self.bert_doc,
            checkpoint_batch_size,
        )
        # Projecting the embedded document representations to a different space.
        return self.project_doc(a_reps)

    # Forward method for the model, which processes both query and document input_ids with their respective masks.
    # Uses the specified BERT models (self.bert_query for queries and self.bert_doc for documents if available).
    # Allows for checkpointing if checkpoint_batch_size is specified.
    ) -> torch.FloatTensor:
        r"""
        Args:
            input_ids_query (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the queries in a batch.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask_query (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            input_ids_doc (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the documents in a batch.
            attention_mask_doc (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on documents padding token indices.
            checkpoint_batch_size (`int`, *optional*, defaults to `-1`):
                If greater than 0, uses gradient checkpointing to only compute sequence representation on
                `checkpoint_batch_size` examples at a time on the GPU. All query representations are still compared to
                all document representations in the batch.

        Return:
            `torch.FloatTensor``: The bidirectional cross-entropy loss obtained while trying to match each query to its
            corresponding document and each document to its corresponding query in the batch
        """
        # 获取输入 query 的设备信息
        device = input_ids_query.device
        # 生成 query 的表示向量
        q_reps = self.embed_questions(input_ids_query, attention_mask_query, checkpoint_batch_size)
        # 生成 document 的表示向量
        a_reps = self.embed_answers(input_ids_doc, attention_mask_doc, checkpoint_batch_size)
        # 计算 query 和 document 之间的相似度分数矩阵
        compare_scores = torch.mm(q_reps, a_reps.t())
        # 计算 query-to-answer 的交叉熵损失
        loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.shape[1]).to(device))
        # 计算 answer-to-query 的交叉熵损失
        loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.shape[0]).to(device))
        # 计算最终的损失，为两种交叉熵损失的平均值
        loss = (loss_qa + loss_aq) / 2
        # 返回损失值
        return loss
```