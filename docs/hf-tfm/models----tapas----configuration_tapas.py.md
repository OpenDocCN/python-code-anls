# `.\models\tapas\configuration_tapas.py`

```
# coding=utf-8
# Copyright 2020 Google Research and The HuggingFace Inc. team.
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
TAPAS configuration. Based on the BERT configuration with added parameters.

Hyperparameters are taken from run_task_main.py and hparam_utils.py of the original implementation. URLS:

- https://github.com/google-research/tapas/blob/master/tapas/run_task_main.py
- https://github.com/google-research/tapas/blob/master/tapas/utils/hparam_utils.py

"""

# 引入PretrainedConfig类，用于构建预训练配置
from ...configuration_utils import PretrainedConfig

# TAPAS预训练模型配置映射表，包含不同预训练模型及其配置文件的URL
TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/tapas-base-finetuned-sqa": (
        "https://huggingface.co/google/tapas-base-finetuned-sqa/resolve/main/config.json"
    ),
    "google/tapas-base-finetuned-wtq": (
        "https://huggingface.co/google/tapas-base-finetuned-wtq/resolve/main/config.json"
    ),
    "google/tapas-base-finetuned-wikisql-supervised": (
        "https://huggingface.co/google/tapas-base-finetuned-wikisql-supervised/resolve/main/config.json"
    ),
    "google/tapas-base-finetuned-tabfact": (
        "https://huggingface.co/google/tapas-base-finetuned-tabfact/resolve/main/config.json"
    ),
}

# TapasConfig类，继承自PretrainedConfig类，用于存储Tapas模型的配置信息
class TapasConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TapasModel`]. It is used to instantiate a TAPAS
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the TAPAS
    [google/tapas-base-finetuned-sqa](https://huggingface.co/google/tapas-base-finetuned-sqa) architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Hyperparameters additional to BERT are taken from run_task_main.py and hparam_utils.py of the original
    implementation. Original implementation available at https://github.com/google-research/tapas/tree/master.

    Example:

    ```python
    >>> from transformers import TapasModel, TapasConfig

    >>> # Initializing a default (SQA) Tapas configuration
    >>> configuration = TapasConfig()
    >>> # Initializing a model from the configuration
    >>> model = TapasModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 指定模型类型为"tapas"
    model_type = "tapas"
    # 初始化函数，用于创建一个新的实例
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制dropout概率，默认为0.1
        max_position_embeddings=1024,  # 最大位置嵌入数，默认为1024
        type_vocab_sizes=[3, 256, 256, 2, 256, 256, 10],  # 类型词汇表大小列表，默认值指定
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化epsilon值，默认为1e-12
        pad_token_id=0,  # 填充token的ID，默认为0
        positive_label_weight=10.0,  # 正标签权重，默认为10.0
        num_aggregation_labels=0,  # 聚合标签数量，默认为0
        aggregation_loss_weight=1.0,  # 聚合损失权重，默认为1.0
        use_answer_as_supervision=None,  # 是否使用答案作为监督信号，默认为None
        answer_loss_importance=1.0,  # 答案损失重要性，默认为1.0
        use_normalized_answer_loss=False,  # 是否使用归一化答案损失，默认为False
        huber_loss_delta=None,  # Huber损失的δ值，默认为None
        temperature=1.0,  # 温度参数，默认为1.0
        aggregation_temperature=1.0,  # 聚合温度参数，默认为1.0
        use_gumbel_for_cells=False,  # 是否为单元使用Gumbel分布，默认为False
        use_gumbel_for_aggregation=False,  # 是否为聚合使用Gumbel分布，默认为False
        average_approximation_function="ratio",  # 平均逼近函数，默认为"ratio"
        cell_selection_preference=None,  # 单元选择偏好，默认为None
        answer_loss_cutoff=None,  # 答案损失截断值，默认为None
        max_num_rows=64,  # 最大行数，默认为64
        max_num_columns=32,  # 最大列数，默认为32
        average_logits_per_cell=False,  # 是否每个单元平均logits，默认为False
        select_one_column=True,  # 是否选择一个列，默认为True
        allow_empty_column_selection=False,  # 是否允许选择空列，默认为False
        init_cell_selection_weights_to_zero=False,  # 是否将单元选择权重初始化为零，默认为False
        reset_position_index_per_cell=True,  # 是否每个单元重置位置索引，默认为True
        disable_per_token_loss=False,  # 是否禁用每个token的损失，默认为False
        aggregation_labels=None,  # 聚合标签列表，默认为None
        no_aggregation_label_index=None,  # 无聚合标签索引，默认为None
        **kwargs,  # 其余关键字参数，用于接收未命名的参数
        ):
        # 调用父类的初始化方法，传入pad_token_id和其它关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 设置BERT模型的超参数
        self.vocab_size = vocab_size                     # 词汇表大小
        self.hidden_size = hidden_size                   # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers       # 隐藏层数量
        self.num_attention_heads = num_attention_heads   # 注意力头数量
        self.hidden_act = hidden_act                     # 隐藏层激活函数
        self.intermediate_size = intermediate_size       # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob   # 隐藏层dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力dropout概率
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.type_vocab_sizes = type_vocab_sizes         # 类型词汇表大小
        self.initializer_range = initializer_range       # 初始化范围
        self.layer_norm_eps = layer_norm_eps             # 层归一化的epsilon值

        # 微调任务的超参数
        self.positive_label_weight = positive_label_weight  # 正类标签权重
        self.num_aggregation_labels = num_aggregation_labels  # 聚合标签数量
        self.aggregation_loss_weight = aggregation_loss_weight  # 聚合损失权重
        self.use_answer_as_supervision = use_answer_as_supervision  # 是否使用答案作为监督
        self.answer_loss_importance = answer_loss_importance  # 答案损失重要性
        self.use_normalized_answer_loss = use_normalized_answer_loss  # 是否使用归一化答案损失
        self.huber_loss_delta = huber_loss_delta         # Huber损失的δ参数
        self.temperature = temperature                   # 温度参数
        self.aggregation_temperature = aggregation_temperature  # 聚合温度参数
        self.use_gumbel_for_cells = use_gumbel_for_cells  # 是否为单元使用Gumbel分布
        self.use_gumbel_for_aggregation = use_gumbel_for_aggregation  # 是否为聚合使用Gumbel分布
        self.average_approximation_function = average_approximation_function  # 平均逼近函数
        self.cell_selection_preference = cell_selection_preference  # 单元选择偏好
        self.answer_loss_cutoff = answer_loss_cutoff    # 答案损失截断
        self.max_num_rows = max_num_rows                # 最大行数
        self.max_num_columns = max_num_columns          # 最大列数
        self.average_logits_per_cell = average_logits_per_cell  # 每个单元的平均logits
        self.select_one_column = select_one_column      # 是否选择一个列
        self.allow_empty_column_selection = allow_empty_column_selection  # 是否允许空列选择
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero  # 初始化单元选择权重为零
        self.reset_position_index_per_cell = reset_position_index_per_cell  # 每个单元重置位置索引
        self.disable_per_token_loss = disable_per_token_loss  # 是否禁用每个token的损失

        # 聚合的超参数
        self.aggregation_labels = aggregation_labels    # 聚合标签
        self.no_aggregation_label_index = no_aggregation_label_index  # 无聚合标签索引

        # 如果聚合标签是字典，则将其键转换为整数类型
        if isinstance(self.aggregation_labels, dict):
            self.aggregation_labels = {int(k): v for k, v in aggregation_labels.items()}
```