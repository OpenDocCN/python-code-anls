# `.\transformers\models\tapas\configuration_tapas.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 Google Research 和 The HuggingFace Inc. team 所有
# 根据 Apache License, Version 2.0 授权使用此文件
# 可以在遵守许可证的情况下使用此文件
# 可以通过该链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得使用此文件
# 根据许可证的规定进行分发，基于"AS IS"的基础分发，没有任何形式的保证或条件，不管是明示的还是默示的
# 请查看许可证以了解特定语言的规则以及许可证下的限制

"""
TAPAS 配置。基于具有添加参数的 BERT 配置。

超参数取自原始实现的 run_task_main.py 和 hparam_utils.py。URLS:
- https://github.com/google-research/tapas/blob/master/tapas/run_task_main.py
- https://github.com/google-research/tapas/blob/master/tapas/utils/hparam_utils.py
"""

# 导入预训练配置 
from ...configuration_utils import PretrainedConfig

# TAPAS 预训练配置归档映射
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

# TAPAS 配置类，用于存储配置的配置类
class TapasConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`TapasModel`] 的配置。根据指定的参数实例化 TAPAS 模型，定义模型架构。使用默认值实例化配置会生成类似 TAPAS
    [google/tapas-base-finetuned-sqa](https://huggingface.co/google/tapas-base-finetuned-sqa) 架构的配置。

    配置对象继承自 [`PreTrainedConfig`]，可用于控制模型输出。阅读来自 [`PretrainedConfig`] 的文档以获取更多信息。

    超参数除了 BERT 外，还取自原始实现的 run_task_main.py 和 hparam_utils.py。原始实现位于 https://github.com/google-research/tapas/tree/master。

    示例:
    ```python
    >>> from transformers import TapasModel, TapasConfig

    >>> # 初始化默认（SQA）TAPAS 配置
    >>> configuration = TapasConfig()
    >>> # 从配置初始化模型
    >>> model = TapasModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    
    # 模型类型
    model_type = "tapas"
    # 初始化函数，设置模型参数的默认数值
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制dropout概率，默认为0.1
        max_position_embeddings=1024,  # 最大位置编码长度，默认为1024
        type_vocab_sizes=[3, 256, 256, 2, 256, 256, 10],  # 类型词表大小列表，默认为指定值
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化epsilon值，默认为1e-12
        pad_token_id=0,  # 填充token的ID，默认为0
        positive_label_weight=10.0,  # 正例标签权重，默认为10.0
        num_aggregation_labels=0,  # 聚合标签数量，默认为0
        aggregation_loss_weight=1.0,  # 聚合损失权重，默认为1.0
        use_answer_as_supervision=None,  # 使用答案作为监督标签，默认为None
        answer_loss_importance=1.0,  # 答案损失重要性，默认为1.0
        use_normalized_answer_loss=False,  # 使用归一化答案损失，默认为False
        huber_loss_delta=None,  # Huber损失delta值，默认为None
        temperature=1.0,  # 温度参数，默认为1.0
        aggregation_temperature=1.0,  # 聚合温度参数，默认为1.0
        use_gumbel_for_cells=False,  # 是否用Gumbel分布生成单元，默认为False
        use_gumbel_for_aggregation=False,  # 是否用Gumbel分布生成聚合操作，默认为False
        average_approximation_function="ratio",  # 平均近似函数，默认为"ratio"
        cell_selection_preference=None,  # 单元选择偏好性，默认为None
        answer_loss_cutoff=None,  # 答案损失截断值，默认为None
        max_num_rows=64,  # 最大行数，默认为64
        max_num_columns=32,  # 最大列数，默认为32
        average_logits_per_cell=False,  # 每个单元的logits是否取平均，默认为False
        select_one_column=True,  # 是否只选择一列，默认为True
        allow_empty_column_selection=False,  # 是否允许空列选择，默认为False
        init_cell_selection_weights_to_zero=False,  # 是否初始化单元选择权重为零，默认为False
        reset_position_index_per_cell=True,  # 每个单元是否重置位置索引，默认为True
        disable_per_token_loss=False,  # 是否禁用每个token的损失计算，默认为False
        aggregation_labels=None,  # 聚合标签列表，默认为None
        no_aggregation_label_index=None,  # 无聚合标签索引，默认为None
        **kwargs,  # 其他关键字参数
    # 初始化父类并传递填充值和其他关键字参数
        ):
            super().__init__(pad_token_id=pad_token_id, **kwargs)
    
            # BERT 超参数定义（包含更新的最大位置嵌入和类型词汇表大小）
            # 定义词汇表的大小
            self.vocab_size = vocab_size
            # 定义隐藏层大小
            self.hidden_size = hidden_size
            # 定义隐藏层的数量
            self.num_hidden_layers = num_hidden_layers
            # 定义注意力头的数量
            self.num_attention_heads = num_attention_heads
            # 定义隐藏层的激活函数
            self.hidden_act = hidden_act
            # 定义中间层大小
            self.intermediate_size = intermediate_size
            # 定义隐藏层的 dropout 概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 定义注意力的 dropout 概率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 定义最大位置嵌入数
            self.max_position_embeddings = max_position_embeddings
            # 定义类型词汇表的大小
            self.type_vocab_sizes = type_vocab_sizes
            # 定义初始化范围
            self.initializer_range = initializer_range
            # 定义层归一化的公差
            self.layer_norm_eps = layer_norm_eps
    
            # 微调任务的超参数定义
            # 正例的权重
            self.positive_label_weight = positive_label_weight
            # 聚合标签的数量
            self.num_aggregation_labels = num_aggregation_labels
            # 聚合损失的权重
            self.aggregation_loss_weight = aggregation_loss_weight
            # 是否使用答案作为监督
            self.use_answer_as_supervision = use_answer_as_supervision
            # 答案损失的重要性
            self.answer_loss_importance = answer_loss_importance
            # 是否使用归一化的答案损失
            self.use_normalized_answer_loss = use_normalized_answer_loss
            # 定义 Huber 损失的 delta
            self.huber_loss_delta = huber_loss_delta
            # 定义温度参数
            self.temperature = temperature
            # 聚合的温度参数
            self.aggregation_temperature = aggregation_temperature
            # 是否使用 Gumbel-softmax 进行单元选择
            self.use_gumbel_for_cells = use_gumbel_for_cells
            # 是否使用 Gumbel-softmax 进行聚合
            self.use_gumbel_for_aggregation = use_gumbel_for_aggregation
            # 平均近似函数
            self.average_approximation_function = average_approximation_function
            # 单元选择偏好
            self.cell_selection_preference = cell_selection_preference
            # 答案损失的截断
            self.answer_loss_cutoff = answer_loss_cutoff
            # 最大行数
            self.max_num_rows = max_num_rows
            # 最大列数
            self.max_num_columns = max_num_columns
            # 每个单元格的平均 logits
            self.average_logits_per_cell = average_logits_per_cell
            # 是否仅选择一个列
            self.select_one_column = select_one_column
            # 是否允许选择空列
            self.allow_empty_column_selection = allow_empty_column_selection
            # 初始化单元格选择权重为零
            self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
            # 是否为每个单元格重置位置索引
            self.reset_position_index_per_cell = reset_position_index_per_cell
            # 是否禁用每个 token 的损失
            self.disable_per_token_loss = disable_per_token_loss
    
            # 聚合相关的超参数定义
            # 定义聚合标签
            self.aggregation_labels = aggregation_labels
            # 无聚合标签的索引
            self.no_aggregation_label_index = no_aggregation_label_index
    
            # 如果聚合标签是一个字典，则将键转换为整型
            if isinstance(self.aggregation_labels, dict):
                self.aggregation_labels = {int(k): v for k, v in aggregation_labels.items()}
```