# `.\models\albert\configuration_albert.py`

```
# 引入 OrderedDict 用于有序字典，Mapping 用于类型提示
from collections import OrderedDict
from typing import Mapping

# 引入预训练配置工具类 PretrainedConfig 和 OnnxConfig
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig

# ALBERT 预训练模型配置文件映射字典，将模型名称映射到对应的配置文件 URL
ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "albert/albert-base-v1": "https://huggingface.co/albert/albert-base-v1/resolve/main/config.json",
    "albert/albert-large-v1": "https://huggingface.co/albert/albert-large-v1/resolve/main/config.json",
    "albert/albert-xlarge-v1": "https://huggingface.co/albert/albert-xlarge-v1/resolve/main/config.json",
    "albert/albert-xxlarge-v1": "https://huggingface.co/albert/albert-xxlarge-v1/resolve/main/config.json",
    "albert/albert-base-v2": "https://huggingface.co/albert/albert-base-v2/resolve/main/config.json",
    "albert/albert-large-v2": "https://huggingface.co/albert/albert-large-v2/resolve/main/config.json",
    "albert/albert-xlarge-v2": "https://huggingface.co/albert/albert-xlarge-v2/resolve/main/config.json",
    "albert/albert-xxlarge-v2": "https://huggingface.co/albert/albert-xxlarge-v2/resolve/main/config.json",
}

# AlbertConfig 类，继承自 PretrainedConfig，用于存储 ALBERT 模型的配置信息
class AlbertConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`AlbertModel`] 或 [`TFAlbertModel`] 的配置。根据指定的参数实例化一个 ALBERT 模型配置，
    定义模型的架构。使用默认参数实例化配置将得到与 ALBERT [albert/albert-xxlarge-v2] 相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    示例：

    ```python
    >>> from transformers import AlbertConfig, AlbertModel

    >>> # 初始化 ALBERT-xxlarge 风格的配置
    >>> albert_xxlarge_configuration = AlbertConfig()

    >>> # 初始化 ALBERT-base 风格的配置
    >>> albert_base_configuration = AlbertConfig(
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     intermediate_size=3072,
    ... )

    >>> # 使用 ALBERT-base 风格的配置初始化一个模型（带有随机权重）

    ```
    ```
    >>> model = AlbertModel(albert_xxlarge_configuration)
    
    
    # 创建一个 AlbertModel 的实例，使用给定的配置 albert_xxlarge_configuration
    model = AlbertModel(albert_xxlarge_configuration)
    
    
    
    >>> # Accessing the model configuration
    >>> configuration = model.config
    
    
    # 获取模型的配置信息并赋值给 configuration 变量
    configuration = model.config
    
    
    
    model_type = "albert"
    
    
    
    def __init__(
        self,
        vocab_size=30000,
        embedding_size=128,
        hidden_size=4096,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=64,
        intermediate_size=16384,
        inner_group_num=1,
        hidden_act="gelu_new",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout_prob=0.1,
        position_embedding_type="absolute",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        **kwargs,
    ):
        # 调用父类的构造函数，并传入相关的特殊 token id 参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    
        # 初始化 AlbertModel 的各种参数
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.position_embedding_type = position_embedding_type
# 从 transformers.models.bert.configuration_bert.BertOnnxConfig 复制并修改为 AlbertOnnxConfig 类，用于处理 Albert 模型的配置
class AlbertOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射，表示输入张量的动态轴
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设定动态轴的不同设置
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回一个有序字典，指定模型输入张量名称与对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 模型输入张量 input_ids 对应的动态轴
                ("attention_mask", dynamic_axis),  # 模型输入张量 attention_mask 对应的动态轴
                ("token_type_ids", dynamic_axis),  # 模型输入张量 token_type_ids 对应的动态轴
            ]
        )
```