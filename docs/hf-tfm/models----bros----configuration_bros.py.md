# `.\transformers\models\bros\configuration_bros.py`

```
# coding=utf-8
# 版权声明
# 本文件使用 Apache 许可证 2.0 授权
# 你可以在遵守许可证的前提下使用本文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 本文件分发的内容基于“原样”提供，不提供任何明示或暗示的保证或条件
# 包括但不限于适销性、特定用途适用性和非侵权性的保证
# 请查阅许可证了解具体的条款和条件

""" Bros 模型配置"""

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射
BROS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "jinho8345/bros-base-uncased": "https://huggingface.co/jinho8345/bros-base-uncased/blob/main/config.json",
    "jinho8345/bros-large-uncased": "https://huggingface.co/jinho8345/bros-large-uncased/blob/main/config.json",
}


class BrosConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`BrosModel`] 或 [`TFBrosModel`] 的配置。它用于根据指定的参数实例化 Bros 模型，
    定义模型架构。使用默认参数实例化配置将产生与 Bros
    [jinho8345/bros-base-uncased](https://huggingface.co/jinho8345/bros-base-uncased) 架构相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读
    [`PretrainedConfig`] 的文档以获取更多信息。
```  
    # Bros模型配置类，定义了Bros模型的各种参数设置
    class BrosConfig:
    
        # Bros模型的词汇表大小，定义了可以由调用BrosModel或TFBrosModel时传递的inputs_ids表示的不同标记数
        def __init__(
            self,
            vocab_size: int = 30522,
            # 编码器层和池化层的维度
            hidden_size: int = 768,
            # Transformer编码器中隐藏层的数量
            num_hidden_layers: int = 12,
            # Transformer编码器中每个注意力层的注意力头数
            num_attention_heads: int = 12,
            # Transformer编码器中“中间”（通常称为前馈）层的维度
            intermediate_size: int = 3072,
            # 编码器和池化层中的非线性激活函数
            hidden_act: Union[str, Callable] = "gelu",
            # 嵌入层、编码器和池化器中所有全连接层的dropout概率
            hidden_dropout_prob: float = 0.1,
            # 注意力概率的dropout比率
            attention_probs_dropout_prob: float = 0.1,
            # 此模型可能与之一起使用的最大序列长度
            max_position_embeddings: int = 512,
            # 调用BrosModel或TFBrosModel时传递的token_type_ids的词汇表大小
            type_vocab_size: int = 2,
            # 用于初始化所有权重矩阵的截断正态初始化器的标准差
            initializer_range: float = 0.02,
            # 层归一化层使用的epsilon
            layer_norm_eps: float = 1e-12,
            # token词汇表中填充标记的索引
            pad_token_id: int = 0,
            # 边界框坐标的维度（x0，y1，x1，y0，x1，y1，x0，y1）
            dim_bbox: int = 8,
            # 边界框坐标的比例因子
            bbox_scale: float = 100.0,
            # SpadeEE（实体提取）、SpadeEL（实体链接）头的关系数量
            n_relations: int = 1,
            # 分类器头的dropout比率
            classifier_dropout_prob: float = 0.1,
        ):
            pass
    
        # 初始化一个BROS jinho8345/bros-base-uncased风格的配置
        def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "PretrainedConfig":
            pass
    >>> configuration = BrosConfig()

    >>> # Initializing a model from the jinho8345/bros-base-uncased style configuration
    >>> model = BrosModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    # 设置模型类型为 "bros"
    model_type = "bros"

    # 定义 BrosConfig 类
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        dim_bbox=8,
        bbox_scale=100.0,
        n_relations=1,
        classifier_dropout_prob=0.1,
        **kwargs,
    ):
        # 调用父类的初始化方法，设置模型的基本参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        # 设置模型的额外参数
        self.dim_bbox = dim_bbox
        self.bbox_scale = bbox_scale
        self.n_relations = n_relations
        self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
        self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
        self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
        self.classifier_dropout_prob = classifier_dropout_prob
```