# `.\models\data2vec\configuration_data2vec_text.py`

```py
# 设置代码文件的编码格式为 UTF-8
# 版权声明，保留所有权利
#
# 根据 Apache 许可证，你只能在遵守许可证规定的情况下使用该文件
# 你可以从以下网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除了适用法律要求或书面同意的情况下，根据许可证分发的软件将按照“原样”分发，
# 不附带任何明示或暗示的担保或条件。请参阅许可证了解特定的语言管理权限以及限制。
""" Data2VecText 配置"""
# 导入所需的模块
from collections import OrderedDict
from typing import Mapping
# 从上级目录中导入 configuration_utils.py 中的 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从上级目录中导入 onnx.py 中的 OnnxConfig 类
from ...onnx import OnnxConfig
# 从上级目录中导入 logging.py 中的 logging 模块
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练配置文件映射
DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/data2vec-text-base": "https://huggingface.co/data2vec/resolve/main/config.json",
}

# Data2VecTextConfig 继承自 PretrainedConfig 类
class Data2VecTextConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Data2VecTextModel`] 和 [`Data2VecTextModel`] 配置的配置类。它用于根据指定的参数实例化
    一个 Data2VecText 模型，定义了模型的架构。使用默认值实例化一个配置将产生类似于 Data2VecText
    [facebook/data2vec-text-base](https://huggingface.co/facebook/data2vec-text-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。请阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import Data2VecTextConfig, Data2VecTextModel
    >>> # 初始化 Data2VecText facebook/data2vec-text-base 风格的配置
    >>> configuration = Data2VecTextConfig()
    >>> # 从 facebook/data2vec-text-base 风格的配置初始化一个模型（带有随机权重）
    >>> model = Data2VecTextModel(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "data2vec-text"

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
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
        # 调用父类的构造函数，传入参数 pad_token_id、bos_token_id、eos_token_id 和 **kwargs
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化 Transformer 的各个参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
```  
# 定义一个数据文本转换为ONNX配置的类，继承自OnnxConfig类
class Data2VecTextOnnxConfig(OnnxConfig):
    # 输入属性的装饰器，返回值为输入映射的字典，键为字符串，值为映射整数和字符串的字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选题，则动态轴为{0:"batch", 1:"choice", 2:"sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        # 否则动态轴为{0:"batch", 1:"sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回有序字典
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),  # 输入IDs，使用动态轴进行映射
                ("attention_mask", dynamic_axis),  # 注意力掩码，使用动态轴进行映射
            ]
        )
```