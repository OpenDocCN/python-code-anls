# `.\models\cpmant\configuration_cpmant.py`

```
# 设置编码为utf-8
# 版权声明
# 根据Apache许可证2.0版许可
# 只能在符合许可证的情况下使用此文件
# 可以在以下网址获取许可证
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非根据适用法律或书面同意所要求，否则将根据“原样”基础分发软件
# 没有任何形式的保证或条件，无论是明示还是暗示的
# 参见许可证以获取有关权限和
# 许可下的限制的具体语言。
"""CPMAnt模型配置"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# CPMAnt预训练配置映射
CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openbmb/cpm-ant-10b": "https://huggingface.co/openbmb/cpm-ant-10b/blob/main/config.json"
    # 查看所有CPMAnt模型，请访问https://huggingface.co/models?filter=cpmant
}


class CpmAntConfig(PretrainedConfig):
    r"""
    这是用于存储[`CpmAntModel`]配置的配置类。它用于根据指定的参数实例化CPMAnt模型，定义模型架构。使用默认值实例化配置将产生与CPMAnt [openbmb/cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b)架构类似的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档以获取更多信息。
    # 定义函数参数
    Args:
        vocab_size (`int`, *optional*, defaults to 30720):
            CPMAnt 模型的词汇大小。定义了在调用 [`CpmAntModel`] 时传递的`input`中可以表示的不同标记的数量。
        hidden_size (`int`, *optional*, defaults to 4096):
            编码器层的维度。
        num_attention_heads (`int`, *optional*, defaults to 32):
            Transformer 编码器中的注意力头数。
        dim_head (`int`, *optional*, defaults to 128):
            Transformer 编码器中每个注意力层的注意力头的维度。
        dim_ff (`int`, *optional*, defaults to 10240):
            Transformer 编码器中“中间”（即前馈）层的维度。
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Transformer 编码器的层数。
        dropout_p (`float`, *optional*, defaults to 0.0):
            用于嵌入、编码器中所有全连接层的 dropout 概率。
        position_bias_num_buckets (`int`, *optional*, defaults to 512):
            position_bias buckets 的数量。
        position_bias_max_distance (`int`, *optional*, defaults to 2048):
            该模型可能被使用的最大序列长度。通常设置为较大的值（例如，512、1024或2048）以防万一。
        eps (`float`, *optional*, defaults to 1e-06):
            层归一化层使用的 epsilon。
        init_std (`float`, *optional*, defaults to 1.0):
            使用 std = init_std 初始化参数。
        prompt_types (`int`, *optional*, defaults to 32):
            prompt 的类型。
        prompt_length (`int`, *optional*, defaults to 32):
            prompt 的长度。
        segment_types (`int`, *optional*, defaults to 32):
            segment 的类型。
        use_cache (`bool`, *optional*, defaults to `True`):
            是否使用缓存。

    Example:

    ```python
    >>> from transformers import CpmAntModel, CpmAntConfig

    >>> # Initializing a CPMAnt cpm-ant-10b style configuration
    >>> configuration = CpmAntConfig()

    >>> # Initializing a model from the cpm-ant-10b style configuration
    >>> model = CpmAntModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    # 设置模型类型
    model_type = "cpmant"
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_size: int = 30720,                         # 词汇表的大小，默认为30720
        hidden_size: int = 4096,                          # 隐藏层的大小，默认为4096
        num_attention_heads: int = 32,                    # 注意力头的数量，默认为32
        dim_head: int = 128,                              # 每个头的维度，默认为128
        dim_ff: int = 10240,                              # 前馈神经网络的维度，默认为10240
        num_hidden_layers: int = 48,                      # 隐藏层的数量，默认为48
        dropout_p: int = 0.0,                             # dropout的概率，默认为0.0
        position_bias_num_buckets: int = 512,             # 位置偏差的桶数，默认为512
        position_bias_max_distance: int = 2048,           # 最大位置偏差，默认为2048
        eps: int = 1e-6,                                  # 一个小的常量数值，默认为1e-6
        init_std: float = 1.0,                            # 初始化标准差，默认为1.0
        prompt_types: int = 32,                           # 提示类型的数量，默认为32
        prompt_length: int = 32,                          # 提示长度，默认为32
        segment_types: int = 32,                          # 分段类型的数量，默认为32
        use_cache: bool = True,                           # 是否使用缓存，默认为True
        **kwargs,                                         # 其他参数
    ):
        # 基类的初始化函数
        super().__init__(**kwargs)
        # 初始化对象的属性
        self.prompt_types = prompt_types                  # 提示类型的数量
        self.prompt_length = prompt_length                # 提示长度
        self.segment_types = segment_types                # 分段类型的数量
        self.hidden_size = hidden_size                    # 隐藏层的大小
        self.num_attention_heads = num_attention_heads    # 注意力头的数量
        self.dim_head = dim_head                          # 每个头的维度
        self.dim_ff = dim_ff                              # 前馈神经网络的维度
        self.num_hidden_layers = num_hidden_layers        # 隐藏层的数量
        self.position_bias_num_buckets = position_bias_num_buckets  # 位置偏差的桶数
        self.position_bias_max_distance = position_bias_max_distance  # 最大位置偏差
        self.dropout_p = dropout_p                        # dropout的概率
        self.eps = eps                                    # 一个小的常量数值
        self.use_cache = use_cache                        # 是否使用缓存
        self.vocab_size = vocab_size                      # 词汇表的大小
        self.init_std = init_std                          # 初始化标准差
```