# `.\models\pegasus\configuration_pegasus.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指出版权所有者和版权年份
# 根据 Apache 许可证 2.0 版本使用该文件
# 在符合许可证条件下，您可以使用该文件；如果不符合，则不允许使用
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“现状”分发此软件
# 无论是明示的还是隐含的，都没有任何形式的保证或条件
# 有关更多信息，请参阅许可证文档
""" PEGASUS model configuration"""

# 从相对路径导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# PEGASUS 预训练模型的配置文件映射，指定模型名称及其对应的配置 JSON 文件 URL
PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pegasus-large": "https://huggingface.co/google/pegasus-large/resolve/main/config.json",
    # 查看所有 PEGASUS 模型，请访问 https://huggingface.co/models?filter=pegasus
}

# PegasusConfig 类，用于存储 PEGASUS 模型的配置信息，继承自 PretrainedConfig
class PegasusConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PegasusModel`]. It is used to instantiate an
    PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PEGASUS
    [google/pegasus-large](https://huggingface.co/google/pegasus-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import PegasusConfig, PegasusModel

    >>> # Initializing a PEGASUS google/pegasus-large style configuration
    >>> configuration = PegasusConfig()

    >>> # Initializing a model (with random weights) from the google/pegasus-large style configuration
    >>> model = PegasusModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 指定模型类型为 PEGASUS
    model_type = "pegasus"
    # 在推理阶段要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，将一些配置项名映射到模型参数中的实际名称
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    #python
    # 初始化函数，用于创建一个新的Transformer模型实例
    def __init__(
        self,
        vocab_size=50265,                            # 词汇表大小，默认为50265
        max_position_embeddings=1024,                # 最大位置编码数，默认为1024
        encoder_layers=12,                           # 编码器层数，默认为12层
        encoder_ffn_dim=4096,                        # 编码器中FeedForward层的维度，默认为4096
        encoder_attention_heads=16,                  # 编码器中注意力头的数量，默认为16个
        decoder_layers=12,                           # 解码器层数，默认为12层
        decoder_ffn_dim=4096,                        # 解码器中FeedForward层的维度，默认为4096
        decoder_attention_heads=16,                  # 解码器中注意力头的数量，默认为16个
        encoder_layerdrop=0.0,                       # 编码器中层Dropout的比例，默认为0.0（无Dropout）
        decoder_layerdrop=0.0,                       # 解码器中层Dropout的比例，默认为0.0（无Dropout）
        use_cache=True,                              # 是否使用缓存，默认为True
        is_encoder_decoder=True,                     # 是否是编码-解码结构，默认为True
        activation_function="gelu",                  # 激活函数类型，默认为"GELU"
        d_model=1024,                                # 模型的维度，默认为1024
        dropout=0.1,                                 # 全局Dropout的比例，默认为0.1
        attention_dropout=0.0,                       # 注意力模块中Dropout的比例，默认为0.0（无Dropout）
        activation_dropout=0.0,                      # 激活函数Dropout的比例，默认为0.0（无Dropout）
        init_std=0.02,                               # 参数初始化的标准差，默认为0.02
        decoder_start_token_id=0,                    # 解码器起始token的ID，默认为0
        scale_embedding=False,                       # 是否对嵌入进行缩放，默认为False
        pad_token_id=0,                              # 填充token的ID，默认为0
        eos_token_id=1,                              # 结束token的ID，默认为1
        forced_eos_token_id=1,                       # 强制结束token的ID，默认为1
        **kwargs,                                    # 其他关键字参数，传递给父类初始化函数
    ):
        self.vocab_size = vocab_size                  # 设置词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 设置最大位置编码数
        self.d_model = d_model                        # 设置模型维度
        self.encoder_ffn_dim = encoder_ffn_dim        # 设置编码器中FeedForward层的维度
        self.encoder_layers = encoder_layers          # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 设置编码器中注意力头的数量
        self.decoder_ffn_dim = decoder_ffn_dim        # 设置解码器中FeedForward层的维度
        self.decoder_layers = decoder_layers          # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器中注意力头的数量
        self.dropout = dropout                        # 设置全局Dropout的比例
        self.attention_dropout = attention_dropout    # 设置注意力模块中Dropout的比例
        self.activation_dropout = activation_dropout  # 设置激活函数Dropout的比例
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std                      # 设置参数初始化的标准差
        self.encoder_layerdrop = encoder_layerdrop    # 设置编码器中层Dropout的比例
        self.decoder_layerdrop = decoder_layerdrop    # 设置解码器中层Dropout的比例
        self.use_cache = use_cache                    # 设置是否使用缓存
        self.num_hidden_layers = encoder_layers       # 设置隐藏层的数量为编码器层数
        self.scale_embedding = scale_embedding        # 设置是否对嵌入进行缩放，如果是则缩放因子为sqrt(d_model)
        super().__init__(                              # 调用父类初始化函数，传递参数给父类
            pad_token_id=pad_token_id,                # 设置填充token的ID
            eos_token_id=eos_token_id,                # 设置结束token的ID
            is_encoder_decoder=is_encoder_decoder,    # 设置是否是编码-解码结构
            decoder_start_token_id=decoder_start_token_id,  # 设置解码器起始token的ID
            forced_eos_token_id=forced_eos_token_id,  # 设置强制结束token的ID
            **kwargs,                                # 传递其他关键字参数给父类初始化函数
        )

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads         # 返回编码器中注意力头的数量

    @property
    def hidden_size(self) -> int:
        return self.d_model                         # 返回模型的维度
```