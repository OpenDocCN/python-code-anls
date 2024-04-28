# `.\models\funnel\configuration_funnel.py`

```
# 设置文件编码为 utf-8
# 版权声明和许可证信息
# 声明配置 Funnel Transformer 模型的类

# 导入父类 PretrainedConfig 和 logging 工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射表
FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/config.json",
    "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/config.json",
    "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/config.json",
    "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/config.json",
    "funnel-transformer/intermediate": (
        "https://huggingface.co/funnel-transformer/intermediate/resolve/main/config.json"
    ),
    "funnel-transformer/intermediate-base": (
        "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/config.json"
    ),
    "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/config.json",
    "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/config.json",
    "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/config.json",
    "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/config.json",
}

# 定义 FunnelConfig 类，继承自 PretrainedConfig 类
class FunnelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FunnelModel`] or a [`TFBertModel`]. It is used to
    instantiate a Funnel Transformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Funnel
    Transformer [funnel-transformer/small](https://huggingface.co/funnel-transformer/small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """
    # 模型类型
    model_type = "funnel"
    # 属性映射
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
    }
    # 初始化方法，用于初始化模型参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        block_sizes=[4, 4, 4],  # Transformer 模块的块大小，默认为[4, 4, 4]
        block_repeats=None,  # Transformer 模块的块重复次数，默认为None
        num_decoder_layers=2,  # 解码器层数，默认为2
        d_model=768,  # 模型维度，默认为768
        n_head=12,  # 注意力头数，默认为12
        d_head=64,  # 每个头的维度，默认为64
        d_inner=3072,  # FeedForward 层内部维度，默认为3072
        hidden_act="gelu_new",  # 激活函数，默认为 "gelu_new"
        hidden_dropout=0.1,  # 隐藏层的 dropout，默认为0.1
        attention_dropout=0.1,  # 注意力层的 dropout，默认为0.1
        activation_dropout=0.0,  # 激活函数的 dropout，默认为0.0
        initializer_range=0.1,  # 初始化范围，默认为0.1
        initializer_std=None,  # 初始化标准差，默认为None
        layer_norm_eps=1e-9,  # LayerNormalization 的 epsilon，默认为1e-9
        pooling_type="mean",  # 池化类型，默认为 "mean"
        attention_type="relative_shift",  # 注意力类型，默认为 "relative_shift"
        separate_cls=True,  # 是否分离CLS，默认为True
        truncate_seq=True,  # 是否截断序列，默认为True
        pool_q_only=True,  # 是否只对查询进行池化，默认为True
        **kwargs,  # 其他参数
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.block_sizes = block_sizes  # 初始化块大小
        self.block_repeats = [1] * len(block_sizes) if block_repeats is None else block_repeats  # 初始化块重复次数
        assert len(block_sizes) == len(
            self.block_repeats
        ), "`block_sizes` and `block_repeats` should have the same length."  # 断言，确保块大小和块重复次数长度一致
        self.num_decoder_layers = num_decoder_layers  # 初始化解码器层数
        self.d_model = d_model  # 初始化模型维度
        self.n_head = n_head  # 初始化注意力头数
        self.d_head = d_head  # 初始化每个头的维度
        self.d_inner = d_inner  # 初始化FeedForward 层内部维度
        self.hidden_act = hidden_act  # 初始化激活函数
        self.hidden_dropout = hidden_dropout  # 初始化隐藏层的 dropout
        self.attention_dropout = attention_dropout  # 初始化注意力层的 dropout
        self.activation_dropout = activation_dropout  # 初始化激活函数的 dropout
        self.initializer_range = initializer_range  # 初始化范围
        self.initializer_std = initializer_std  # 初始化标准差
        self.layer_norm_eps = layer_norm_eps  # 初始化LayerNormalization 的 epsilon
        assert pooling_type in [
            "mean",
            "max",
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."  # 断言，确保池化类型支持
        self.pooling_type = pooling_type  # 初始化池化类型
        assert attention_type in [
            "relative_shift",
            "factorized",
        ], f"Got {attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."  # 断言，确保注意力类型支持
        self.attention_type = attention_type  # 初始化注意力类型
        self.separate_cls = separate_cls  # 初始化是否分离CLS
        self.truncate_seq = truncate_seq  # 初始化是否截断序列
        self.pool_q_only = pool_q_only  # 初始化是否只对查询进行池化

        super().__init__(**kwargs)  # 调用父类的初始化方法

    # 获取隐藏层的总数
    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    # 设置隐藏层的总数，但此模型不支持设置隐藏层的总数
    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `block_sizes`."
        )

    # 获取块的总数
    @property
    def num_blocks(self):
        return len(self.block_sizes)

    # 设置块的总数，但此模型不支持设置块的总数
    @num_blocks.setter
    def num_blocks(self, value):
        raise NotImplementedError("This model does not support the setting of `num_blocks`. Please set `block_sizes`.")
```