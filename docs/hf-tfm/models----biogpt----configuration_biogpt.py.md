# `.\models\biogpt\configuration_biogpt.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science All rights reserved.
# 上面是版权声明和编码声明

# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义一个字典，映射 BioGPT 预训练模型名称到配置文件的 URL
BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/config.json",
    # 可以在 https://huggingface.co/models?filter=biogpt 查看所有的 BioGPT 模型
}

# 定义 BioGptConfig 类，继承自 PretrainedConfig 类
class BioGptConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BioGptModel`]. It is used to instantiate an
    BioGPT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the BioGPT
    [microsoft/biogpt](https://huggingface.co/microsoft/biogpt) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义了 BioGPT 模型的配置类，包含了模型的各种参数设置
    
    class BioGptConfig:
        # 构造函数，初始化配置参数
        def __init__(
            self,
            # 词汇表大小，默认为 42384，定义了模型可以表示的不同 token 的数量
            vocab_size=42384,
            # 隐藏层大小，默认为 1024，定义了编码器层和池化层的维度
            hidden_size=1024,
            # Transformer 编码器中的隐藏层数，默认为 24
            num_hidden_layers=24,
            # Transformer 编码器中每个注意力层的注意力头数，默认为 16
            num_attention_heads=16,
            # Transformer 编码器中“中间”（即前馈）层的维度，默认为 4096
            intermediate_size=4096,
            # 编码器和池化器中的非线性激活函数，默认为 "gelu"
            hidden_act="gelu",
            # 嵌入层、编码器和池化器中所有全连接层的 dropout 概率，默认为 0.1
            hidden_dropout_prob=0.1,
            # 注意力概率的 dropout 比率，默认为 0.1
            attention_probs_dropout_prob=0.1,
            # 可能会用到的最大序列长度，通常设置为较大值（例如 512、1024 或 2048）
            max_position_embeddings=1024,
            # 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为 0.02
            initializer_range=0.02,
            # 层归一化层使用的 epsilon，默认为 1e-12
            layer_norm_eps=1e-12,
            # 是否通过将嵌入除以 sqrt(d_model) 来缩放嵌入，默认为 True
            scale_embedding=True,
            # 模型是否应返回最后一组键/值注意力，默认为 True，仅在 config.is_decoder=True 时相关
            use_cache=True,
            # LayerDrop 参数，请参考论文 https://arxiv.org/abs/1909.11556 获取详细信息，默认为 0.0
            layerdrop=0.0,
            # 全连接层内部激活的 dropout 比率，默认为 0.0
            activation_dropout=0.0,
            # 填充标记的 id，默认为 1
            pad_token_id=1,
            # 流的起始标记 id，默认为 0
            bos_token_id=0,
            # 流的结束标记 id，默认为 2
            eos_token_id=2,
        ):
            # 将各个参数赋值给对应的实例变量
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.scale_embedding = scale_embedding
            self.use_cache = use_cache
            self.layerdrop = layerdrop
            self.activation_dropout = activation_dropout
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
    # Initializing a BioGPT configuration object with default parameters
    configuration = BioGptConfig()
    
    # Initializing a BioGPT model using the configuration object created above
    model = BioGptModel(configuration)
    
    # Accessing the configuration attributes from the model
    configuration = model.config
```