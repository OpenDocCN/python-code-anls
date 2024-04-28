# `.\transformers\models\lilt\configuration_lilt.py`

```
# 导入所需的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# LiLT 预训练模型配置文件的映射字典，将模型名称映射到对应的配置文件链接
LILT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "SCUT-DLVCLab/lilt-roberta-en-base": (
        "https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base/resolve/main/config.json"
    ),
}

# LiLT 配置类，用于存储 LiLT 模型的配置信息
class LiltConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LiltModel`]. It is used to instantiate a LiLT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LiLT
    [SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import LiltConfig, LiltModel

    >>> # Initializing a LiLT SCUT-DLVCLab/lilt-roberta-en-base style configuration
    >>> configuration = LiltConfig()
    >>> # Randomly initializing a model from the SCUT-DLVCLab/lilt-roberta-en-base style configuration
    >>> model = LiltModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    model_type = "lilt"  # 模型类型为 "lilt"

    # 初始化 LiLT 配置类
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小
        hidden_size=768,  # 隐藏层大小
        num_hidden_layers=12,  # 隐藏层层数
        num_attention_heads=12,  # 注意力头数
        intermediate_size=3072,  # 中间层大小
        hidden_act="gelu",  # 隐藏层激活函数
        hidden_dropout_prob=0.1,  # 隐藏层 dropout 概率
        attention_probs_dropout_prob=0.1,  # 注意力 dropout 概率
        max_position_embeddings=512,  # 最大位置嵌入
        type_vocab_size=2,  # 类型词汇表大小
        initializer_range=0.02,  # 初始化范围
        layer_norm_eps=1e-12,  # 层归一化 epsilon 值
        pad_token_id=0,  # 填充 token 的 id
        position_embedding_type="absolute",  # 位置嵌入类型
        classifier_dropout=None,  # 分类器 dropout
        channel_shrink_ratio=4,  # 通道压缩比
        max_2d_position_embeddings=1024,  # 最大二维位置嵌入
        **kwargs,  # 其他参数
    # 初始化函数
    
    
    def __init__(self,
                     pad_token_id=None,
                     vocab_size=30522,
                     hidden_size=768,
                     num_hidden_layers=12,
                     num_attention_heads=12,
                     hidden_act="gelu",
                     intermediate_size=3072,
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=2,
                     initializer_range=0.02,
                     layer_norm_eps=1e-12,
                     position_embedding_type='absolute',
                     classifier_dropout=0.1,
                     channel_shrink_ratio=2,
                     max_2d_position_embeddings=8192
                     ):
    
    
    ## 注释：
    
    
    # 初始化函数
    # 设置默认参数
    def __init__(self,
                     pad_token_id=None,  # padding的token id
                     vocab_size=30522,  # 词汇表的大小
                     hidden_size=768,  # 隐藏层大小
                     num_hidden_layers=12,  # 隐藏层的数量
                     num_attention_heads=12,  # attention头的数量
                     hidden_act="gelu",  # 隐藏层激活函数的名称
                     intermediate_size=3072,  # 中间层大小
                     hidden_dropout_prob=0.1,  # 隐藏层dropout的概率
                     attention_probs_dropout_prob=0.1,  # attention层dropout的概率
                     max_position_embeddings=512,  # 最大位置嵌入的数量
                     type_vocab_size=2,  # 类型词汇表的大小
                     initializer_range=0.02,  # 初始化范围
                     layer_norm_eps=1e-12,  # LayerNorm层的epsilon值
                     position_embedding_type='absolute',  # 位置嵌入的类型
                     classifier_dropout=0.1,  # 分类器dropout的概率
                     channel_shrink_ratio=2,  # 通道收缩比例
                     max_2d_position_embeddings=8192  # 最大2D位置嵌入的数量
                     ):
```