# `.\transformers\models\sew\configuration_sew.py`

```py
# coding=utf-8
# 定义了文件编码格式为 UTF-8

# 引入 functools 和 operator 模块
import functools
import operator

# 从 transformers 包中引入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从 transformers 包中引入 logging 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 SEW 预训练模型及其配置文件的映射关系
SEW_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "asapp/sew-tiny-100k": "https://huggingface.co/asapp/sew-tiny-100k/resolve/main/config.json",
    # 在 https://huggingface.co/models?filter=sew 查看所有 SEW 模型
}

# SEWConfig 继承自 PretrainedConfig 类，用于存储 SEWModel 的配置信息
class SEWConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SEWModel`]. It is used to instantiate a SEW model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SEW
    [asapp/sew-tiny-100k](https://huggingface.co/asapp/sew-tiny-100k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import SEWConfig, SEWModel

    >>> # Initializing a SEW asapp/sew-tiny-100k style configuration
    >>> configuration = SEWConfig()

    >>> # Initializing a model (with random weights) from the asapp/sew-tiny-100k style configuration
    >>> model = SEWModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "sew"
    model_type = "sew"
    # 初始化编解码器模型的参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数量，默认为12
        num_attention_heads=12,  # 注意力头数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        squeeze_factor=2,  # 挤压因子，默认为2
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层丢弃率，默认为0.1
        activation_dropout=0.1,  # 激活函数丢弃率，默认为0.1
        attention_dropout=0.1,  # 注意力丢弃率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影丢弃率，默认为0.0
        final_dropout=0.1,  # 最终丢弃率，默认为0.1
        layerdrop=0.1,  # 层间丢弃率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层标准化阈值，默认为1e-5
        feat_extract_norm="group",  # 特征提取标准化方式，默认为group
        feat_extract_activation="gelu",  # 特征提取激活函数，默认为gelu
        conv_dim=(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512),  # 卷积层维度，默认为给定的值
        conv_stride=(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1),  # 卷积层步长，默认为给定的值
        conv_kernel=(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1),  # 卷积核大小，默认为给定的值
        conv_bias=False,  # 是否包含卷积层偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置编码数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置编码分组数量，默认为16
        apply_spec_augment=True,  # 是否应用特定数据增强技术，默认为True
        mask_time_prob=0.05,  # 时间掩码概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 时间掩码最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码最小数量，默认为0
        ctc_loss_reduction="mean",  # CTC损失降维方式，默认为mean
        ctc_zero_infinity=False,  # 是否将无穷值作为零处理，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        pad_token_id=0,  # 填充令牌ID，默认为0
        bos_token_id=1,  # 起始令牌ID，默认为1
        eos_token_id=2,  # 结束令牌ID，默认为2
        **kwargs,  # 其他参数
    # 调用父类初始化方法，并传入关键字参数
    super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
    # 设置隐藏层大小
    self.hidden_size = hidden_size
    # 特征提取的归一化方式
    self.feat_extract_norm = feat_extract_norm
    # 特征提取的激活函数
    self.feat_extract_activation = feat_extract_activation
    # 卷积层维度
    self.conv_dim = list(conv_dim)
    # 卷积层步幅
    self.conv_stride = list(conv_stride)
    # 卷积核大小
    self.conv_kernel = list(conv_kernel)
    # 是否包含卷积层偏置项
    self.conv_bias = conv_bias
    # 卷积位置编码的数量
    self.num_conv_pos_embeddings = num_conv_pos_embeddings
    # 卷积位置编码的分组数
    self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
    # 特征提取层数量
    self.num_feat_extract_layers = len(self.conv_dim)
    # 隐藏层数量
    self.num_hidden_layers = num_hidden_layers
    # 中间层大小
    self.intermediate_size = intermediate_size
    # 压缩因子
    self.squeeze_factor = squeeze_factor
    # 隐藏层激活函数
    self.hidden_act = hidden_act
    # 注意力头数量
    self.num_attention_heads = num_attention_heads
    # 隐藏层丢弃率
    self.hidden_dropout = hidden_dropout
    # 注意力丢弃率
    self.attention_dropout = attention_dropout
    # 激活函数丢弃率
    self.activation_dropout = activation_dropout
    # 特征投影丢弃率
    self.feat_proj_dropout = feat_proj_dropout
    # 最终输出丢弃率
    self.final_dropout = final_dropout
    # 层丢弃
    self.layerdrop = layerdrop
    # 层标准化的 epsilon
    self.layer_norm_eps = layer_norm_eps
    # 初始化范围
    self.initializer_range = initializer_range
    # 词汇表大小
    self.vocab_size = vocab_size

    # 检查卷积层配置是否正确
    if (
        (len(self.conv_stride) != self.num_feat_extract_layers)
        or (len(self.conv_kernel) != self.num_feat_extract_layers)
        or (len(self.conv_dim) != self.num_feat_extract_layers)
    ):
        raise ValueError(
            "Configuration for convolutional layers is incorrect. "
            "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`, "
            f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride) "
            f"= {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
        )

    # 为 SpecAugment 设置微调配置参数
    self.apply_spec_augment = apply_spec_augment
    self.mask_time_prob = mask_time_prob
    self.mask_time_length = mask_time_length
    self.mask_time_min_masks = mask_time_min_masks
    self.mask_feature_prob = mask_feature_prob
    self.mask_feature_length = mask_feature_length
    self.mask_feature_min_masks = mask_feature_min_masks

    # CTC 损失
    self.ctc_loss_reduction = ctc_loss_reduction
    self.ctc_zero_infinity = ctc_zero_infinity

    # 序列分类
    self.use_weighted_layer_sum = use_weighted_layer_sum
    self.classifier_proj_size = classifier_proj_size

@property
# 计算输入到logits比率
def inputs_to_logits_ratio(self):
    return functools.reduce(operator.mul, self.conv_stride, 1)
```