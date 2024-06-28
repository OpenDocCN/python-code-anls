# `.\models\hubert\configuration_hubert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明此代码版权归 Fairseq 作者和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；除非符合许可证条件，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律另有规定或书面同意，否则按“原样”分发软件
# 无论是明示还是暗示的条件，都没有担保或条件，包括但不限于
# 适销性或特定用途适用性的保证。详细信息请参阅许可证。
""" Hubert model configuration"""

# 导入 functools 和 operator 模块
import functools
import operator

# 从 configuration_utils.py 中导入 PretrainedConfig 类
# 从 utils.py 中导入 logging 模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger 实例，用于记录日志信息
logger = logging.get_logger(__name__)

# Hubert 预训练配置文件的映射表，将模型名称映射到其配置文件的 URL
HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/hubert-base-ls960": "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json",
    # 查看所有 Hubert 模型的列表：https://huggingface.co/models?filter=hubert
}


class HubertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`HubertModel`] 配置的类。用于根据指定的参数实例化 Hubert 模型，
    定义模型架构。使用默认值实例化配置将产生类似于 Hubert
    [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请阅读
    [`PretrainedConfig`] 的文档。

    Example:

    ```python
    >>> from transformers import HubertModel, HubertConfig

    >>> # 初始化一个 Hubert facebook/hubert-base-ls960 风格的配置
    >>> configuration = HubertConfig()

    >>> # 使用配置初始化一个模型，其模型风格为 facebook/hubert-base-ls960
    >>> model = HubertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 指定模型类型为 "hubert"
    model_type = "hubert"
    # 定义一个初始化方法，用于创建一个 Transformer 模型的实例
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer 模型中的隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # Transformer 中间层的大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_dropout=0.1,  # 隐藏层的 dropout 概率，默认为0.1
        activation_dropout=0.1,  # 激活函数的 dropout 概率，默认为0.1
        attention_dropout=0.1,  # 注意力机制的 dropout 概率，默认为0.1
        feat_proj_layer_norm=True,  # 是否对特征投影进行层归一化，默认为True
        feat_proj_dropout=0.0,  # 特征投影的 dropout 概率，默认为0.0
        final_dropout=0.1,  # 最终输出层的 dropout 概率，默认为0.1
        layerdrop=0.1,  # LayerDrop 的概率，默认为0.1
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon 值，默认为1e-5
        feat_extract_norm="group",  # 特征提取的归一化方式，默认为"group"
        feat_extract_activation="gelu",  # 特征提取的激活函数，默认为 GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积层的卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积层的偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数量，默认为16
        do_stable_layer_norm=False,  # 是否进行稳定层归一化，默认为False
        apply_spec_augment=True,  # 是否应用音频增强技术，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小数量，默认为0
        ctc_loss_reduction="sum",  # CTC 损失的归并方式，默认为"sum"
        ctc_zero_infinity=False,  # 是否将无穷值视为零，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影的大小，默认为256
        pad_token_id=0,  # 填充标记的 ID，默认为0
        bos_token_id=1,  # 起始标记的 ID，默认为1
        eos_token_id=2,  # 结束标记的 ID，默认为2
        **kwargs,  # 其他未指定的参数
    ):
        ):
        # 调用父类的初始化方法，并传递额外的关键字参数
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 特征提取层的归一化方法
        self.feat_extract_norm = feat_extract_norm
        # 特征提取层的激活函数
        self.feat_extract_activation = feat_extract_activation
        # 卷积层的维度列表
        self.conv_dim = list(conv_dim)
        # 卷积层的步长列表
        self.conv_stride = list(conv_stride)
        # 卷积核大小的列表
        self.conv_kernel = list(conv_kernel)
        # 是否包含卷积层的偏置
        self.conv_bias = conv_bias
        # 卷积位置嵌入的数量
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        # 卷积位置嵌入的组数
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        # 特征提取层的数量
        self.num_feat_extract_layers = len(self.conv_dim)
        # 隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 中间层的大小
        self.intermediate_size = intermediate_size
        # 隐藏层的激活函数
        self.hidden_act = hidden_act
        # 注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 隐藏层的丢弃率
        self.hidden_dropout = hidden_dropout
        # 注意力丢弃率
        self.attention_dropout = attention_dropout
        # 激活函数的丢弃率
        self.activation_dropout = activation_dropout
        # 特征投影层的层归一化
        self.feat_proj_layer_norm = feat_proj_layer_norm
        # 特征投影层的丢弃率
        self.feat_proj_dropout = feat_proj_dropout
        # 最终输出的丢弃率
        self.final_dropout = final_dropout
        # 层丢弃（LayerDrop）的比例
        self.layerdrop = layerdrop
        # 层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 初始化器的范围
        self.initializer_range = initializer_range
        # 词汇表大小
        self.vocab_size = vocab_size
        # 是否使用稳定层归一化
        self.do_stable_layer_norm = do_stable_layer_norm
        # 是否使用加权层求和
        self.use_weighted_layer_sum = use_weighted_layer_sum
        # 分类器投影层的大小
        self.classifier_proj_size = classifier_proj_size

        # 检查卷积层配置的有效性
        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            # 如果卷积层配置不正确，则抛出值错误异常
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # 针对 SpecAugment 进行微调配置参数，参考论文 https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        # 时间遮蔽的概率
        self.mask_time_prob = mask_time_prob
        # 时间遮蔽的长度
        self.mask_time_length = mask_time_length
        # 时间遮蔽的最小遮蔽数量
        self.mask_time_min_masks = mask_time_min_masks
        # 特征遮蔽的概率
        self.mask_feature_prob = mask_feature_prob
        # 特征遮蔽的长度
        self.mask_feature_length = mask_feature_length
        # 特征遮蔽的最小遮蔽数量
        self.mask_feature_min_masks = mask_feature_min_masks

        # CTC（Connectionist Temporal Classification）损失函数的配置
        self.ctc_loss_reduction = ctc_loss_reduction
        # CTC 损失函数中的零无穷值
        self.ctc_zero_infinity = ctc_zero_infinity

    @property
    # 计算输入到 logits 的比例
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)
```