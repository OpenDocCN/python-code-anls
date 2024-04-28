# `.\transformers\models\unispeech\configuration_unispeech.py`

```
# 设定字符编码格式为utf-8
# 版权声明和许可证信息
# 引入必要的库和模块
import functools
import operator

# 从模块中引入所需的配置对象类和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 UniSpeech 模型的预训练配置文件映射
UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/unispeech-large-1500h-cv": (
        "https://huggingface.co/microsoft/unispeech-large-1500h-cv/resolve/main/config.json"
    ),
    # 查看所有 UniSpeech 模型的链接地址
}

# 定义 UniSpeechConfig 类，用于存储 UniSpeech 模型的配置信息
class UniSpeechConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UniSpeechModel`]. It is used to instantiate an
    UniSpeech model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the UniSpeech
    [microsoft/unispeech-large-1500h-cv](https://huggingface.co/microsoft/unispeech-large-1500h-cv) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import UniSpeechConfig, UniSpeechModel

    >>> # Initializing a UniSpeech facebook/unispeech-base-960h style configuration
    >>> configuration = UniSpeechConfig()

    >>> # Initializing a model (with random weights) from the facebook/unispeech-base-960h style configuration
    >>> model = UniSpeechModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 指定模型类型为'unispeech'
    model_type = "unispeech"
    # 初始化函数，设置模型的各种参数，可以通过参数传入或者默认数值
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层的dropout比例，默认为0.1
        activation_dropout=0.1,  # 激活函数的dropout比例，默认为0.1
        attention_dropout=0.1,  # 注意力的dropout比例，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影的dropout比例，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化的dropout比例，默认为0.0
        final_dropout=0.1,  # 最终的dropout比例，默认为0.1
        layerdrop=0.1,  # 层级dropout比例，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取的归一化方式，默认为group
        feat_extract_activation="gelu",  # 特征提取的激活函数，默认为gelu
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数量，默认为16
        do_stable_layer_norm=False,  # 是否进行稳定层归一化，默认为False
        apply_spec_augment=True,  # 是否应用频谱增强，默认为True
        mask_time_prob=0.05,  # 时间屏蔽的概率，默认为0.05
        mask_time_length=10,  # 时间屏蔽的长度，默认为10
        mask_time_min_masks=2,  # 时间屏蔽的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征屏蔽的概率，默认为0.0
        mask_feature_length=10,  # 特征屏蔽的长度，默认为10
        mask_feature_min_masks=0,  # 特征屏蔽的最小数量，默认为0
        num_codevectors_per_group=320,  # 每个分组的编码向量数量，默认为320
        num_codevector_groups=2,  # 编码向量分组数量，默认为2
        contrastive_logits_temperature=0.1,  # 对数温度，默认为0.1
        num_negatives=100,  # 负样本数量，默认为100
        codevector_dim=256,  # 编码向量维���，默认为256
        proj_codevector_dim=256,  # 投影编码向量维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失权重，默认为0.1
        ctc_loss_reduction="mean",  # CTC损失的缩减方式，默认为"mean"
        ctc_zero_infinity=False,  # CTC是否将无穷视为零，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        num_ctc_classes=80,  # CTC类别数量，默认为80
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 起始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        replace_prob=0.5,  # 替换概率，默认为0.5
        **kwargs,  # 其他参数
    @property
    # 计算输入与logits的比例
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)  # 返回卷积步长的乘积
```