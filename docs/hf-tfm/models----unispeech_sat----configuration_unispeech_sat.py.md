# `.\transformers\models\unispeech_sat\configuration_unispeech_sat.py`

```
# 引入必要的库和模块
# coding=utf-8
# 导入预训练模型的配置基类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# UniSpeechSat 预训练模型的配置文件路径映射
UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/unispeech-sat-base-100h-libri-ft": (
        "https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft/resolve/main/config.json"
    ),
    # 查看所有 UniSpeechSat 模型：https://huggingface.co/models?filter=unispeech_sat
}


# 定义 UniSpeechSat 的配置类，继承自预训练模型的配置基类 PretrainedConfig
class UniSpeechSatConfig(PretrainedConfig):
    r"""
    该类用于存储 [`UniSpeechSatModel`] 的配置。它被用于根据指定的参数实例化 UniSpeechSat 模型，定义模型的架构。
    使用默认配置实例化一个配置对象将得到与 UniSpeechSat
    [microsoft/unispeech-sat-base-100h-libri-ft](https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft)
    模型架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    示例:

    ```python
    >>> from transformers import UniSpeechSatModel, UniSpeechSatConfig

    >>> # 初始化一个 UniSpeechSat 配置，使用 microsoft/unispeech-sat-base-100h-libri-ft 风格的配置
    >>> configuration = UniSpeechSatConfig()

    >>> # 使用 UniSpeechSat 配置初始化一个模型
    >>> model = UniSpeechSatModel(configuration)

    >>> # 访问模型的配置
    >>> configuration = model.config
    ```"""

    # 模型类型为 "unispeech-sat"
    model_type = "unispeech-sat"
    # 初始化函数，用于创建一个新的对象实例，设置模型的各种参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层丢弃率，默认为0.1
        activation_dropout=0.1,  # 激活函数丢弃率，默认为0.1
        attention_dropout=0.1,  # 注意力丢弃率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影丢弃率，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化器丢弃率，默认为0.0
        final_dropout=0.1,  # 最终丢弃率，默认为0.1
        layerdrop=0.1,  # 层丢弃率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为1e-5
        feat_extract_norm="group",  # 特征提取的归一化方法，默认为"group"
        feat_extract_activation="gelu",  # 特征提取的激活函数，默认为"gelu"
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度，默认为一组(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步长，默认为一组(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为一组(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入组数量，默认为16
        do_stable_layer_norm=False,  # 是否进行稳定的层归一化，默认为False
        apply_spec_augment=True,  # 是否应用频谱增强，默认为True
        mask_time_prob=0.05,  # 时间掩码概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 最小时间掩码数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 最小特征掩码数量，默认为0
        num_codevectors_per_group=320,  # 每组代码向量数量，默认为320
        num_codevector_groups=2,  # 代码向量组数量，默认为2
        contrastive_logits_temperature=0.1,  # 对比日志温度，默认为0.1
        num_negatives=100,  # 负样本数量，默认为100
        codevector_dim=256,  # 代码向量维度，默认为256
        proj_codevector_dim=256,  # 投影代码向量维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失权重，默认为0.1
        ctc_loss_reduction="mean",  # CTC 损失的减少方式，默认为"mean"
        ctc_zero_infinity=False,  # CTC 零无穷，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN 维度，默认为一组(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN 卷积核大小，默认为一组(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN 膨胀率，默认为一组(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # xvector 输出维度，默认为512
        pad_token_id=0,  # 填充标记 ID，默认为0
        bos_token_id=1,  # 起始标记 ID，默认为1
        eos_token_id=2,  # 结束标记 ID，默认为2
        num_clusters=504,  # 聚类数量，默认为504
        **kwargs,  # 其它关键字参数
        @property
        def inputs_to_logits_ratio(self):  # 属性方法，计算输入到对数比
            return functools.reduce(operator.mul, self.conv_stride, 1)  # 计算输入到对数的比率，乘以每个卷积步长，默认为1
```