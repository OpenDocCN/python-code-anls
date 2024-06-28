# `.\models\wav2vec2_conformer\configuration_wav2vec2_conformer.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，声明代码版权归 Fairseq 作者和 HuggingFace 团队所有，保留所有权利

# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0

# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不提供任何形式的担保或条件
# 请参阅许可证获取更多信息

""" Wav2Vec2Conformer 模型配置"""

# 导入 functools 和 operator 模块
import functools
import operator

# 从配置工具中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从工具模块中导入日志记录
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射字典，指定了模型名称和其对应的配置文件 URL
WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/wav2vec2-conformer-rel-pos-large": (
        "https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large/resolve/main/config.json"
    ),
}


class Wav2Vec2ConformerConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Wav2Vec2ConformerModel`] 配置的类。它用于根据指定的参数实例化一个 Wav2Vec2Conformer 模型，
    定义模型架构。使用默认值实例化配置将产生类似于 Wav2Vec2Conformer
    [facebook/wav2vec2-conformer-rel-pos-large](https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。更多信息请阅读 [`PretrainedConfig`] 的文档。

    示例：

    ```
    >>> from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel

    >>> # 初始化一个 Wav2Vec2Conformer facebook/wav2vec2-conformer-rel-pos-large 风格的配置
    >>> configuration = Wav2Vec2ConformerConfig()

    >>> # 从该配置初始化一个模型（具有随机权重），使用 facebook/wav2vec2-conformer-rel-pos-large 风格的配置
    >>> model = Wav2Vec2ConformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "wav2vec2-conformer"
    model_type = "wav2vec2-conformer"
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_size=None,  # 词汇表大小，默认为None
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer模型中隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # Feedforward层的中间维度大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout=0.1,  # 隐藏层的Dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数的Dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力层的Dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层的Dropout率，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化器的Dropout率，默认为0.0
        final_dropout=0.1,  # 最终输出层的Dropout率，默认为0.1
        layerdrop=0.1,  # 层级Dropout率，默认为0.1
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层级归一化的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取层的归一化方式，默认为"group"
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为GELU
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积层的卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积层的偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数量，默认为16
        apply_spec_augment=True,  # 是否应用频谱增强，默认为True
        mask_time_prob=0.05,  # 遮盖时间的概率，默认为0.05
        mask_time_length=10,  # 遮盖时间的长度，默认为10
        mask_time_min_masks=2,  # 遮盖时间的最小数量，默认为2
        mask_feature_prob=0.0,  # 遮盖特征的概率，默认为0.0
        mask_feature_length=10,  # 遮盖特征的长度，默认为10
        mask_feature_min_masks=0,  # 遮盖特征的最小数量，默认为0
        num_codevectors_per_group=320,  # 每组编码向量的数量，默认为320
        num_codevector_groups=2,  # 编码向量组的数量，默认为2
        contrastive_logits_temperature=0.1,  # 对比日志的温度参数，默认为0.1
        num_negatives=100,  # 负样本数量，默认为100
        codevector_dim=256,  # 编码向量的维度，默认为256
        proj_codevector_dim=256,  # 投影编码向量的维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失的权重，默认为0.1
        ctc_loss_reduction="sum",  # CTC损失的减少方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC损失中零是否为无穷，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层和，默认为False
        classifier_proj_size=256,  # 分类器投影的大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN层的维度，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN层的卷积核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN层的膨胀系数，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # X向量输出的维度，默认为512
        pad_token_id=0,  # 填充标记的ID，默认为0
        bos_token_id=1,  # 开始标记的ID，默认为1
        eos_token_id=2,  # 结束标记的ID，默认为2
        add_adapter=False,  # 是否添加适配器层，默认为False
        adapter_kernel_size=3,  # 适配器层的卷积核大小，默认为3
        adapter_stride=2,  # 适配器层的步长，默认为2
        num_adapter_layers=3,  # 适配器层的数量，默认为3
        output_hidden_size=None,  # 输出隐藏层的大小，默认为None
        position_embeddings_type="relative",  # 位置嵌入的类型，默认为"relative"
        rotary_embedding_base=10000,  # 旋转嵌入的基数，默认为10000
        max_source_positions=5000,  # 最大源位置，默认为5000
        conv_depthwise_kernel_size=31,  # 深度卷积核的大小，默认为31
        conformer_conv_dropout=0.1,  # Conformer模型的卷积Dropout率，默认为0.1
        **kwargs,  # 其他未命名的参数
    ):
        # 计算输入到logits比例的属性，即所有卷积步长的乘积
        @property
        def inputs_to_logits_ratio(self):
            return functools.reduce(operator.mul, self.conv_stride, 1)
```