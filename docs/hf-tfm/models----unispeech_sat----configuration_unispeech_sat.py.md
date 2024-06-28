# `.\models\unispeech_sat\configuration_unispeech_sat.py`

```
# 定义 UniSpeechSatConfig 类，用于存储 UniSpeechSat 模型的配置信息
class UniSpeechSatConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`UniSpeechSatModel`] 的配置信息。根据指定的参数实例化 UniSpeechSat 模型，定义模型架构。
    使用默认配置实例化将产生与 UniSpeechSat [microsoft/unispeech-sat-base-100h-libri-ft](https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft)
    架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import UniSpeechSatModel, UniSpeechSatConfig

    >>> # 初始化一个 UniSpeechSat microsoft/unispeech-sat-base-100h-libri-ft 风格的配置
    >>> configuration = UniSpeechSatConfig()

    >>> # 从 microsoft/unispeech-sat-base-100h-libri-ft 风格的配置初始化一个模型
    >>> model = UniSpeechSatModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    # 初始化函数，用于创建一个新的对象实例，设置各种模型参数和配置
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层的dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数的dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力层的dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层的dropout率，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化器的dropout率，默认为0.0
        final_dropout=0.1,  # 最终输出层的dropout率，默认为0.1
        layerdrop=0.1,  # 层级丢弃率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取层的归一化方式，默认为"group"
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为gelu
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积层的维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积层的步幅，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积层的偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的分组数，默认为16
        do_stable_layer_norm=False,  # 是否进行稳定的层归一化，默认为False
        apply_spec_augment=True,  # 是否应用语音数据增强，默认为True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小数量，默认为0
        num_codevectors_per_group=320,  # 每组码向量的数量，默认为320
        num_codevector_groups=2,  # 码向量的组数，默认为2
        contrastive_logits_temperature=0.1,  # 对比损失的温度参数，默认为0.1
        num_negatives=100,  # 负样本数量，默认为100
        codevector_dim=256,  # 码向量的维度，默认为256
        proj_codevector_dim=256,  # 投影码向量的维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失的权重，默认为0.1
        ctc_loss_reduction="mean",  # CTC损失的减少方式，默认为"mean"
        ctc_zero_infinity=False,  # CTC损失是否将无穷值设为零，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN层的维度，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN层的卷积核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN层的膨胀率，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # X向量的输出维度，默认为512
        pad_token_id=0,  # 填充token的ID，默认为0
        bos_token_id=1,  # 起始token的ID，默认为1
        eos_token_id=2,  # 结束token的ID，默认为2
        num_clusters=504,  # 聚类中心的数量，默认为504
        **kwargs,  # 其他可选参数
    ):
        # 计算输入到logits的比率，即卷积步幅的乘积
        @property
        def inputs_to_logits_ratio(self):
            return functools.reduce(operator.mul, self.conv_stride, 1)
```