# `.\models\sew\configuration_sew.py`

```
# 设置编码为UTF-8，确保代码可以处理Unicode字符
# 版权声明，指明代码的版权归属和许可协议
# 导入 functools 和 operator 模块，用于后续操作的支持函数
# 从相对路径导入 PretrainedConfig 类
# 从相对路径导入 logging 工具，用于记录日志信息

# 获取 SEWConfig 类专用的日志记录器
logger = logging.get_logger(__name__)

# SEW_PRETRAINED_CONFIG_ARCHIVE_MAP 是一个字典，映射了 SEW 模型的预训练配置文件的名称到对应的 URL
SEW_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "asapp/sew-tiny-100k": "https://huggingface.co/asapp/sew-tiny-100k/resolve/main/config.json",
    # 可以在 https://huggingface.co/models?filter=sew 查看所有 SEW 模型的列表
}

# SEWConfig 类继承自 PretrainedConfig 类，用于存储 SEW 模型的配置信息
class SEWConfig(PretrainedConfig):
    r"""
    这是 SEWModel 的配置类，用于存储 SEW 模型的配置信息。根据指定的参数实例化配置对象，定义模型的架构。
    使用默认参数实例化配置对象将得到与 asapp/sew-tiny-100k 架构类似的配置。

    配置对象继承自 PretrainedConfig，可用于控制模型的输出。详细信息请参阅 PretrainedConfig 的文档。

    Example:

    ```python
    >>> from transformers import SEWConfig, SEWModel

    >>> # 初始化一个 SEW 类型的配置，例如 asapp/sew-tiny-100k
    >>> configuration = SEWConfig()

    >>> # 使用 SEW 类型的配置初始化一个模型（随机权重）
    >>> model = SEWModel(configuration)

    >>> # 访问模型的配置信息
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "sew"
    model_type = "sew"
    # 初始化函数，用于创建一个 Transformer 模型的实例
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # Transformer 中的隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层的大小，默认为3072
        squeeze_factor=2,  # 压缩因子，用于某些特征压缩，默认为2
        hidden_act="gelu",  # 隐藏层激活函数，默认为 GELU
        hidden_dropout=0.1,  # 隐藏层的 dropout 概率，默认为0.1
        activation_dropout=0.1,  # 激活函数的 dropout 概率，默认为0.1
        attention_dropout=0.1,  # 注意力层的 dropout 概率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影层的 dropout 概率，默认为0.0
        final_dropout=0.1,  # 最终输出的 dropout 概率，默认为0.1
        layerdrop=0.1,  # 层级 dropout 概率，默认为0.1
        initializer_range=0.02,  # 参数初始化的范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为1e-5
        feat_extract_norm="group",  # 特征提取层的归一化方式，默认为 group
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为 GELU
        conv_dim=(64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512),  # 卷积层的维度列表
        conv_stride=(5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1),  # 卷积层的步长列表
        conv_kernel=(10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1),  # 卷积层的卷积核大小列表
        conv_bias=False,  # 是否使用卷积层的偏置，默认为 False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入的数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入的组数，默认为16
        apply_spec_augment=True,  # 是否应用特定的数据增强，默认为 True
        mask_time_prob=0.05,  # 时间掩码的概率，默认为0.05
        mask_time_length=10,  # 时间掩码的长度，默认为10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码的概率，默认为0.0
        mask_feature_length=10,  # 特征掩码的长度，默认为10
        mask_feature_min_masks=0,  # 特征掩码的最小数量，默认为0
        ctc_loss_reduction="mean",  # CTC 损失的归并方式，默认为 mean
        ctc_zero_infinity=False,  # CTC 损失是否将无限值设为0，默认为 False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为 False
        classifier_proj_size=256,  # 分类器投影层的大小，默认为256
        pad_token_id=0,  # 填充标记的 ID，默认为0
        bos_token_id=1,  # 开始标记的 ID，默认为1
        eos_token_id=2,  # 结束标记的 ID，默认为2
        **kwargs,  # 其他未列出的参数，使用关键字参数传递
        ):
            # 调用父类的初始化方法，并传递关键字参数
            super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
            # 设置隐藏层大小
            self.hidden_size = hidden_size
            # 特征提取层的归一化方式
            self.feat_extract_norm = feat_extract_norm
            # 特征提取层的激活函数
            self.feat_extract_activation = feat_extract_activation
            # 卷积层的维度列表
            self.conv_dim = list(conv_dim)
            # 卷积层的步长列表
            self.conv_stride = list(conv_stride)
            # 卷积层的卷积核大小列表
            self.conv_kernel = list(conv_kernel)
            # 卷积层是否包含偏置
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
            # 压缩因子
            self.squeeze_factor = squeeze_factor
            # 隐藏层的激活函数
            self.hidden_act = hidden_act
            # 注意力头的数量
            self.num_attention_heads = num_attention_heads
            # 隐藏层的dropout比率
            self.hidden_dropout = hidden_dropout
            # 注意力机制的dropout比率
            self.attention_dropout = attention_dropout
            # 激活函数的dropout比率
            self.activation_dropout = activation_dropout
            # 特征投影的dropout比率
            self.feat_proj_dropout = feat_proj_dropout
            # 最终输出的dropout比率
            self.final_dropout = final_dropout
            # 层间的dropout
            self.layerdrop = layerdrop
            # 层归一化的epsilon值
            self.layer_norm_eps = layer_norm_eps
            # 初始化范围
            self.initializer_range = initializer_range
            # 词汇表大小
            self.vocab_size = vocab_size

            # 检查卷积层配置参数是否正确
            if (
                (len(self.conv_stride) != self.num_feat_extract_layers)
                or (len(self.conv_kernel) != self.num_feat_extract_layers)
                or (len(self.conv_dim) != self.num_feat_extract_layers)
            ):
                # 如果不正确，则抛出数值错误异常
                raise ValueError(
                    "Configuration for convolutional layers is incorrect. "
                    "It is required that `len(config.conv_dim)` == `len(config.conv_stride)` == `len(config.conv_kernel)`, "
                    f"but is `len(config.conv_dim) = {len(self.conv_dim)}`, `len(config.conv_stride) "
                    f"= {len(self.conv_stride)}`, `len(config.conv_kernel) = {len(self.conv_kernel)}`."
                )

            # 针对 SpecAugment 进行微调的配置参数：https://arxiv.org/abs/1904.08779
            # 是否应用 SpecAugment
            self.apply_spec_augment = apply_spec_augment
            # 时间遮盖的概率
            self.mask_time_prob = mask_time_prob
            # 时间遮盖的长度
            self.mask_time_length = mask_time_length
            # 最小时间遮盖数
            self.mask_time_min_masks = mask_time_min_masks
            # 特征遮盖的概率
            self.mask_feature_prob = mask_feature_prob
            # 特征遮盖的长度
            self.mask_feature_length = mask_feature_length
            # 最小特征遮盖数
            self.mask_feature_min_masks = mask_feature_min_masks

            # CTC损失配置
            # 减少的方式
            self.ctc_loss_reduction = ctc_loss_reduction
            # CTC无穷远
            self.ctc_zero_infinity = ctc_zero_infinity

            # 序列分类
            # 是否使用加权层求和
            self.use_weighted_layer_sum = use_weighted_layer_sum
            # 分类器投影的大小
            self.classifier_proj_size = classifier_proj_size

        @property
        def inputs_to_logits_ratio(self):
            # 返回输入到logits比率，通过乘以所有卷积步长的结果得到
            return functools.reduce(operator.mul, self.conv_stride, 1)
```