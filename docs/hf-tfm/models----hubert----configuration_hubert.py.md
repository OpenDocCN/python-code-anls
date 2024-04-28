# `.\models\hubert\configuration_hubert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Fairseq 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" Hubert 模型配置"""

# 导入必要的库
import functools
import operator

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# Hubert 预训练配置文件映射
HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/hubert-base-ls960": "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json",
    # 查看所有 Hubert 模型：https://huggingface.co/models?filter=hubert
}

# Hubert 配置类，继承自预训练配置类
class HubertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`HubertModel`] 配置的配置类。根据指定的参数实例化 Hubert 模型，定义模型架构。使用默认值实例化配置将产生类似于 Hubert [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import HubertModel, HubertConfig

    >>> # 初始化一个 Hubert facebook/hubert-base-ls960 风格的配置
    >>> configuration = HubertConfig()

    >>> # 使用 facebook/hubert-base-ls960 风格的配置初始化模型
    >>> model = HubertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型为 "hubert"
    model_type = "hubert"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=32,  # 词汇表大小，默认为32
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力dropout率，默认为0.1
        feat_proj_layer_norm=True,  # 特征投影层是否使用LayerNorm，默认为True
        feat_proj_dropout=0.0,  # 特征投影层dropout率，默认为0.0
        final_dropout=0.1,  # 最终输出层dropout率，默认为0.1
        layerdrop=0.1,  # 层级dropout率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # LayerNorm的epsilon值，默认为1e-5
        feat_extract_norm="group",  # 特征提取层的归一化方式，默认为group
        feat_extract_activation="gelu",  # 特征提取层的激活函数，默认为gelu
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入分组数量，默认为16
        do_stable_layer_norm=False,  # 是否使用稳定的LayerNorm，默认为False
        apply_spec_augment=True,  # 是否应用特定数据增强，默认为True
        mask_time_prob=0.05,  # 时间掩码概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 最小时间掩码数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 最小特征掩码数量，默认为0
        ctc_loss_reduction="sum",  # CTC损失函数的减少方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC损失函数中是否将无穷值设为零，默认为False
        use_weighted_layer_sum=False,  # 是���使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 起始标记ID，默认为1
        eos_token_id=2,  # 结束标记ID，默认为2
        **kwargs,  # 其他参数
    # 初始化函数，继承父类的属性，并设置一些模型的参数
    ):
        # 调用父类的初始化函数，传入kwargs参数以及特殊的pad_token_id、bos_token_id、eos_token_id参数
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 特征提取的归一化方式
        self.feat_extract_norm = feat_extract_norm
        # 特征提取的激活函数
        self.feat_extract_activation = feat_extract_activation
        # 卷积层的维度
        self.conv_dim = list(conv_dim)
        # 卷积层的步长
        self.conv_stride = list(conv_stride)
        # 卷积核的大小
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
        # 隐藏层的dropout
        self.hidden_dropout = hidden_dropout
        # 注意力的dropout
        self.attention_dropout = attention_dropout
        # 激活函数的dropout
        self.activation_dropout = activation_dropout
        # 特征投影的层归一化
        self.feat_proj_layer_norm = feat_proj_layer_norm
        # 特征投影的dropout
        self.feat_proj_dropout = feat_proj_dropout
        # 最终的dropout
        self.final_dropout = final_dropout
        # 层丢弃
        self.layerdrop = layerdrop
        # 层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 初始化范围
        self.initializer_range = initializer_range
        # 词汇表的大小
        self.vocab_size = vocab_size
        # 是否使用稳定的层归一化
        self.do_stable_layer_norm = do_stable_layer_norm
        # 是否使用加权层求和
        self.use_weighted_layer_sum = use_weighted_layer_sum
        # 分类器投影的大小
        self.classifier_proj_size = classifier_proj_size

        # 检查卷积层配置是否正确
        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # 针对SpecAugment的微调配置参数：https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # CTC损失
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

    # 计算输入到logits的比例
    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)
```