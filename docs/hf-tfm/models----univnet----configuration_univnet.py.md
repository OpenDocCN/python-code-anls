# `.\models\univnet\configuration_univnet.py`

```py
# 版权声明和信息，指出此代码版权归HuggingFace团队所有，并使用Apache许可证2.0授权
#
# 在遵守许可证的前提下，您可以使用此文件。您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 本软件是基于"原样"提供的，没有任何明示或暗示的担保或条件。详情请查看许可证。
""" UnivNetModel 模型配置"""

# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取用于日志记录的记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射，将模型名称映射到对应的配置文件URL
UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "dg845/univnet-dev": "https://huggingface.co/dg845/univnet-dev/resolve/main/config.json",
}

# UnivNetConfig 类，继承自 PretrainedConfig 类
class UnivNetConfig(PretrainedConfig):
    r"""
    这是用于存储 [`UnivNetModel`] 配置的类。它用于根据指定参数实例化 UnivNet 语音合成模型，定义模型架构。
    使用默认值实例化配置会生成类似于 UnivNet [dg845/univnet-dev](https://huggingface.co/dg845/univnet-dev)
    架构的配置，对应于 [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/master/config/default_c32.yaml)
    中的 'c32' 架构。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。

    示例：

    ```
    >>> from transformers import UnivNetModel, UnivNetConfig

    >>> # 初始化 Tortoise TTS 风格的配置
    >>> configuration = UnivNetConfig()

    >>> # 从 Tortoise TTS 风格的配置初始化一个模型（带有随机权重）
    >>> model = UnivNetModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型标识符
    model_type = "univnet"

    # 初始化方法，定义 UnivNetConfig 的各种参数
    def __init__(
        self,
        model_in_channels=64,
        model_hidden_channels=32,
        num_mel_bins=100,
        resblock_kernel_sizes=[3, 3, 3],
        resblock_stride_sizes=[8, 8, 4],
        resblock_dilation_sizes=[[1, 3, 9, 27], [1, 3, 9, 27], [1, 3, 9, 27]],
        kernel_predictor_num_blocks=3,
        kernel_predictor_hidden_channels=64,
        kernel_predictor_conv_size=3,
        kernel_predictor_dropout=0.0,
        initializer_range=0.01,
        leaky_relu_slope=0.2,
        **kwargs,
        ):
            如果 `resblock_kernel_sizes`、`resblock_stride_sizes`、`resblock_dilation_sizes` 的长度不相等，
            抛出 ValueError 异常，提示这三个参数必须具有相同的长度，这个长度也将是模型中 ResNet 块的数量。
        self.model_in_channels = model_in_channels
            设置模型的输入通道数。
        self.model_hidden_channels = model_hidden_channels
            设置模型的隐藏通道数。
        self.num_mel_bins = num_mel_bins
            设置 Mel 频谱的频段数。
        self.resblock_kernel_sizes = resblock_kernel_sizes
            设置 ResNet 块的卷积核大小列表。
        self.resblock_stride_sizes = resblock_stride_sizes
            设置 ResNet 块的步幅大小列表。
        self.resblock_dilation_sizes = resblock_dilation_sizes
            设置 ResNet 块的扩张（dilation）大小列表。
        self.kernel_predictor_num_blocks = kernel_predictor_num_blocks
            设置核预测器中的块数量。
        self.kernel_predictor_hidden_channels = kernel_predictor_hidden_channels
            设置核预测器中的隐藏通道数。
        self.kernel_predictor_conv_size = kernel_predictor_conv_size
            设置核预测器中的卷积大小。
        self.kernel_predictor_dropout = kernel_predictor_dropout
            设置核预测器中的 dropout 概率。
        self.initializer_range = initializer_range
            设置模型参数的初始化范围。
        self.leaky_relu_slope = leaky_relu_slope
            设置 Leaky ReLU 激活函数的斜率。
        super().__init__(**kwargs)
            调用父类的构造函数，传递可能的关键字参数。
```