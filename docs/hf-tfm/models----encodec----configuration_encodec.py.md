# `.\models\encodec\configuration_encodec.py`

```
# 设置编码格式为 UTF-8，确保代码可以正确处理各种字符
# 版权声明，指出版权归 Meta Platforms, Inc. 及其关联公司和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，只有在符合许可证的情况下才能使用此文件
# 可以通过链接获取许可证的副本
# 根据适用法律或书面同意，软件根据“原样”分发，无任何明示或暗示的保证或条件
# 请参阅许可证了解具体语言的规定，以及许可证下的限制
""" EnCodec model configuration"""

# 导入数学库
import math
# 导入类型提示模块，用于类型注解
from typing import Optional

# 导入 numpy 库，用于数值操作
import numpy as np

# 导入配置工具中的预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射，将模型名称映射到其配置文件的 URL
ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/encodec_24khz": "https://huggingface.co/facebook/encodec_24khz/resolve/main/config.json",
    "facebook/encodec_48khz": "https://huggingface.co/facebook/encodec_48khz/resolve/main/config.json",
}

# EncodecConfig 类，用于存储 Encodec 模型的配置信息
class EncodecConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`EncodecModel`]. It is used to instantiate a
    Encodec model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import EncodecModel, EncodecConfig

    >>> # Initializing a "facebook/encodec_24khz" style configuration
    >>> configuration = EncodecConfig()

    >>> # Initializing a model (with random weights) from the "facebook/encodec_24khz" style configuration
    >>> model = EncodecModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 encodec
    model_type = "encodec"

    # 构造方法，初始化 EncodecConfig 实例
    def __init__(
        self,
        target_bandwidths=[1.5, 3.0, 6.0, 12.0, 24.0],
        sampling_rate=24_000,
        audio_channels=1,
        normalize=False,
        chunk_length_s=None,
        overlap=None,
        hidden_size=128,
        num_filters=32,
        num_residual_layers=1,
        upsampling_ratios=[8, 5, 4, 2],
        norm_type="weight_norm",
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        dilation_growth_rate=2,
        use_causal_conv=True,
        pad_mode="reflect",
        compress=2,
        num_lstm_layers=2,
        trim_right_ratio=1.0,
        codebook_size=1024,
        codebook_dim=None,
        use_conv_shortcut=True,
        **kwargs,
    ):
        self.target_bandwidths = target_bandwidths
        self.sampling_rate = sampling_rate
        self.audio_channels = audio_channels
        self.normalize = normalize
        self.chunk_length_s = chunk_length_s
        self.overlap = overlap
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.last_kernel_size = last_kernel_size
        self.residual_kernel_size = residual_kernel_size
        self.dilation_growth_rate = dilation_growth_rate
        self.use_causal_conv = use_causal_conv
        self.pad_mode = pad_mode
        self.compress = compress
        self.num_lstm_layers = num_lstm_layers
        self.trim_right_ratio = trim_right_ratio
        self.codebook_size = codebook_size
        # 设置 codebook_dim，如果未指定，则使用 hidden_size
        self.codebook_dim = codebook_dim if codebook_dim is not None else hidden_size
        self.use_conv_shortcut = use_conv_shortcut

        # 检查 norm_type 是否为支持的类型，否则抛出 ValueError 异常
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        # 调用父类的构造方法，传入其他未明确列出的关键字参数
        super().__init__(**kwargs)

    # 由于 chunk_length_s 可能会在运行时更改，所以这是一个属性
    @property
    def chunk_length(self) -> Optional[int]:
        # 如果 chunk_length_s 为 None，则返回 None
        if self.chunk_length_s is None:
            return None
        else:
            # 否则返回计算得到的 chunk_length
            return int(self.chunk_length_s * self.sampling_rate)

    # 由于 chunk_length_s 和 overlap 可能会在运行时更改，所以这是一个属性
    @property
    def chunk_stride(self) -> Optional[int]:
        # 如果 chunk_length_s 或 overlap 为 None，则返回 None
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            # 否则返回计算得到的 chunk_stride
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    # 计算并返回帧率，这是一个属性
    @property
    def frame_rate(self) -> int:
        hop_length = np.prod(self.upsampling_ratios)  # 计算 upsampling_ratios 的乘积
        return math.ceil(self.sampling_rate / hop_length)  # 计算并返回帧率

    # 返回 quantizer 的数量，这是一个属性
    @property
    def num_quantizers(self) -> int:
        return int(1000 * self.target_bandwidths[-1] // (self.frame_rate * 10))
```