# `.\models\encodec\feature_extraction_encodec.py`

```
# 指定代码文件的编码格式为UTF-8

# 版权声明，声明此代码版权归HuggingFace Inc.团队所有，保留所有权利

# 根据Apache License, Version 2.0许可证使用本文件。您除非遵守许可证，否则不得使用本文件。
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，软件按"原样"分发，不附带任何明示或暗示的担保或条件。

# 导入必要的库
"""Feature extractor class for EnCodec."""

from typing import List, Optional, Union

import numpy as np  # 导入NumPy库

# 导入相关的特征提取工具和实用函数
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义EnCodecFeatureExtractor类，继承自SequenceFeatureExtractor类
class EncodecFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an EnCodec feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
    [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) architecture.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        chunk_length_s (`float`, *optional*):
            If defined the audio is pre-processed into chunks of lengths `chunk_length_s` and then encoded.
        overlap (`float`, *optional*):
            Defines the overlap between each chunk. It is used to compute the `chunk_stride` using the following
            formulae : `int((1.0 - self.overlap) * self.chunk_length)`.

    """

    # 模型输入的名称列表
    model_input_names = ["input_values", "padding_mask"]

    # 构造函数，初始化EnCodecFeatureExtractor对象
    def __init__(
        self,
        feature_size: int = 1,  # 特征维度，默认为1
        sampling_rate: int = 24000,  # 采样率，默认为24000
        padding_value: float = 0.0,  # 填充值，默认为0.0
        chunk_length_s: float = None,  # 分块长度（秒），可选参数
        overlap: float = None,  # 分块重叠度，可选参数
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的构造函数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 设置分块长度属性
        self.chunk_length_s = chunk_length_s
        # 设置分块重叠度属性
        self.overlap = overlap

    # chunk_length_s属性的getter，作为属性，可以动态更改chunk_length_s的值
    @property
    # 如果未设置 chunk_length_s，则返回 None，表示长度未定义
    def chunk_length(self) -> Optional[int]:
        if self.chunk_length_s is None:
            return None
        else:
            # 计算并返回采样率乘以 chunk_length_s 的整数值，作为 chunk 的长度
            return int(self.chunk_length_s * self.sampling_rate)

    # 这是一个属性，因为你可能想动态更改 chunk_length_s
    @property
    def chunk_stride(self) -> Optional[int]:
        # 如果 chunk_length_s 或 overlap 未定义，则返回 None，表示步长未定义
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            # 计算并返回步长值，确保至少为 1，根据 overlap 和 chunk_length 计算得出
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    # 函数调用运算符重载，用于将音频数据处理成模型所需格式
    def __call__(
        self,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
```