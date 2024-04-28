# `.\models\encodec\feature_extraction_encodec.py`

```
# 设置编码为 UTF-8，确保支持中文注释等特殊字符
# 版权声明，指明版权归 The HuggingFace Inc. team 所有
# 使用 Apache 许可证 2.0 版本，允许自由使用，但需包含许可证和版权声明
# 获取完整的许可证内容
# 如果符合适用法律要求或以书面形式同意，则按"原样"提供，不提供任何明示或暗示的保证或条件
# 根据许可证分发的软件是按"原样"分发的，不附带任何明示或暗示的保证或条件，包括但不限于适销性、特定用途适用性和非侵权性。
# 详见许可证
"""EnCodec 的特征提取器类。"""

# 引入需要的类型提示
from typing import List, Optional, Union

# 引入 NumPy 库
import numpy as np

# 引入相关的特征提取工具
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
# 引入填充策略和张量类型
from ...utils import PaddingStrategy, TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义 EncodecFeatureExtractor 类，继承自 SequenceFeatureExtractor 类
class EncodecFeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建一个 EnCodec 特征提取器。

    这个特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大多数主要方法。用户应该参考此超类以获取有关这些方法的更多信息。

    使用默认值实例化特征提取器将产生类似于 [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz) 架构的配置。

    参数:
        feature_size (`int`, *可选*, 默认为 1):
            提取特征的特征维度。对于单声道使用 1，立体声使用 2。
        sampling_rate (`int`, *可选*, 默认为 24000):
            应该数字化音频波形的采样率，以赫兹（Hz）表示。
        padding_value (`float`, *可选*, 默认为 0.0):
            用于填充值的值。
        chunk_length_s (`float`, *可选*):
            如果定义了，则音频将预处理为长度为 `chunk_length_s` 的块，然后进行编码。
        overlap (`float`, *可选*):
            定义每个块之间的重叠。它用于计算 `chunk_stride`，使用以下公式：`int((1.0 - self.overlap) * self.chunk_length)`。
    """

    # 模型输入的名称列表
    model_input_names = ["input_values", "padding_mask"]

    # 初始化方法
    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 24000,
        padding_value: float = 0.0,
        chunk_length_s: float = None,
        overlap: float = None,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 设置块长度（秒）
        self.chunk_length_s = chunk_length_s
        # 设置重叠率
        self.overlap = overlap

    # 这是一个属性，因为你可能想动态更改 chunk_length_s
    @property
    # 返回数据块的长度，以采样率为单位，如果数据块长度未定义则返回空值
    def chunk_length(self) -> Optional[int]:
        if self.chunk_length_s is None:
            return None
        else:
            return int(self.chunk_length_s * self.sampling_rate)

    # 这是一个属性，因为您可能希望动态更改chunk_length_s
    @property
    # 返回数据块的步幅，以采样率为单位，如果数据块长度或重叠未定义则返回空值
    def chunk_stride(self) -> Optional[int]:
        if self.chunk_length_s is None or self.overlap is None:
            return None
        else:
            return max(1, int((1.0 - self.overlap) * self.chunk_length))

    # 声明一个调用方法，接受原始音频数据、填充、截断、最大长度、返回的张量类型和采样率等参数
    def __call__(
        self,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
```