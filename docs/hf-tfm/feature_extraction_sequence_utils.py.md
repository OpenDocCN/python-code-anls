# `.\transformers\feature_extraction_sequence_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""
用于常见特征提取器预处理序列的序列特征提取类。
"""
from typing import Dict, List, Optional, Union

import numpy as np

from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .utils import PaddingStrategy, TensorType, is_tf_tensor, is_torch_tensor, logging, to_numpy

# 获取日志记录器
logger = logging.get_logger(__name__)

# 序列特征提取器类，继承自特征提取混合类
class SequenceFeatureExtractor(FeatureExtractionMixin):
    """
    这是用于语音识别的通用特征提取类。

    Args:
        feature_size (`int`):
            提取特征的特征维度。
        sampling_rate (`int`):
            应以其数字化的音频文件的采样率表示，单位为赫兹（Hz）。
        padding_value (`float`):
            用于填充填充值/向量的值。
    """

    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs):
        # 初始化特征维度、采样率和填充值
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        # 设置填充方向和是否返回注意力掩码
        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)

        super().__init__(**kwargs)

    # 对处理后的特征进行填充
    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    def _pad(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    def _truncate(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        truncation: Optional[bool] = None,
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features(`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional`):
                maximum length of the returned list and optionally padding length (see below)
            pad_to_multiple_of (`int`, *optional*) :
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            truncation (`bool`, *optional`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        # 如果不需要截断，则直接返回处理后的特征
        if not truncation:
            return processed_features
        # 如果需要截断但未定义最大长度，则抛出异常
        elif truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        required_input = processed_features[self.model_input_names[0]]

        # 找到适合 `pad_to_multiple_of` 的 `max_length`
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length

        # 如果需要截断
        if needs_to_be_truncated:
            # 截断输入到指定的最大长度
            processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
            # 如果存在 "attention_mask"，也截断它
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]

        # 返回处理后的特征
        return processed_features
    # 获取填充策略的函数，用于确定是否需要填充以及填充的方式
    def _get_padding_strategies(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """

        # 获取填充策略
        if padding is not False:
            # 如果 padding 为 True，则默认选择将批次中的序列填充到最长的序列长度
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            # 如果 padding 不是 PaddingStrategy 类型，则根据传入的字符串创建填充策略
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            # 如果 padding 是 PaddingStrategy 类型，则直接使用传入的填充策略
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            # 如果不需要填充，则填充策略为 DO_NOT_PAD
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # 如果需要指定最大长度，则检查是否设置了 max_length
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                # 如果设置了 padding_strategy 为 MAX_LENGTH，但未指定 max_length，则报错
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        # 检查是否有填充值
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            # 如果需要填充但没有指定填充值，则报错
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy
```