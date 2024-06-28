# `.\feature_extraction_sequence_utils.py`

```py
`
"""
Sequence feature extraction class for common feature extractors to preprocess sequences.
"""
# 导入必要的模块和库
from typing import Dict, List, Optional, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库

from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin  # 导入自定义模块
from .utils import PaddingStrategy, TensorType, is_tf_tensor, is_torch_tensor, logging, to_numpy  # 导入自定义工具模块

logger = logging.get_logger(__name__)  # 获取日志记录器对象


class SequenceFeatureExtractor(FeatureExtractionMixin):
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (`int`):
            The feature dimension of the extracted features.
        sampling_rate (`int`):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs):
        self.feature_size = feature_size  # 初始化特征维度大小
        self.sampling_rate = sampling_rate  # 初始化采样率
        self.padding_value = padding_value  # 初始化填充值

        self.padding_side = kwargs.pop("padding_side", "right")  # 初始化填充位置，默认为右侧
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)  # 是否返回注意力掩码，默认为 True

        super().__init__(**kwargs)  # 调用父类初始化方法

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
    ):
        """
        Pad sequences of features to the same length.

        Args:
            processed_features (Union[BatchFeature, List[BatchFeature], Dict[str, BatchFeature], ...]):
                The processed features to be padded.
            padding (Union[bool, str, PaddingStrategy]):
                Strategy for padding. Can be a boolean, string, or enum from PaddingStrategy.
            max_length (Optional[int]):
                Maximum length to pad or truncate the sequences.
            truncation (bool):
                Whether to truncate sequences that exceed `max_length`.
            pad_to_multiple_of (Optional[int]):
                Pad to a multiple of this value.
            return_attention_mask (Optional[bool]):
                Whether to return attention masks.
            return_tensors (Optional[Union[str, TensorType]]):
                The type of tensor(s) to be returned.

        Returns:
            Padded sequences of features.
        """
        pass  # Placeholder for method implementation

    def _pad(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        """
        Internal method for padding sequences of features.

        Args:
            processed_features (Union[Dict[str, np.ndarray], BatchFeature]):
                The processed features to be padded.
            max_length (Optional[int]):
                Maximum length to pad or truncate the sequences.
            padding_strategy (PaddingStrategy):
                Strategy for padding. Default is DO_NOT_PAD.
            pad_to_multiple_of (Optional[int]):
                Pad to a multiple of this value.
            return_attention_mask (Optional[bool]):
                Whether to return attention masks.
        """
        pass  # Placeholder for method implementation

    def _truncate(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        truncation: Optional[bool] = None,
    ):
        """
        Internal method for truncating sequences of features.

        Args:
            processed_features (Union[Dict[str, np.ndarray], BatchFeature]):
                The processed features to be truncated.
            max_length (Optional[int]):
                Maximum length to truncate the sequences.
            pad_to_multiple_of (Optional[int]):
                Pad to a multiple of this value.
            truncation (Optional[bool]):
                Whether to truncate sequences that exceed `max_length`.
        """
        pass  # Placeholder for method implementation
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features(`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional*):
                maximum length of the returned list and optionally padding length (see below)
            pad_to_multiple_of (`int`, *optional*) :
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        # 如果不进行截断，则直接返回处理后的特征
        if not truncation:
            return processed_features
        # 如果需要截断但未指定最大长度，则抛出数值错误异常
        elif truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        # 获取需要进行截断的输入数据
        required_input = processed_features[self.model_input_names[0]]

        # 根据 `pad_to_multiple_of` 找到适合的 `max_length`
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        # 判断是否需要进行截断操作
        needs_to_be_truncated = len(required_input) > max_length

        # 如果需要截断，则对输入数据进行截断操作
        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
            # 如果存在 `attention_mask`，则同步截断 `attention_mask`
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]

        # 返回处理后的特征
        return processed_features
    def _get_padding_strategies(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """

        # 获取填充策略
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # 默认为将批次中的序列填充到最长的序列长度
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD  # 不进行填充

        # 如果需要，设置最大长度
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        # 检查是否有填充值
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy
```