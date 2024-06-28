# `.\models\wav2vec2\feature_extraction_wav2vec2.py`

```
# coding=utf-8
# 版权所有 2021 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件根据"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
"""
Wav2Vec2 的特征提取器类
"""

from typing import List, Optional, Union

import numpy as np

# 导入序列特征提取器和批处理特征
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)


class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):
    r"""
    构建一个 Wav2Vec2 特征提取器。

    此特征提取器继承自 [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`]，其中包含大部分主要方法。
    用户应参考此超类以获取关于这些方法的更多信息。

    Args:
        feature_size (`int`, 默认为 1):
            提取特征的特征维度。
        sampling_rate (`int`, 默认为 16000):
            音频文件的数字化采样率，以赫兹（Hz）表示。
        padding_value (`float`, 默认为 0.0):
            用于填充填充值的值。
        do_normalize (`bool`, *可选*, 默认为 `True`):
            是否对输入进行零均值单位方差归一化。归一化可以显著提高某些模型的性能，
            例如 [wav2vec2-lv60](https://huggingface.co/models?search=lv60)。
        return_attention_mask (`bool`, *可选*, 默认为 `False`):
            是否 [`~Wav2Vec2FeatureExtractor.__call__`] 应返回 `attention_mask`。

            <Tip>

            对于设置了 `config.feat_extract_norm == "group"` 的 Wav2Vec2 模型，例如
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h)，**没有** 使用
            `attention_mask` 进行训练。对于这样的模型，`input_values` 应仅用 0 填充，不应传递 `attention_mask`。

            对于设置了 `config.feat_extract_norm == "layer"` 的 Wav2Vec2 模型，例如
            [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)，应传递 `attention_mask`
            以进行批处理推断。

            </Tip>
    """

    model_input_names = ["input_values", "attention_mask"]
    # 初始化方法，设置特征大小、采样率、填充值等参数，并调用父类的初始化方法
    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 是否返回注意力掩码
        self.return_attention_mask = return_attention_mask
        # 是否进行归一化处理
        self.do_normalize = do_normalize

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        每个数组都被归一化为零均值和单位方差
        """
        if attention_mask is not None:
            # 将注意力掩码转换为numpy数组类型
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            # 对于输入值列表中的每个向量和对应的长度进行循环
            for vector, length in zip(input_values, attention_mask.sum(-1)):
                # 计算切片的归一化值，确保长度外的部分使用填充值
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                # 将归一化后的切片添加到结果列表中
                normed_input_values.append(normed_slice)
        else:
            # 对于没有注意力掩码的情况，直接对每个输入值进行归一化处理
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        # 返回归一化后的输入值列表
        return normed_input_values

    # 调用方法，接收原始语音数据及相关参数，并进行相应的数据处理和转换
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
```