# `.\transformers\models\wav2vec2\feature_extraction_wav2vec2.py`

```py
# coding=utf-8  # 设置代码文件的编码格式为 UTF-8
# Copyright 2021 The HuggingFace Inc. team.  # 版权声明，版权归 HuggingFace 公司所有
# Licensed under the Apache License, Version 2.0 (the "License");  # 使用 Apache License 2.0 许可证
# you may not use this file except in compliance with the License;  # 除非遵循许可证规定，否则不得使用该文件
# You may obtain a copy of the License at  # 可在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software  # 除非适用的法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 根据许可证分发的软件是基于“原样基础”分发的
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 没有任何形式的担保或条件，明示或暗示
# See the License for the specific language governing permissions and  # 有关特定语言的权限和限制，请参阅许可证
# limitations under the License.
"""
Feature extractor class for Wav2Vec2  # Wav2Vec2 的特征提取器类
"""

from typing import List, Optional, Union  # 导入必要的类型提示

import numpy as np  # 导入第三方库 numpy

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor  # 从 HuggingFace 库中导入 SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature  # 从 HuggingFace 库中导入 BatchFeature
from ...utils import PaddingStrategy, TensorType, logging  # 从 HuggingFace 库中导入 PaddingStrategy, TensorType, logging

logger = logging.get_logger(__name__)  # 获取 logger 对象

class Wav2Vec2FeatureExtractor(SequenceFeatureExtractor):  # 创建 Wav2Vec2FeatureExtractor 类并继承自 SequenceFeatureExtractor 类
    r"""
    Constructs a Wav2Vec2 feature extractor.  # 构造一个 Wav2Vec2 特征提取器

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains  # 这个特征提取器继承自 SequenceFeatureExtractor 类
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:  # 参数
        feature_size (`int`, defaults to 1):  # 特征的维度，默认为1
            The feature dimension of the extracted features.  # 提取特征的特征维度
        sampling_rate (`int`, defaults to 16000):  # 采样率，默认为16000
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).  # 音频文件应该被数字化的采样率（赫兹）
        padding_value (`float`, defaults to 0.0):  # 填充值，默认为0.0
            The value that is used to fill the padding values.  # 用于填充填充值的值
        do_normalize (`bool`, *optional*, defaults to `True`):  # 是否进行规范化，可选，默认为 True
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly  # 是否对输入进行零均值单位方差规范化。规范化可以显著改善某些模型的性能
            improve the performance for some models, *e.g.*,  # 某些模型的性能可以通过规范化显著提高，例如
            [wav2vec2-lv60](https://huggingface.co/models?search=lv60).
        return_attention_mask (`bool`, *optional*, defaults to `False`):  # 返回注意力蒙版，可选，默认为 False
            Whether or not [`~Wav2Vec2FeatureExtractor.__call__`] should return `attention_mask`.  # [`~Wav2Vec2FeatureExtractor.__call__`] 是否应该返回 `attention_mask`。

            <Tip>

            Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as  # 设置了 `config.feat_extract_norm == "group"` 的 Wav2Vec2 模型
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using  # 例如 [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h)，**未**使用注意力蒙版进行训练
            `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask`  # 对于这些模型，`input_values` 应该简单地用 0 填充，不需要 `attention_mask`
            should be passed.

            For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as  # 设置了 `config.feat_extract_norm == "layer"` 的 Wav2Vec2 模型
            [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should be  # 例如 [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)，应该传入 `attention_mask` 进行批量推断

            passed for batched inference.

            </Tip>
    """

    model_input_names = ["input_values", "attention_mask"]  # 模型输入的名称
    # 初始化方法，设置默认特征大小、采样率、填充值、是否返回注意力掩码和是否进行标准化处理等参数
    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs,
    ):
        # 调用父类初始化方法，设置特征大小、采样率、填充值等参数
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        # 设置是否返回注意力掩码
        self.return_attention_mask = return_attention_mask
        # 设置是否进行标准化处理
        self.do_normalize = do_normalize

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        每个列表中的数组都被标准化为零均值和单位方差
        """
        # 如果存在注意力掩码
        if attention_mask is not None:
            # 将注意力掩码转换为整数类型的数组
            attention_mask = np.array(attention_mask, np.int32)
            # 存储标准化后的输入值
            normed_input_values = []

            # 对每个输入值和对应的长度进行处理
            for vector, length in zip(input_values, attention_mask.sum(-1)):
                # 标准化处理，计算均值和方差
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                # 如果长度小于标准化后的切片长度，使用填充值填充
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                # 存储标准化后的切片
                normed_input_values.append(normed_slice)
        else:
            # 如果不存在注意力掩码，直接对每个输入值进行标准化处理
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    # 调用方法，对原始语音进行预处理
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