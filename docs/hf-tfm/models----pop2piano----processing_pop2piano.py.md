# `.\models\pop2piano\processing_pop2piano.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Processor class for Pop2Piano."""

import os
from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy
from ...utils import TensorType


class Pop2PianoProcessor(ProcessorMixin):
    r"""
    Constructs an Pop2Piano processor which wraps a Pop2Piano Feature Extractor and Pop2Piano Tokenizer into a single
    processor.

    [`Pop2PianoProcessor`] offers all the functionalities of [`Pop2PianoFeatureExtractor`] and [`Pop2PianoTokenizer`].
    See the docstring of [`~Pop2PianoProcessor.__call__`] and [`~Pop2PianoProcessor.decode`] for more information.

    Args:
        feature_extractor (`Pop2PianoFeatureExtractor`):
            An instance of [`Pop2PianoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Pop2PianoTokenizer`):
            An instance of ['Pop2PianoTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Pop2PianoFeatureExtractor"
    tokenizer_class = "Pop2PianoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: Union[np.ndarray, List[float], List[np.ndarray]] = None,
        sampling_rate: Union[int, List[int]] = None,
        steps_per_beat: int = 2,
        resample: Optional[bool] = True,
        notes: Union[List, TensorType] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Call method to process input audio data into features suitable for Pop2Piano model.

        Args:
            audio (Union[np.ndarray, List[float], List[np.ndarray]], optional):
                Input audio data. Can be a numpy array, list of floats, or list of numpy arrays.
            sampling_rate (Union[int, List[int]], optional):
                Sampling rate of the input audio. Can be an integer or a list of integers.
            steps_per_beat (int, optional):
                Number of steps per beat in the musical sequence.
            resample (bool, optional):
                Whether to resample the input audio to the specified sampling rate.
            notes (Union[List, TensorType], optional):
                Musical notes associated with the audio data. Can be a list or TensorType.
            padding (Union[bool, str, PaddingStrategy], optional):
                Padding strategy to apply to the input data.
            truncation (Union[bool, str, TruncationStrategy], optional):
                Truncation strategy to apply to the input data.
            max_length (int, optional):
                Maximum length of the output sequence.
            pad_to_multiple_of (int, optional):
                Pad the sequence length to be a multiple of this value.
            verbose (bool, optional):
                Whether to print verbose information during processing.

            **kwargs:
                Additional keyword arguments for processing.

        Returns:
            BatchEncoding:
                Processed batch of encoded inputs suitable for Pop2Piano model.
        """
        # Implementation details for processing audio data using the provided feature extractor and tokenizer
        pass  # Placeholder for actual implementation
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        使用 [`Pop2PianoFeatureExtractor.__call__`] 方法准备模型的对数梅尔频谱图(log-mel-spectrograms)，
        并使用 [`Pop2PianoTokenizer.__call__`] 方法从音符中准备 token_ids。

        请查阅上述两个方法的文档字符串以获取更多信息。
        """

        # 因为特征提取器需要音频和采样率，而分词器需要 token_ids 和特征提取器的输出，所以必须同时检查两者。
        if (audio is None and sampling_rate is None) and (notes is None):
            raise ValueError(
                "You have to specify at least audios and sampling_rate in order to use feature extractor or "
                "notes to use the tokenizer part."
            )

        if audio is not None and sampling_rate is not None:
            # 调用特征提取器，生成模型的输入
            inputs = self.feature_extractor(
                audio=audio,
                sampling_rate=sampling_rate,
                steps_per_beat=steps_per_beat,
                resample=resample,
                **kwargs,
            )
        if notes is not None:
            # 调用分词器，生成音符的 token_ids
            encoded_token_ids = self.tokenizer(
                notes=notes,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                verbose=verbose,
                **kwargs,
            )

        if notes is None:
            # 如果没有音符，返回特征提取器生成的输入
            return inputs

        elif audio is None or sampling_rate is None:
            # 如果没有音频或采样率，返回分词器生成的 token_ids
            return encoded_token_ids

        else:
            # 否则，将分词器生成的 token_ids 添加到特征提取器生成的输入中，并返回
            inputs["token_ids"] = encoded_token_ids["token_ids"]
            return inputs

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ) -> BatchEncoding:
        """
        使用 [`Pop2PianoTokenizer.batch_decode`] 方法将模型生成的 token_ids 转换为 midi_notes。

        请查阅上述方法的文档字符串以获取更多信息。
        """

        return self.tokenizer.batch_decode(
            token_ids=token_ids, feature_extractor_output=feature_extractor_output, return_midi=return_midi
        )

    @property
    def model_input_names(self):
        """
        返回模型输入的名称列表，包括分词器和特征提取器的输入名称。

        使用 `self.tokenizer.model_input_names` 和 `self.feature_extractor.model_input_names` 获取输入名称列表，
        并将两者合并后去除重复项后返回。
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        """
        将模型的预训练文件保存到指定目录中。

        如果 `save_directory` 是文件而不是目录，将引发 ValueError。
        如果目录不存在，则创建目录。
        最后，调用父类的 `save_pretrained` 方法保存预训练文件。

        Args:
            save_directory (str): 要保存预训练文件的目录路径。
            **kwargs: 其他参数传递给 `save_pretrained` 方法。

        Returns:
            Any: `save_pretrained` 方法的返回值。

        Raises:
            ValueError: 如果 `save_directory` 是文件路径而不是目录路径。
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    # 类方法，用于从预训练模型名称或路径中获取参数，并返回一个参数列表
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 调用类方法 _get_arguments_from_pretrained，获取预训练模型的参数
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 使用获取到的参数列表创建并返回当前类的实例
        return cls(*args)
```