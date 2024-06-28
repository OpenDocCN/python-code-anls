# `.\models\tvp\processing_tvp.py`

```py
# coding=utf-8
# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for TVP.
"""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

class TvpProcessor(ProcessorMixin):
    """
    Constructs an TVP processor which wraps a TVP image processor and a Bert tokenizer into a single processor.

    [`TvpProcessor`] offers all the functionalities of [`TvpImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~TvpProcessor.__call__`] and [`~TvpProcessor.decode`] for more information.

    Args:
        image_processor ([`TvpImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "TvpImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        """
        Initialize the TVP processor with an image processor and a tokenizer.

        Args:
            image_processor ([`TvpImageProcessor`], *optional*):
                The image processor is a required input.
            tokenizer ([`BertTokenizerFast`], *optional*):
                The tokenizer is a required input.

        Raises:
            ValueError: If either `image_processor` or `tokenizer` is not provided.
        """
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def batch_decode(self, *args, **kwargs):
        """
        Forward all arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`] method.

        Returns:
            Decoded outputs corresponding to the input tokens.

        See Also:
            [`~PreTrainedTokenizer.batch_decode`] for more details.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        Forward all arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`] method.

        Returns:
            Decoded string corresponding to the input token.

        See Also:
            [`~PreTrainedTokenizer.decode`] for more details.
        """
        return self.tokenizer.decode(*args, **kwargs)
    # 定义一个方法，用于处理视频定位的后处理，计算视频的开始和结束时间
    def post_process_video_grounding(self, logits, video_durations):
        """
        Compute the time of the video.

        Args:
            logits (`torch.Tensor`):
                The logits output of TvpForVideoGrounding.
            video_durations (`float`):
                The video's duration.

        Returns:
            start (`float`):
                The start time of the video.
            end (`float`):
                The end time of the video.
        """
        # 从 logits 中提取开始时间和结束时间，并乘以视频的总时长，保留一位小数
        start, end = (
            round(logits.tolist()[0][0] * video_durations, 1),
            round(logits.tolist()[0][1] * video_durations, 1),
        )

        return start, end

    @property
    # 从 transformers.models.blip.processing_blip.BlipProcessor.model_input_names 复制而来
    # 定义一个属性，返回模型输入的名称列表，合并并去重 tokenizer 和 image_processor 的输入名称
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```