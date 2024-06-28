# `.\models\trocr\processing_trocr.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Processor class for TrOCR.
"""
import warnings
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin


class TrOCRProcessor(ProcessorMixin):
    r"""
    Constructs a TrOCR processor which wraps a vision image processor and a TrOCR tokenizer into a single processor.

    [`TrOCRProcessor`] offers all the functionalities of [`ViTImageProcessor`/`DeiTImageProcessor`] and
    [`RobertaTokenizer`/`XLMRobertaTokenizer`]. See the [`~TrOCRProcessor.__call__`] and [`~TrOCRProcessor.decode`] for
    more information.

    Args:
        image_processor ([`ViTImageProcessor`/`DeiTImageProcessor`], *optional*):
            An instance of [`ViTImageProcessor`/`DeiTImageProcessor`]. The image processor is a required input.
        tokenizer ([`RobertaTokenizer`/`XLMRobertaTokenizer`], *optional*):
            An instance of [`RobertaTokenizer`/`XLMRobertaTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 如果提供了`feature_extractor`参数，则发出警告并将其转换为`image_processor`
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未提供`image_processor`，则使用`feature_extractor`，否则会引发错误
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，将`image_processor`和`tokenizer`传入
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为`image_processor`
        self.current_processor = self.image_processor
        # 设置目标上下文管理器状态为False
        self._in_target_context_manager = False
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~TrOCRProcessor.as_target_processor`] this method forwards all its arguments to TrOCRTokenizer's
        [`~TrOCRTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        # 如果在目标处理器上下文中，则调用当前处理器对象处理输入参数并返回结果
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 从 kwargs 中弹出 'images' 和 'text' 参数
        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)

        # 如果没有传入参数，则检查是否有位置参数传入并赋值给 images，同时将其余的位置参数赋给 args
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        # 如果 images 和 text 都为 None，则抛出 ValueError
        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        # 如果 images 不为 None，则使用 image_processor 处理 images 和其他参数
        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)

        # 如果 text 不为 None，则使用 tokenizer 处理 text 和其他参数
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        # 如果 text 为 None，则返回 inputs；如果 images 为 None，则返回 encodings；否则将 labels 添加到 inputs 后返回
        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 调用 tokenizer 的 batch_decode 方法并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TrOCRTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        # 调用 tokenizer 的 decode 方法并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        # 发出警告信息，说明这个方法即将被移除，建议使用 __call__ 方法的 text 参数处理标签
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your images inputs, or in a separate call."
        )
        # 设置 _in_target_context_manager 标志为 True，设置当前处理器为 tokenizer，并在退出时恢复为 image_processor
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    @property
    def feature_extractor_class(self):
        """
        Warns about deprecation of `feature_extractor_class` and suggests using `image_processor_class`.
        """
        # 发出警告信息，说明 feature_extractor_class 将在 v5 版本移除，建议使用 image_processor_class 替代
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回 image_processor_class 属性
        return self.image_processor_class
    def feature_extractor(self):
        # 发出警告，提醒用户 `feature_extractor` 方法已经废弃，将在 v5 版本中移除，请使用 `image_processor` 方法代替。
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回当前对象的 `image_processor` 属性作为特征提取器
        return self.image_processor
```