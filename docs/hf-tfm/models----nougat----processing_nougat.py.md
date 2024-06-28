# `.\models\nougat\processing_nougat.py`

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
"""
Processor class for Nougat.
"""

from typing import Dict, List, Optional, Union

# 导入所需的类型和工具类
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy

# 导入自定义的混合处理器和工具
from ...processing_utils import ProcessorMixin
from ...utils import PaddingStrategy, TensorType


class NougatProcessor(ProcessorMixin):
    r"""
    Constructs a Nougat processor which wraps a Nougat image processor and a Nougat tokenizer into a single processor.

    [`NougatProcessor`] offers all the functionalities of [`NougatImageProcessor`] and [`NougatTokenizerFast`]. See the
    [`~NougatProcessor.__call__`] and [`~NougatProcessor.decode`] for more information.

    Args:
        image_processor ([`NougatImageProcessor`]):
            An instance of [`NougatImageProcessor`]. The image processor is a required input.
        tokenizer ([`NougatTokenizerFast`]):
            An instance of [`NougatTokenizerFast`]. The tokenizer is a required input.
    """

    # 定义类属性，包括可以访问的属性列表和默认的处理器类名称
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        # 调用父类的构造函数初始化混合处理器
        super().__init__(image_processor, tokenizer)
        # 将当前处理器设置为图像处理器
        self.current_processor = self.image_processor
    # 定义一个方法，使对象可以像函数一样被调用
    def __call__(
        self,
        images=None,
        text=None,
        do_crop_margin: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: "PILImageResampling" = None,  # noqa: F821
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional["ChannelDimension"] = "channels_first",  # noqa: F821
        input_data_format: Optional[Union[str, "ChannelDimension"]] = None,  # noqa: F821
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ):
        if images is None and text is None:
            # 如果既没有指定 images 也没有指定 text 输入，则抛出数值错误异常
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            # 如果指定了 images 输入，则使用 image_processor 处理图片数据
            inputs = self.image_processor(
                images,
                do_crop_margin=do_crop_margin,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_thumbnail=do_thumbnail,
                do_align_long_axis=do_align_long_axis,
                do_pad=do_pad,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                return_tensors=return_tensors,
                data_format=data_format,
                input_data_format=input_data_format,
            )
        if text is not None:
            # 如果指定了 text 输入，则使用 tokenizer 处理文本数据
            encodings = self.tokenizer(
                text,
                text_pair=text_pair,
                text_target=text_target,
                text_pair_target=text_pair_target,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
            )

        if text is None:
            # 如果没有 text 输入，则返回处理过的 images 输入数据
            return inputs
        elif images is None:
            # 如果没有 images 输入，则返回处理过的 text 输入数据
            return encodings
        else:
            # 如果既有 images 又有 text 输入，则将 labels 添加到 inputs 字典中并返回
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 将所有参数转发给 NougatTokenizer 的 batch_decode 方法，并返回其结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 将所有参数转发给 NougatTokenizer 的 decode 方法，并返回其结果
        return self.tokenizer.decode(*args, **kwargs)
    def post_process_generation(self, *args, **kwargs):
        """
        将所有参数转发到 NougatTokenizer 的 [`~PreTrainedTokenizer.post_process_generation`] 方法。
        请参考该方法的文档字符串获取更多信息。
        """
        # 调用内部的 NougatTokenizer 对象的 post_process_generation 方法，并返回其结果
        return self.tokenizer.post_process_generation(*args, **kwargs)
```