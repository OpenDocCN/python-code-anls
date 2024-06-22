# `.\transformers\models\nougat\processing_nougat.py`

```py
# 代码头部包含版权声明和许可信息
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

# 该文件定义了 NougatProcessor 类，它封装了 NougatImageProcessor 和 NougatTokenizerFast 两个处理器
"""
Processor class for Nougat.
"""

# 导入所需的类型和模块
from typing import Dict, List, Optional, Union

from transformers.tokenization_utils_base import PreTokenizedInput, TextInput, TruncationStrategy

from ...processing_utils import ProcessorMixin
from ...utils import PaddingStrategy, TensorType


# 定义 NougatProcessor 类，继承自 ProcessorMixin
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

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # 初始化方法
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor


该代码定义了一个 `NougatProcessor` 类，它是 `ProcessorMixin` 类的子类。这个处理器类封装了 `NougatImageProcessor` 和 `NougatTokenizerFast` 两个处理器，提供了统一的接口。

主要功能包括:
1. 设置了类属性 `attributes`、`image_processor_class` 和 `tokenizer_class`。
2. 在初始化方法中，接受 `NougatImageProcessor` 和 `NougatTokenizerFast` 两个实例作为输入参数，并将 `image_processor` 设置为当前处理器。

总的来说，这个类提供了一个统一的接口来处理 Nougat 相关的图像和文本数据。
    # 定义一个接收输入数据并处理的函数
    def __call__(
        self,
        # 图像数据，可以是一个或多个图片
        images=None,
        # 文本数据
        text=None,
        # 是否裁剪边距的标志
        do_crop_margin: bool = None,
        # 是否调整大小的标志
        do_resize: bool = None,
        # 尺寸大小的字典
        size: Dict[str, int] = None,
        # 重采样方式
        resample: "PILImageResampling" = None,  # noqa: F821
        # 是否生成缩略图的标志
        do_thumbnail: bool = None,
        # 是否沿着长轴对齐的标志
        do_align_long_axis: bool = None,
        # 是否填充的标志
        do_pad: bool = None,
        # 是否重新缩放的标志
        do_rescale: bool = None,
        # 重新缩放的因子
        rescale_factor: Union[int, float] = None,
        # 是否归一化的标志
        do_normalize: bool = None,
        # 图像平均值，可以是一个值或列表
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差，可以是一个值或列表
        image_std: Optional[Union[float, List[float]]] = None,
        # 数据格式，通道维度的设置
        data_format: Optional["ChannelDimension"] = "channels_first",  # noqa: F821
        # 输入数据格式，通道维度的设置
        input_data_format: Optional[Union[str, "ChannelDimension"]] = None,  # noqa: F821
        # 文本对，支持不同的输入格式
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        # 文本目标，支持不同的输入格式
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # 文本对目标，支持不同的输入格式
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        # 是否添加特殊token的标志
        add_special_tokens: bool = True,
        # 填充的策略
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断的策略
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长
        stride: int = 0,
        # 是否分词的标志
        is_split_into_words: bool = False,
        # 填充至的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型ids的标志
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力遮罩的标志
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的tokens
        return_overflowing_tokens: bool = False,
        # 是否返回特殊token的遮罩
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回长度
        return_length: bool = False,
        # 是否详细打印信息的标志
        verbose: bool = True,
        if images is None and text is None:
            # 如果既没有图像输入也没有文本输入，则抛出数值错误
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            # 如果有图像输入，则使用图像处理器对图像进行处理
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
            # 如果有文本输入，则使用分词器对文本进行编码
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
            # 如果没有文本输入，则返回图像处理器处理后的结果
            return inputs
        elif images is None:
            # 如果没有图像输入，则返回分词器编码后的结果
            return encodings
        else:
            # 如果既有文本输入又有图像输入，则将图像处理器处理后的结果中的标签设为分词器编码后的输入 ID，然后返回
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 此方法将所有参数转发给 NougatTokenizer 的 `PreTrainedTokenizer.batch_decode` 方法，请参考该方法的文档字符串以获取更多信息
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 此方法将所有参数转发给 NougatTokenizer 的 `PreTrainedTokenizer.decode` 方法，请参考该方法的文档字符串以获取更多信息
        return self.tokenizer.decode(*args, **kwargs)
    # 这个函数用于对生成的输出进行后处理
    def post_process_generation(self, *args, **kwargs):
        """
        # 这个注释解释了这个方法的作用:
        # 这个方法会将所有的参数转发给 NougatTokenizer 的 post_process_generation 方法
        # 更多信息请参考该方法的docstring
        """
        # 将参数转发给 NougatTokenizer 的 post_process_generation 方法并返回结果
        return self.tokenizer.post_process_generation(*args, **kwargs)
```