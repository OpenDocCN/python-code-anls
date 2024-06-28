# `.\models\udop\processing_udop.py`

```py
# coding=utf-8
# 设置文件编码格式为 UTF-8，确保代码中的中文和其他非ASCII字符能正确处理
# Copyright 2024 The HuggingFace Inc. team.
# 版权声明，指明代码的版权归属于 HuggingFace Inc. 团队
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 授权，可以自由使用、修改和分发本代码，但需遵守许可证规定
# you may not use this file except in compliance with the License;
# 除非遵守许可证规定，否则不得使用本文件
# You may obtain a copy of the License at
# 可在上述链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证的详细信息链接
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则本软件按"原样"提供，不附带任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解具体的语言及权限的约束和限制
"""
Processor class for UDOP.
UDOP 的处理器类。
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

# 导入必要的模块和类

class UdopProcessor(ProcessorMixin):
    r"""
    Constructs a UDOP processor which combines a LayoutLMv3 image processor and a UDOP tokenizer into a single processor.

    构建 UDOP 处理器，将 LayoutLMv3 图像处理器和 UDOP 分词器结合成一个单一的处理器。

    [`UdopProcessor`] offers all the functionalities you need to prepare data for the model.

    [`UdopProcessor`] 提供了准备数据以供模型使用的所有功能。

    It first uses [`LayoutLMv3ImageProcessor`] to resize, rescale and normalize document images, and optionally applies OCR
    to get words and normalized bounding boxes. These are then provided to [`UdopTokenizer`] or [`UdopTokenizerFast`],
    which turns the words and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
    Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
    classification tasks (such as FUNSD, CORD).

    首先使用 [`LayoutLMv3ImageProcessor`] 调整、重新缩放和归一化文档图像，可选地应用 OCR
    获取单词和归一化边界框。然后提供给 [`UdopTokenizer`] 或 [`UdopTokenizerFast`]，
    将单词和边界框转换为令牌级别的 `input_ids`、`attention_mask`、`token_type_ids` 和 `bbox`。
    可选地，可以提供整数 `word_labels`，将其转换为令牌级别的标签，用于令牌分类任务（如 FUNSD、CORD）。

    Additionally, it also supports passing `text_target` and `text_pair_target` to the tokenizer, which can be used to
    prepare labels for language modeling tasks.

    此外，还支持将 `text_target` 和 `text_pair_target` 传递给分词器，可用于准备语言建模任务的标签。

    Args:
        image_processor (`LayoutLMv3ImageProcessor`):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`UdopTokenizer` or `UdopTokenizerFast`):
            An instance of [`UdopTokenizer`] or [`UdopTokenizerFast`]. The tokenizer is a required input.

    参数:
        image_processor (`LayoutLMv3ImageProcessor`):
            [`LayoutLMv3ImageProcessor`] 的一个实例。图像处理器是必需的输入。
        tokenizer (`UdopTokenizer` or `UdopTokenizerFast`):
            [`UdopTokenizer`] 或 [`UdopTokenizerFast`] 的一个实例。分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]
    # 类属性，包含字符串列表，列出了 UDOP 处理器的属性
    image_processor_class = "LayoutLMv3ImageProcessor"
    # 类属性，指明了图像处理器的类名字串为 "LayoutLMv3ImageProcessor"
    tokenizer_class = ("UdopTokenizer", "UdopTokenizerFast")
    # 类属性，指明了分词器的类名字串为 ("UdopTokenizer", "UdopTokenizerFast")

    def __init__(self, image_processor, tokenizer):
        # 初始化方法，接收图像处理器和分词器作为参数
        super().__init__(image_processor, tokenizer)
        # 调用父类 ProcessorMixin 的初始化方法，传入图像处理器和分词器
    # 定义 __call__ 方法，允许将对象实例作为函数调用
    def __call__(
        self,
        images: Optional[ImageInput] = None,  # 图像输入，可选
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,  # 文本输入，支持多种类型
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,  # 第二段文本输入，可选
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,  # 包含边界框的列表，支持多种维度
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,  # 单词标签列表，可选
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,  # 目标文本输入，支持多种类型
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,  # 第二段目标文本输入，可选
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为 True
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，支持布尔值、字符串或填充策略对象
        truncation: Union[bool, str, TruncationStrategy] = False,  # 截断策略，支持布尔值、字符串或截断策略对象
        max_length: Optional[int] = None,  # 最大长度限制，可选
        stride: int = 0,  # 步长，默认为 0
        pad_to_multiple_of: Optional[int] = None,  # 填充到的倍数，可选
        return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型 ID，可选
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，可选
        return_overflowing_tokens: bool = False,  # 是否返回溢出的标记，默认为 False
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为 False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为 False
        return_length: bool = False,  # 是否返回长度，默认为 False
        verbose: bool = True,  # 是否详细输出信息， 默认为 True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 是否返回张量，可选
    ):
        """
        Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.__call__
        Method defining the behavior of the object when called as a function.
        """

        # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.get_overflowing_images
        def get_overflowing_images(self, images, overflow_to_sample_mapping):
            """
            This method ensures each `input_ids` sample is mapped to its corresponding image in case of overflow.
            """
            images_with_overflow = []
            for sample_idx in overflow_to_sample_mapping:
                images_with_overflow.append(images[sample_idx])

            if len(images_with_overflow) != len(overflow_to_sample_mapping):
                raise ValueError(
                    "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                    f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
                )

            return images_with_overflow

        # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.batch_decode
        def batch_decode(self, *args, **kwargs):
            """
            This method forwards all its arguments to PreTrainedTokenizer's `batch_decode`.
            Please refer to the docstring of that method for more information.
            """
            return self.tokenizer.batch_decode(*args, **kwargs)

        # Copied from transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.decode
        def decode(self, *args, **kwargs):
            """
            This method forwards all its arguments to PreTrainedTokenizer's `decode`.
            Please refer to the docstring of that method for more information.
            """
            return self.tokenizer.decode(*args, **kwargs)

        @property
    # 从 transformers.models.layoutlmv3.processing_layoutlmv3.LayoutLMv3Processor.model_input_names 复制而来的方法
    def model_input_names(self):
        # 返回一个包含固定字符串列表的列表，这些字符串代表模型的输入名称
        return ["input_ids", "bbox", "attention_mask", "pixel_values"]
```