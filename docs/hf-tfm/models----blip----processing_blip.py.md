# `.\transformers\models\blip\processing_blip.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，此代码归 The HuggingFace Inc. 团队所有

# Apache 许可证版本 2.0，除非遵守许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，软件将按"原样"分发，
# 不附带任何形式的保证或条件。
# 有关许可证的更多信息，请参阅许可证。

"""
Blip 的处理器类。
"""

from typing import List, Optional, Union  # 导入类型提示

from ...image_utils import ImageInput  # 导入图像输入模块
from ...processing_utils import ProcessorMixin  # 导入处理器混合模块
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy  # 导入令牌化工具基础模块
from ...utils import TensorType  # 导入张量类型


class BlipProcessor(ProcessorMixin):  # 定义 BlipProcessor 类，继承自 ProcessorMixin 类
    r"""
    构建一个 BLIP 处理器，将 BERT 分词器和 BLIP 图像处理器封装成一个单一处理器。

    [`BlipProcessor`] 提供了 [`BlipImageProcessor`] 和 [`BertTokenizerFast`] 的所有功能。更多信息请参见 [`~BlipProcessor.__call__`] 和 [`~BlipProcessor.decode`] 的文档字符串。

    Args:
        image_processor (`BlipImageProcessor`):
            [`BlipImageProcessor`] 的一个实例。图像处理器是必需的输入。
        tokenizer (`BertTokenizerFast`):
            ['BertTokenizerFast`] 的一个实例。分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]  # 定义属性列表

    # 图像处理器类和分词器类的定义
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor, tokenizer):  # 定义初始化方法
        tokenizer.return_token_type_ids = False  # 禁用分词器的返回令牌类型 ID 功能
        super().__init__(image_processor, tokenizer)  # 调用父类的初始化方法
        self.current_processor = self.image_processor  # 设置当前处理器为图像处理器

    def __call__(  # 定义调用方法
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
```  
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        # 如果既没有传入图片也没有传入文本，则抛出数值错误
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # 仅获取文本
        if images is None:
            self.current_processor = self.tokenizer
            # 使用 tokenizer 处理文本，返回文本编码
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            return text_encoding

        # 添加像素值
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)

        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_encoding = None

        # 如果存在文本编码，则更新图像处理器
        if text_encoding is not None:
            encoding_image_processor.update(text_encoding)

        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 将所有参数传递给 BertTokenizerFast 的 batch_decode 方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 BertTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法。请参阅该方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        # 调用 BertTokenizerFast 实例的 decode 方法，并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，包括 tokenizer 和 image_processor 的输入名称
    @property
    def model_input_names(self):
        # 获取 tokenizer 的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取 image_processor 的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将两个列表合并并去重，返回结果
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```