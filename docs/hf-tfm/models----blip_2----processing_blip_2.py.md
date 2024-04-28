# `.\transformers\models\blip_2\processing_blip_2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非适用法律要求或书面同意，否则软件按"原样"分发，不提供任何形式的担保或条件
# 请查看许可证以获取更多信息
"""
BLIP-2 的处理器类。
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class Blip2Processor(ProcessorMixin):
    r"""
    构建一个 BLIP-2 处理器，将 BLIP 图像处理器和 OPT/T5 分词器封装成一个单一处理器。

    [`BlipProcessor`] 提供了 [`BlipImageProcessor`] 和 [`AutoTokenizer`] 的所有功能。更多信息请参阅 [`~BlipProcessor.__call__`] 和 [`~BlipProcessor.decode`] 的文档字符串。

    Args:
        image_processor (`BlipImageProcessor`):
            [`BlipImageProcessor`] 的一个实例。图像处理器是一个必需的输入。
        tokenizer (`AutoTokenizer`):
            [`PreTrainedTokenizer`] 的一个实例。分词器是一个必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # 从 transformers.models.blip.processing_blip.BlipProcessor.__init__ 复制而来
    def __init__(self, image_processor, tokenizer):
        # 禁用分词器返回 token_type_ids
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为图像处理器
        self.current_processor = self.image_processor

    # 从 transformers.models.blip.processing_blip.BlipProcessor.__call__ 复制而来
    def __call__(
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
    ) -> BatchEncoding:
        """
        这个方法使用 [`BlipImageProcessor.__call__`] 方法来准备图像以供模型使用，
        并使用 [`BertTokenizerFast.__call__`] 方法来准备文本以供模型使用。

        请参考上述两个方法的文档字符串以获取更多信息。
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # 仅获取文本
        if images is None:
            self.current_processor = self.tokenizer
            # 使用 tokenizer 处理文本，返回编码结果
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
            # 使用 tokenizer 处理文本，返回编码结果
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

        if text_encoding is not None:
            # 更新编码结果
            encoding_image_processor.update(text_encoding)

        return encoding_image_processor

    # 从 transformers.models.blip.processing_blip.BlipProcessor.batch_decode 复制而来，替换 BertTokenizerFast 为 PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        这个方法将其所有参数转发给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.batch_decode`] 方法。
        请参考该方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
```  
    # 从transformers.models.blip.processing_blip.BlipProcessor.decode中复制过来，使用BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        此方法将所有参数转发到PreTrainedTokenizer的[`~PreTrainedTokenizer.decode`]。请参考
        此方法的文档字符串以获取更多信息。
        """
        # 调用PreTrainedTokenizer的decode方法进行解码，并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # 从transformers.models.blip.processing_blip.BlipProcessor.model_input_names中复制过来
    def model_input_names(self):
        # 获取tokenizer的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取image_processor的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将tokenizer和image_processor的模型输入名称列表合并成一个列表，并去除重复的元素
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```