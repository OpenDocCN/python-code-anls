# `.\models\flava\processing_flava.py`

```
# 设置文件编码为UTF-8
# 版权声明
# 根据Apache License, Version 2.0 (许可证)的规定，除非符合许可证的条款，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发，
# 没有任何明示或暗示的保证或条件。请参阅许可证获取特定语言的权限和限制

"""
Image/Text processor class for FLAVA
"""

# 导入模块
import warnings
from typing import List, Optional, Union

# 导入自定义模块
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

# 定义FlavaProcessor类，并继承ProcessorMixin
class FlavaProcessor(ProcessorMixin):
    r"""
    Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

    [`FlavaProcessor`] offers all the functionalities of [`FlavaImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~FlavaProcessor.__call__`] and [`~FlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`FlavaImageProcessor`], *optional*): The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*): The tokenizer is a required input.
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FlavaImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    # 定义初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果kwargs中包含feature_extractor，则发出警告，此参数将在v5中被弃用
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果image_processor未传入，则使用feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果image_processor仍然为None，则抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果tokenizer未传入，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类初始化方法，传入image_processor和tokenizer
        super().__init__(image_processor, tokenizer)
        # 初始化current_processor属性为image_processor
        self.current_processor = self.image_processor
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_image_mask: Optional[bool] = None,
        return_codebook_pixels: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """
        This method uses [`FlavaImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """

        # 检查是否同时未指定文本和图像，若是，则抛出数值错误异常
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        # 如果存在文本，则使用 tokenizer 处理文本数据
        if text is not None:
            encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        
        # 如果存在图像，则使用 image_processor 处理图像数据
        if images is not None:
            image_features = self.image_processor(
                images,
                return_image_mask=return_image_mask,
                return_codebook_pixels=return_codebook_pixels,
                return_tensors=return_tensors,
                **kwargs,
            )

        # 如果同时存在文本和图像，则将图像特征更新到文本编码中并返回结果
        if text is not None and images is not None:
            encoding.update(image_features)
            return encoding
        # 如果仅存在文本，则直接返回文本编码结果
        elif text is not None:
            return encoding
        # 如果仅存在图像，则创建一个 BatchEncoding 对象返回图像特征
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
    # 将所有参数转发给 BertTokenizerFast 的 `~PreTrainedTokenizer.batch_decode` 方法，并返回结果
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 将所有参数转发给 BertTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法，并返回结果
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，这里使用了去重操作
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回特征提取器的类别，已被标记为废弃，建议使用 `image_processor_class` 替代
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器，已被标记为废弃，建议使用 `image_processor` 替代
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```