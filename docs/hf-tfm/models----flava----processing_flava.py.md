# `.\models\flava\processing_flava.py`

```py
# 指定文件编码为 UTF-8
# 版权声明和许可信息
"""
FLAVA 的图像/文本处理类
"""

# 导入警告模块
import warnings
# 导入类型提示模块
from typing import List, Optional, Union

# 导入图像工具模块
from ...image_utils import ImageInput
# 导入处理工具模块
from ...processing_utils import ProcessorMixin
# 导入基础标记化工具模块
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 导入实用工具模块
from ...utils import TensorType


# 定义 FLAVA 处理器类，继承自 ProcessorMixin
class FlavaProcessor(ProcessorMixin):
    """
    构建一个 FLAVA 处理器，将 FLAVA 图像处理器和 FLAVA 标记器包装成单个处理器。

    [`FlavaProcessor`] 提供了 [`FlavaImageProcessor`] 和 [`BertTokenizerFast`] 的所有功能。参见
    [`~FlavaProcessor.__call__`] 和 [`~FlavaProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`FlavaImageProcessor`], *optional*): 图像处理器是必需的输入。
        tokenizer ([`BertTokenizerFast`], *optional*): 标记器是必需的输入。
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FlavaImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    # 初始化方法
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 检查是否传入了 feature_extractor 参数，若有则发出警告并将其移除
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 若未提供 image_processor 参数，则尝试使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果没有提供 image_processor 参数，则抛出 ValueError
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果没有提供 tokenizer 参数，则抛出 ValueError
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入图像处理器和标记器
        super().__init__(image_processor, tokenizer)
        # 设置当前处理器为图像处理器
        self.current_processor = self.image_processor
    # 定义一个类方法，接受多个参数，用于处理图像和文本输入，准备输入数据用于模型
    def __call__(
        self,
        images: Optional[ImageInput] = None,  # 图像输入，可选参数
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,  # 文本输入，可选参数
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为True
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充，填充策略
        truncation: Union[bool, str, TruncationStrategy] = False,  # 是否进行截断，截断策略
        max_length: Optional[int] = None,  # 最大长度，可选参数
        stride: int = 0,  # 步幅，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到多少的倍数，可选参数
        return_image_mask: Optional[bool] = None,  # 返回图像遮罩，可选参数
        return_codebook_pixels: Optional[bool] = None,  # 返回码本像素，可选参数
        return_token_type_ids: Optional[bool] = None,  # 返回token类型ID，可选参数
        return_attention_mask: Optional[bool] = None,  # 返回注意力遮罩，可选参数
        return_overflowing_tokens: bool = False,  # 返回溢出的token，默认为False
        return_special_tokens_mask: bool = False,  # 返回特殊token的遮罩，默认为False
        return_offsets_mapping: bool = False,  # 返回偏移映射，默认为False
        return_length: bool = False,  # 返回长度，默认为False
        verbose: bool = True,  # 详细模式，默认为True
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量，可选参数
        **kwargs,  # 其他关键字参数
    ):
        """
        This method uses [`FlavaImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """

        # 如果既没有文本也没有图像输入，则抛出数值错误
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        # 如果有文本输入
        if text is not None:
            # 使用tokenizer方法处理文本
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
        
        # 如果有图像输入
        if images is not None:
            # 使用image_processor方法处理图像
            image_features = self.image_processor(
                images,
                return_image_mask=return_image_mask,
                return_codebook_pixels=return_codebook_pixels,
                return_tensors=return_tensors,
                **kwargs,
            )

        # 如果既有文本输入也有图像输入
        if text is not None and images is not None:
            # 将图像特征合并到编码中，并返回结果
            encoding.update(image_features)
            return encoding
        # 如果只有文本输入
        elif text is not None:
            return encoding
        # 如果只有图像输入
        else:
            # 返回批处理编码
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
    # 将所有参数转发到BertTokenizerFast的`~PreTrainedTokenizer.batch_decode`方法中，详情请参考该方法的文档字符串
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 将所有参数转发到BertTokenizerFast的`~PreTrainedTokenizer.decode`方法中，详情请参考该方法的文档字符串
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入名称的列表，合并了tokenizer和image_processor的模型输入名称后去重
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回特征提取器的类名，会发出未来警告提示，建议使用`image_processor_class`代替
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器对象，会发出未来警告提示，建议使用`image_processor`代替
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```