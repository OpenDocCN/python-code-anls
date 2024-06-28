# `.\models\align\processing_align.py`

```py
"""
Image/Text processor class for ALIGN
"""

# 导入必要的模块和类
from ...processing_utils import ProcessorMixin  # 导入处理工具混合类
from ...tokenization_utils_base import BatchEncoding  # 导入批量编码类


class AlignProcessor(ProcessorMixin):
    r"""
    Constructs an ALIGN processor which wraps [`EfficientNetImageProcessor`] and
    [`BertTokenizer`]/[`BertTokenizerFast`] into a single processor that interits both the image processor and
    tokenizer functionalities. See the [`~AlignProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more
    information.

    Args:
        image_processor ([`EfficientNetImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`BertTokenizer`, `BertTokenizerFast`]):
            The tokenizer is a required input.
    """

    # 定义类属性
    attributes = ["image_processor", "tokenizer"]  # 类属性列表，包含图像处理器和分词器
    image_processor_class = "EfficientNetImageProcessor"  # 图像处理器类名
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")  # 分词器类名（元组形式）

    def __init__(self, image_processor, tokenizer):
        """
        Initialize the AlignProcessor with image_processor and tokenizer.

        Args:
            image_processor: Instance of EfficientNetImageProcessor.
            tokenizer: Instance of BertTokenizer or BertTokenizerFast.
        """
        super().__init__(image_processor, tokenizer)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[str]: Decoded texts.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Property that combines model input names from both tokenizer and image processor.

        Returns:
            list: List of unique model input names.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```