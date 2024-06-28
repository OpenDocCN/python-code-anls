# `.\models\bros\processing_bros.py`

```
# 设置编码格式为 UTF-8，确保支持中文等多种字符
# 版权声明，指明此代码版权归 HuggingFace Inc. 团队所有，遵循 Apache License 2.0
# 详细许可条款可以在 http://www.apache.org/licenses/LICENSE-2.0 查看
# 如果符合许可协议，可以自由使用、修改和分发本代码
"""
Processor class for Bros.
"""

# 从相关模块导入所需的类和函数
from typing import List, Optional, Union

# 导入自定义的处理工具类
from ...processing_utils import ProcessorMixin
# 导入与标记化相关的基础工具类
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 导入张量类型定义
from ...utils import TensorType


# BrosProcessor 类，继承自 ProcessorMixin 类
class BrosProcessor(ProcessorMixin):
    r"""
    Constructs a Bros processor which wraps a BERT tokenizer.

    [`BrosProcessor`] offers all the functionalities of [`BertTokenizerFast`]. See the docstring of
    [`~BrosProcessor.__call__`] and [`~BrosProcessor.decode`] for more information.

    Args:
        tokenizer (`BertTokenizerFast`, *optional*):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """

    # 类属性，指定可访问的属性名列表
    attributes = ["tokenizer"]
    # 类属性，指定支持的 tokenizer 类名
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    # 初始化方法
    def __init__(self, tokenizer=None, **kwargs):
        # 如果未提供 tokenizer 参数，则抛出 ValueError 异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入 tokenizer 参数
        super().__init__(tokenizer)

    # 实例调用方法，处理文本和标记化的主要功能
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
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
        Processes input text or pre-tokenized input into a format suitable for BERT models.

        Args:
            text (Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], optional):
                The input text to process, can be single or batched inputs.
            add_special_tokens (bool, optional):
                Whether to add special tokens like [CLS], [SEP].
            padding (Union[bool, str, PaddingStrategy], optional):
                Strategy for padding sequences to the same length.
            truncation (Union[bool, str, TruncationStrategy], optional):
                Strategy for truncating sequences to a maximum length.
            max_length (int, optional):
                Maximum length of the returned sequences after truncation and padding.
            stride (int, optional):
                Stride for splitting text into chunks when truncation is applied.
            pad_to_multiple_of (int, optional):
                Pad all sequences to a multiple of this value.
            return_token_type_ids (bool, optional):
                Whether to return token type IDs.
            return_attention_mask (bool, optional):
                Whether to return attention masks.
            return_overflowing_tokens (bool, optional):
                Whether to return overflowing tokens that were truncated.
            return_special_tokens_mask (bool, optional):
                Whether to return a mask indicating special tokens.
            return_offsets_mapping (bool, optional):
                Whether to return offsets mapping tokenized input to original text.
            return_length (bool, optional):
                Whether to return the length of the output sequence.
            verbose (bool, optional):
                Whether to print informative messages during processing.
            return_tensors (Optional[Union[str, TensorType]], optional):
                Type of tensor to return (e.g., 'pt' for PyTorch tensors).

            **kwargs:
                Additional keyword arguments passed to the tokenizer.

        Returns:
            BatchEncoding:
                Processed batch encoding containing tokenized inputs and relevant masks/tensors.
        """
        pass  # 方法体为空，实际功能由子类实现
    ) -> BatchEncoding:
        """
        This method uses `BertTokenizerFast.__call__` to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        # 使用BertTokenizerFast的__call__方法对文本进行处理，生成模型的输入编码
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

        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's `~PreTrainedTokenizer.batch_decode`. Please
        refer to the docstring of this method for more information.
        """
        # 将所有参数转发到BertTokenizerFast的~PreTrainedTokenizer.batch_decode方法
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's `~PreTrainedTokenizer.decode`. Please refer to
        the docstring of this method for more information.
        """
        # 将所有参数转发到BertTokenizerFast的~PreTrainedTokenizer.decode方法
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取tokenizer的模型输入名称列表，并移除重复项后返回
        return list(dict.fromkeys(tokenizer_input_names))
```