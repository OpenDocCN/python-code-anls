# `.\models\layoutxlm\tokenization_layoutxlm_fast.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本使用代码
# 在遵守许可证的前提下，您可以使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发
# 没有任何担保或条件，无论是明示的还是暗示的
# 请查阅许可证了解详细信息
""" LayoutXLM 模型的标记类."""


# 导入必要的库
import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

# 导入 HuggingFace 库中的相关模块
from ...tokenization_utils import AddedToken
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, is_sentencepiece_available, logging
from ..xlm_roberta.tokenization_xlm_roberta_fast import (
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    PRETRAINED_VOCAB_FILES_MAP,
    VOCAB_FILES_NAMES,
)

# 检查是否安装了 SentencePiece
if is_sentencepiece_available():
    # 如果安装了 SentencePiece，则导入 LayoutXLMTokenizer
    from .tokenization_layoutxlm import LayoutXLMTokenizer
else:
    # 如果没有安装 SentencePiece，则将 LayoutXLMTokenizer 设置为 None
    LayoutXLMTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)


class LayoutXLMTokenizerFast(PreTrainedTokenizerFast):
    """
    构建“快速”LayoutXLM标记器（由HuggingFace的*tokenizers*库支持）。从
    [`RobertaTokenizer`] 和 [`XLNetTokenizer`]进行调整。基于
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models)。

    该标记器继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应
    参考此超类以获取有关这些方法的更多信息。

    """

    # 定义词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名称
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速标记器类设置为 LayoutXLMTokenizer
    slow_tokenizer_class = LayoutXLMTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        **kwargs,
``` 
    ):
        # Mask token behave like a normal word, i.e. include the space before it
            # 如果是字符串类型的mask_token，则表示mask token与其前面的空格都会被包含进来

        # Create a Mask Token object to handle the mask token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
            # 如果mask_token是字符串类型，则将其转换为AddedToken对象处理，保留空格

        # Call the constructor of the parent class
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            **kwargs,
        )

        # Set the vocab_file attribute
        self.vocab_file = vocab_file
            # 设置vocab_file属性为传入的vocab_file值

        # Set additional properties
        self.cls_token_box = cls_token_box
            # 设置cls_token_box属性为传入的cls_token_box值
        self.sep_token_box = sep_token_box
            # 设置sep_token_box属性为传入的sep_token_box值
        self.pad_token_box = pad_token_box
            # 设置pad_token_box属性为传入的pad_token_box值
        self.pad_token_label = pad_token_label
            # 设置pad_token_label属性为传入的pad_token_label值
        self.only_label_first_subword = only_label_first_subword
            # 设置only_label_first_subword属性为传入的only_label_first_subword值

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # Check if vocab_file exists
        return os.path.isfile(self.vocab_file) if self.vocab_file else False
            # 检查vocab_file属性是否是一个文件存在，如果存在则返回True，否则返回False

    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        # Convert the input text to tokens
        def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
            # If pair exists, create a batched input with text and pair
            batched_input = [(text, pair)] if pair else [text]
            # Encode the batched input using the tokenizer
            encodings = self._tokenizer.encode_batch(
                batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
            )
            # Return the tokens from the first encoding
            return encodings[0].tokens
    # 定义一个私有方法，用于批量编码文本或文本对
    def _batch_encode_plus(
        self,  # 表示该方法是类的方法
        batch_text_or_text_pairs: Union[  # 输入参数，可以是文本列表、文本对列表或预分词后的文本列表
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 表示输入是否是文本对
        boxes: Optional[List[List[List[int]]] = None,  # 包围框的坐标信息
        word_labels: Optional[List[List[int]]] = None,  # 单词级别的标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度
        return_tensors: Optional[str] = None,  # 返回张量
        return_token_type_ids: Optional[bool] = None,  # 返回令牌类型 ID
        return_attention_mask: Optional[bool] = None,  # 返回注意力掩码
        return_overflowing_tokens: bool = False,  # 返回溢出的令牌
        return_special_tokens_mask: bool = False,  # 返回特殊标记掩码
        return_offsets_mapping: bool = False,  # 返回偏移映射
        return_length: bool = False,  # 返回长度
        verbose: bool = True,  # 是否详细输出信息
        **kwargs,  # 其他关键字参数
    def _encode_plus(
        self,  # 表示该方法是类的方法
        text: Union[TextInput, PreTokenizedInput],  # 输入的文本
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的文本对
        boxes: Optional[List[List[int]]] = None,  # 包围框的坐标信息
        word_labels: Optional[List[int]] = None,  # 单词级别的标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度
        return_tensors: Optional[bool] = None,  # 返回张量
        return_token_type_ids: Optional[bool] = None,  # 返回令牌类型 ID
        return_attention_mask: Optional[bool] = None,  # 返回注意力掩码
        return_overflowing_tokens: bool = False,  # 返回溢出的令牌
        return_special_tokens_mask: bool = False,  # 返回特殊标记掩码
        return_offsets_mapping: bool = False,  # 返回偏移映射
        return_length: bool = False,  # 返回长度
        verbose: bool = True,  # 是否详细输出信息
        **kwargs,  # 其他关键字参数
        ) -> BatchEncoding:
        # 将输入组成批量输入
        # 有两种选项：
        # 1) 只有文本，此时文本必须是一个字符串列表
        # 2) 文本 + 文本对，此时文本为一个字符串，文本对为一个字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]  # 将框的信息组成列表
        batched_word_labels = [word_labels] if word_labels is not None else None  # 将单词标签信息组成列表
        # 通过_batch_encode_plus方法批量编码输入
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),
            boxes=batched_boxes,
            word_labels=batched_word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,  # 其他关键字参数
        )

        # 如果返回的张量为None，则可以移除前导的批量轴
        # 如果溢出的标记返回为一批输出，则在这种情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            # 如果返回的批处理输出是列表中的第一个元素，且该元素是列表类型，则取其第一个元素
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,  # 使用批处理输出的编码
            )

        # 检查是否可能存在过长的序列，并发出警告
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output  # 返回批处理输出

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    # 使用特殊标记构建输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def special_tokens(self) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
```