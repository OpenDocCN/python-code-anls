# `.\models\layoutxlm\tokenization_layoutxlm_fast.py`

```
# 导入必要的库和模块
import os  # 导入操作系统相关功能
from shutil import copyfile  # 导入复制文件功能
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关模块

# 导入 tokenization_utils 中的部分功能和类
from ...tokenization_utils import AddedToken  
from ...tokenization_utils_base import (
    BatchEncoding,  # 批编码相关类
    EncodedInput,  # 编码输入相关类
    PreTokenizedInput,  # 预分词输入相关类
    TextInput,  # 文本输入相关类
    TextInputPair,  # 文本对输入相关类
    TruncationStrategy,  # 截断策略相关类
)
# 导入 tokenization_utils_fast 中的 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  
# 导入工具函数和常量定义
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, is_sentencepiece_available, logging  
# 从 xlm_roberta.tokenization_xlm_roberta_fast 导入相关常量
from ..xlm_roberta.tokenization_xlm_roberta_fast import (
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,  # 预训练位置嵌入尺寸
    PRETRAINED_VOCAB_FILES_MAP,  # 预训练词汇文件映射
    VOCAB_FILES_NAMES,  # 词汇文件名列表
)

# 如果 sentencepiece 可用，则导入 LayoutXLMTokenizer 类，否则设为 None
if is_sentencepiece_available():
    from .tokenization_layoutxlm import LayoutXLMTokenizer
else:
    LayoutXLMTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)
        # Mask token behave like a normal word, i.e. include the space before it
        # 如果 mask token 是字符串，则将其作为 AddedToken 对象，保留其前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化方法，设置各种特殊标记和文件路径
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

        # 将初始化参数中的词汇文件路径存储在对象属性中
        self.vocab_file = vocab_file

        # 设置额外的属性值
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查词汇文件是否存在，以确定是否可以保存缓慢的分词器
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

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
        # 调用对象本身，实现文本编码和处理的方法
        pass  # 这里只是占位符，实际应该有具体的处理逻辑

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 创建批量输入列表
        batched_input = [(text, pair)] if pair else [text]
        # 使用底层的分词器对批量输入进行编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        # 返回第一个编码结果的 token 列表
        return encodings[0].tokens
    # 定义一个方法 _batch_encode_plus，用于批量编码文本或文本对
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 标志是否为文本对
        boxes: Optional[List[List[List[int]]]] = None,  # 文本框的坐标信息
        word_labels: Optional[List[List[int]]] = None,  # 单词标签列表
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到的倍数
        return_tensors: Optional[str] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细输出
        **kwargs,  # 其他关键字参数
    ):
    
        # 定义一个方法 _encode_plus，用于编码单个文本或文本对
        def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput],  # 文本输入
            text_pair: Optional[PreTokenizedInput] = None,  # 第二个文本输入（可选）
            boxes: Optional[List[List[int]]] = None,  # 文本框的坐标信息
            word_labels: Optional[List[int]] = None,  # 单词标签列表
            add_special_tokens: bool = True,  # 是否添加特殊标记
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
            max_length: Optional[int] = None,  # 最大长度限制
            stride: int = 0,  # 步长
            pad_to_multiple_of: Optional[int] = None,  # 填充到的倍数
            return_tensors: Optional[bool] = None,  # 返回的张量类型
            return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID
            return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
            return_overflowing_tokens: bool = False,  # 是否返回溢出的token
            return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
            return_offsets_mapping: bool = False,  # 是否返回偏移映射
            return_length: bool = False,  # 是否返回长度
            verbose: bool = True,  # 是否详细输出
            **kwargs,  # 其他关键字参数
        ):
    ) -> BatchEncoding:
        # 将输入处理为批量输入
        # 两个选项：
        # 1) 只有文本，如果文本是一个字符串列表
        # 2) 文本 + 文本对，其中文本是一个字符串，文本对是一个字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]  # 将盒子（框）数据转为列表形式
        batched_word_labels = [word_labels] if word_labels is not None else None  # 将单词标签数据转为列表形式或保持为None
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),  # 判断是否有文本对
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
            **kwargs,
        )

        # 如果返回的张量为None，则删除前导的批处理轴
        # 如果有溢出的tokens，则以批处理形式返回它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        # 对编码后的输入进行填充处理
        # encoded_inputs: 包含编码输入的字典或BatchEncoding对象
        # max_length: 最大长度限制（可选）
        # padding_strategy: 填充策略，默认不填充
        # pad_to_multiple_of: 填充到的倍数（可选）
        # return_attention_mask: 是否返回注意力掩码（可选）
        ...
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        # 构建带有特殊标记的输入
        # token_ids_0: 第一个句子的标记ID列表
        # token_ids_1: 第二个句子的标记ID列表（可选）
        ...
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens added.
        """

        # If only one sequence is provided, add `<s>` (CLS) token, sequence tokens, and `</s>` (SEP) token
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # For pairs of sequences, concatenate tokens with appropriate special tokens
        cls = [self.cls_token_id]  # CLS token
        sep = [self.sep_token_id]  # SEP token
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
            `List[int]`: List of zeros indicating the token types.

        """

        sep = [self.sep_token_id]  # SEP token
        cls = [self.cls_token_id]  # CLS token

        # If only one sequence is provided, return a list of zeros for token type ids
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # For pairs of sequences, return a list of zeros for token type ids
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer vocabulary to the specified directory.

        Args:
            save_directory (str):
                Directory where the vocabulary will be saved.
            filename_prefix (str, *optional*):
                Prefix to prepend to the saved vocabulary filename.

        Returns:
            Tuple containing the path to the saved vocabulary file.
        """

        # Check if the fast tokenizer can save the vocabulary for a slow tokenizer
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Check if save_directory exists and is a directory
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return  # Return if directory is not valid

        # Construct the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocabulary file path is not the same as the desired output path, copy the vocabulary file
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```