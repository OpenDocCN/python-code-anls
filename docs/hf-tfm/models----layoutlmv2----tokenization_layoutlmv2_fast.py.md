# `.\models\layoutlmv2\tokenization_layoutlmv2_fast.py`

```
# 设定编码方式为 UTF-8
# 版权声明 2021 年 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何明示或暗示的保证或条件。
# 有关更多详细信息，请参阅许可证。
"""
LayoutLMv2 的快速分词器类。覆盖了慢分词器类的两个方法：_batch_encode_plus 和 _encode_plus，其中使用了 Rust 分词器。
"""

import json
from typing import Dict, List, Optional, Tuple, Union

# 导入正则化工具
from tokenizers import normalizers

# 导入基础分词器和快速分词器的相关工具和类
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    PreTokenizedInput,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

# 导入 LayoutLMv2 的快速分词器类
from ...tokenization_utils_fast import PreTrainedTokenizerFast

# 导入日志工具和 LayoutLMv2 分词器的相关类
from ...utils import add_end_docstrings, logging
from .tokenization_layoutlmv2 import (
    LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING,
    LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    LayoutLMv2Tokenizer,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇表和分词器文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇表文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv2-base-uncased": (
            "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "microsoft/layoutlmv2-base-uncased": (
            "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv2-base-uncased": 512,
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlmv2-base-uncased": {"do_lower_case": True},
}


class LayoutLMv2TokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个基于 HuggingFace 的 *tokenizers* 库支持的"快速" LayoutLMv2 分词器。基于 WordPiece。

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取更多关于这些方法的信息。
    # 初始化词汇文件名列表，使用预定义的全局常量
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射，包含文件名到预训练模型配置的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型初始化的配置，包含了预定义的配置参数
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 将预训练模型的位置嵌入大小赋值给 max_model_input_sizes
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 将 LayoutLMv2Tokenizer 类赋值给 slow_tokenizer_class
    slow_tokenizer_class = LayoutLMv2Tokenizer

    # 初始化函数，用于创建一个 LayoutLMv2Tokenizer 对象
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 调用父类的初始化方法，设置相关属性
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # 从 backend_tokenizer 中获取当前的标准化状态
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())

        # 检查预处理器的小写和去重音选项是否与参数中的设置一致，若不一致则更新预处理器状态
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            # 获取预处理器的类，并更新参数
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            # 实例化新的预处理器对象
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        # 设置实例的属性
        self.do_lower_case = do_lower_case
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

    # 将函数的装饰器添加到当前类中
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 使用装饰器添加文档字符串，其中包含 LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING 和 LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING 的内容
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
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
    # batch_encode_plus 方法用于批量编码文本或文本对，并返回编码后的结果
        ) -> BatchEncoding:
        # 为了向后兼容 'truncation_strategy', 'pad_to_max_length' 参数
        # 调用内部方法获取填充和截断策略以及其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法进行批量编码处理，并返回结果
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
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

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 将输入文本和可选的配对文本构成批量输入
        batched_input = [(text, pair)] if pair else [text]
        # 使用内部的分词器对批量输入进行编码处理
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        # 返回第一个编码结果的 tokens 属性，即分词后的文本列表
        return encodings[0].tokens

    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
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
        # 使用特定的文本和配对文本、框、单词标签等信息进行编码处理
        # 设置默认添加特殊标记，以及填充和截断策略
        # 返回编码后的结果，根据参数选择是否返回张量形式的数据
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及其他相关参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_encode_plus` 进行编码
        return self._encode_plus(
            text=text,
            boxes=boxes,
            text_pair=text_pair,
            word_labels=word_labels,
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

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 定义函数参数text，可以是单文本或预分词文本输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选参数，用于处理文本对
        boxes: Optional[List[List[int]]] = None,  # 可选参数，用于处理边界框信息
        word_labels: Optional[List[int]] = None,  # 可选参数，用于处理单词级别标签
        add_special_tokens: bool = True,  # 是否添加特殊token，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 可选参数，最大长度限制
        stride: int = 0,  # 步长，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 可选参数，填充到某个倍数
        return_tensors: Optional[bool] = None,  # 可选参数，返回张量形式
        return_token_type_ids: Optional[bool] = None,  # 可选参数，返回token类型IDs
        return_attention_mask: Optional[bool] = None,  # 可选参数，返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token，默认不返回
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码，默认不返回
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认不返回
        return_length: bool = False,  # 是否返回长度，默认不返回
        verbose: bool = True,  # 是否显示详细信息，默认为True
        **kwargs,  # 其他未指定的关键字参数
    ) -> BatchEncoding:
        # 将输入文本处理为批次输入
        # 有两种选项：
        # 1) 只有text，此时text必须是str的列表
        # 2) text + text_pair，此时text是str，text_pair是str的列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        # 将边界框信息处理为批次边界框
        batched_boxes = [boxes]
        # 将单词级别标签处理为批次标签
        batched_word_labels = [word_labels] if word_labels is not None else None
        # 使用_batch_encode_plus方法处理批次输入
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),  # 是否是文本对
            boxes=batched_boxes,  # 批次边界框信息
            word_labels=batched_word_labels,  # 批次单词级别标签
            add_special_tokens=add_special_tokens,  # 是否添加特殊token
            padding_strategy=padding_strategy,  # 填充策略
            truncation_strategy=truncation_strategy,  # 截断策略
            max_length=max_length,  # 最大长度限制
            stride=stride,  # 步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到某个倍数
            return_tensors=return_tensors,  # 是否返回张量形式
            return_token_type_ids=return_token_type_ids,  # 是否返回token类型IDs
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的token
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊token的掩码
            return_offsets_mapping=return_offsets_mapping,  # 是否返回偏移映射
            return_length=return_length,  # 是否返回长度
            verbose=verbose,  # 是否显示详细信息
            **kwargs,  # 其他未指定的关键字参数
        )

        # 如果返回的张量为None，并且不返回溢出的token，则移除批次输出的前导批次轴
        # 如果返回的值为批次的输出，则在这种情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        # 检查并警告处理后序列过长的情况
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        # 返回处理后的批次输出
        return batched_output
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        """
        Pad encoded inputs according to specified parameters.

        Args:
            encoded_inputs (Union[Dict[str, EncodedInput], BatchEncoding]):
                Dictionary or batch encoding containing encoded inputs.
            max_length (Optional[int], *optional*):
                Maximum length to pad or truncate the sequences.
            padding_strategy (PaddingStrategy):
                Strategy for padding the sequences.
            pad_to_multiple_of (Optional[int], *optional*):
                Pad to a multiple of this value.
            return_attention_mask (Optional[bool], *optional*):
                Whether to return attention mask.

        Returns:
            Union[Dict[str, torch.Tensor], BatchEncoding]:
                Padded and encoded inputs.
        """

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequences by adding special tokens.

        Args:
            token_ids_0 (List[int]):
                List of IDs for the first sequence.
            token_ids_1 (List[int], *optional*):
                Optional list of IDs for the second sequence.

        Returns:
            List[int]: List of input IDs with added special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from sequences for sequence-pair classification tasks.

        Args:
            token_ids_0 (List[int]):
                List of IDs for the first sequence.
            token_ids_1 (List[int], *optional*):
                Optional list of IDs for the second sequence.

        Returns:
            List[int]: List of token type IDs indicating the sequence segments.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the tokenizer model.

        Args:
            save_directory (str):
                Directory to save the vocabulary files.
            filename_prefix (Optional[str], *optional*):
                Prefix for the vocabulary filenames.

        Returns:
            Tuple[str]: Tuple containing the paths of the saved files.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```