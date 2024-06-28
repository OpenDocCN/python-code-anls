# `.\models\layoutxlm\tokenization_layoutxlm.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
""" Tokenization classes for LayoutXLM model."""


import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as spm  # 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
from ..xlm_roberta.tokenization_xlm_roberta import (  # 导入 XLM-Roberta 的 tokenization 模块相关内容
    PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES,
    PRETRAINED_VOCAB_FILES_MAP,
    SPIECE_UNDERLINE,
    VOCAB_FILES_NAMES,
)


logger = logging.get_logger(__name__)  # 获取 logger 对象


class LayoutXLMTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件名
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置最大模型输入大小
    model_input_names = ["input_ids", "attention_mask"]  # 设置模型输入名称列表

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",  # 开始词标记
        eos_token="</s>",  # 结束词标记
        sep_token="</s>",  # 分隔词标记
        cls_token="<s>",  # 类别标记
        unk_token="<unk>",  # 未知词标记
        pad_token="<pad>",  # 填充词标记
        mask_token="<mask>",  # 掩码词标记
        cls_token_box=[0, 0, 0, 0],  # 类别标记边界框
        sep_token_box=[1000, 1000, 1000, 1000],  # 分隔词标记边界框
        pad_token_box=[0, 0, 0, 0],  # 填充词标记边界框
        pad_token_label=-100,  # 填充词标签
        only_label_first_subword=True,  # 仅标记第一个子词
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # sentencepiece 模型参数
        **kwargs,  # 其他关键字参数
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        # 如果 mask_token 是字符串，则创建一个带有特殊属性且会去除左侧空格的 AddedToken 对象
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        # 如果 sp_model_kwargs 为 None，则初始化为空字典，否则使用传入的参数
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 使用 SentencePieceProcessor 初始化 self.sp_model 对象，并加载给定的 vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # 确保 fairseq 的词汇表和 spm 的词汇表必须是“对齐”的关系
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # fairseq 词汇表的偏移量，用于实现 token-to-id 对齐
        self.fairseq_offset = 1

        # 添加 "<mask>" token 到 fairseq 的词汇表映射中，并计算其对应的 id
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 设置额外的属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

        # 调用父类的初始化方法，传递参数和关键字参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def __getstate__(self):
        # 创建当前对象的状态字典副本
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None，以防止序列化时保存 SentencePieceProcessor 对象
        state["sp_model"] = None
        # 将 sp_model_proto 设置为当前 sp_model 的序列化模型协议
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        # 恢复对象的状态字典
        self.__dict__ = d

        # 兼容旧版本的代码，如果不存在 sp_model_kwargs，则初始化为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 重新创建 sp_model 对象，并从序列化模型协议中加载 sp_model 的状态
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate token type IDs from a pair of sequences. Token type IDs distinguish between two sequences in a model input.
        For XLM-RoBERTa, token type IDs are:

        - single sequence: 0s for all tokens
        - pair of sequences: 0s for the tokens from the first sequence, 1s for the tokens from the second sequence

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs for the second sequence in a pair.

        Returns:
            `List[int]`: List of token type IDs indicating the sequence membership of each token.
        """

        # If only one sequence is provided, return token type IDs with only the special tokens
        if token_ids_1 is None:
            return [0] * len(token_ids_0)

        # For a pair of sequences, generate token type IDs distinguishing the two sequences
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)
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

        # Initialize special tokens
        sep = [self.sep_token_id]  # List containing the separator token ID
        cls = [self.cls_token_id]  # List containing the classification token ID

        if token_ids_1 is None:
            # If only one sequence is provided, return a list of zeros for its combined length with special tokens
            return len(cls + token_ids_0 + sep) * [0]
        else:
            # If two sequences are provided, return a list of zeros for their combined length with special tokens
            return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # Calculate and return the vocabulary size including an additional token for <mask>
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        # Generate and return a dictionary mapping tokens to their IDs
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)  # Update with any additional tokens
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize the input text using SentencePiece model and return a list of tokens as strings
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]  # Return ID if token is in pre-defined mapping
        spm_id = self.sp_model.PieceToId(token)

        # Return ID adjusted by fairseq offset for unknown tokens returned by SentencePiece
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]  # Return token if index is in pre-defined mapping
        return self.sp_model.IdToPiece(index - self.fairseq_offset)  # Return token from SentencePiece model

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            # Log an error if the save directory is not valid
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return  # Return None if directory is invalid
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            # Copy the existing vocabulary file if paths differ and the current file exists
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            # Otherwise, write the serialized SentencePiece model to the output vocabulary file
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    # 定义一个方法，使对象可被调用，接受多种文本输入形式和相关参数
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
        # 实现文本和文本对的编码，并根据需求进行特殊标记、填充和截断
        ...

    # 定义一个批处理编码方法，接受多个文本或文本对输入及相关参数
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
        # 对批量文本或文本对进行编码，支持对特殊标记、填充策略和截断策略的控制
        ...
    ) -> BatchEncoding:
        # 如果用户请求返回偏移映射，则抛出未实现的错误，因为 Python tokenizers 不支持这个功能
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用 _batch_prepare_for_model 方法，准备输入数据并返回模型输入的批编码
        batch_outputs = self._batch_prepare_for_model(
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
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # 将批处理输出包装成 BatchEncoding 对象并返回
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,
        is_pair: bool = None,
        boxes: Optional[List[List[int]]] = None,
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
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        # Initialize an empty dictionary to store batch outputs
        batch_outputs = {}

        # Iterate over examples in the batch, where each example consists of text or text pairs and associated boxes
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            batch_text_or_text_pair, boxes_example = example

            # Prepare inputs for the model
            outputs = self.prepare_for_model(
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,  # First sequence or single sequence
                batch_text_or_text_pair[1] if is_pair else None,  # Second sequence (if pair) or None
                boxes_example,  # Boxes associated with the example
                word_labels=word_labels[idx] if word_labels is not None else None,  # Word labels if provided
                add_special_tokens=add_special_tokens,  # Whether to add special tokens
                padding=PaddingStrategy.DO_NOT_PAD.value,  # Padding strategy
                truncation=truncation_strategy.value,  # Truncation strategy
                max_length=max_length,  # Maximum sequence length
                stride=stride,  # Stride for overflowing tokens
                pad_to_multiple_of=None,  # Pad to multiple of this value (will pad in batch)
                return_attention_mask=False,  # Do not return attention masks here (batch level operation)
                return_token_type_ids=return_token_type_ids,  # Whether to return token type IDs
                return_overflowing_tokens=return_overflowing_tokens,  # Whether to return overflowing tokens
                return_special_tokens_mask=return_special_tokens_mask,  # Whether to return special tokens mask
                return_length=return_length,  # Whether to return length of sequences
                return_tensors=None,  # Convert the batch to tensors at the end
                prepend_batch_axis=False,  # Do not prepend batch axis
                verbose=verbose,  # Verbosity level
            )

            # Aggregate outputs into batch_outputs dictionary
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Pad the batch outputs according to specified padding strategy and parameters
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,  # Padding strategy
            max_length=max_length,  # Maximum sequence length for padding
            pad_to_multiple_of=pad_to_multiple_of,  # Pad to multiple of this value
            return_attention_mask=return_attention_mask,  # Whether to return attention mask
        )

        # Convert batch_outputs dictionary to BatchEncoding object with specified tensor type
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return the prepared batch outputs
        return batch_outputs
    # 定义一个方法 `_encode_plus`，用于将文本和可能的配对文本、文本框、词标签等编码为模型输入
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 主要文本输入，可以是文本或预分词的输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的配对文本输入，预分词的输入格式
        boxes: Optional[List[List[int]]] = None,  # 可选的文本框列表，每个文本框由四个整数表示
        word_labels: Optional[List[int]] = None,  # 可选的词标签列表，整数表示每个词的标签
        add_special_tokens: bool = True,  # 是否添加特殊的标记符号（如CLS、SEP）
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口的步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回序列长度
        verbose: bool = True,  # 是否详细输出信息
        **kwargs,  # 其他关键字参数
    ) -> BatchEncoding:
        # 如果设置了返回偏移映射，则抛出未实现错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 调用 `prepare_for_model` 方法，准备输入以供模型使用，返回 `BatchEncoding` 对象
        return self.prepare_for_model(
            text=text,  # 主要文本输入
            text_pair=text_pair,  # 配对文本输入
            boxes=boxes,  # 文本框列表
            word_labels=word_labels,  # 词标签列表
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding=padding_strategy.value,  # 填充策略的值
            truncation=truncation_strategy.value,  # 截断策略的值
            max_length=max_length,  # 最大长度限制
            stride=stride,  # 滑动窗口步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到指定的倍数
            return_tensors=return_tensors,  # 返回的张量类型
            prepend_batch_axis=True,  # 是否在返回的张量中添加批次维度
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回token类型IDs
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的token
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊token的掩码
            return_length=return_length,  # 是否返回序列长度
            verbose=verbose,  # 是否详细输出信息
        )

    # 使用装饰器添加关于 `LAYOUTXLM_ENCODE_KWARGS_DOCSTRING` 的文档字符串
    @add_end_docstrings(LAYOUTXLM_ENCODE_KWARGS_DOCSTRING)
    def prepare_for_model(
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
        prepend_batch_axis: bool = False,
        **kwargs,
    ):
        """
        准备输入以供模型使用的方法。

        参数:
        - text: 输入文本，可以是未分词或预分词的输入。
        - text_pair: 可选的第二个文本输入，用于处理文本对（如句子对任务）。
        - boxes: 可选的边界框列表，用于处理与文本相关的图像区域。
        - word_labels: 可选的单词级别标签列表。
        - add_special_tokens: 是否添加特殊的语言模型令牌（如CLS和SEP）。
        - padding: 控制填充输入序列的方式，可以是布尔值、字符串或填充策略对象。
        - truncation: 控制截断输入序列的方式，可以是布尔值、字符串或截断策略对象。
        - max_length: 输入序列的最大长度限制。
        - stride: 截断或填充时的步长。
        - pad_to_multiple_of: 如果指定，将输入填充到该数的倍数。
        - return_tensors: 控制返回的张量类型。
        - return_token_type_ids: 是否返回token_type_ids。
        - return_attention_mask: 是否返回attention_mask。
        - return_overflowing_tokens: 是否返回溢出的token。
        - return_special_tokens_mask: 是否返回特殊token的mask。
        - return_offsets_mapping: 是否返回token在原始输入中的偏移映射。
        - return_length: 是否返回输入长度。
        - verbose: 是否打印详细信息。
        - prepend_batch_axis: 是否在返回张量中添加批处理维度。
        - **kwargs: 其他未明确列出的参数。
        """
        ...

    def truncate_sequences(
        self,
        ids: List[int],
        token_boxes: List[List[int]],
        pair_ids: Optional[List[int]] = None,
        pair_token_boxes: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ):
        """
        截断序列的方法。

        参数:
        - ids: 输入序列的token IDs。
        - token_boxes: 每个token的边界框。
        - pair_ids: 可选的第二个序列的token IDs，用于处理序列对。
        - pair_token_boxes: 可选的第二个序列的边界框列表。
        - labels: 可选的标签列表。
        - num_tokens_to_remove: 要移除的token数量。
        - truncation_strategy: 截断策略，如"longest_first"等。
        - stride: 截断时的步长。
        """
        ...

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        """
        填充方法，用于将输入编码填充到相同长度。

        参数:
        - encoded_inputs: 编码后的输入，可以是单个EncodedInput对象或BatchEncoding对象。
        - max_length: 填充后的最大长度限制。
        - padding_strategy: 填充策略对象，控制如何进行填充。
        - pad_to_multiple_of: 如果指定，将填充到该数的倍数。
        - return_attention_mask: 是否返回attention_mask。
        """
        ...
```