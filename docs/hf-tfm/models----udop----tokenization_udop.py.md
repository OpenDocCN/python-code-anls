# `.\models\udop\tokenization_udop.py`

```py
# 定义了编码为 UTF-8
# 版权声明 2024 年由 HuggingFace Inc. 团队所有
# 使用 Apache 许可证 2.0 版本授权，除非符合许可证，否则不得使用此文件
# 可以在以下网址获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，按“原样”分发软件，无任何形式的明示或暗示担保或条件
# 详见许可证，限制和限制条件
""" Tokenization classes for UDOP model."""

# 导入标准库模块
import os
import re
import warnings
# 从 shutil 模块中导入 copyfile 函数
from shutil import copyfile
# 从 typing 模块导入各种类型注解
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 sentencepiece 库，用于分词
import sentencepiece as spm

# 导入 HuggingFace 库中的相关模块和类
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
# 导入 HuggingFace 库中的工具类和函数
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging

# 获取 logger 实例，用于记录日志
logger = logging.get_logger(__name__)

# SentencePiece 模型中用于表示单词起始的符号
SPIECE_UNDERLINE = "▁"

# SentencePiece 模型的文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型的文件映射，包括 spiece.model 和 tokenizer.json 文件的下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/udop-large": "https://huggingface.co/microsoft/udop-large/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "microsoft/udop-large": "https://huggingface.co/microsoft/udop-large/resolve/main/tokenizer.json",
    },
}

# UdopTokenizer 类，继承自 PreTrainedTokenizer 类
class UdopTokenizer(PreTrainedTokenizer):
    """
    从 LayoutXLMTokenizer 和 T5Tokenizer 改编而来。基于 SentencePiece 实现的 tokenizer。

    继承自 PreTrainedTokenizer 类，该类包含大多数主要方法。用户应参考超类以获取有关这些方法的更多信息。

    属性:
        sp_model (`SentencePieceProcessor`):
            每次转换（字符串、token 和 ID）所使用的 SentencePiece 处理器。
    """

    # 词汇文件的名称映射
    vocab_files_names = VOCAB_FILES_NAMES

    # 预训练模型的文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        legacy=True,
        add_prefix_space=True,
        **kwargs,
    ) -> None:
        # 如果 eos_token 是字符串，则将其封装成特殊的 AddedToken 对象，否则保持原样
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则将其封装成特殊的 AddedToken 对象，否则保持原样
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 sep_token 是字符串，则将其封装成特殊的 AddedToken 对象，否则保持原样
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果 pad_token 是字符串，则将其封装成特殊的 AddedToken 对象，否则保持原样
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

        # 设置 legacy 属性
        self.legacy = legacy
        # 设置 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space
        # 如果 sp_model_kwargs 为 None，则设置为空字典，否则使用传入的 sp_model_kwargs
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置 vocab_file 属性
        self.vocab_file = vocab_file

        # 使用 sp_model_kwargs 创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载指定的 vocab_file 到 SentencePieceProcessor 对象中
        self.sp_model.Load(vocab_file)

        # 设置额外的属性
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

        # 调用父类的初始化方法，传入相应参数
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    # 返回当前 SentencePieceProcessor 对象的词汇大小
    def vocab_size(self):
        return len(self.sp_model)

    # 从 T5Tokenizer 类中复制而来的方法，获取词汇表的字典表示
    def get_vocab(self):
        # 创建词汇表的字典，将 token 到 id 的映射关系逆转
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 添加额外的特殊 token 编码器映射关系
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 从 T5Tokenizer 类中复制而来的方法，获取特殊 token 的 mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # If the token list already has special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Normal case: adding special tokens to token_ids_0 and token_ids_1
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]  # Append 1 after token_ids_0 for the special token
        else:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]  # Append 1 after both token_ids_0 and token_ids_1 for their special tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_sentinel_tokens
    def get_sentinel_tokens(self):
        """
        Retrieves sentinel tokens from the list of additional special tokens.

        Returns:
            list: List of sentinel tokens identified by regex pattern "<extra_id_\d+>".
        """
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)), self.additional_special_tokens))
        )

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_sentinel_token_ids
    def get_sentinel_token_ids(self):
        """
        Retrieves token IDs for sentinel tokens using the tokenizer's vocabulary.

        Returns:
            list: List of token IDs corresponding to sentinel tokens.
        """
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._add_eos_if_not_present
    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """
        Adds an end-of-sequence (EOS) token to token_ids if it's not already present.

        Args:
            token_ids (List[int]): List of token IDs.

        Returns:
            List[int]: List of token IDs with EOS appended if not already present.
        """
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            # Warn if the sequence already ends with EOS
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            # Append EOS token to the end of token_ids
            return token_ids + [self.eos_token_id]

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates token type IDs for sequences, distinguishing between token_ids_0 and token_ids_1.

        Args:
            token_ids_0 (List[int]): List of token IDs for the first sequence.
            token_ids_1 (List[int], optional): List of token IDs for the second sequence (if exists).

        Returns:
            List[int]: List of token type IDs where 0 corresponds to token_ids_0 and 1 to token_ids_1 (if provided).
        """
    # 返回一个用于序列对分类任务的掩码。T5 不使用 token type ids，因此返回一个全为零的列表。
    def create_model_input_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """

        # EOS 标记列表，包含一个结束标记的 ID
        eos = [self.eos_token_id]

        # 如果没有第二个序列，则返回第一个序列加上 EOS 标记的长度的零列表
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]

        # 否则，返回两个序列加上各自的 EOS 标记的长度的零列表
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    # 从 T5Tokenizer 类的 build_inputs_with_special_tokens 方法复制而来
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果 token_ids_1 为 None，则直接返回 token_ids_0
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            # 否则为 token_ids_1 也添加 EOS 标记后返回两个列表的连接
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    # 从 T5Tokenizer 类的 __getstate__ 方法复制而来
    def __getstate__(self):
        """
        Serialize the T5Tokenizer instance, preparing it for pickling.
        """
        # 复制实例字典并设置 sp_model 为 None，然后返回状态
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 从 T5Tokenizer 类的 __setstate__ 方法复制而来
    def __setstate__(self, d):
        """
        Deserialize and restore a previously serialized T5Tokenizer instance.
        """
        # 使用传入的字典 d 恢复实例的状态，然后重新加载 sp_model
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 从 T5Tokenizer 类的 tokenize 方法复制而来
    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        # 如果 legacy 标志为真或者文本长度为零，则调用父类的 tokenize 方法并返回结果
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        # 替换文本中的 SPIECE_UNDERLINE 为空格
        text = text.replace(SPIECE_UNDERLINE, " ")

        # 如果 add_prefix_space 为真，则在文本前添加 SPIECE_UNDERLINE
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text

        # 调用父类的 tokenize 方法获取 token 列表
        tokens = super().tokenize(text, **kwargs)

        # 如果 tokens 长度大于 1 并且第一个 token 是 SPIECE_UNDERLINE 且第二个 token 是特殊 token，则去掉第一个 token
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]

        return tokens
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 sentencepiece 模型对文本进行编码，返回字符串类型的 token 列表
        tokens = self.sp_model.encode(text, out_type=str)
        
        # 检查是否为旧版本或者文本不以 SPIECE_UNDERLINE 或空格开头
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 对字符串添加前缀，例如 "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从编码后的 token 列表中移除 self.unk_token
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用 vocab 将 token 转换为对应的 id
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 vocab 将 index 转换为对应的 token
        return self.sp_model.IdToPiece(index)

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 因为我们手动添加了前缀空格，所以在解码时需要将其移除
        if tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊 token 不使用 sentencepiece 模型进行解码
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构造输出词汇表文件路径，根据是否提供文件名前缀决定文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件与输出文件不是同一个文件且当前词汇表文件存在，则复制当前词汇表文件到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化后的 sp_model 内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出文件路径的元组形式
        return (out_vocab_file,)

    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        **kwargs,
    ) -> BatchEncoding:
        # 检查是否同时提供了 text 和 text_target，若未提供则引发 ValueError
        if text is None and text_target is None:
            raise ValueError("You need to specify either `text` or `text_target`.")
        
        # 如果提供了 text，则根据当前上下文切换输入模式，并调用 call_boxes 方法获取编码结果
        if text is not None:
            # 如果当前不处于目标文本上下文管理器中，则切换到输入模式
            if not self._in_target_context_manager:
                self._switch_to_input_mode()
            encodings = self.call_boxes(text=text, text_pair=text_pair, boxes=boxes, word_labels=word_labels, **kwargs)
        
        # 如果提供了 text_target，则切换到目标文本模式，并调用 _call_one 方法获取目标文本编码结果
        if text_target is not None:
            self._switch_to_target_mode()
            target_encodings = self._call_one(text=text_target, text_pair=text_pair_target, **kwargs)
        
        # 离开目标标记器回到输入模式
        self._switch_to_input_mode()

        # 根据是否提供了 text_target 决定返回编码结果或者目标编码结果或者混合编码结果
        if text_target is None:
            return encodings
        elif text is None:
            return target_encodings
        else:
            encodings["labels"] = target_encodings["input_ids"]
            return encodings
    # 定义一个方法用于处理文本、文本对、文本序列的输入，并根据需要添加边界框信息
    def call_boxes(
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
        # 执行文本和文本对的编码与边界框集成处理
        # 略...
    
    # 批量处理文本或文本对列表，可以选择是否为文本对，同时处理边界框信息
    def batch_encode_plus_boxes(
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
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        # 批量处理输入文本或文本对，并可选地处理它们的边界框标签
        # 略...
        ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_batch_encode_plus_boxes` 进行批量编码
        return self._batch_encode_plus_boxes(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
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

    def encode_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Args:
            Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. Same as doing
            `self.convert_tokens_to_ids(self.tokenize(text))`.
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        # Call the `encode_plus_boxes` method to encode text and optional text_pair with additional parameters
        encoded_inputs = self.encode_plus_boxes(
            text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        # Return only the 'input_ids' from the encoded inputs
        return encoded_inputs["input_ids"]

    def encode_plus_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
        """
        Encode text inputs with associated boxes, word labels, and other optional parameters into model inputs.
        
        Args:
            text: Input text or tokenized input.
            text_pair: Optional second input text or tokenized input.
            boxes: Optional list of bounding boxes for each token.
            word_labels: Optional list of labels corresponding to each token.
            add_special_tokens: Whether to add special tokens (like [CLS], [SEP]) to the encoded inputs.
            padding: Strategy for padding sequences to a certain length.
            truncation: Strategy for truncating sequences longer than `max_length`.
            max_length: Maximum length of the sequences after padding/truncation.
            stride: Stride for splitting the sequence into smaller parts.
            is_split_into_words: Whether the input is already split into words.
            pad_to_multiple_of: Pad the sequence length to a multiple of this value.
            return_tensors: Whether to return tensors (e.g., PyTorch tensors) as outputs.
            return_token_type_ids: Whether to return token type ids as part of the outputs.
            return_attention_mask: Whether to return attention masks as part of the outputs.
            return_overflowing_tokens: Whether to return overflowing tokens beyond max_length.
            return_special_tokens_mask: Whether to return a mask indicating special tokens.
            return_offsets_mapping: Whether to return offsets mapping from original text to tokens.
            return_length: Whether to return the length of the encoded inputs.
            verbose: Whether to print verbose information during encoding.
            **kwargs: Additional keyword arguments for specific encoders.

        Returns:
            Dictionary containing the encoded inputs with specified model inputs (like 'input_ids', 'attention_mask', etc.).
        """
        # Implementation details of encoding process are handled internally by this method
        # and depend on the specific tokenizer and encoding strategy used.
        # Detailed processing steps are not commented here to maintain brevity.
        pass
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            text (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """

        # 获取填充和截断策略，以及其他相关参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_encode_plus_boxes` 进行编码和特殊标记添加
        return self._encode_plus_boxes(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
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
    def _batch_encode_plus_boxes(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],  # 输入参数：可以是单文本、文本对或预处理过的输入列表
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 是否是文本对的标志，可以为None
        boxes: Optional[List[List[List[int]]]] = None,  # 文本框的位置信息，可选参数，默认为None
        word_labels: Optional[List[List[int]]] = None,  # 单词标签，可选参数，默认为None
        add_special_tokens: bool = True,  # 是否添加特殊token，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，可选参数，默认为None
        stride: int = 0,  # 步长，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，可选参数，默认为None
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型IDs，可选参数，默认为None
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，可选参数，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊token掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否显示详细信息，默认为True
        **kwargs,
    ) -> BatchEncoding:
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用_batch_prepare_for_model_boxes方法进行批量编码准备
        batch_outputs = self._batch_prepare_for_model_boxes(
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

        # 返回批量编码结果的BatchEncoding对象
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    # 准备数据以批量输入模型，处理文本或文本对
    def _batch_prepare_for_model_boxes(
        self,
        batch_text_or_text_pairs,  # 批量文本或文本对输入
        is_pair: bool = None,  # 是否为文本对
        boxes: Optional[List[List[int]]] = None,  # 文本框坐标（可选）
        word_labels: Optional[List[List[int]]] = None,  # 单词标签（可选）
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度（可选）
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数（可选）
        return_tensors: Optional[str] = None,  # 返回的张量类型（可选）
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码（可选）
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token掩码
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细输出信息
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

        # Iterate over each index and example in the zipped batch_text_or_text_pairs and boxes
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            batch_text_or_text_pair, boxes_example = example
            
            # Determine if the example is a pair of texts or a single text
            if is_pair:
                text_or_text_pair = batch_text_or_text_pair[0]
            else:
                text_or_text_pair = batch_text_or_text_pair

            # Prepare inputs for the model, including handling special tokens, padding, truncation, etc.
            outputs = self.prepare_for_model_boxes(
                text_or_text_pair,
                batch_text_or_text_pair[1] if is_pair else None,
                boxes_example,
                word_labels=word_labels[idx] if word_labels is not None else None,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # Aggregate outputs into batch_outputs dictionary
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Pad the batch outputs according to specified padding strategy and max length
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # Convert the batch outputs into a BatchEncoding object with specified tensor type
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return the final batch outputs
        return batch_outputs
    def _encode_plus_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
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
    ) -> BatchEncoding:
        # 如果设置了 return_offsets_mapping 参数，则抛出未实现错误，因为 Python tokenizer 不支持此特性
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 调用实例方法 prepare_for_model_boxes 来准备输入数据并编码成模型所需的格式
        return self.prepare_for_model_boxes(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,  # 根据指定的填充策略进行填充
            truncation=truncation_strategy.value,  # 根据指定的截断策略进行截断
            max_length=max_length,  # 设置最大长度限制
            stride=stride,  # 设置步进值
            pad_to_multiple_of=pad_to_multiple_of,  # 设置填充到的倍数
            return_tensors=return_tensors,  # 控制返回的张量类型
            prepend_batch_axis=True,  # 在返回的张量中添加批次维度
            return_attention_mask=return_attention_mask,  # 控制是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 控制是否返回 token 类型 IDs
            return_overflowing_tokens=return_overflowing_tokens,  # 控制是否返回溢出的 tokens
            return_special_tokens_mask=return_special_tokens_mask,  # 控制是否返回特殊 tokens 掩码
            return_length=return_length,  # 控制是否返回编码长度
            verbose=verbose,  # 控制是否打印详细信息
        )

    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    # 定义方法 `prepare_for_model_boxes`，准备输入数据以供模型处理
    def prepare_for_model_boxes(
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
    # 从 `transformers.models.layoutxlm.tokenization_layoutxlm.LayoutXLMTokenizer.truncate_sequences` 复制的方法
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
    # 从 `transformers.models.layoutxlm.tokenization_layoutxlm.LayoutXLMTokenizer._pad` 复制的方法
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
```