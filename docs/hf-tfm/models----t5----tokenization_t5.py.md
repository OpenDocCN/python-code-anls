# `.\models\t5\tokenization_t5.py`

```
# coding=utf-8
# Copyright 2018 T5 Authors and HuggingFace Inc. team.
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
# limitations under the License.
""" Tokenization class for model T5."""

# Import necessary standard library modules
import os                       # 导入操作系统相关功能的模块
import re                       # 导入正则表达式模块
import warnings                 # 导入警告处理模块
from shutil import copyfile     # 从 shutil 模块导入 copyfile 函数
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple   # 导入类型提示相关的模块

# Import SentencePiece library for tokenization
import sentencepiece as spm    # 导入 SentencePiece 库

# Import functions and classes from Transformers package
from ...convert_slow_tokenizer import import_protobuf     # 从特定路径导入 import_protobuf 函数
from ...tokenization_utils import PreTrainedTokenizer    # 从 tokenization_utils 模块导入 PreTrainedTokenizer 类
from ...tokenization_utils_base import AddedToken         # 从 tokenization_utils_base 模块导入 AddedToken 类

# Check type hints only during static type checking
if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput    # 仅在静态类型检查时导入 TextInput 类
from ...utils import logging    # 从 utils 模块导入 logging 模块

# Get logger instance for logging messages
logger = logging.get_logger(__name__)

# Define constant for vocabulary file names
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# Define mapping of pretrained model names to their respective vocabulary file URLs
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google-t5/t5-small": "https://huggingface.co/google-t5/t5-small/resolve/main/spiece.model",
        "google-t5/t5-base": "https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model",
        "google-t5/t5-large": "https://huggingface.co/google-t5/t5-large/resolve/main/spiece.model",
        "google-t5/t5-3b": "https://huggingface.co/google-t5/t5-3b/resolve/main/spiece.model",
        "google-t5/t5-11b": "https://huggingface.co/google-t5/t5-11b/resolve/main/spiece.model",
    }
}

# TODO(PVP) - this should be removed in Transformers v5
# Define sizes of positional embeddings for different pretrained T5 models
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google-t5/t5-small": 512,
    "google-t5/t5-base": 512,
    "google-t5/t5-large": 512,
    "google-t5/t5-3b": 512,
    "google-t5/t5-11b": 512,
}

# Define a special token used by SentencePiece for word beginning
SPIECE_UNDERLINE = "▁"


class T5Tokenizer(PreTrainedTokenizer):
    """
    Construct a T5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES     # Assign constant for vocabulary file names to class attribute
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP   # Assign vocabulary file URL mapping to class attribute
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES    # Assign positional embeddings sizes to class attribute
    model_input_names = ["input_ids", "attention_mask"]   # Define input names required by the model
    # 初始化方法，用于创建一个新的 T5Tokenizer 对象
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        legacy=None,
        add_prefix_space=True,
        **kwargs,
    ):
        # 从参数中初始化 T5Tokenizer 对象的属性
        # 设置结束符号，默认为 "</s>"
        # 设置未知符号，默认为 "<unk>"
        # 设置填充符号，默认为 "<pad>"
        # 设置额外的 ID 数量，默认为 100
        # 设置额外的特殊标记列表
        # 设置 SentencePiece 模型的关键字参数
        # 设置是否使用旧版本兼容模式，默认为 None
        # 设置是否在空格前添加前缀，默认为 True
        pass  # 这里是一个占位符，表示初始化方法暂时不执行任何操作

    # 静态方法，返回 SentencePiece 处理器对象
    @staticmethod
    def get_spm_processor(self, from_slow=False):
        # 根据给定的 sp_model_kwargs 创建 SentencePieceProcessor 对象
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        
        # 如果处于兼容模式或者 from_slow 为 True，则不依赖于 protobuf，直接加载词汇文件
        if self.legacy or from_slow:
            # 从磁盘加载 SentencePiece 词汇文件
            tokenizer.Load(self.vocab_file)
            return tokenizer

        # 否则，使用新的行为，依赖于 protobuf 加载模型
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            # 导入 protobuf 模型定义
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            # 反序列化 protobuf 模型
            model = model_pb2.ModelProto.FromString(sp_model)
            # 设置规范化器规范
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            # 序列化模型为字节流
            sp_model = model.SerializeToString()
            # 使用序列化后的模型初始化 tokenizer
            tokenizer.LoadFromSerializedProto(sp_model)
        
        return tokenizer

    # 属性方法，返回词汇表大小
    @property
    def vocab_size(self):
        # 获取 SentencePieceProcessor 对象的词汇大小
        return self.sp_model.get_piece_size()
    def get_vocab(self):
        """
        构建词汇表字典，将词汇索引映射到对应的词汇。
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 将额外添加的特殊词汇编码器更新到词汇表中
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        检索没有添加特殊标记的令牌列表的序列ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经包含了模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 普通情况：一些特殊标记
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def get_sentinel_tokens(self):
        """
        返回包含特殊标记的令牌列表。
        """
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )

    def get_sentinel_token_ids(self):
        """
        返回特殊标记的令牌 ID 列表。
        """
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """
        如果用户尚未添加 EOS 标记，则不再添加 EOS。
        """
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列中创建令牌类型 ID 列表。
        """
    def create_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence for sequence pairs.

        Returns:
            `List[int]`: List of zeros representing the mask.
        """
        # Define the end-of-sequence token
        eos = [self.eos_token_id]

        # If only one sequence is provided, return a mask for it
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        
        # If two sequences are provided, concatenate them with special tokens and return a mask for the combined sequence
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens added.
        """
        # Add end-of-sequence token if not already present
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        
        # If only one sequence is provided, return it with special tokens added
        if token_ids_1 is None:
            return token_ids_0
        else:
            # Add end-of-sequence token if not already present for the second sequence
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            # Concatenate both sequences with special tokens added
            return token_ids_0 + token_ids_1

    def __getstate__(self):
        # Copy the object's state dictionary
        state = self.__dict__.copy()
        # Set 'sp_model' attribute to None to avoid serializing it
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # Restore object's state from the dictionary 'd'
        self.__dict__ = d

        # For backward compatibility, initialize 'sp_model_kwargs' if not already present
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # Load SentencePiece processor from 'vocab_file'
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. Adds a prefix token if `self.legacy` is False and the first token is not special.
        """
        # Call superclass's tokenize method if legacy mode is enabled or text is empty
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        # Replace SPIECE_UNDERLINE with space in the text
        text = text.replace(SPIECE_UNDERLINE, " ")

        # Add SPIECE_UNDERLINE prefix to the text if required
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text

        # Tokenize the text using superclass's tokenize method
        tokens = super().tokenize(text, **kwargs)

        # Remove the prefix token if it is the first token and not a special token
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]

        return tokens

    @property
    def unk_token_length(self):
        """
        Calculate the length of the unknown token.
        """
        return len(self.sp_model.encode(str(self.unk_token)))
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 sentencepiece 模型对文本进行编码，返回字符串类型的编码结果
        tokens = self.sp_model.encode(text, out_type=str)
        # 如果是传统模式或者文本不以 SPIECE_UNDERLINE 或空格开头，则直接返回编码结果
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 将文本添加 unk_token 前缀，例如 "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从编码结果中去除 unk_token，例如 ['<','unk','>', '▁Hey']
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 index 转换为对应的 token
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 因为我们手动添加了前缀空格，所以在解码时需要将其去除
        if tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊 token 不通过 sentencepiece 模型进行解码
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
    # 保存词汇表到指定目录，返回保存的文件路径元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建输出的词汇表文件路径，包括可选的前缀和固定的文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标路径不同，并且当前词汇表文件存在，则复制当前词汇表文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的词汇表模型内容写入到目标文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的词汇表文件路径的元组
        return (out_vocab_file,)
```