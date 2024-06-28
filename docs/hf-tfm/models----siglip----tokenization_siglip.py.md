# `.\models\siglip\tokenization_siglip.py`

```py
# coding=utf-8
# 设定文件编码为 UTF-8

# Copyright 2024 The HuggingFace Inc. team.
# 版权声明，版权归 The HuggingFace Inc. 团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 授权许可

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则依据 "AS IS" 分发本软件，
# 无论是明示的还是隐含的，不包括任何形式的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证以了解有关权限和限制的详细信息

""" Tokenization class for SigLIP model."""
# 为 SigLIP 模型设计的分词类

import os
import re
import string
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入 sentencepiece 库

from ...convert_slow_tokenizer import import_protobuf  # 导入从 protobuf 导入的函数
from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器基类
from ...tokenization_utils_base import AddedToken  # 导入添加的 token 类型


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput  # 导入文本输入类型

from ...utils import logging, requires_backends  # 导入日志记录和后端要求


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}  # 词汇文件的名称映射

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/siglip-base-patch16-224": "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/spiece.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/siglip-base-patch16-224": 256,
}

SPIECE_UNDERLINE = "▁"  # SentencePiece 使用的前缀符号


class SiglipTokenizer(PreTrainedTokenizer):
    """
    Construct a Siglip tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # Siglip 分词器的构造函数，基于 SentencePiece

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        # 初始化函数，设置分词器的各种参数
        pass
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"</s>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        model_max_length (`int`, *optional*, defaults to 64):
            The maximum length (in number of tokens) for model inputs.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
    """

    # Define constants related to tokenizer vocabulary files and models
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # Constructor method for the Tokenizer class
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="</s>",
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        model_max_length=64,
        do_lower_case=True,
        **kwargs,
    ):
    @property
    # 返回词汇表的大小，基于 SentencePiece 模型的词汇量
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab
    def get_vocab(self):
        # 创建词汇表字典，将词汇 ID 映射到对应的词汇
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新词汇表字典，加入额外的特殊标记的映射
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> torch.Tensor:
        # 返回特殊标记的掩码张量，用于标记哪些是特殊标记
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
        # If the token list already has special tokens, delegate to the base class's method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Calculate special tokens mask for the normal case (with special tokens)
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]  # No sequence pair, return mask for token_ids_0
        else:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]  # Return mask for token_ids_0 and token_ids_1

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._add_eos_if_not_present
    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        # Check if eos_token_id is already present at the end of token_ids
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            # Warn if eos_token is already present to prevent duplication in future versions
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids  # Return unchanged token_ids if eos_token is already present
        else:
            return token_ids + [self.eos_token_id]  # Add eos_token_id to token_ids and return

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
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
        eos = [self.eos_token_id]  # Create a list containing eos_token_id

        # Calculate token type ids assuming the presence of EOS tokens
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]  # Return zero mask for token_ids_0 + eos
        else:
            return len(token_ids_0 + eos + token_ids_1 + eos) * [0]  # Return zero mask for token_ids_0 + eos + token_ids_1 + eos

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        Build model inputs from a sequence or a pair of sequences, including adding special tokens.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs for the second sequence in a pair.

        Returns:
            `List[int]`: A list of token IDs with added special tokens.
        """
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
        # Ensure the first sequence ends with an end-of-sequence token if not already present
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        
        # If only one sequence is provided, return it with added special tokens
        if token_ids_1 is None:
            return token_ids_0
        else:
            # Ensure the second sequence ends with an end-of-sequence token if not already present
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            # Concatenate both sequences with their respective special tokens added
            return token_ids_0 + token_ids_1

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.__getstate__
    def __getstate__(self):
        # Create a copy of the object's state dictionary
        state = self.__dict__.copy()
        # Set the 'sp_model' attribute to None to avoid pickling issues
        state["sp_model"] = None
        return state

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.__setstate__
    def __setstate__(self, d):
        # Restore the object's state from the provided dictionary
        self.__dict__ = d

        # Ensure backward compatibility by initializing 'sp_model_kwargs' if absent
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # Initialize 'sp_model' using SentencePieceProcessor with saved parameters
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def remove_punctuation(self, text: str) -> str:
        # Remove all punctuation characters from the input text
        return text.translate(str.maketrans("", "", string.punctuation))

    # source: https://github.com/google-research/big_vision/blob/3b8e5ab6ad4f96e32b32826f9e1b8fd277914f9c/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94
    def canonicalize_text(self, text, *, keep_punctuation_exact_string=None):
        """Returns canonicalized `text` (puncuation removed).

        Args:
            text (`str`):
                String to be canonicalized.
            keep_punctuation_exact_string (`str`, *optional*):
                If provided, then this exact string is kept. For example providing '{}' will keep any occurrences of '{}'
                (but will still remove '{' and '}' that appear separately).
        """
        if keep_punctuation_exact_string:
            # Replace occurrences of 'keep_punctuation_exact_string' with itself after removing punctuation
            text = keep_punctuation_exact_string.join(
                self.remove_punctuation(part) for part in text.split(keep_punctuation_exact_string)
            )
        else:
            # Remove all punctuation characters from the entire text
            text = self.remove_punctuation(text)
        
        # Replace multiple spaces with a single space, then strip leading and trailing spaces
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
    # 将文本转换为标记列表的方法
    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens.
        """
        # 使用父类的方法将文本转换为标记列表
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        # 如果标记数大于1且第一个标记是SPIECE_UNDERLINE，并且第二个标记是特殊标记之一，则移除第一个标记
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    @property
    # 从transformers.models.t5.tokenization_t5.T5Tokenizer.unk_token_length中复制而来
    def unk_token_length(self):
        # 返回未知标记的编码长度
        return len(self.sp_model.encode(str(self.unk_token)))

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE.

        For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give `['H', 'e', 'y']` instead of `['▁He', 'y']`.

        Thus we always encode `f"{unk_token}text"` and strip the `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 规范化文本，保持标点符号的精确性
        text = self.canonicalize_text(text, keep_punctuation_exact_string=None)
        # 使用句子片段模型对文本进行编码
        tokens = self.sp_model.encode(text, out_type=str)

        # 1. 编码字符串 + 前缀，例如 "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从 ['<','unk','>', '▁Hey'] 中移除 self.unk_token
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer._convert_token_to_id中复制而来
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将标记转换为标识符
        return self.sp_model.piece_to_id(token)

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer._convert_id_to_token中复制而来
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将标识符转换为标记
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊标记不使用句子片段模型解码
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
    # 从 transformers.models.t5.tokenization_t5.T5Tokenizer.save_vocabulary 复制而来的方法
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 拼接输出的词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 如果当前词汇文件路径与输出路径不同，并且当前词汇文件存在，则复制当前词汇文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇文件不存在，则将序列化后的模型内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        
        # 返回输出文件的路径元组
        return (out_vocab_file,)
```