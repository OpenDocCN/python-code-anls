# `.\models\rembert\tokenization_rembert.py`

```
# coding=utf-8
# 声明编码方式为UTF-8，确保支持各种字符集

# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
# 版权声明：HuggingFace 团队和 HuggingFace 公司保留所有权利。

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 根据 Apache License, Version 2.0 许可协议授权，详见上述链接

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非有适用法律要求或书面同意，本软件按"原样"分发，不附带任何明示或暗示的担保或条件。
# 请参阅许可协议了解更多信息。

"""Tokenization classes for RemBERT."""
# 用于 RemBERT 的分词类

import os
from shutil import copyfile
from typing import List, Optional, Tuple

import sentencepiece as spm

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 导入所需模块和类

logger = logging.get_logger(__name__)

# 获取 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model"}

# 定义词汇文件的名称映射，指定为 sentencepiece.model

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/rembert": "https://huggingface.co/google/rembert/resolve/main/sentencepiece.model",
    },
}

# 预训练模型的词汇文件映射，指定 RemBERT 模型对应的 sentencepiece.model 的下载地址

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/rembert": 256,
}

# 预训练模型的位置嵌入尺寸映射，指定 RemBERT 模型的位置嵌入尺寸为 256

class RemBertTokenizer(PreTrainedTokenizer):
    """
    Construct a RemBERT tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 构建 RemBERT 分词器，基于 SentencePiece

    def __init__(
        self,
        vocab_file,
        *,
        tokenizer_file=None,
        do_lower_case=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        **kwargs
    ):
        # 初始化方法，接收参数包括词汇文件路径、是否转换为小写、特殊标记等

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            **kwargs
        )

        # 调用父类的初始化方法，设置分词器的基本属性
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """
    # 定义类变量，包含预定义的文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 包含预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 包含预训练位置嵌入大小的字典
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 初始化方法，接收多个参数，包括词汇文件和特殊标记的设置
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
        # 如果 mask_token 是字符串类型，则设置 lstrip=True 和 rstrip=False 的 AddedToken 对象
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 初始化模型的参数
        self.do_lower_case = do_lower_case  # 是否将输入文本转换为小写
        self.remove_space = remove_space    # 是否移除输入文本中的空格
        self.keep_accents = keep_accents    # 是否保留输入文本中的重音符号
        self.vocab_file = vocab_file        # 词汇表文件的路径

        # 使用 SentencePieceProcessor 初始化 self.sp_model 对象，并加载词汇表文件
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化方法，设置模型的基本参数
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小，即 self.sp_model 中词汇的数量
        return len(self.sp_model)

    def get_vocab(self):
        # 构建并返回词汇表，包括从 id 到 token 的映射和额外添加的 tokens 编码器
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        # 返回对象的状态字典，不包括 sp_model，用于序列化对象
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        # 从状态字典中恢复对象的状态，重新加载 sp_model
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text, sample=False):
        """Tokenize a string."""
        # 使用 sp_model 对文本进行分词，返回分词后的结果 pieces
        pieces = self.sp_model.EncodeAsPieces(text)
        return pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据 token 转换成对应的 id，使用 sp_model 的 PieceToId 方法
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 id 转换成对应的 token，使用 sp_model 的 IdToPiece 方法
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        # 将 tokens 转换成字符串，使用 sp_model 的 decode_pieces 方法
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        # 构建包含特殊 token 的输入，这里不包含实现细节，只是声明方法签名
        pass
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from two sequences. Token type IDs are binary masks identifying the type of each token
        in the sequence: 0 for the first sequence, 1 for the second sequence.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs where each ID corresponds to the type of its respective token.
        """

        if token_ids_1 is None:
            # If there's no second sequence, all tokens belong to the first sequence (type ID 0)
            return [0] * len(token_ids_0)

        # Create token type IDs for a pair of sequences
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RemBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define the separation token ID and the classification token ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If there is no second sequence provided, return a mask with 0s for only the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Otherwise, return a mask with 0s for the first sequence and 1s for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if the save_directory exists; if not, log an error and return None
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Copy the vocabulary file to the output directory if it's different from the current location
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # If the vocabulary file doesn't exist, write the serialized sp_model to the output file
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```