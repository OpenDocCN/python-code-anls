# `.\models\albert\tokenization_albert.py`

```py
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
""" Tokenization classes for ALBERT model."""

# 引入必要的库和模块
import os
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入句子分词模块

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入自定义的分词工具类
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)
# 定义 ALBERT 模型的词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert/albert-base-v1": "https://huggingface.co/albert/albert-base-v1/resolve/main/spiece.model",
        "albert/albert-large-v1": "https://huggingface.co/albert/albert-large-v1/resolve/main/spiece.model",
        "albert/albert-xlarge-v1": "https://huggingface.co/albert/albert-xlarge-v1/resolve/main/spiece.model",
        "albert/albert-xxlarge-v1": "https://huggingface.co/albert/albert-xxlarge-v1/resolve/main/spiece.model",
        "albert/albert-base-v2": "https://huggingface.co/albert/albert-base-v2/resolve/main/spiece.model",
        "albert/albert-large-v2": "https://huggingface.co/albert/albert-large-v2/resolve/main/spiece.model",
        "albert/albert-xlarge-v2": "https://huggingface.co/albert/albert-xlarge-v2/resolve/main/spiece.model",
        "albert/albert-xxlarge-v2": "https://huggingface.co/albert/albert-xxlarge-v2/resolve/main/spiece.model",
    }
}

# 预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "albert/albert-base-v1": 512,
    "albert/albert-large-v1": 512,
    "albert/albert-xlarge-v1": 512,
    "albert/albert-xxlarge-v1": 512,
    "albert/albert-base-v2": 512,
    "albert/albert-large-v2": 512,
    "albert/albert-xlarge-v2": 512,
    "albert/albert-xxlarge-v2": 512,
}

# SentencePiece 分词器特有的下划线符号
SPIECE_UNDERLINE = "▁"

# ALBERT 模型的分词器类，继承自 PreTrainedTokenizer 类
class AlbertTokenizer(PreTrainedTokenizer):
    """
    Construct an ALBERT tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # 词汇文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # sp_model 是 SentencePieceProcessor 对象，用于字符串、token 和 ID 的转换
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sp_model = None  # 初始化 sp_model 为空
    # 初始化函数，用于初始化一个新的对象实例
    def __init__(
        self,
        vocab_file,  # 词汇文件的路径
        do_lower_case=True,  # 是否将输入文本转换为小写，默认为True
        remove_space=True,  # 是否移除输入文本中的空格，默认为True
        keep_accents=False,  # 是否保留输入文本中的重音符号，默认为False
        bos_token="[CLS]",  # 开始标记（Beginning of Sentence），默认为"[CLS]"
        eos_token="[SEP]",  # 结束标记（End of Sentence），默认为"[SEP]"
        unk_token="<unk>",  # 未知标记（Unknown Token），默认为"<unk>"
        sep_token="[SEP]",  # 分隔标记（Separator Token），默认为"[SEP]"
        pad_token="<pad>",  # 填充标记（Padding Token），默认为"<pad>"
        cls_token="[CLS]",  # 类别标记（Class Token），默认为"[CLS]"
        mask_token="[MASK]",  # 掩码标记（Mask Token），默认为"[MASK]"
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # SentencePiece 模型的参数，可选字典类型，默认为None
        **kwargs,  # 其他额外的关键字参数
    ) -> None:
        # 将掩码标记（mask_token）处理成一个 AddedToken 对象，具有特定的处理属性
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 如果未提供 sp_model_kwargs，则初始化为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置对象的各种属性值
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # 使用 SentencePieceProcessor 初始化一个 sp_model 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 调用父类的初始化方法，传递参数和额外的关键字参数
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
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    # vocab_size 属性，返回 sp_model 中词汇的数量
    @property
    def vocab_size(self) -> int:
        return len(self.sp_model)

    # 获取词汇表的方法，返回词汇到索引的字典
    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # __getstate__ 方法，用于对象的序列化状态，排除 sp_model 以防止对象过大
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None  # 将 sp_model 设置为 None，不包含在序列化状态中
        return state

    # __setstate__ 方法，用于对象的反序列化，重新初始化 sp_model
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容，如果对象没有 sp_model_kwargs 属性，则设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 SentencePieceProcessor 重新初始化 sp_model 并加载 vocab_file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 文本预处理方法，根据对象的属性对输入文本进行处理并返回处理后的文本
    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())  # 移除多余的空格
        else:
            outputs = inputs

        outputs = outputs.replace("``", '"').replace("''", '"')  # 替换双引号

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)  # 标准化 unicode 字符串
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])  # 移除重音符号

        if self.do_lower_case:
            outputs = outputs.lower()  # 将文本转换为小写

        return outputs
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string."""
        # 对输入文本进行预处理
        text = self.preprocess_text(text)
        # 使用句子片段模型对文本进行编码，输出为字符串列表
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 处理特殊情况的逻辑，参见 https://github.com/google-research/bert/blob/master/README.md#tokenization
                # 当遇到形如 `9,9` 的情况时，确保正确分割为 ['▁9', ',', '9']，而非 [`_9,`, '9']
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将token转换为对应的ID
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将ID转换为对应的token
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊的token不被句子片段模型解码
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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        """Build model inputs from a sequence or a pair of sequence for BERT."""
        # 实现构建适用于BERT模型的特殊token输入
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from token list indices.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: A list of token type IDs (0 or 1) corresponding to each token in the input sequences.
        """
        # Define token type ID for the first sequence (0)
        token_type_ids = [0] * len(token_ids_0)
        
        if token_ids_1 is not None:
            # Define token type ID for the second sequence (1)
            token_type_ids += [1] * len(token_ids_1)
        
        return token_type_ids
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of token IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of token IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define separation and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one sequence is provided (token_ids_1 is None), return a mask with 0s for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # Otherwise, concatenate both sequences and return a mask with 0s for the first sequence and 1s for the second
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files to the specified directory.

        Args:
            save_directory (`str`):
                Directory path where the vocabulary files will be saved.
            filename_prefix (`str`, *optional*):
                Optional prefix to prepend to the vocabulary file names.

        Returns:
            `Tuple[str]`: Tuple containing the path of the saved vocabulary file.
        """
        # Check if the save directory exists; if not, log an error and return
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # Determine the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocabulary file is not the same as the output file and exists, copy it to the output location
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # If the current vocabulary file doesn't exist, write the serialized model to the output file
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # Return the path of the saved vocabulary file
        return (out_vocab_file,)
```