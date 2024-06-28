# `.\models\xglm\tokenization_xglm.py`

```py
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for ."""

# 导入标准库 os 和 shutil 中的 copyfile 函数
import os
from shutil import copyfile
# 导入类型提示模块中的相关对象
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，用于处理基于 SentencePiece 的 tokenization
import sentencepiece as spm

# 导入父类 PreTrainedTokenizer 和 logging 工具
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取当前模块的 logger 对象
logger = logging.get_logger(__name__)

# 定义 SentencePiece 使用的特殊 token
SPIECE_UNDERLINE = "▁"

# 定义 vocab 文件的名称映射，包含一个 vocab 文件的标准名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 定义预训练模型中的 vocab 文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/sentencepiece.bpe.model",
    }
}

# 定义预训练模型的位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/xglm-564M": 2048,
}

# 定义 XGLMTokenizer 类，继承自 PreTrainedTokenizer 类
class XGLMTokenizer(PreTrainedTokenizer):
    """
    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
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

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
    """

    # 从全局常量中获取词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射，指定了每个特殊词汇的文件路径
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练的位置嵌入大小赋值给max_model_input_sizes变量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数，用于创建一个新的实例对象
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果sp_model_kwargs为None，则设为一个空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 兼容性处理，与原始分词器的兼容
        self.num_madeup_words = 7
        madeup_words = [f"<madeupword{i}>" for i in range(self.num_madeup_words)]

        # 获取kwargs中的additional_special_tokens列表，如果不存在则创建一个空列表
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        # 将madeup_words中未在additional_special_tokens中的单词添加到additional_special_tokens中
        kwargs["additional_special_tokens"] += [
            word for word in madeup_words if word not in kwargs["additional_special_tokens"]
        ]

        # 使用指定的参数初始化SentencePieceProcessor对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到self.sp_model中
        self.sp_model.Load(str(vocab_file))
        # 将vocab_file保存到self.vocab_file中
        self.vocab_file = vocab_file

        # 原始fairseq词汇表和spm词汇表必须是“对齐”的：
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # 在原始fairseq词汇表和spm词汇表之间进行对齐，第一个“真实”标记“,”在fairseq词汇表中位置为4，在spm词汇表中位置为3
        self.fairseq_offset = 1

        # 模仿fairseq的token-to-id对齐，对前4个token进行映射
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # 计算spm词汇表的大小
        sp_size = len(self.sp_model)
        # 创建一个字典，将madeup_words映射到fairseq词汇表之后的位置
        madeup_words = {f"<madeupword{i}>": sp_size + i + self.fairseq_offset for i in range(self.num_madeup_words)}
        self.fairseq_tokens_to_ids.update(madeup_words)

        # 创建一个反向映射，从token id到token的映射
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 调用父类的初始化方法，传入相应的参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    # 序列化对象时调用的方法，返回对象的状态信息
    def __getstate__(self):
        state = self.__dict__.copy()
        # 将self.sp_model设置为None，因为它不能直接被序列化
        state["sp_model"] = None
        # 将self.sp_model的序列化模型信息保存到state中
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # 反序列化对象时调用的方法，用于恢复对象的状态信息
    def __setstate__(self, d):
        # 恢复对象的状态信息
        self.__dict__ = d

        # 向后兼容性处理
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用保存的sp_model_proto信息重新初始化self.sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列或序列对构建用于序列分类任务的模型输入，通过连接并添加特殊标记。XLM-RoBERTa 序列的格式如下：

        - 单序列： `<s> X </s>`
        - 序列对： `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 带有适当特殊标记的输入 ID 列表。
        """

        if token_ids_1 is None:
            # 如果只有一个序列，返回带有 SEP 特殊标记的 token_ids_0
            return [self.sep_token_id] + token_ids_0
        sep = [self.sep_token_id]
        # 如果有两个序列，返回连接的序列，每个序列末尾带有两个 SEP 特殊标记
        return sep + token_ids_0 + sep + sep + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的 token 列表中检索序列 ID。在使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                token 列表是否已经使用特殊标记格式化为模型。

        Returns:
            `List[int]`: 整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            # 如果已经有特殊标记，调用父类方法获取特殊标记掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # 如果只有一个序列，返回一个序列首部带有特殊标记的掩码
            return [1] + ([0] * len(token_ids_0))
        # 如果有两个序列，返回连接的序列，每个序列首尾带有特殊标记的掩码
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        从序列或序列对创建用于区分 token 类型的 token 类型 ID。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个序列的 ID 列表，用于序列对。

        Returns:
            无返回值，该方法会生成用于区分 token 类型的 token 类型 ID。
        """
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

        # Define a separator token list containing `self.sep_token_id`
        sep = [self.sep_token_id]

        # Check if token_ids_1 is None; if so, return a list of zeros based on the length of `sep + token_ids_0`
        if token_ids_1 is None:
            return len(sep + token_ids_0) * [0]
        
        # If token_ids_1 is provided, return a list of zeros based on the extended length of tokens including separators
        return len(sep + token_ids_0 + sep + sep + token_ids_1) * [0]

    @property
    def vocab_size(self):
        # Calculate and return the total vocabulary size, including fairseq offsets and made-up words
        return len(self.sp_model) + self.fairseq_offset + self.num_madeup_words

    def get_vocab(self):
        # Create a dictionary mapping from token strings to their corresponding IDs within the vocabulary
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)  # Update with additional tokens from `added_tokens_encoder`
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize the input `text` using `sp_model` and return a list of token strings
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Return the offset ID for unknown tokens if SP model returns 0 (indicating unknown token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) into a single string."""
        # Concatenate tokens into a single string, replacing SPIECE_UNDERLINE with spaces and stripping leading/trailing spaces
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Ensure `save_directory` exists; if not, log an error and return None
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Define the output vocabulary file path based on `save_directory` and `filename_prefix`
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current `vocab_file` path is different from `out_vocab_file` and exists, copy `vocab_file` to `out_vocab_file`
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # If `vocab_file` does not exist, write `sp_model.serialized_model_proto()` content to `out_vocab_file`
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
```